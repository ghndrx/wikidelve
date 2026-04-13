"""
In-memory full-text search.

Replaces SQLite FTS5. The index is rebuilt from the storage backend on
every cold start (and on demand) and lives entirely in process memory.
This works because:

  - Article corpus is small (~10MB raw text per KB at our scale).
  - We don't have a row-level update problem — articles are append-only
    from the user's perspective and rebuilds are cheap.
  - It removes a hard dependency on SQLite-on-disk and unblocks the
    cloud-native (S3 + DynamoDB) backend, which can't ship FTS5.

The implementation:

  - Inverted index: term → list of (doc_id, term frequency) postings
  - Doc index:      doc_id → article metadata + body
  - BM25 ranking with k1=1.5, b=0.75
  - Title / summary / tags get a 3× term-frequency boost over body terms
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
from collections import defaultdict

from app import storage
from app.wiki import parse_frontmatter, _to_str, _to_str_list

logger = logging.getLogger("kb-service.search")


# BM25 parameters — tuned for technical markdown corpora where most
# documents are long (1-2k words), mixed with short stub articles.
#
# - k1 controls term-frequency saturation. Lower values (~1.2) saturate
#   TF earlier, which rewards documents that mention a query term a few
#   times without heavily over-weighting documents that repeat it
#   dozens of times. 1.5 is the BM25 textbook default.
# - b controls length normalisation. 0.75 is the standard; lowering it
#   toward 0 would stop penalising long articles (useful if our corpus
#   has wildly variable doc lengths), raising it toward 1 would penalise
#   long articles more aggressively.
#
# Both are env-overridable so you can A/B tune without touching code.
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "to", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "at",
    "from", "this", "that", "these", "those", "it", "its", "but", "not",
    "if", "then", "than", "so", "do", "does", "did", "have", "has", "had",
})


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in _STOP_WORDS
    ]


# ---------------------------------------------------------------------------
# Index data structures
# ---------------------------------------------------------------------------


class _Index:
    """Single in-memory inverted index across every KB."""

    def __init__(self) -> None:
        # doc_id → metadata
        self.docs: list[dict] = []
        # term → list of (doc_id, weighted tf)
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        # doc_id → total weighted token count (for BM25)
        self.doc_lens: list[float] = []
        # term → idf (cached)
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.built_at: float = 0.0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.docs)

    # --- Build / update ----------------------------------------------------

    def add_doc(self, meta: dict, body: str) -> int:
        title_terms = _tokenize(meta.get("title", "")) * 3
        summary_terms = _tokenize(meta.get("summary", "")) * 3
        tags_terms = _tokenize(" ".join(meta.get("tags", []))) * 3
        body_terms = _tokenize(body)

        all_terms = title_terms + summary_terms + tags_terms + body_terms

        tf: dict[str, float] = defaultdict(float)
        for t in all_terms:
            tf[t] += 1.0

        doc_id = len(self.docs)
        self.docs.append({
            "slug": meta.get("slug", ""),
            "kb": meta.get("kb", ""),
            "title": meta.get("title", ""),
            "summary": meta.get("summary", ""),
            "tags": meta.get("tags", []),
            "body": body,
        })
        self.doc_lens.append(float(len(all_terms)))

        for term, freq in tf.items():
            self.postings[term].append((doc_id, freq))

        return doc_id

    def finalize(self) -> None:
        n = len(self.docs)
        self.avgdl = (sum(self.doc_lens) / n) if n else 0.0
        self.idf.clear()
        for term, postings in self.postings.items():
            df = len(postings)
            # BM25 idf with the +1 smoothing variant
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        import time
        self.built_at = time.time()

    # --- Query -------------------------------------------------------------

    def query(
        self,
        text: str,
        kb: str | None = None,
        limit: int = 20,
    ) -> list[tuple[float, int]]:
        terms = _tokenize(text)
        if not terms:
            return []

        k1 = BM25_K1
        b = BM25_B
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id, tf in postings:
                if kb and self.docs[doc_id]["kb"] != kb:
                    continue
                dl = self.doc_lens[doc_id] or 1.0
                norm = 1.0 - b + b * (dl / (self.avgdl or 1.0))
                denom = tf + k1 * norm
                scores[doc_id] += idf * (tf * (k1 + 1.0)) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [(score, doc_id) for doc_id, score in ranked]


# ---------------------------------------------------------------------------
# Module-level index + lock
# ---------------------------------------------------------------------------


_index: _Index = _Index()
_build_lock = asyncio.Lock()


def _normalize_doc(slug: str, raw: str, kb: str) -> tuple[dict, str]:
    meta_dict, body = parse_frontmatter(raw)
    meta = {
        "slug": slug,
        "kb": kb,
        "title": _to_str(meta_dict.get("title")) or slug.replace("-", " ").title(),
        "summary": _to_str(meta_dict.get("summary")),
        "tags": _to_str_list(meta_dict.get("tags")),
    }
    return meta, body


def _build_sync() -> _Index:
    """Synchronous rebuild — runs on a worker thread so it doesn't
    block the asyncio event loop. Called via ``asyncio.to_thread``."""
    new_index = _Index()
    for kb_name in storage.list_kbs():
        for slug, raw in storage.iter_articles(kb_name, subdir="wiki"):
            meta, body = _normalize_doc(slug, raw, kb_name)
            new_index.add_doc(meta, body)
    new_index.finalize()
    return new_index


async def build_search_index() -> int:
    """(Re)build the in-memory FTS index from every KB in the storage backend.

    Runs the rebuild on a worker thread so the event loop stays responsive
    while the (thread-pooled) S3 fetches happen. Safe to call concurrently.
    """
    async with _build_lock:
        new_index = await asyncio.to_thread(_build_sync)
        global _index
        _index = new_index
        logger.info("FTS in-memory index rebuilt: %d docs", len(new_index))
        return len(new_index)


async def update_article_index(kb_name: str, slug: str) -> None:
    """Refresh a single article in the index. Currently triggers a full
    rebuild — cheap at our scale (~700 docs ⇒ <2s) and avoids tombstone
    bookkeeping for the inverted lists."""
    await build_search_index()


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def _make_snippet(body: str, query: str, length: int = 240) -> str:
    if not body:
        return ""
    terms = [t for t in _tokenize(query) if t]
    body_lower = body.lower()
    pos = -1
    for term in terms:
        pos = body_lower.find(term)
        if pos != -1:
            break
    if pos == -1:
        return (body[:length] + "...") if len(body) > length else body
    start = max(0, pos - length // 2)
    end = min(len(body), start + length)
    snippet = body[start:end].replace("\n", " ")
    if start > 0:
        snippet = "..." + snippet
    if end < len(body):
        snippet = snippet + "..."
    return snippet


async def search_fts(query: str, limit: int = 20, kb: str | None = None) -> list[dict]:
    """Search the in-memory index. Async signature kept for compatibility
    with callers that may add async work later."""
    if not query or not query.strip():
        return []
    if len(_index) == 0:
        await build_search_index()
    hits = _index.query(query, kb=kb, limit=limit)
    out: list[dict] = []
    for score, doc_id in hits:
        d = _index.docs[doc_id]
        out.append({
            "slug": d["slug"],
            "kb": d["kb"],
            "title": d["title"],
            "summary": d["summary"],
            "tags": d["tags"],
            "snippet": _make_snippet(d["body"], query),
            "score": float(score),
        })
    return out


async def search_autocomplete(prefix: str, limit: int = 8) -> list[dict]:
    """Title-prefix autocomplete over the in-memory index."""
    if not prefix or len(prefix) < 2:
        return []
    if len(_index) == 0:
        await build_search_index()
    needle = prefix.lower()
    out: list[dict] = []
    for d in _index.docs:
        if d["title"].lower().startswith(needle):
            out.append({"slug": d["slug"], "kb": d["kb"], "title": d["title"]})
            if len(out) >= limit:
                break
    return out


def search_kb(query: str) -> list[dict]:
    """Backward-compat sync wrapper used by older web routes."""
    if not query or not query.strip():
        return []
    if len(_index) == 0:
        # We can't await from a sync function — just return empty until the
        # async startup hook has built the index.
        return []
    hits = _index.query(query, limit=20)
    out: list[dict] = []
    for score, doc_id in hits:
        d = _index.docs[doc_id]
        out.append({
            "slug": d["slug"],
            "kb": d["kb"],
            "title": d["title"],
            "summary": d["summary"],
            "tags": d["tags"],
            "snippet": _make_snippet(d["body"], query),
            "score": float(score),
        })
    return out


# Legacy alias kept so older code that imported search_kb_basic still works
search_kb_basic = search_kb


def index_size() -> int:
    return len(_index)
