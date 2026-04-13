"""
Wiki quality improvement workflows.

Three passes:
1. Dedup: Find redundant/overlapping articles and merge them
2. Enrich: Rewrite shallow articles with proper structure
3. Crosslink: Add wikilinks between related articles

Triggered via API or worker task.
"""

import re
import json
import logging
import time
import httpx
from difflib import SequenceMatcher
from datetime import datetime, timezone


# In-process TTL cache for the expensive quality scans. Admin-page reads
# read through this; article writes call ``invalidate_quality_cache`` to
# drop stale entries. TTL is a last-resort ceiling, not a correctness
# guarantee — writes should invalidate eagerly.
_QUALITY_CACHE_TTL = 300.0  # 5 minutes
_quality_cache: dict[tuple[str, str], tuple[float, object]] = {}


def _quality_cache_get(kind: str, kb_name: str):
    entry = _quality_cache.get((kind, kb_name))
    if entry is None:
        return None
    ts, value = entry
    if (time.monotonic() - ts) > _QUALITY_CACHE_TTL:
        _quality_cache.pop((kind, kb_name), None)
        return None
    return value


def _quality_cache_set(kind: str, kb_name: str, value) -> None:
    _quality_cache[(kind, kb_name)] = (time.monotonic(), value)


def invalidate_quality_cache(kb_name: str | None = None) -> None:
    """Drop cached quality scan results. Pass None to clear every KB."""
    if kb_name is None:
        _quality_cache.clear()
        return
    for key in [k for k in _quality_cache if k[1] == kb_name]:
        _quality_cache.pop(key, None)

from app.config import (
    SERPER_API_KEY,
)
from app.llm import llm_chat

SERPER_URL = "https://google.serper.dev/search"
from app.wiki import (
    get_articles,
    parse_frontmatter,
    _serialize_frontmatter,
    _wikilink_to_slug,
    read_article_text,
    write_article_text,
)
from app import db

logger = logging.getLogger("kb-service.quality")


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _chat(prompt: str, system: str = "", max_tokens: int = 3000) -> str:
    """Call the configured LLM provider and return content."""
    return await llm_chat(system, prompt, max_tokens, 0.2)


# ─── Article Quality Scoring ────────────────────────────────────────────────

def score_article_quality(kb_name: str, slug: str) -> dict:
    """Compute a 0-100 quality score for a single article."""
    text = read_article_text(kb_name, slug)
    if not text:
        return {"slug": slug, "score": 0, "breakdown": {}}
    meta, body = parse_frontmatter(text)
    return _score_from_parsed(slug, meta, body)


def _score_from_parsed(slug: str, meta: dict, body: str) -> dict:
    """Score helper that takes already-parsed frontmatter + body.

    Lets the batch path (``score_all_articles``) reuse the text fetched
    by ``storage.iter_articles`` instead of re-reading every article
    one-by-one from S3.
    """
    word_count = len(body.split()) if body else 0

    breakdown = {}

    # Word count (0-25)
    if word_count >= 1500:
        breakdown["words"] = 25
    elif word_count >= 800:
        breakdown["words"] = 20
    elif word_count >= 400:
        breakdown["words"] = 15
    elif word_count >= 200:
        breakdown["words"] = 8
    else:
        breakdown["words"] = max(0, word_count // 25)

    # Structure — H2 headers (0-15)
    h2_count = len(re.findall(r'^## ', body, re.MULTILINE)) if body else 0
    breakdown["structure"] = min(15, h2_count * 3)

    # Links (0-15)
    wikilinks = len(re.findall(r'\[\[.*?\]\]', body)) if body else 0
    ext_links = len(re.findall(r'\[.*?\]\(http', body)) if body else 0
    breakdown["links"] = min(15, (wikilinks * 2) + ext_links)

    # Code blocks (0-10)
    code_blocks = len(re.findall(r'```', body)) // 2 if body else 0
    breakdown["code"] = min(10, code_blocks * 3)

    # Frontmatter completeness (0-15)
    fm_score = 0
    if meta.get("title"): fm_score += 3
    if meta.get("summary"): fm_score += 4
    if meta.get("tags"): fm_score += 3
    if meta.get("source_files"): fm_score += 3
    if meta.get("updated"): fm_score += 2
    breakdown["frontmatter"] = fm_score

    # Tables (0-5)
    tables = len(re.findall(r'\|.*\|.*\|', body)) if body else 0
    breakdown["tables"] = min(5, tables)

    # Lists (0-5)
    lists = len(re.findall(r'^[-*] ', body, re.MULTILINE)) if body else 0
    breakdown["lists"] = min(5, lists)

    # Freshness (0-10)
    updated = str(meta.get("updated", ""))
    if "2026" in updated:
        breakdown["freshness"] = 10
    elif "2025" in updated:
        breakdown["freshness"] = 5
    else:
        breakdown["freshness"] = 0

    total = min(100, sum(breakdown.values()))

    return {
        "slug": slug,
        "title": meta.get("title", slug),
        "score": total,
        "word_count": word_count,
        "breakdown": breakdown,
    }


def score_all_articles(kb_name: str) -> list[dict]:
    """Score every article in a KB, sorted by score ascending (worst first).

    Streams bodies via ``storage.iter_articles`` so the whole KB is
    fetched with the backend's thread pool instead of paying N
    sequential S3 round-trips.
    """
    cached = _quality_cache_get("scores", kb_name)
    if cached is not None:
        return cached

    from app import storage

    scored: list[dict] = []
    for slug, text in storage.iter_articles(kb_name, subdir="wiki"):
        meta, body = parse_frontmatter(text)
        scored.append(_score_from_parsed(slug, meta, body))

    result = sorted(scored, key=lambda x: x["score"])
    _quality_cache_set("scores", kb_name, result)
    return result


# ─── Pass 1: Find Duplicates ────────────────────────────────────────────────

def find_duplicates(kb_name: str, threshold: float = 0.5) -> list[dict]:
    """Find pairs of articles with overlapping titles or content.

    Uses a tag-bucket prefilter so the expensive SequenceMatcher only
    runs on pairs that share at least one tag OR at least one title word
    — on a multi-hundred-article KB this cuts the pair space by orders
    of magnitude vs. the naive O(N²) scan.
    """
    cache_key = f"duplicates:{threshold}"
    cached = _quality_cache_get(cache_key, kb_name)
    if cached is not None:
        return cached
    articles = get_articles(kb_name)

    # Precompute cheap per-article fingerprints once.
    _STOP_WORDS = {
        "the", "a", "an", "and", "or", "of", "for", "in", "on", "to", "with",
        "how", "what", "why", "vs", "versus", "best", "guide", "tutorial",
    }
    entries: list[dict] = []
    for a in articles:
        title_words = {
            w for w in re.findall(r"[a-z0-9]+", (a.get("title") or "").lower())
            if len(w) > 2 and w not in _STOP_WORDS
        }
        entries.append({
            "slug": a["slug"],
            "title": a.get("title", ""),
            "title_lower": (a.get("title") or "").lower(),
            "tags": set(a.get("tags") or []),
            "title_words": title_words,
        })

    # Build inverted indices by tag and by title word. A candidate pair
    # is (i, j) that appears together under any index key.
    by_tag: dict[str, list[int]] = {}
    for idx, e in enumerate(entries):
        for tag in e["tags"]:
            by_tag.setdefault(tag, []).append(idx)
    by_word: dict[str, list[int]] = {}
    for idx, e in enumerate(entries):
        for word in e["title_words"]:
            by_word.setdefault(word, []).append(idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for bucket in list(by_tag.values()) + list(by_word.values()):
        if len(bucket) < 2 or len(bucket) > 200:
            # Skip empty singletons and huge tag-buckets that would
            # degenerate back to N² ("python", "aws", etc.).
            continue
        for i_idx, i in enumerate(bucket):
            for j in bucket[i_idx + 1:]:
                if i < j:
                    candidate_pairs.add((i, j))
                else:
                    candidate_pairs.add((j, i))

    dupes: list[dict] = []
    for i, j in candidate_pairs:
        a, b = entries[i], entries[j]

        title_ratio = SequenceMatcher(None, a["title_lower"], b["title_lower"]).ratio()

        tags_a, tags_b = a["tags"], b["tags"]
        tag_overlap = len(tags_a & tags_b) / max(len(tags_a | tags_b), 1)

        score = (title_ratio * 0.6) + (tag_overlap * 0.4)
        if score >= threshold:
            dupes.append({
                "article_a": a["slug"],
                "title_a": a["title"],
                "article_b": b["slug"],
                "title_b": b["title"],
                "score": round(score, 3),
                "title_similarity": round(title_ratio, 3),
                "tag_overlap": round(tag_overlap, 3),
            })

    result = sorted(dupes, key=lambda d: d["score"], reverse=True)
    _quality_cache_set(cache_key, kb_name, result)
    return result


# ─── Pass 1.5: Broken wikilinks ─────────────────────────────────────────────

_WIKILINK_RE = re.compile(r"\[\[([^\]\n|]+)(?:\|[^\]\n]*)?\]\]")


def _wikilink_slug(title: str) -> str:
    """Match app.wiki._wikilink_to_slug — kept local so this module
    doesn't need to import wiki.py (which would be a cycle)."""
    safe = "".join(c if c.isalnum() or c in " -" else "" for c in title)
    return safe.lower().strip().replace(" ", "-")


def find_broken_wikilinks(kb_name: str) -> dict:
    """Scan every article body in ``kb_name`` for ``[[Wikilinks]]``
    whose target slug has no matching article. Returns a dict with:

        {
            "broken_count": int,
            "affected_articles": int,
            "by_target": [
                {"target": "gpu cloud platforms", "slug": "gpu-cloud-platforms",
                 "count": 3, "sources": [{"slug": ..., "title": ...}, ...]},
                ...
            ],
            "by_article": [
                {"slug": ..., "title": ...,
                 "broken": [{"target": ..., "slug": ...}, ...]},
                ...
            ],
        }

    Results are cached under ``_quality_cache`` with a 5-minute TTL.
    """
    from app import storage

    cache_key = "broken_wikilinks"
    cached = _quality_cache_get(cache_key, kb_name)
    if cached is not None:
        return cached

    articles = get_articles(kb_name)
    existing_slugs = {a["slug"] for a in articles}
    title_by_slug = {a["slug"]: a.get("title", a["slug"]) for a in articles}

    # target_slug -> {"target": label, "count": n, "sources": [{slug, title}]}
    by_target: dict[str, dict] = {}
    # source_slug -> list[{"target": label, "slug": target_slug}]
    by_article: dict[str, list[dict]] = {}

    for src_slug, body in storage.iter_articles(kb_name, subdir="wiki"):
        if src_slug not in existing_slugs:
            continue
        _, body_text = parse_frontmatter(body)
        seen_in_article: set[str] = set()
        for m in _WIKILINK_RE.finditer(body_text):
            raw_target = m.group(1).strip()
            if not raw_target:
                continue
            target_slug = _wikilink_slug(raw_target)
            if not target_slug or target_slug in existing_slugs:
                continue

            # Dedupe multiple occurrences of the same link in one article
            # — still report "count" at the target level though.
            dedup_key = target_slug
            bucket = by_target.setdefault(
                target_slug,
                {"target": raw_target, "slug": target_slug, "count": 0, "sources": []},
            )
            bucket["count"] += 1
            if dedup_key not in seen_in_article:
                seen_in_article.add(dedup_key)
                bucket["sources"].append({
                    "slug": src_slug,
                    "title": title_by_slug.get(src_slug, src_slug),
                })
                by_article.setdefault(src_slug, []).append({
                    "target": raw_target,
                    "slug": target_slug,
                })

    broken_by_target = sorted(
        by_target.values(),
        key=lambda x: (-x["count"], x["target"].lower()),
    )
    broken_by_article = [
        {
            "slug": src_slug,
            "title": title_by_slug.get(src_slug, src_slug),
            "broken": broken_list,
        }
        for src_slug, broken_list in sorted(
            by_article.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )
    ]

    result = {
        "kb": kb_name,
        "broken_count": sum(b["count"] for b in broken_by_target),
        "affected_articles": len(broken_by_article),
        "unique_targets": len(broken_by_target),
        "by_target": broken_by_target,
        "by_article": broken_by_article,
    }
    _quality_cache_set(cache_key, kb_name, result)
    return result


# ─── Pass 2: Identify Shallow Articles ──────────────────────────────────────

_TECH_TAGS = frozenset({
    "go", "python", "typescript", "bash", "docker", "kubernetes", "terraform",
})


def find_shallow_articles(kb_name: str, min_words: int = 300) -> list[dict]:
    """Find articles that are too short, have no structure, or lack quality.

    Fast path: when ``get_articles`` returns rows carrying derived
    signal columns (h2_count / link_count / code_count / has_tech_tag,
    populated by the DynamoDB metadata index), we can compute every
    issue from those columns without reading a single article body.
    Falls back to streaming bodies via ``storage.iter_articles`` if
    the signals are missing (pre-index deployment or empty index).
    """
    from app import storage

    cache_key = f"shallow:{min_words}"
    cached = _quality_cache_get(cache_key, kb_name)
    if cached is not None:
        return cached

    articles = get_articles(kb_name)
    shallow: list[dict] = []

    # Detect whether the index is populated with derived signals. A
    # single row that has ``h2_count`` as a populated int means the
    # whole set came from the index path.
    has_signals = any("h2_count" in a and a.get("h2_count", 0) > 0 for a in articles)

    if has_signals:
        for meta in articles:
            slug = meta.get("slug") or ""
            issues: list[str] = []
            word_count = int(meta.get("word_count", 0))
            h2_count = int(meta.get("h2_count", 0))
            link_count = int(meta.get("link_count", 0))
            code_count = int(meta.get("code_count", 0))
            has_tech = bool(meta.get("has_tech_tag", False))

            if word_count < min_words:
                issues.append(f"only {word_count} words")
            if h2_count < 2 and word_count > 200:
                issues.append(f"only {h2_count} sections")
            if link_count < 2:
                issues.append(f"only {link_count} links")
            if code_count == 0 and has_tech:
                issues.append("no code examples")

            if issues:
                shallow.append({
                    "slug": slug,
                    "title": meta.get("title", slug),
                    "word_count": word_count,
                    "issues": issues,
                    "kb": kb_name,
                })
    else:
        # Pre-index fallback: stream bodies from storage.
        metas = {a["slug"]: a for a in articles}
        for slug, text in storage.iter_articles(kb_name, subdir="wiki"):
            meta = metas.get(slug)
            if meta is None:
                continue

            issues = []
            word_count = meta.get("word_count", 0)
            if word_count < min_words:
                issues.append(f"only {word_count} words")

            _, body = parse_frontmatter(text)

            h2_count = len(re.findall(r"^## ", body, re.MULTILINE))
            if h2_count < 2 and word_count > 200:
                issues.append(f"only {h2_count} sections")

            link_count = len(re.findall(r"\[.*?\]\(.*?\)", body))
            if link_count < 2:
                issues.append(f"only {link_count} links")

            if "```" not in body and any(t in _TECH_TAGS for t in meta.get("tags", [])):
                issues.append("no code examples")

            if issues:
                shallow.append({
                    "slug": slug,
                    "title": meta.get("title", slug),
                    "word_count": word_count,
                    "issues": issues,
                    "kb": kb_name,
                })

    result = sorted(shallow, key=lambda s: len(s["issues"]), reverse=True)
    _quality_cache_set(cache_key, kb_name, result)
    return result


# ─── Pass 3: Enrich a Single Article ────────────────────────────────────────

async def enrich_article(kb_name: str, slug: str) -> dict:
    """Rewrite a shallow article to be comprehensive and well-structured."""
    text = read_article_text(kb_name, slug)
    if not text:
        return {"error": f"Article not found: {slug}"}

    meta, body = parse_frontmatter(text)
    title = meta.get("title", slug.replace("-", " ").title())

    prompt = f"""Rewrite and improve this knowledge base article. The current version is shallow and needs more depth, structure, and quality.

CURRENT ARTICLE TITLE: {title}

CURRENT CONTENT:
{body[:4000]}

REWRITE RULES:
1. Keep the same topic and all existing factual content. Do NOT invent new facts.
2. Add proper structure with ## headers for major sections
3. Add more depth and context to each section (explain why things matter, not just what they are)
4. Add practical examples and code snippets where relevant to the topic
5. Only add comparison tables when genuinely comparing alternatives
6. Add a "Recommendations" section with actionable guidance
7. Don't pad with exhaustive reference data. Focus on insights.
8. Preserve all existing source citations [Source](url). Do not remove or fabricate citations.
9. End with a "See Also" section listing 3-5 related topics as [[Wikilinks]]
10. Keep it under 2000 words. Quality over quantity.
11. Clean markdown formatting. No stray artifacts or incomplete sections.
12. Adapt to the topic: non-technical topics don't need code blocks.

Output ONLY the rewritten article content (no frontmatter, no title — those are managed separately)."""

    system = "You are improving articles for a personal knowledge base. Preserve all existing facts and citations. Add depth and structure without inventing information. Write clearly and concisely. IMPORTANT: Write ONLY in English."

    new_content = await _chat(prompt, system=system, max_tokens=4000)

    if not new_content or len(new_content) < 100:
        return {"error": "LLM returned insufficient content"}

    # Update the article
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta["updated"] = today
    if meta.get("status") == "draft":
        meta["status"] = "review"

    fm_str = _serialize_frontmatter(meta)
    write_article_text(kb_name, slug, fm_str + "\n\n" + new_content)

    new_word_count = len(new_content.split())
    logger.info("Enriched article %s/%s: %d -> %d words", kb_name, slug, len(body.split()), new_word_count)

    return {
        "slug": slug,
        "title": title,
        "old_words": len(body.split()),
        "new_words": new_word_count,
        "status": "enriched",
    }


# ─── Pass 4: Add Crosslinks ────────────────────────────────────────────────

async def add_crosslinks(kb_name: str, slug: str) -> dict:
    """Analyze an article and add wikilinks to related articles."""
    text = read_article_text(kb_name, slug)
    if not text:
        return {"error": f"Article not found: {slug}"}

    meta, body = parse_frontmatter(text)

    # Get all article titles for crosslinking
    all_articles = get_articles(kb_name)
    other_titles = [a["title"] for a in all_articles if a["slug"] != slug]

    if not other_titles:
        return {"slug": slug, "links_added": 0}

    prompt = f"""Given this knowledge base article, identify mentions of topics that match other articles in the wiki. Add [[Wikilinks]] where appropriate.

ARTICLE:
{body[:3000]}

OTHER ARTICLES IN THE WIKI (available for linking):
{chr(10).join(f"- {t}" for t in other_titles[:50])}

RULES:
1. Find phrases in the article that match or closely relate to other article titles
2. Replace the first mention of each related topic with a [[Wikilink]]
3. Don't over-link — max 10 wikilinks per article
4. Only link on first mention, not every occurrence
5. Don't link inside code blocks or headers
6. Return the FULL article with wikilinks added (no frontmatter)"""

    system = "You are editing a wiki article to add internal cross-references. Only add [[Wikilinks]] where they naturally fit."

    linked_content = await _chat(prompt, system=system, max_tokens=4000)

    if not linked_content or len(linked_content) < 100:
        return {"error": "Minimax returned insufficient content"}

    # Count new wikilinks added
    old_links = len(re.findall(r'\[\[.*?\]\]', body))
    new_links = len(re.findall(r'\[\[.*?\]\]', linked_content))

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta["updated"] = today

    fm_str = _serialize_frontmatter(meta)
    write_article_text(kb_name, slug, fm_str + "\n\n" + linked_content)

    logger.info("Crosslinked article %s/%s: %d -> %d wikilinks", kb_name, slug, old_links, new_links)

    return {
        "slug": slug,
        "old_links": old_links,
        "new_links": new_links,
        "links_added": new_links - old_links,
    }


# ─── Full Quality Pass ─────────────────────────────────────────────────────

async def run_quality_pass(kb_name: str, max_articles: int = 10) -> dict:
    """Run a full quality improvement pass on a KB.

    1. Find shallow articles
    2. Enrich the worst ones
    3. Add crosslinks to all enriched articles

    Returns a summary of changes made.
    """
    results = {
        "kb": kb_name,
        "shallow_found": 0,
        "articles_enriched": 0,
        "articles_crosslinked": 0,
        "total_words_added": 0,
        "total_links_added": 0,
        "details": [],
    }

    # Find shallow articles
    shallow = find_shallow_articles(kb_name)
    results["shallow_found"] = len(shallow)

    # Enrich the worst ones
    targets = shallow[:max_articles]
    for article in targets:
        try:
            enrich_result = await enrich_article(kb_name, article["slug"])
            if "error" not in enrich_result:
                results["articles_enriched"] += 1
                words_added = enrich_result.get("new_words", 0) - enrich_result.get("old_words", 0)
                results["total_words_added"] += max(0, words_added)
                results["details"].append(enrich_result)

                # Crosslink the enriched article
                try:
                    link_result = await add_crosslinks(kb_name, article["slug"])
                    if "error" not in link_result:
                        results["articles_crosslinked"] += 1
                        results["total_links_added"] += link_result.get("links_added", 0)
                except Exception as exc:
                    logger.warning("Crosslink failed for %s: %s", article["slug"], exc)
            else:
                results["details"].append(enrich_result)
        except Exception as exc:
            logger.warning("Enrich failed for %s: %s", article["slug"], exc)
            results["details"].append({"slug": article["slug"], "error": str(exc)})

    return results


# ─── Serper Search Helper ───────────────────────────────────────────────────

async def _serper_search(query: str, num: int = 5) -> list[dict]:
    """Quick Serper search, returns list of {title, snippet, url}."""
    if not SERPER_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                SERPER_URL,
                json={"q": query, "num": num},
                headers={"X-API-KEY": SERPER_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "url": r.get("link", ""),
                }
                for r in data.get("organic", [])
            ]
    except Exception as exc:
        logger.warning("Serper search failed: %s", exc)
        return []


# ─── Pass 5: Fact Check ────────────────────────────────────────────────────

async def fact_check_article(kb_name: str, slug: str) -> dict:
    """Extract key claims from an article, verify each against Serper results.

    Returns a report of verified, outdated, and unverifiable claims.

    If ``article_claims`` already contains stored claims for this article
    (populated by the synthesis critique pass), re-verify those instead
    of re-extracting from the raw markdown each run. This is much cheaper
    for periodic fact-check crons because the slow-and-expensive
    extraction step only ever runs once per article.
    """
    text = read_article_text(kb_name, slug)
    if not text:
        return {"error": f"Article not found: {slug}"}

    meta, body = parse_frontmatter(text)
    title = meta.get("title", slug)

    # Try the cheap path first — read pre-computed claims from
    # article_claims and re-verify them. Falls through to extraction if
    # nothing is stored yet.
    claims: list[dict] = []
    stored = []
    try:
        stored = await db.get_claims_for_article(kb_name, slug)
    except Exception as exc:
        logger.warning("Failed to load stored claims for %s/%s: %s", kb_name, slug, exc)
        stored = []

    if stored:
        for c in stored:
            txt = c.get("claim_text") or ""
            if not txt.strip():
                continue
            # Build a verification query from the claim text. The search
            # is the model's job to refine — we just hand over the claim.
            claims.append({
                "claim": txt,
                "search_query": txt[:200],
                "type": c.get("claim_type") or "general",
                "_stored_id": c.get("id"),
            })
        logger.info(
            "fact_check %s/%s: re-verifying %d stored claims",
            kb_name, slug, len(claims),
        )

    if not claims:
        # Step 1: Extract verifiable claims using Minimax (legacy path)
        extract_prompt = f"""Extract the key verifiable factual claims from this article. Focus on:
- Version numbers (e.g., "v1.9.3", "Python 3.12")
- Pricing (e.g., "$5/month", "free tier includes 100 devices")
- Dates and deadlines (e.g., "released January 2026", "deadline March 2027")
- Specific metrics (e.g., "30 tok/s", "95% accuracy")
- Feature claims (e.g., "supports GPU autoscaling", "includes built-in TURN server")
- Comparisons (e.g., "faster than X", "replaces deprecated Y")

ARTICLE TITLE: {title}

ARTICLE CONTENT:
{body[:4000]}

Return a JSON array of objects, each with:
- "claim": the specific factual claim (one sentence)
- "search_query": a targeted search query to verify this claim
- "type": one of "version", "pricing", "date", "metric", "feature", "comparison"

Return ONLY the JSON array, max 8 claims. Focus on claims most likely to change over time."""

        claims_raw = await _chat(extract_prompt, max_tokens=1500)

        # Parse claims
        try:
            match = re.search(r'\[[\s\S]*\]', claims_raw)
            claims = json.loads(match.group()) if match else []
        except (json.JSONDecodeError, AttributeError):
            claims = []

    if not claims:
        return {"slug": slug, "title": title, "claims_checked": 0, "results": []}

    # Step 2: Search each claim
    check_results = []
    for claim in claims[:8]:
        query = claim.get("search_query", claim.get("claim", ""))
        search_results = await _serper_search(query, num=3)

        if not search_results:
            check_results.append({
                "claim": claim.get("claim", ""),
                "type": claim.get("type", "unknown"),
                "status": "unverifiable",
                "reason": "No search results found",
            })
            continue

        # Step 3: Ask Minimax to compare claim against search results
        snippets = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in search_results
        )
        verify_prompt = f"""Compare this claim from our knowledge base against current search results. Is it still accurate?

CLAIM: {claim.get('claim', '')}
CLAIM TYPE: {claim.get('type', '')}

CURRENT SEARCH RESULTS:
{snippets}

Respond with ONLY a JSON object:
{{
  "status": "verified" | "outdated" | "partially_correct" | "unverifiable",
  "current_info": "what the search results actually say (one sentence)",
  "correction": "what should be updated if outdated (or null if verified)"
}}"""

        verify_raw = await _chat(verify_prompt, max_tokens=300)
        try:
            match = re.search(r'\{[\s\S]*\}', verify_raw)
            verdict = json.loads(match.group()) if match else {}
        except (json.JSONDecodeError, AttributeError):
            verdict = {"status": "unverifiable", "current_info": verify_raw[:200]}

        check_results.append({
            "claim": claim.get("claim", ""),
            "type": claim.get("type", "unknown"),
            "status": verdict.get("status", "unverifiable"),
            "current_info": verdict.get("current_info", ""),
            "correction": verdict.get("correction"),
            "sources": [r["url"] for r in search_results[:2]],
        })

        # If this claim was loaded from article_claims (reflection path),
        # persist the new verdict back so future re-checks see the updated
        # last_checked_at + status. Maps verify-step verdicts onto the
        # claims-table status vocabulary.
        stored_id = claim.get("_stored_id")
        if stored_id:
            verdict_status = verdict.get("status", "unverifiable")
            confidence_map = {
                "verified": 0.95,
                "outdated": 0.2,
                "partially_correct": 0.5,
                "unverifiable": 0.4,
            }
            try:
                await db.update_claim_status(
                    stored_id,
                    verdict_status,
                    confidence_map.get(verdict_status, 0.5),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to update stored claim %s: %s", stored_id, exc,
                )

    # Summary
    verified = sum(1 for r in check_results if r["status"] == "verified")
    outdated = sum(1 for r in check_results if r["status"] == "outdated")
    partial = sum(1 for r in check_results if r["status"] == "partially_correct")

    return {
        "slug": slug,
        "title": title,
        "claims_checked": len(check_results),
        "verified": verified,
        "outdated": outdated,
        "partially_correct": partial,
        "results": check_results,
    }


# ─── Pass 6: Freshness Audit + Auto-Update ─────────────────────────────────

async def freshness_audit(kb_name: str, slug: str, auto_update: bool = True) -> dict:
    """Check an article for stale content and optionally apply updates.

    1. Extract time-sensitive claims (versions, pricing, dates)
    2. Search for current info on each
    3. If outdated, generate a diff and apply it
    """
    # First run fact check
    check = await fact_check_article(kb_name, slug)

    if "error" in check:
        return check

    outdated_claims = [r for r in check.get("results", []) if r["status"] in ("outdated", "partially_correct")]

    if not outdated_claims:
        return {
            "slug": slug,
            "title": check.get("title", slug),
            "status": "fresh",
            "claims_checked": check["claims_checked"],
            "outdated": 0,
            "updates_applied": 0,
        }

    if not auto_update:
        return {
            "slug": slug,
            "title": check.get("title", slug),
            "status": "stale",
            "claims_checked": check["claims_checked"],
            "outdated": len(outdated_claims),
            "outdated_details": outdated_claims,
            "updates_applied": 0,
        }

    # Auto-update: read article, ask Minimax to apply corrections
    text = read_article_text(kb_name, slug)
    if not text:
        return {"error": "Could not read article for updating"}

    meta, body = parse_frontmatter(text)

    corrections = "\n".join(
        f"- OUTDATED: \"{c['claim']}\"\n  CORRECTION: {c.get('correction') or c.get('current_info', 'needs update')}\n  SOURCE: {c.get('sources', [''])[0]}"
        for c in outdated_claims
    )

    update_prompt = f"""Update this knowledge base article by applying the corrections listed below. Only change the specific outdated facts — don't rewrite the entire article.

CORRECTIONS TO APPLY:
{corrections}

CURRENT ARTICLE:
{body[:5000]}

RULES:
1. Only modify sentences containing the outdated claims
2. Replace old facts with the corrected information
3. Add [Updated YYYY-MM-DD] after each corrected sentence
4. Keep everything else exactly the same
5. Return the FULL article content (not just the changed parts)"""

    system = "You are a precise editor. Apply minimal, targeted corrections to factual claims. Do not change writing style, structure, or content beyond the specific corrections listed."

    updated_body = await _chat(update_prompt, system=system, max_tokens=4000)

    if not updated_body or len(updated_body) < 100:
        return {"error": "Minimax returned insufficient content for update"}

    # Write updated article
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta["updated"] = today

    fm_str = _serialize_frontmatter(meta)
    write_article_text(kb_name, slug, fm_str + "\n\n" + updated_body)

    logger.info("Freshness audit updated %s/%s: %d corrections applied", kb_name, slug, len(outdated_claims))

    return {
        "slug": slug,
        "title": check.get("title", slug),
        "status": "updated",
        "claims_checked": check["claims_checked"],
        "outdated": len(outdated_claims),
        "updates_applied": len(outdated_claims),
        "corrections": outdated_claims,
    }


# ─── Batch Freshness Audit ─────────────────────────────────────────────────

async def run_freshness_audit(kb_name: str, max_articles: int = 10, auto_update: bool = True) -> dict:
    """Run freshness audit across the stalest articles in a KB."""
    articles = get_articles(kb_name)

    # Prioritize articles with version numbers, pricing, dates
    scored = []
    for a in articles:
        text = read_article_text(kb_name, a["slug"])
        if not text:
            continue
        _, body = parse_frontmatter(text)

        # Count time-sensitive content
        versions = len(re.findall(r'v\d+\.\d+', body))
        prices = len(re.findall(r'\$\d+', body))
        dates = len(re.findall(r'20\d{2}[-/]\d{2}', body))
        score = versions * 2 + prices * 3 + dates * 1

        if score > 0:
            scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    targets = [a for _, a in scored[:max_articles]]

    results = {
        "kb": kb_name,
        "articles_audited": 0,
        "articles_updated": 0,
        "total_corrections": 0,
        "details": [],
    }

    for article in targets:
        try:
            audit = await freshness_audit(kb_name, article["slug"], auto_update=auto_update)
            results["articles_audited"] += 1
            if audit.get("updates_applied", 0) > 0:
                results["articles_updated"] += 1
                results["total_corrections"] += audit["updates_applied"]
            results["details"].append(audit)
        except Exception as exc:
            logger.warning("Freshness audit failed for %s: %s", article["slug"], exc)
            results["details"].append({"slug": article["slug"], "error": str(exc)})

    return results


# ─── Pass 7: Wikilink Integrity Check ─────────────────────────────────────

def check_wikilinks(kb_name: str) -> dict:
    """Scan all articles for broken [[Wikilinks]] and report issues.

    For each wikilink, checks if the target slug exists. If not, attempts
    fuzzy matching against existing article titles/slugs.
    """
    articles = get_articles(kb_name)
    slug_to_title = {a["slug"]: a["title"] for a in articles}
    title_to_slug = {a["title"].lower(): a["slug"] for a in articles}
    all_slugs = set(slug_to_title.keys())

    broken_links: list[dict] = []
    total_links = 0
    valid_links = 0

    for article in articles:
        text = read_article_text(kb_name, article["slug"])
        if not text:
            continue
        _, body = parse_frontmatter(text)

        wikilinks = re.findall(r'\[\[([^\]]+)\]\]', body)
        for link_text in wikilinks:
            total_links += 1
            target_slug = _wikilink_to_slug(link_text)

            if target_slug in all_slugs:
                valid_links += 1
                continue

            suggestion = _fuzzy_match_slug(link_text, slug_to_title, title_to_slug)
            broken_links.append({
                "source_slug": article["slug"],
                "source_title": article["title"],
                "link_text": link_text,
                "expected_slug": target_slug,
                "suggestion": suggestion,
            })

    return {
        "kb": kb_name,
        "total_links": total_links,
        "valid_links": valid_links,
        "broken_links": len(broken_links),
        "details": broken_links,
    }


def _fuzzy_match_slug(
    link_text: str,
    slug_to_title: dict[str, str],
    title_to_slug: dict[str, str],
) -> dict | None:
    """Try to find a matching article for a broken wikilink."""
    link_lower = link_text.lower().strip()

    # Exact title match (case-insensitive)
    if link_lower in title_to_slug:
        slug = title_to_slug[link_lower]
        return {"slug": slug, "title": slug_to_title[slug], "confidence": 1.0}

    # Fuzzy match against all titles
    best_ratio = 0.0
    best_slug = None
    for title_lower, slug in title_to_slug.items():
        ratio = SequenceMatcher(None, link_lower, title_lower).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_slug = slug

    if best_slug and best_ratio >= 0.6:
        return {
            "slug": best_slug,
            "title": slug_to_title[best_slug],
            "confidence": round(best_ratio, 3),
        }
    return None


def fix_wikilinks(kb_name: str, slug: str | None = None, auto_remove: bool = True) -> dict:
    """Fix broken wikilinks in one or all articles.

    - Fuzzy match >= 0.7 confidence: replace with corrected link
    - No match + auto_remove: convert to plain text
    - No match + not auto_remove: leave unchanged
    """
    articles = get_articles(kb_name)
    slug_to_title = {a["slug"]: a["title"] for a in articles}
    title_to_slug = {a["title"].lower(): a["slug"] for a in articles}
    all_slugs = set(slug_to_title.keys())

    targets = [a for a in articles if a["slug"] == slug] if slug else articles
    if slug and not targets:
        return {"error": f"Article not found: {slug}"}

    total_fixed = 0
    total_removed = 0
    fixed_articles: list[dict] = []

    for article in targets:
        text = read_article_text(kb_name, article["slug"])
        if not text:
            continue
        meta, body = parse_frontmatter(text)
        original_body = body
        fixes_in_article = 0
        removals_in_article = 0

        def _replace_wikilink(m: re.Match) -> str:
            nonlocal fixes_in_article, removals_in_article
            link_text = m.group(1)
            target_slug = _wikilink_to_slug(link_text)

            if target_slug in all_slugs:
                return m.group(0)

            suggestion = _fuzzy_match_slug(link_text, slug_to_title, title_to_slug)
            if suggestion and suggestion["confidence"] >= 0.7:
                fixes_in_article += 1
                return f"[[{suggestion['title']}]]"

            if auto_remove:
                removals_in_article += 1
                return link_text

            return m.group(0)

        new_body = re.sub(r'\[\[([^\]]+)\]\]', _replace_wikilink, body)

        if new_body != original_body:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            meta["updated"] = today
            fm_str = _serialize_frontmatter(meta)
            write_article_text(kb_name, article["slug"], fm_str + "\n\n" + new_body)

            total_fixed += fixes_in_article
            total_removed += removals_in_article
            fixed_articles.append({
                "slug": article["slug"],
                "title": article["title"],
                "links_fixed": fixes_in_article,
                "links_removed": removals_in_article,
            })
            logger.info(
                "Fixed wikilinks in %s/%s: %d corrected, %d removed",
                kb_name, article["slug"], fixes_in_article, removals_in_article,
            )

    return {
        "kb": kb_name,
        "articles_modified": len(fixed_articles),
        "links_fixed": total_fixed,
        "links_removed": total_removed,
        "details": fixed_articles,
    }
