"""
Chat retrieval, prompt assembly, and tool dispatch.

A RAG chat layer built on the existing primitives:
  - hybrid_search (FTS + vector + KG) for candidate articles
  - get_article for full markdown
  - embed_text / embed_texts + cosine similarity for chunk reranking
  - llm_chat_stream / llm_chat_tools for model calls
  - sse_response for the streaming wire format
  - chat_sessions / chat_messages tables for history
  - topic_candidates for chat-gap feedback into auto-discovery

This module is pure orchestration — every helper delegates to an existing
function. No new business logic.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app import db

# ---------------------------------------------------------------------------
# Prompt-injection detection
#
# Scans retrieved RAG passages for common prompt-injection markers before
# the LLM sees them. This is the OWASP LLM Top 10 2026 / LLM01 concern:
# malicious text embedded in indexed documents can hijack the model's
# instructions. We can't perfectly detect injection (adversarial LLM
# outputs can evade any static rule), but flagging obvious patterns is
# the cheap first layer of defense — we log hits + let the caller
# decide whether to drop or down-rank the chunk.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above)",
        r"forget\s+(all\s+)?(previous|prior|above|everything)",
        r"new\s+instructions?\s*[:;]",
        r"(?:^|\n)\s*system\s*[:;]\s",
        r"(?:^|\n)\s*assistant\s*[:;]\s",
        r"you\s+are\s+now\s+(a|an)\s+\w+",
        r"your\s+new\s+(role|task|instructions?)\s+(is|are)",
        r"reveal\s+(your|the)\s+(system\s+)?prompt",
        r"print\s+(your|the)\s+(system\s+)?prompt",
        r"act\s+as\s+(a|an)\s+(jailbroken|unfiltered|unrestricted)",
        r"\bBEGIN\s+ADVERSARIAL\b",
        r"<\s*\|?im_start\|?\s*>",  # chatml-style hijack
        r"<\s*\|?system\|?\s*>",
    )
)


def detect_prompt_injection(text: str) -> list[str]:
    """Return the list of injection patterns that matched ``text``.

    An empty list means the text is clean against our static ruleset.
    A non-empty list names the rules that fired (for logging). The
    caller decides whether to drop, down-rank, or keep-and-flag.
    """
    if not text:
        return []
    hits: list[str] = []
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits
from app.embeddings import (
    embed_text,
    embed_texts,
    _dot_product,
    _normalize_vector,
)
from app.hybrid_search import hybrid_search
from app.knowledge_graph import get_related_by_graph
from app.wiki import find_related_article, get_article

logger = logging.getLogger("kb-service.chat")


# ---------------------------------------------------------------------------
# Token estimation (no tiktoken dependency — 4 chars ≈ 1 token for English)
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate. Good enough for context-budget bookkeeping."""
    if not text:
        return 0
    return len(text) // 4


# ---------------------------------------------------------------------------
# Chunking — modeled on app/embeddings.py:_chunk_text
# ---------------------------------------------------------------------------


def _chunk_for_retrieval(
    body: str,
    max_chars: int = 2000,
    overlap_chars: int = 200,
) -> list[str]:
    """Split an article body into retrieval chunks respecting section
    boundaries.

    Layered strategy:

    1. **Section boundaries** — split on ``^## `` (H2 headers). Each
       section becomes an independent chunk group so a single chunk
       never spans two topics.
    2. **Paragraph boundaries within a section** — if a section
       exceeds ``max_chars``, split on blank lines.
    3. **Hard fallback** — if a single paragraph exceeds ``max_chars``
       (rare — huge code blocks), split at the char boundary.
    4. **Rolling overlap** — each chunk carries the last
       ``overlap_chars`` chars of the previous one as a prefix. This
       helps retrieval recall when the answer straddles a boundary.

    ~2000 chars ≈ 500 tokens, so a top-k of 6 fits comfortably under
    a 3000-token passages budget.
    """
    if not body:
        return []
    body = body.strip()
    if not body:
        return []

    # Slice on H2 headers so a chunk never crosses topics. The ``##``
    # line itself stays with the following content as the section title.
    sections: list[str] = []
    lines = body.split("\n")
    current: list[str] = []
    for line in lines:
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    sections = [s for s in sections if s]

    raw_chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            raw_chunks.append(section)
            continue

        # Section too big — fall through to paragraph packing within it.
        current_text = ""
        for para in section.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(current_text) + len(para) + 2 > max_chars:
                if current_text:
                    raw_chunks.append(current_text.strip())
                if len(para) > max_chars:
                    # Huge paragraph (probably a code block). Hard-split.
                    for i in range(0, len(para), max_chars):
                        raw_chunks.append(para[i : i + max_chars].strip())
                    current_text = ""
                else:
                    current_text = para
            else:
                current_text = (
                    current_text + "\n\n" + para if current_text else para
                )
        if current_text.strip():
            raw_chunks.append(current_text.strip())

    raw_chunks = [c for c in raw_chunks if c]
    if overlap_chars <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    # Rolling overlap: each chunk after the first gets the tail of the
    # previous chunk as a prefix, making retrieval robust to answers
    # that straddle chunk boundaries.
    overlapped: list[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i - 1][-overlap_chars:]
        overlapped.append((prev_tail + "\n\n" + raw_chunks[i]).strip())
    return overlapped


# ---------------------------------------------------------------------------
# 1.1 — Retrieval composer
# ---------------------------------------------------------------------------


async def _decompose_query(question: str, max_subqs: int = 3) -> list[str]:
    """Break a complex question into focused sub-questions via the LLM.

    Returns an empty list when decomposition isn't useful — short
    questions, single-intent queries, or LLM failures. Callers should
    always combine with the original question so retrieval never loses
    the user's exact phrasing.
    """
    question = (question or "").strip()
    if not question or len(question.split()) < 8:
        return []

    from app.llm import llm_chat

    system_msg = (
        "You split complex research questions into focused sub-questions "
        "for a retrieval system. Return ONE sub-question per line, no "
        "numbering, no preamble, no quotes. Return NOTHING if the input "
        "is already a single focused question."
    )
    user_msg = (
        f"Original question: {question}\n\n"
        f"Return up to {max_subqs} sub-questions that together cover "
        f"the original. Each should be self-contained and searchable."
    )
    try:
        response = await llm_chat(
            system_msg=system_msg,
            user_msg=user_msg,
            max_tokens=300,
            temperature=0.2,
        )
    except Exception as exc:
        logger.debug("query decomposition failed: %s", exc)
        return []

    lines = [ln.strip(" -•*").strip() for ln in (response or "").splitlines()]
    subs = [ln for ln in lines if 5 < len(ln) < 200 and ln.endswith(("?", ".")) or len(ln) > 20]
    # Drop duplicates + the original.
    seen: set[str] = set()
    unique: list[str] = []
    q_lower = question.lower().strip()
    for s in subs[:max_subqs]:
        key = s.lower().strip()
        if key == q_lower or key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


async def retrieve_context(
    question: str,
    kb: str | None = None,
    *,
    k: int = 8,
    article_limit: int = 10,
) -> list[dict]:
    """Return top-k passages from the KB for a given question.

    Pipeline:
      1. hybrid_search → metadata-only candidates (slug, kb, score, ...)
      2. get_article on the top ``article_limit`` to fetch raw markdown
      3. _chunk_for_retrieval splits each article into ~500-tok passages
      4. embed_text(question) + embed_texts(chunks) → cosine similarity
      5. return top-k passages with {slug, kb, title, chunk_text, url, score}

    When ``QUERY_DECOMPOSITION=true`` is set, the question is first
    decomposed into sub-questions and each sub-question runs its own
    hybrid search; the candidate pool is then merged before chunking
    + re-ranking. Useful for compound questions like "compare X and Y
    and explain when to use each".

    Returns an empty list if hybrid_search returns nothing or the question
    is empty. Errors fetching individual articles are logged and skipped.
    """
    if not question or not question.strip():
        return []

    import os as _os
    use_decomp = _os.getenv("QUERY_DECOMPOSITION", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    candidate_pool: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    async def _hybrid_once(q: str, limit: int) -> None:
        try:
            results = await hybrid_search(q, kb_name=kb, limit=limit)
        except Exception as exc:
            logger.warning("hybrid_search failed in retrieve_context: %s", exc)
            return
        for r in results:
            key = (r.get("kb", ""), r.get("slug", ""))
            if not key[1] or key in seen_keys:
                continue
            seen_keys.add(key)
            candidate_pool.append(r)

    await _hybrid_once(question, limit=15)

    if use_decomp:
        subs = await _decompose_query(question)
        if subs:
            logger.info("query decomposed into %d sub-questions", len(subs))
            # Smaller limit per sub-question — they're additive.
            for sub in subs:
                await _hybrid_once(sub, limit=8)

    if not candidate_pool:
        return []

    candidates = candidate_pool

    if not candidates:
        return []

    # Collect chunks from the top N articles.
    chunks_meta: list[dict] = []  # {slug, kb, title, chunk_text, url}
    chunk_seen: set[tuple[str, str]] = set()
    for cand in candidates[:article_limit]:
        slug = cand.get("slug")
        cand_kb = cand.get("kb")
        if not slug or not cand_kb:
            continue
        key = (cand_kb, slug)
        if key in chunk_seen:
            continue
        chunk_seen.add(key)

        try:
            article = get_article(cand_kb, slug)
        except Exception as exc:
            logger.debug("get_article failed for %s/%s: %s", cand_kb, slug, exc)
            continue
        if not article:
            continue

        body = article.get("raw_markdown", "") or ""
        title = article.get("title") or slug.replace("-", " ").title()
        for chunk_text in _chunk_for_retrieval(body):
            chunks_meta.append(
                {
                    "slug": slug,
                    "kb": cand_kb,
                    "title": title,
                    "chunk_text": chunk_text,
                    "url": f"/wiki/{cand_kb}/{slug}",
                }
            )

    if not chunks_meta:
        return []

    # Score chunks by cosine similarity to the question
    try:
        question_vec = await embed_text(question, embed_type="query")
        question_vec = _normalize_vector(question_vec)
    except Exception as exc:
        logger.warning("Failed to embed question for retrieval: %s", exc)
        # Fall back to RRF order from hybrid_search — return the first k chunks
        return [{**c, "score": 0.0} for c in chunks_meta[:k]]

    try:
        chunk_vecs = await embed_texts([c["chunk_text"] for c in chunks_meta])
    except Exception as exc:
        logger.warning("Failed to embed chunks for retrieval: %s", exc)
        return [{**c, "score": 0.0} for c in chunks_meta[:k]]

    scored = []
    flagged_count = 0
    dropped_count = 0
    for meta, vec in zip(chunks_meta, chunk_vecs):
        if not vec:
            continue
        injection_hits = detect_prompt_injection(meta["chunk_text"])
        if injection_hits:
            flagged_count += 1
            # Two or more injection markers in a single chunk is almost
            # never a false positive — drop it outright. A single hit
            # still enters the ranking but is logged + flagged so the
            # caller can surface a warning in the trace.
            if len(injection_hits) >= 2:
                dropped_count += 1
                logger.warning(
                    "retrieve_context: dropped chunk %s/%s (injection hits: %s)",
                    meta.get("kb"), meta.get("slug"), injection_hits,
                )
                continue
            logger.info(
                "retrieve_context: flagged chunk %s/%s (injection hit: %s)",
                meta.get("kb"), meta.get("slug"), injection_hits[0],
            )
        score = _dot_product(question_vec, _normalize_vector(vec))
        entry = {**meta, "score": float(score)}
        if injection_hits:
            entry["injection_flags"] = injection_hits
        scored.append(entry)

    if flagged_count:
        logger.info(
            "retrieve_context: %d/%d chunks flagged, %d dropped",
            flagged_count, len(chunks_meta), dropped_count,
        )

    scored.sort(key=lambda r: r["score"], reverse=True)

    # Optional LLM cross-encoder rerank — opt-in via CROSS_ENCODER_RERANK.
    # Takes the top ``rerank_pool`` passages, asks the LLM to score each
    # one 0-10 for relevance to the question, and blends the new score
    # with the original cosine score. Cheap (1 LLM call per query) and
    # high-signal for multi-intent questions where bi-encoder similarity
    # can be fooled by surface-form matches.
    import os as _os
    if _os.getenv("CROSS_ENCODER_RERANK", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            scored[: k * 2] = await _rerank_with_llm(question, scored[: k * 2])
            scored.sort(key=lambda r: r["score"], reverse=True)
        except Exception as exc:
            logger.debug("cross-encoder rerank skipped: %s", exc)

    return scored[:k]


async def _rerank_with_llm(question: str, candidates: list[dict]) -> list[dict]:
    """Ask the LLM to score (query, passage) pairs and blend with existing scores.

    Sends a compact prompt listing every candidate passage, asks for a
    JSON array of relevance scores in [0, 10], and computes a blended
    final score as ``0.4 * old_normalized + 0.6 * new_normalized``.
    Bi-encoder similarity stays partially in play so the rerank can't
    catastrophically reorder results on a noisy LLM response.
    """
    import json as _json
    from app.llm import llm_chat

    if not candidates:
        return candidates

    # Build the passage list with stable numeric IDs.
    items = []
    for i, c in enumerate(candidates):
        body = (c.get("chunk_text") or "")[:600]
        items.append(f"[{i}] {c.get('title', '')}: {body}")
    passages_block = "\n\n".join(items)

    system_msg = (
        "You score retrieval passages 0-10 for how well each answers a "
        "user question. Return ONLY a JSON array of objects: "
        '[{"id": 0, "score": 8}, {"id": 1, "score": 3}, ...]. '
        "No prose, no preamble, no code fences."
    )
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"PASSAGES:\n{passages_block}\n\n"
        "Score each passage by how directly it answers the question."
    )

    raw = await llm_chat(
        system_msg=system_msg,
        user_msg=user_msg,
        max_tokens=500,
        temperature=0.0,
    )

    raw = (raw or "").strip()
    # Strip any ``` fences.
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```"))
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        return candidates
    try:
        data = _json.loads(raw[start : end + 1])
    except _json.JSONDecodeError:
        return candidates
    if not isinstance(data, list):
        return candidates

    new_scores: dict[int, float] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("id", -1))
            score = float(entry.get("score", 0)) / 10.0
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates):
            new_scores[idx] = max(0.0, min(1.0, score))

    if not new_scores:
        return candidates

    # Normalise the existing cosine scores to [0, 1] for a fair blend.
    old_max = max((c.get("score", 0.0) for c in candidates), default=1.0) or 1.0
    blended: list[dict] = []
    for i, c in enumerate(candidates):
        old_norm = float(c.get("score", 0.0)) / old_max
        new_norm = new_scores.get(i, old_norm)
        c_copy = dict(c)
        c_copy["score"] = 0.4 * old_norm + 0.6 * new_norm
        c_copy["rerank_score"] = new_scores.get(i)
        blended.append(c_copy)
    return blended


# ---------------------------------------------------------------------------
# 1.4 — Conversation memory + token budget
# ---------------------------------------------------------------------------


async def get_recent_history(
    session_id: str, *, max_tokens: int = 2000,
) -> list[dict]:
    """Load recent chat history under a token budget.

    Walks newest-first, accumulating until the budget is exhausted, then
    returns the messages oldest-first as ``{role, content}`` dicts ready
    for the LLM. The current user turn is NOT included — the caller adds
    it after building the prompt.
    """
    try:
        all_msgs = await db.get_chat_messages(session_id)
    except Exception as exc:
        logger.debug("get_chat_messages failed for %s: %s", session_id, exc)
        return []

    if not all_msgs:
        return []

    kept: list[dict] = []
    used_tokens = 0
    # Walk newest-first
    for msg in reversed(all_msgs):
        content = msg.get("content") or ""
        cost = _estimate_tokens(content)
        if used_tokens + cost > max_tokens and kept:
            break
        used_tokens += cost
        kept.append({"role": msg.get("role", "user"), "content": content})

    kept.reverse()  # Restore chronological order
    return kept


# ---------------------------------------------------------------------------
# 1.1 — Prompt assembly
# ---------------------------------------------------------------------------


CHAT_SYSTEM_PROMPT = (
    "You are wikidelve's research assistant. You answer questions about a "
    "self-hosted knowledge base of articles the user has researched.\n\n"
    "RULES:\n"
    "- Ground every factual claim in one or more of the provided passages.\n"
    "- Cite sources inline using markdown links: [Title](url) where url is "
    "the passage's url field. Cite at the end of the sentence the claim "
    "appears in.\n"
    "- If the passages don't contain enough information to fully answer, "
    "say so explicitly and end your response with a line of the form "
    "`[GAP] <specific topic the KB should research next>` (one per gap, "
    "each on its own line). The auto-discovery system will queue these.\n"
    "- Prefer concise, direct answers over padding.\n"
    "- Do not invent URLs or titles. Only cite passages that were actually "
    "given to you.\n"
    "- Write in English."
)


def build_chat_prompt(
    question: str,
    passages: list[dict],
    history: list[dict],
) -> tuple[str, str]:
    """Assemble the (system, user) message pair for the LLM.

    The system prompt is fixed (`CHAT_SYSTEM_PROMPT`).
    The user message stitches together history + retrieved passages + the
    new question. Passages are formatted as numbered blocks with the URL
    inline so the model can cite them.
    """
    system_msg = CHAT_SYSTEM_PROMPT

    parts: list[str] = []

    if history:
        parts.append("## Conversation history")
        for msg in history:
            role = msg.get("role", "user").upper()
            parts.append(f"**{role}:** {msg.get('content', '').strip()}")
        parts.append("")

    if passages:
        parts.append("## Retrieved passages from your knowledge base")
        for i, p in enumerate(passages, 1):
            parts.append(
                f"### Passage {i}: {p.get('title', '?')} "
                f"({p.get('url', '')})"
            )
            parts.append(p.get("chunk_text", "").strip())
            parts.append("")
    else:
        parts.append("## Retrieved passages from your knowledge base")
        parts.append("(no passages matched the question)")
        parts.append("")

    parts.append("## Current question")
    parts.append(question.strip())

    user_msg = "\n".join(parts)
    return system_msg, user_msg


# ---------------------------------------------------------------------------
# 1.6 — [GAP] sentinel parser
# ---------------------------------------------------------------------------


_GAP_RE = re.compile(r"^\s*\[GAP\]\s*(.+?)\s*$", re.MULTILINE)


def parse_gap_topics(text: str) -> list[str]:
    """Extract `[GAP] <topic>` lines from an LLM response.

    Returns a list of topic strings (deduplicated, in order of first
    appearance). Tolerates trailing whitespace and multiple gaps.
    """
    if not text:
        return []
    seen: list[str] = []
    for match in _GAP_RE.finditer(text):
        topic = match.group(1).strip()
        if topic and topic not in seen and len(topic) >= 5:
            seen.append(topic)
    return seen


# ---------------------------------------------------------------------------
# 1.3 — Tool definitions + dispatcher
# ---------------------------------------------------------------------------


CHAT_TOOLS: list[dict] = [
    {
        "name": "search_kb",
        "description": "Hybrid search the wikidelve knowledge base. Returns "
        "the top-N matching articles with title, slug, kb, snippet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "kb": {
                    "type": "string",
                    "description": "Optional KB name; omit to search all KBs",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_article",
        "description": "Fetch the full markdown content of an article by KB + slug.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kb": {"type": "string"},
                "slug": {"type": "string"},
            },
            "required": ["kb", "slug"],
        },
    },
    {
        "name": "find_related",
        "description": "Fuzzy-match a topic against existing article titles. "
        "Use this to check whether a topic is already in the KB before "
        "queuing new research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kb": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["kb", "topic"],
        },
    },
    {
        "name": "get_graph_neighbors",
        "description": "Walk the knowledge graph from an article and return "
        "related articles within `depth` hops.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kb": {"type": "string"},
                "slug": {"type": "string"},
                "depth": {"type": "integer", "default": 2},
            },
            "required": ["kb", "slug"],
        },
    },
    {
        "name": "enqueue_research",
        "description": "Queue a new research_task for a topic the KB doesn't "
        "yet cover. Costs Serper API calls — use sparingly. Returns the "
        "job_id of the queued task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "kb": {"type": "string"},
            },
            "required": ["topic", "kb"],
        },
    },
    {
        "name": "refine_research",
        "description": "Queue a follow-up research_task scoped to a "
        "specific section of an existing article. Use this when the user "
        "asks to 'dig deeper into the Limitations section' or 'expand "
        "the troubleshooting part'. Costs Serper API calls. The scoped "
        "topic is built from the parent article title + the section "
        "name so the synthesis pass merges the new content into the "
        "right place.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kb": {"type": "string"},
                "slug": {
                    "type": "string",
                    "description": "Slug of the parent article to refine",
                },
                "section": {
                    "type": "string",
                    "description": "Name of the section to expand "
                    "(e.g. 'Limitations', 'Practical Recommendations')",
                },
            },
            "required": ["kb", "slug", "section"],
        },
    },
]


async def run_tool(name: str, tool_input: dict, *, redis: Any = None) -> dict:
    """Dispatch a tool call to its real implementation.

    Returns a dict shaped for the LLM to consume. Errors are caught and
    returned as ``{"error": str}`` so the model can reason about them
    instead of crashing the conversation.
    """
    try:
        if name == "search_kb":
            query = tool_input.get("query") or ""
            kb = tool_input.get("kb")
            limit = int(tool_input.get("limit") or 10)
            results = await hybrid_search(query, kb_name=kb, limit=limit)
            return {"results": results}

        if name == "get_article":
            kb = tool_input["kb"]
            slug = tool_input["slug"]
            article = get_article(kb, slug)
            if not article:
                return {"error": f"Article not found: {kb}/{slug}"}
            return {
                "kb": kb,
                "slug": slug,
                "title": article.get("title"),
                "summary": article.get("summary"),
                "tags": article.get("tags", []),
                "raw_markdown": article.get("raw_markdown", ""),
                "word_count": article.get("word_count", 0),
            }

        if name == "find_related":
            kb = tool_input["kb"]
            topic = tool_input["topic"]
            match = find_related_article(kb, topic)
            return {"match": match}

        if name == "get_graph_neighbors":
            kb = tool_input["kb"]
            slug = tool_input["slug"]
            depth = int(tool_input.get("depth") or 2)
            results = await get_related_by_graph(slug, kb_name=kb, depth=depth)
            return {"neighbors": results}

        if name == "enqueue_research":
            if redis is None:
                return {"error": "Tool execution requires a redis pool"}
            topic = tool_input["topic"]
            kb = tool_input["kb"]
            # Cooldown check — same as the auto-discovery enqueue path
            existing = await db.check_cooldown(topic)
            if existing:
                return {
                    "error": "cooldown",
                    "existing_job_id": existing["id"],
                }
            job_id = await db.create_job(topic)
            from app.config import ARQ_QUEUE_NAME
            await redis.enqueue_job(
                "research_task",
                topic,
                job_id,
                kb,
                _queue_name=ARQ_QUEUE_NAME,
            )
            return {"job_id": job_id, "status": "queued", "topic": topic, "kb": kb}

        if name == "refine_research":
            # Scoped follow-up research for an existing article.
            if redis is None:
                return {"error": "Tool execution requires a redis pool"}
            kb = tool_input["kb"]
            slug = tool_input["slug"]
            section = tool_input["section"]
            # Resolve the article so we can build a meaningful research topic.
            article = get_article(kb, slug)
            if not article:
                return {"error": f"Article not found: {kb}/{slug}"}
            title = article.get("title") or slug.replace("-", " ").title()
            # Compose a scoped topic. fuzzy-merge in create_or_update_article
            # will route the resulting content back into this same article.
            scoped_topic = f"{title}: deeper dive on {section}"
            existing = await db.check_cooldown(scoped_topic)
            if existing:
                return {"error": "cooldown", "existing_job_id": existing["id"]}
            job_id = await db.create_job(scoped_topic)
            from app.config import ARQ_QUEUE_NAME
            await redis.enqueue_job(
                "research_task",
                scoped_topic,
                job_id,
                kb,
                _queue_name=ARQ_QUEUE_NAME,
            )
            return {
                "job_id": job_id,
                "status": "queued",
                "topic": scoped_topic,
                "parent_slug": slug,
                "kb": kb,
            }

        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool %s failed: %s", name, exc)
        return {"error": str(exc)}
