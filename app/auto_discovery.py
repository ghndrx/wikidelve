"""
Auto-discovery: continuous research via Serper budget.

Two phases:
  * Refill  -- generate candidate topics from each enabled KB (KG entities
               and/or LLM follow-ups from recent articles), filtered by the
               KB's seed topics if configured. Writes to topic_candidates.
  * Enqueue -- drain the candidate queue per KB into research_task jobs,
               respecting the per-KB daily Serper budget and hourly cap.

Configuration lives per-KB in the auto_discovery_config table. A single
global env var AUTO_DISCOVERY_ENABLED acts as a master kill switch.
"""

from __future__ import annotations

import json
import logging
import random
import re
from datetime import datetime, timedelta, timezone

import aiosqlite

from app import db, storage
from app.config import (
    ARQ_QUEUE_NAME,
    AUTO_DISCOVERY_ENABLED,
    DB_PATH,
    SERPER_CALLS_PER_JOB_ESTIMATE,
)
from app.llm import llm_chat
from app.wiki import find_related_article, get_article, get_articles

logger = logging.getLogger("kb-service.auto_discovery")


# --- KG discovery -----------------------------------------------------------

async def _kg_entities_for_seed_articles(
    kb: str, seed_slugs: list[str] | None,
) -> list[dict]:
    """Fetch KG entities co-occurring with the given article slugs.

    If ``seed_slugs`` is None, returns the globally top entities (ordered by
    article_count desc). Otherwise restricts to entities that appear as
    source or target in kg_edges for any of the seed article slugs (in this
    KB).
    """
    aconn = await aiosqlite.connect(str(DB_PATH))
    aconn.row_factory = aiosqlite.Row
    try:
        if seed_slugs:
            placeholders = ",".join("?" * len(seed_slugs))
            query = f"""
                SELECT DISTINCT e.id, e.name, e.type, e.article_count
                FROM kg_entities e
                JOIN kg_edges ed
                  ON e.id = ed.source_entity_id OR e.id = ed.target_entity_id
                WHERE ed.kb = ? AND ed.article_slug IN ({placeholders})
                ORDER BY e.article_count DESC
            """
            cursor = await aconn.execute(query, [kb, *seed_slugs])
        else:
            cursor = await aconn.execute(
                "SELECT id, name, type, article_count FROM kg_entities "
                "ORDER BY article_count DESC LIMIT 500"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await aconn.close()


def _resolve_seed_slugs(kb: str, seed_topics: list[str] | None) -> list[str] | None:
    """Turn seed topic strings into a list of article slugs via fuzzy match.

    Returns None when seed_topics is None/empty (= whole-KB mode).
    """
    if not seed_topics:
        return None

    slugs: set[str] = set()
    for seed in seed_topics:
        if not seed or not seed.strip():
            continue
        # Direct fuzzy match
        match = find_related_article(kb, seed)
        if match:
            slugs.add(match["slug"])
        # Also grab any article whose tags contain the seed substring
        seed_lower = seed.lower().strip()
        for art in get_articles(kb):
            tags = [t.lower() for t in art.get("tags", []) if isinstance(t, str)]
            if any(seed_lower in tag for tag in tags):
                slugs.add(art["slug"])
            elif seed_lower in art.get("title", "").lower():
                slugs.add(art["slug"])

    return sorted(slugs)


async def discover_from_kg_entities(
    kb: str, seed_topics: list[str] | None, limit: int = 50,
) -> int:
    """Insert candidate topics from KG entities that lack a wiki article.

    Returns number of new candidates inserted.
    """
    seed_slugs = _resolve_seed_slugs(kb, seed_topics)
    if seed_topics and not seed_slugs:
        logger.info(
            "auto_discovery[kg] kb=%s: seed_topics %r matched no articles; skipping",
            kb, seed_topics,
        )
        return 0

    entities = await _kg_entities_for_seed_articles(kb, seed_slugs)
    if not entities:
        logger.info("auto_discovery[kg] kb=%s: no entities found", kb)
        return 0

    inserted = 0
    for entity in entities:
        if inserted >= limit:
            break
        name = (entity.get("name") or "").strip()
        if not name or len(name) < 3:
            continue
        # Needs to look like a meaningful research topic
        if len(name) > 120:
            continue

        # Skip if an article on this topic already exists
        if find_related_article(kb, name):
            continue
        # Skip if recently researched (respects the 7-day cooldown)
        if await db.check_cooldown(name):
            continue

        did_insert = await db.insert_topic_candidate(
            kb, name, "kg_entity", name, float(entity.get("article_count") or 0),
        )
        if did_insert:
            inserted += 1

    logger.info("auto_discovery[kg] kb=%s: %d candidates inserted", kb, inserted)
    return inserted


# --- LLM follow-ups ---------------------------------------------------------

_FOLLOWUP_SYSTEM = (
    "You propose focused follow-up research topics. Return ONLY the topics, "
    "one per line, no numbering, no bullets, no commentary."
)


async def _generate_followup_topics(
    title: str, content: str, per_article: int,
) -> list[str]:
    """Ask the LLM for N broader follow-up research topics about an article.

    Modeled on _generate_sub_questions in app/research.py: same llm_chat
    shape, tuned for open-ended research expansion (not verification).
    """
    truncated = content[:4000]
    prompt = (
        f'An article titled "{title}" has been researched. Propose exactly '
        f"{per_article} distinct NEW research topics that would naturally "
        f"extend or deepen the knowledge base beyond this article. Each "
        f"topic must be a standalone, search-friendly phrase (8-80 chars), "
        f"NOT a question. Do not repeat the article's own topic.\n\n"
        f"Article excerpt:\n{truncated}"
    )
    try:
        response = await llm_chat(
            system_msg=_FOLLOWUP_SYSTEM,
            user_msg=prompt,
            max_tokens=300,
            temperature=0.4,
        )
    except Exception as exc:
        logger.warning("Follow-up topic generation failed: %s", exc)
        return []

    topics: list[str] = []
    for line in response.strip().split("\n"):
        t = line.strip().lstrip("0123456789.-) ").strip("\"'")
        if 10 <= len(t) <= 120:
            topics.append(t)
        if len(topics) >= per_article:
            break
    return topics


async def _pick_source_articles(
    kb: str, seed_topics: list[str] | None, sample_size: int,
) -> list[dict]:
    """Select articles from the KB to feed to the follow-up LLM."""
    all_articles = get_articles(kb)
    if not all_articles:
        return []

    candidates = all_articles
    if seed_topics:
        seed_slugs = set(_resolve_seed_slugs(kb, seed_topics) or [])
        candidates = [a for a in all_articles if a["slug"] in seed_slugs]
        if not candidates:
            return []

    # Prefer more substantive articles (higher word count) but inject some
    # randomness so every refill isn't identical.
    candidates = sorted(
        candidates, key=lambda a: a.get("word_count", 0), reverse=True,
    )[: sample_size * 3]
    random.shuffle(candidates)
    return candidates[:sample_size]


async def discover_from_llm_followups(
    kb: str,
    seed_topics: list[str] | None,
    sample_size: int,
    per_article: int = 4,
) -> int:
    """Generate candidate topics by asking the LLM for follow-ups on sampled articles."""
    source_articles = await _pick_source_articles(kb, seed_topics, sample_size)
    if not source_articles:
        logger.info("auto_discovery[llm] kb=%s: no source articles", kb)
        return 0

    # We need the raw body for context, which get_articles doesn't include.
    # Import lazily to avoid a cycle at module load.
    from app.wiki import get_article

    inserted = 0
    HARD_CAP = 30
    for art in source_articles:
        if inserted >= HARD_CAP:
            break
        full = get_article(kb, art["slug"])
        if not full:
            continue
        body = full.get("raw_markdown") or full.get("body") or ""
        if not body:
            continue

        topics = await _generate_followup_topics(
            full.get("title", art["slug"]), body, per_article,
        )
        for topic in topics:
            if inserted >= HARD_CAP:
                break
            if find_related_article(kb, topic):
                continue
            if await db.check_cooldown(topic):
                continue
            did_insert = await db.insert_topic_candidate(
                kb, topic, "llm_followup", art["slug"], 1.0,
            )
            if did_insert:
                inserted += 1

    logger.info("auto_discovery[llm] kb=%s: %d candidates inserted", kb, inserted)
    return inserted


# ---------------------------------------------------------------------------
# Self-mining discovery strategies
# ---------------------------------------------------------------------------
#
# These strategies mine signal from data the wiki already has, without
# burning extra Serper budget at discovery time. They all write to the
# same ``topic_candidates`` queue the auto-discovery enqueue loop drains,
# distinguished by ``source``:
#
#     contradiction   → article-pair contradictions
#     stale           → 90-day-old high-velocity articles
#     orphan_entity   → entities mentioned in ≥3 articles, no own page
#     question        → unanswered ?-ending sentences in bodies
#     blind_spot      → sparse research-history clusters
#     stub            → TOC bullets that never got a real section
#     wikilink        → [[broken wikilinks]]
#
# Strategies are deliberately defensive: every LLM/DB call is wrapped so a
# single bad article can't kill a whole discovery pass.


# A small set of tags that indicate the topic moves fast and is worth
# re-researching periodically. Used by the stale-article rerun strategy.
_HIGH_VELOCITY_TAGS = {
    "ai", "ml", "llm", "security", "kubernetes", "k8s", "rust", "go",
    "python", "javascript", "browser", "react", "nextjs", "node",
    "docker", "linux", "cloud", "aws", "gcp", "azure", "blockchain",
    "crypto", "frontend", "backend", "devops", "infrastructure",
    "framework", "runtime", "compiler",
}


# --- 4.1: Contradiction harvester ------------------------------------------


_CONTRADICTION_SYSTEM = (
    "You compare two short article excerpts about the same topic and "
    "decide whether they make claims that contradict each other. Return "
    "ONLY a JSON object: "
    '{"contradicts": true|false, "topic": "<short topic that needs '
    "fresh research to resolve>\", \"reason\": \"<one sentence>\"}. "
    "Mark contradicts=true only if there's a clear factual conflict "
    "(numbers, dates, version specs, behavior). Differences in scope or "
    "framing don't count."
)


async def _find_article_pairs_with_overlap(
    kb: str, max_pairs: int = 8,
) -> list[tuple[dict, dict]]:
    """Return article pairs that share at least one tag AND one KG entity.

    Cheap heuristic for "these talk about the same thing": tag overlap is
    the first filter (so we don't run an O(N²) loop on the whole KB), and
    KG entity overlap is the second (so we only ask the LLM about pairs
    that almost certainly cover the same ground).
    """
    articles = get_articles(kb)
    if len(articles) < 2:
        return []

    # Bucket articles by tag → list of slugs
    by_tag: dict[str, list[dict]] = {}
    for art in articles:
        for tag in art.get("tags", []) or []:
            if not isinstance(tag, str) or not tag.strip():
                continue
            by_tag.setdefault(tag.lower().strip(), []).append(art)

    # Pull KG edges for this KB so we can filter on entity overlap.
    aconn = await aiosqlite.connect(str(DB_PATH))
    aconn.row_factory = aiosqlite.Row
    try:
        cursor = await aconn.execute(
            "SELECT article_slug, source_entity_id, target_entity_id "
            "FROM kg_edges WHERE kb = ?",
            (kb,),
        )
        rows = await cursor.fetchall()
    finally:
        await aconn.close()

    entities_by_slug: dict[str, set[int]] = {}
    for r in rows:
        slug = r["article_slug"]
        if not slug:
            continue
        entities_by_slug.setdefault(slug, set()).update(
            x for x in (r["source_entity_id"], r["target_entity_id"]) if x
        )

    seen_pairs: set[frozenset] = set()
    pairs: list[tuple[dict, dict]] = []
    for tag, members in by_tag.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = frozenset((a["slug"], b["slug"]))
                if key in seen_pairs:
                    continue
                ent_a = entities_by_slug.get(a["slug"], set())
                ent_b = entities_by_slug.get(b["slug"], set())
                if not (ent_a & ent_b):
                    continue
                seen_pairs.add(key)
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


async def discover_from_contradictions(
    kb: str, max_pairs: int = 6,
) -> int:
    """Mine article pairs for contradictions; queue resolutions as candidates."""
    pairs = await _find_article_pairs_with_overlap(kb, max_pairs=max_pairs)
    if not pairs:
        logger.info("auto_discovery[contradiction] kb=%s: no overlapping pairs", kb)
        return 0

    inserted = 0
    for a, b in pairs:
        full_a = get_article(kb, a["slug"])
        full_b = get_article(kb, b["slug"])
        if not (full_a and full_b):
            continue
        body_a = (full_a.get("raw_markdown") or "")[:3000]
        body_b = (full_b.get("raw_markdown") or "")[:3000]
        if not body_a or not body_b:
            continue

        user_msg = (
            f'Article A: "{full_a.get("title", a["slug"])}"\n{body_a}\n\n'
            f'Article B: "{full_b.get("title", b["slug"])}"\n{body_b}\n\n'
            f"Do these contradict on any factual claim?"
        )
        try:
            raw = await llm_chat(
                system_msg=_CONTRADICTION_SYSTEM,
                user_msg=user_msg,
                max_tokens=300,
                temperature=0.1,
            )
        except Exception as exc:
            logger.warning("Contradiction LLM call failed: %s", exc)
            continue

        # Tolerant JSON parse
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("\n", 1)
            if len(parts) == 2:
                text = parts[1].rstrip("`").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            continue
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not obj.get("contradicts"):
            continue
        topic = (obj.get("topic") or "").strip()
        if not topic or len(topic) < 8:
            continue
        if find_related_article(kb, topic):
            continue
        if await db.check_cooldown(topic):
            continue
        ref = f"{a['slug']}|{b['slug']}"
        did_insert = await db.insert_topic_candidate(
            kb, topic, "contradiction", ref, 2.0,
        )
        if did_insert:
            inserted += 1

    logger.info("auto_discovery[contradiction] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- 4.2: Stale article rerun ----------------------------------------------


async def discover_from_stale(kb: str, days: int = 90) -> int:
    """Queue 90-day-old high-velocity articles for re-research."""
    articles = get_articles(kb)
    if not articles:
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

    inserted = 0
    for art in articles:
        updated = (art.get("updated") or "").strip()[:10]
        if not updated or updated >= cutoff:
            continue
        tags = {
            (t or "").lower().strip()
            for t in art.get("tags", []) or []
            if isinstance(t, str)
        }
        if not (tags & _HIGH_VELOCITY_TAGS):
            continue
        # Topic = article title; the resulting research will merge into the
        # existing slug via create_or_update_article's fuzzy match.
        topic = (art.get("title") or art.get("slug") or "").strip()
        if not topic:
            continue
        if await db.check_cooldown(topic):
            continue
        did_insert = await db.insert_topic_candidate(
            kb, topic, "stale", art["slug"], 1.5,
        )
        if did_insert:
            inserted += 1

    logger.info("auto_discovery[stale] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- 4.3: Orphan entity mining ---------------------------------------------


async def discover_from_orphan_mentions(
    kb: str, min_articles: int = 3, limit: int = 30,
) -> int:
    """Entities referenced in ≥N articles but with no dedicated wiki page."""
    aconn = await aiosqlite.connect(str(DB_PATH))
    aconn.row_factory = aiosqlite.Row
    try:
        # For each entity in this KB, count distinct articles that reference it.
        cursor = await aconn.execute(
            """
            SELECT e.id, e.name, COUNT(DISTINCT ed.article_slug) AS article_count
            FROM kg_entities e
            JOIN kg_edges ed
              ON e.id = ed.source_entity_id OR e.id = ed.target_entity_id
            WHERE ed.kb = ?
            GROUP BY e.id, e.name
            HAVING article_count >= ?
            ORDER BY article_count DESC
            LIMIT ?
            """,
            (kb, min_articles, limit * 3),
        )
        candidates = await cursor.fetchall()
    finally:
        await aconn.close()

    inserted = 0
    for row in candidates:
        if inserted >= limit:
            break
        name = (row["name"] or "").strip()
        if not name or len(name) < 3 or len(name) > 120:
            continue
        if find_related_article(kb, name):
            continue
        if await db.check_cooldown(name):
            continue
        did_insert = await db.insert_topic_candidate(
            kb, name, "orphan_entity", name, float(row["article_count"]),
        )
        if did_insert:
            inserted += 1

    logger.info("auto_discovery[orphan_entity] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- 4.4: Question miner ---------------------------------------------------


# Match a sentence-ending question. Lookbehind to avoid matching "?" inside
# code blocks or URLs (very rough heuristic, but good enough).
_QUESTION_RE = re.compile(
    r"(?<![/\w])([A-Z][^.?!\n]{15,200}\?)",
)


async def discover_from_questions(kb: str, limit: int = 20) -> int:
    """Mine ?-ending sentences from article bodies as research candidates."""
    articles = get_articles(kb)
    if not articles:
        return 0

    inserted = 0
    for art in articles:
        if inserted >= limit:
            break
        full = get_article(kb, art["slug"])
        if not full:
            continue
        body = full.get("raw_markdown") or ""
        if not body or len(body) < 200:
            continue

        # Strip code blocks before scanning so we don't pick up snippets
        without_code = re.sub(r"```[\s\S]*?```", " ", body)
        for match in _QUESTION_RE.finditer(without_code):
            if inserted >= limit:
                break
            question = match.group(1).strip()
            # Skip rhetorical / common stop questions
            lower = question.lower()
            if lower.startswith(("but ", "so ", "now ", "and ", "or ")):
                continue
            if find_related_article(kb, question):
                continue
            if await db.check_cooldown(question):
                continue
            did_insert = await db.insert_topic_candidate(
                kb, question[:200], "question", art["slug"], 0.8,
            )
            if did_insert:
                inserted += 1

    logger.info("auto_discovery[question] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- 4.5: Research history clustering --------------------------------------


async def cluster_research_history(kb: str, limit: int = 10) -> int:
    """Find sparse clusters in completed research history → blind spots.

    Pulls embeddings for recent completed research jobs in this KB. Runs a
    cheap O(N²) similarity comparison (we're not at the scale where a
    proper k-means matters yet). Topics whose nearest neighbour is far
    away (low max-similarity score) are "lonely" — likely under-served
    areas the KB should expand around.

    Each lonely topic gets a candidate inserted with a placeholder topic
    derived from its nearest neighbour, so the LLM follow-up generator
    has something to work with.
    """
    aconn = await aiosqlite.connect(str(DB_PATH))
    aconn.row_factory = aiosqlite.Row
    try:
        cursor = await aconn.execute(
            "SELECT id, topic, content FROM research_jobs "
            "WHERE status = 'complete' AND wiki_kb = ? AND content IS NOT NULL "
            "ORDER BY id DESC LIMIT 50",
            (kb,),
        )
        jobs = await cursor.fetchall()
    finally:
        await aconn.close()

    if len(jobs) < 5:
        # Not enough data to cluster meaningfully.
        return 0

    # Embed each job's content. Use embed_texts (batch) for efficiency.
    from app.embeddings import embed_texts, _normalize_vector, _dot_product

    texts = [(j["topic"] or "")[:500] for j in jobs]
    try:
        vectors = await embed_texts(texts)
    except Exception as exc:
        logger.warning("Cluster embeddings failed: %s", exc)
        return 0
    if not vectors or len(vectors) != len(jobs):
        return 0

    normalized = [_normalize_vector(v) for v in vectors]

    # For each job, find its highest similarity to any other job.
    inserted = 0
    sparse_jobs: list[tuple[float, dict]] = []
    for i, vi in enumerate(normalized):
        best = 0.0
        for j, vj in enumerate(normalized):
            if i == j:
                continue
            sim = _dot_product(vi, vj)
            if sim > best:
                best = sim
        sparse_jobs.append((best, dict(jobs[i])))

    # Lowest max-similarity → most isolated topics
    sparse_jobs.sort(key=lambda x: x[0])
    for sim, job in sparse_jobs[:limit]:
        topic = (job.get("topic") or "").strip()
        if not topic:
            continue
        # Use the LLM to propose a related-but-different research topic
        # for this isolated subject. Cheap fallback: prepend "Deep dive:"
        # so we don't immediately re-research the same topic.
        candidate_topic = f"Adjacent topics to {topic}"[:200]
        if await db.check_cooldown(candidate_topic):
            continue
        did_insert = await db.insert_topic_candidate(
            kb, candidate_topic, "blind_spot", topic, float(1.0 - sim),
        )
        if did_insert:
            inserted += 1

    logger.info("auto_discovery[blind_spot] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- 4.6: TOC stub miner ---------------------------------------------------


# Match `- bullet text` followed by either a header (next ##) or another
# bullet — i.e. a bullet that has no real prose under it.
_STUB_BULLET_RE = re.compile(
    r"^\s*[-*]\s+(.{8,200}?)\s*$", re.MULTILINE,
)


async def discover_from_stubs(kb: str, limit: int = 20) -> int:
    """Mine bullets that look like topics but never got a real section."""
    articles = get_articles(kb)
    if not articles:
        return 0

    inserted = 0
    for art in articles:
        if inserted >= limit:
            break
        full = get_article(kb, art["slug"])
        if not full:
            continue
        body = full.get("raw_markdown") or ""
        if not body:
            continue

        title = full.get("title") or art["slug"]
        # Find bullets in the body and treat any that look like a topic
        # phrase (capitalized noun phrase, no verb endings) as a stub.
        for match in _STUB_BULLET_RE.finditer(body):
            if inserted >= limit:
                break
            stub = match.group(1).strip()
            # Skip prose-y bullets (sentences with periods or ending in a
            # verb-like word) — we want short noun-phrase stubs.
            if len(stub) > 80 or "." in stub:
                continue
            # Combine with parent article title for context
            topic = f"{stub} (in context of {title})"[:200]
            if find_related_article(kb, stub):
                continue
            if await db.check_cooldown(topic):
                continue
            did_insert = await db.insert_topic_candidate(
                kb, topic, "stub", art["slug"], 0.7,
            )
            if did_insert:
                inserted += 1

    logger.info("auto_discovery[stub] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- 4.7: Broken wikilink miner --------------------------------------------


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def _wikilink_to_slug_local(title: str) -> str:
    """Mirror app/wiki.py:_wikilink_to_slug without importing the private name."""
    safe = "".join(c if c.isalnum() or c in " -" else "" for c in title)
    return safe.lower().strip().replace(" ", "-")


async def discover_from_broken_wikilinks(kb: str, limit: int = 30) -> int:
    """Parse [[X]] wikilinks; any X with no resolving slug becomes a candidate.

    This is the link-integrity helper the MEMORY note flagged. Walks every
    article's RAW file (not through ``get_article``, which rewrites
    wikilinks to markdown links before returning), extracts ``[[X]]``
    tokens, slugifies them, and queues anything that doesn't resolve.
    """
    articles = get_articles(kb)
    if not articles:
        return 0
    known_slugs = {a["slug"] for a in articles}

    inserted = 0
    for art in articles:
        if inserted >= limit:
            break
        # Read the raw file directly so [[X]] tokens are still intact.
        slug = art["slug"]
        body = storage.read_text(kb, f"wiki/{slug}.md")
        if not body:
            continue

        for match in _WIKILINK_RE.finditer(body):
            if inserted >= limit:
                break
            link_title = match.group(1).strip()
            if not link_title or len(link_title) < 3:
                continue
            target_slug = _wikilink_to_slug_local(link_title)
            if not target_slug or target_slug in known_slugs:
                continue
            # Skip if a related article exists under a different slug
            if find_related_article(kb, link_title):
                continue
            if await db.check_cooldown(link_title):
                continue
            did_insert = await db.insert_topic_candidate(
                kb, link_title[:200], "wikilink", art["slug"], 1.2,
            )
            if did_insert:
                inserted += 1

    logger.info("auto_discovery[wikilink] kb=%s: %d candidates", kb, inserted)
    return inserted


# --- Discovery dispatch -----------------------------------------------------

# Self-mining discovery strategies (run individually or via "all" mode).
_MINING_STRATEGIES = (
    "contradiction", "stale", "orphan_entity",
    "question", "blind_spot", "stub", "wikilink",
)


def _parse_seed_topics(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    seeds = [str(s).strip() for s in parsed if str(s).strip()]
    return seeds or None


async def run_discovery_for_kb(kb: str) -> dict:
    """Refill candidates for a single KB according to its config.

    Strategy values:
        kg_entities  → KG-entity discovery only
        llm          → LLM follow-ups only
        hybrid       → KG + LLM
        all          → hybrid PLUS every self-mining strategy
        <name>       → run only the named self-mining strategy
    """
    cfg = await db.get_auto_discovery_config(kb)
    if not cfg or not cfg.get("enabled"):
        return {"kb": kb, "skipped": "disabled"}

    seed_topics = _parse_seed_topics(cfg.get("seed_topics"))
    strategy = (cfg.get("strategy") or "hybrid").lower()
    llm_sample = int(cfg.get("llm_sample") or 5)

    counts: dict[str, int] = {}

    # KG / LLM strategies (back-compat with kg_entities / llm / hybrid)
    if strategy in ("kg_entities", "hybrid", "all"):
        try:
            counts["kg_entities"] = await discover_from_kg_entities(kb, seed_topics)
        except Exception as exc:
            logger.exception("kg_entities discovery failed: %s", exc)
            counts["kg_entities"] = 0
    if strategy in ("llm", "hybrid", "all"):
        try:
            counts["llm_followup"] = await discover_from_llm_followups(
                kb, seed_topics, sample_size=llm_sample,
            )
        except Exception as exc:
            logger.exception("llm follow-up discovery failed: %s", exc)
            counts["llm_followup"] = 0

    # Self-mining strategies — run all on "all", or pick a single one by name
    mining_dispatch = {
        "contradiction": discover_from_contradictions,
        "stale": discover_from_stale,
        "orphan_entity": discover_from_orphan_mentions,
        "question": discover_from_questions,
        "blind_spot": cluster_research_history,
        "stub": discover_from_stubs,
        "wikilink": discover_from_broken_wikilinks,
    }

    if strategy == "all":
        for name, fn in mining_dispatch.items():
            try:
                counts[name] = await fn(kb)
            except Exception as exc:
                logger.exception("%s discovery failed: %s", name, exc)
                counts[name] = 0
    elif strategy in mining_dispatch:
        try:
            counts[strategy] = await mining_dispatch[strategy](kb)
        except Exception as exc:
            logger.exception("%s discovery failed: %s", strategy, exc)
            counts[strategy] = 0

    total_pending = await db.count_pending_candidates(kb)
    return {
        "kb": kb,
        "strategy": strategy,
        "counts": counts,
        # Back-compat fields for the existing admin UI
        "kg_inserted": counts.get("kg_entities", 0),
        "llm_inserted": counts.get("llm_followup", 0),
        "total_pending": total_pending,
    }


async def run_discovery_all() -> dict:
    """Refill candidates for every enabled KB."""
    if not AUTO_DISCOVERY_ENABLED:
        return {"skipped": "global_disabled"}

    configs = await db.list_enabled_auto_discovery_configs()
    results = []
    for cfg in configs:
        kb = cfg["kb"]
        try:
            results.append(await run_discovery_for_kb(kb))
        except Exception as exc:
            logger.exception("auto_discovery refill failed for kb=%s", kb)
            results.append({"kb": kb, "error": str(exc)})
    return {"results": results}


# --- Enqueue loop -----------------------------------------------------------

async def _set_last_run(arq_pool, kb: str, payload: dict) -> None:
    """Write a per-KB status snapshot to Redis for the admin UI."""
    try:
        key = f"auto_discovery:last_run:{kb}"
        payload_with_ts = dict(payload)
        payload_with_ts["at"] = datetime.now(timezone.utc).isoformat()
        await arq_pool.set(key, json.dumps(payload_with_ts))
    except Exception as exc:
        logger.debug("auto_discovery: failed to write last_run snapshot: %s", exc)


async def enqueue_next_candidates_for_kb(arq_pool, kb: str) -> dict:
    """Drain up to N candidates from the KB's queue into research_task jobs."""
    if not AUTO_DISCOVERY_ENABLED:
        return {"kb": kb, "skipped": "global_disabled"}

    cfg = await db.get_auto_discovery_config(kb)
    if not cfg or not cfg.get("enabled"):
        return {"kb": kb, "skipped": "disabled"}

    daily_budget = int(cfg["daily_budget"])
    used = await db.serper_calls_today(kb)
    remaining = max(0, daily_budget - used)

    if remaining < SERPER_CALLS_PER_JOB_ESTIMATE:
        result = {
            "kb": kb, "skipped": "budget", "remaining": remaining, "used": used,
        }
        await _set_last_run(arq_pool, kb, result)
        return result

    max_per_hour = int(cfg["max_per_hour"])
    budget_slots = remaining // SERPER_CALLS_PER_JOB_ESTIMATE
    max_this_run = min(max_per_hour, budget_slots)

    candidates = await db.get_pending_candidates(kb, max_this_run)
    if not candidates:
        result = {"kb": kb, "enqueued": 0, "reason": "no_candidates"}
        await _set_last_run(arq_pool, kb, result)
        return result

    enqueued = 0
    skipped_cooldown = 0
    skipped_exists = 0
    for cand in candidates:
        topic = cand["topic"]
        cand_id = cand["id"]

        if await db.check_cooldown(topic):
            await db.mark_candidate_skipped(cand_id, "cooldown")
            skipped_cooldown += 1
            continue
        if find_related_article(kb, topic):
            await db.mark_candidate_skipped(cand_id, "exists")
            skipped_exists += 1
            continue

        job_id = await db.create_job(topic)
        await arq_pool.enqueue_job(
            "research_task",
            topic,
            job_id,
            kb,
            _queue_name=ARQ_QUEUE_NAME,
        )
        await db.mark_candidate_enqueued(cand_id, job_id)
        enqueued += 1

    result = {
        "kb": kb,
        "enqueued": enqueued,
        "skipped_cooldown": skipped_cooldown,
        "skipped_exists": skipped_exists,
        "remaining_budget": remaining - enqueued * SERPER_CALLS_PER_JOB_ESTIMATE,
    }
    await _set_last_run(arq_pool, kb, result)
    logger.info("auto_discovery[enqueue] %s", result)
    return result


async def enqueue_next_candidates_all(arq_pool) -> dict:
    """Enqueue candidates for every enabled KB."""
    if not AUTO_DISCOVERY_ENABLED:
        return {"skipped": "global_disabled"}

    configs = await db.list_enabled_auto_discovery_configs()
    results = []
    for cfg in configs:
        kb = cfg["kb"]
        try:
            results.append(await enqueue_next_candidates_for_kb(arq_pool, kb))
        except Exception as exc:
            logger.exception("auto_discovery enqueue failed for kb=%s", kb)
            results.append({"kb": kb, "error": str(exc)})
    return {"results": results}


async def get_status_for_kb(kb: str, arq_pool=None) -> dict:
    """Return a status dict for the admin UI / API."""
    cfg = await db.get_auto_discovery_config(kb)
    used = await db.serper_calls_today(kb)
    pending = await db.count_pending_candidates(kb)
    seed_topics = _parse_seed_topics(cfg.get("seed_topics")) if cfg else None

    last_run = None
    if arq_pool is not None:
        try:
            raw = await arq_pool.get(f"auto_discovery:last_run:{kb}")
            if raw:
                last_run = json.loads(raw)
        except Exception:
            last_run = None

    return {
        "kb": kb,
        "enabled": bool(cfg.get("enabled")) if cfg else False,
        "daily_budget": int(cfg["daily_budget"]) if cfg else 0,
        "max_per_hour": int(cfg["max_per_hour"]) if cfg else 0,
        "strategy": cfg.get("strategy") if cfg else None,
        "seed_topics": seed_topics,
        "llm_sample": int(cfg.get("llm_sample") or 0) if cfg else 0,
        "used_today": used,
        "remaining": max(0, (int(cfg["daily_budget"]) - used) if cfg else 0),
        "pending_candidates": pending,
        "last_run": last_run,
    }
