"""
Palace Mode: MemPalace-inspired spatial hierarchy for WikiDelve.

Organizes articles into a navigable structure:
  - Wings  = KBs (personal, pluto, etc.)
  - Halls  = Article types (how-to, architecture, comparison, etc.)
  - Rooms  = Topic clusters derived from knowledge graph entities
  - Palace Map = Lightweight JSON index for navigation and search boosting

Classification is regex/keyword-based (zero LLM calls).
"""

import asyncio
import logging
import re
import time
from collections import defaultdict

from app import db, storage

logger = logging.getLogger("kb-service.palace")

# --- Hall types and colors ---------------------------------------------------

HALL_TYPES = [
    "how-to", "architecture", "comparison", "reference",
    "troubleshooting", "deep-dive", "integration", "release-notes",
]

HALL_COLORS = {
    "how-to": "#2563eb",
    "architecture": "#7c3aed",
    "comparison": "#ea580c",
    "reference": "#059669",
    "troubleshooting": "#dc2626",
    "deep-dive": "#4338ca",
    "integration": "#0891b2",
    "release-notes": "#ca8a04",
}

HALL_LABELS = {
    "how-to": "How-To",
    "architecture": "Architecture",
    "comparison": "Comparison",
    "reference": "Reference",
    "troubleshooting": "Troubleshooting",
    "deep-dive": "Deep Dive",
    "integration": "Integration",
    "release-notes": "Release Notes",
}

# --- Classification rules ----------------------------------------------------

HALL_RULES = {
    "comparison": {
        "title_patterns": [r"\bvs\b", r"\bcompar", r"\balternative", r"\bversus\b", r"\bpros\b.*\bcons\b"],
        "tag_patterns": [r"\bvs\b", r"\bcompar", r"\balternative"],
        "body_patterns": [r"\|.*\|.*\|", r"pros.*cons", r"advantages.*disadvantages", r"trade.?off"],
        "header_patterns": [r"##\s.*comparison", r"##\s.*vs\b", r"##\s.*alternative"],
    },
    "how-to": {
        "title_patterns": [r"\bsetup\b", r"\bconfigur", r"\binstall", r"\bdeploy", r"\bhow[- ]to\b",
                           r"\bguide\b", r"\btutorial\b", r"\bpattern", r"\bbest[- ]practice"],
        "tag_patterns": [r"\bsetup\b", r"\bconfig", r"\bdeploy", r"\bguide", r"\btutorial"],
        "body_patterns": [r"```(?:bash|sh|yaml|shell)", r"step \d", r"first.*then", r"mkdir\b", r"apt\s+install"],
        "header_patterns": [r"##\s.*setup", r"##\s.*install", r"##\s.*configur", r"##\s.*prerequisites",
                            r"##\s.*getting started", r"##\s.*quick start"],
    },
    "troubleshooting": {
        "title_patterns": [r"\btroubleshoot", r"\bdebug", r"\berror", r"\bfix\b", r"\bissue", r"\bincident"],
        "tag_patterns": [r"\bdebug", r"\btroubleshoot", r"\berror", r"\bfix"],
        "body_patterns": [r"error:", r"traceback", r"stack trace", r"root cause", r"work.?around",
                          r"the fix", r"solution:", r"symptom"],
        "header_patterns": [r"##\s.*error", r"##\s.*debug", r"##\s.*troubleshoot", r"##\s.*common issues",
                            r"##\s.*problem", r"##\s.*fix"],
    },
    "architecture": {
        "title_patterns": [r"\barchitect", r"\bdesign\b", r"\binfrastructur", r"\bscal",
                           r"\bsystem\b", r"\boverview\b"],
        "tag_patterns": [r"\barchitect", r"\bdesign", r"\binfrastructure", r"\bscalability"],
        "body_patterns": [r"component", r"layer", r"service mesh", r"microservice", r"monolith",
                          r"event.driven", r"distributed"],
        "header_patterns": [r"##\s.*architect", r"##\s.*design", r"##\s.*component", r"##\s.*overview"],
    },
    "reference": {
        "title_patterns": [r"\breference\b", r"\bcheat[- ]sheet", r"\bapi\b", r"\bsecurity\b",
                           r"\bperformance\b", r"\boptimiz"],
        "tag_patterns": [r"\breference", r"\bapi\b", r"\bsecurity", r"\bperformance"],
        "body_patterns": [r"##\s+API", r"##\s+Methods", r"##\s+Parameters", r"##\s+Options"],
        "header_patterns": [r"##\s.*recommendations", r"##\s.*key takeaways", r"##\s.*summary"],
    },
    "deep-dive": {
        "title_patterns": [r"\bdeep[- ]dive", r"\badvanced\b", r"\binternals\b", r"\bunder the hood",
                           r"\bcomprehensive\b"],
        "tag_patterns": [r"\badvanced", r"\binternals", r"\bdeep.dive"],
        "body_patterns": [],
        "header_patterns": [r"##\s.*internals", r"##\s.*how it works", r"##\s.*under the hood"],
    },
    "integration": {
        "title_patterns": [r"\bintegrat", r"\bbridge\b", r"\bconnect", r"\bwebhook", r"\bpipeline",
                           r"\bmigrat", r"\bfederat"],
        "tag_patterns": [r"\bintegrat", r"\bbridge", r"\bpipeline", r"\bfederat"],
        "body_patterns": [r"webhook", r"event.driven", r"trigger", r"middleware", r"adapter"],
        "header_patterns": [r"##\s.*integrat", r"##\s.*connect", r"##\s.*bridge"],
    },
    "release-notes": {
        "title_patterns": [r"\brelease\b", r"\bchangelog", r"\bwhat.s new", r"\bv\d+\.\d+",
                           r"\bupdate\b", r"\bupgrade\b"],
        "tag_patterns": [r"\brelease", r"\bchangelog", r"\bupgrade"],
        "body_patterns": [r"breaking change", r"migration guide", r"new feature", r"deprecated"],
        "header_patterns": [r"##\s.*breaking changes", r"##\s.*migration", r"##\s.*what.s new",
                            r"##\s.*changelog"],
    },
}

# Scoring weights
TITLE_WEIGHT = 10
TAG_WEIGHT = 5
BODY_WEIGHT = 3
HEADER_WEIGHT = 7
MIN_THRESHOLD = 5


# --- Classification engine ---------------------------------------------------

def classify_article(title: str, tags: list[str], body: str, word_count: int = 0) -> tuple[str, float]:
    """Classify an article into a hall using regex/keyword scoring.

    Returns (hall_name, confidence).
    """
    title_lower = title.lower()
    tags_str = " ".join(t.lower() for t in tags) if tags else ""
    body_lower = body[:10000].lower()  # only scan first 10k chars
    headers = "\n".join(re.findall(r"^##\s+.+$", body, re.MULTILINE)).lower()

    scores: dict[str, int] = {}

    for hall, rules in HALL_RULES.items():
        score = 0

        for pat in rules.get("title_patterns", []):
            if re.search(pat, title_lower):
                score += TITLE_WEIGHT

        for pat in rules.get("tag_patterns", []):
            if re.search(pat, tags_str):
                score += TAG_WEIGHT

        for pat in rules.get("body_patterns", []):
            matches = len(re.findall(pat, body_lower))
            score += BODY_WEIGHT * min(matches, 3)  # cap at 3 hits per pattern

        for pat in rules.get("header_patterns", []):
            if re.search(pat, headers):
                score += HEADER_WEIGHT

        scores[hall] = score

    # Bonus: deep-dive for long, well-structured articles with no strong match
    h2_count = len(re.findall(r"^##\s+", body, re.MULTILINE))
    if word_count > 1200 and h2_count >= 4:
        scores["deep-dive"] = scores.get("deep-dive", 0) + 8

    # Find winner
    best_hall = max(scores, key=scores.get)
    best_score = scores[best_hall]

    if best_score < MIN_THRESHOLD:
        best_hall = "reference"
        best_score = MIN_THRESHOLD

    # Calculate confidence (max possible ~50 for a hall with 5 patterns per category)
    max_possible = max(
        len(rules.get("title_patterns", [])) * TITLE_WEIGHT +
        len(rules.get("tag_patterns", [])) * TAG_WEIGHT +
        min(len(rules.get("body_patterns", [])), 3) * BODY_WEIGHT * 3 +
        len(rules.get("header_patterns", [])) * HEADER_WEIGHT
        for rules in HALL_RULES.values()
    )
    confidence = min(1.0, best_score / max(max_possible * 0.4, 1))

    return best_hall, round(confidence, 3)


# --- Batch classification ----------------------------------------------------

async def classify_all_articles(kb_name: str) -> dict:
    """Classify all articles in a KB into halls. Returns stats."""
    from app.wiki import get_articles

    articles = get_articles(kb_name)
    classified = 0
    hall_counts: dict[str, int] = defaultdict(int)

    for article in articles:
        title = article.get("title", "")
        tags = article.get("tags", [])
        body = article.get("raw_markdown", article.get("body", ""))
        word_count = article.get("word_count", 0)
        slug = article.get("slug", "")

        if not slug:
            continue

        hall, confidence = classify_article(title, tags, body, word_count)
        await db.upsert_classification(slug, kb_name, hall, confidence)

        hall_counts[hall] += 1
        classified += 1

    logger.info("Classified %d articles in %s: %s", classified, kb_name, dict(hall_counts))
    return {"kb": kb_name, "classified": classified, "halls": dict(hall_counts)}


# --- Room clustering ---------------------------------------------------------

async def cluster_rooms(kb_name: str) -> dict:
    """Build rooms from knowledge graph entities. Returns stats."""
    # Clear existing rooms for this KB
    await db.clear_rooms(kb_name)

    # Get entities with article connections
    conn = await db._get_db()
    try:
        # Get entities ordered by how many articles reference them
        cursor = await conn.execute("""
            SELECT e.id, e.name, e.type, COUNT(DISTINCT ke.article_slug) as article_count
            FROM kg_entities e
            JOIN kg_edges ke ON (ke.source_entity_id = e.id OR ke.target_entity_id = e.id)
            WHERE ke.kb = ?
            GROUP BY e.id
            HAVING article_count >= 2
            ORDER BY article_count DESC
        """, (kb_name,))
        entities = [dict(r) for r in await cursor.fetchall()]

        # Track which articles are assigned to rooms
        assigned: dict[str, int] = defaultdict(int)  # slug -> number of room assignments
        rooms_created = 0
        total_memberships = 0

        for entity in entities:
            # Find articles connected to this entity
            cursor = await conn.execute("""
                SELECT DISTINCT ke.article_slug
                FROM kg_edges ke
                WHERE (ke.source_entity_id = ? OR ke.target_entity_id = ?)
                AND ke.kb = ?
            """, (entity["id"], entity["id"], kb_name))
            article_slugs = [row["article_slug"] for row in await cursor.fetchall()]

            if not article_slugs:
                continue

            # Skip if >60% of connected articles already have 2+ room assignments
            heavily_assigned = sum(1 for s in article_slugs if assigned.get(s, 0) >= 2)
            if len(article_slugs) > 0 and heavily_assigned / len(article_slugs) > 0.6:
                continue

            # Create the room
            room_id = await db.upsert_room(kb_name, entity["name"], entity["id"], len(article_slugs))

            for slug in article_slugs:
                # Relevance decreases with more room assignments
                relevance = 1.0 / (1 + assigned.get(slug, 0))
                await db.add_room_member(room_id, slug, kb_name, relevance)
                assigned[slug] = assigned.get(slug, 0) + 1
                total_memberships += 1

            rooms_created += 1

        # Handle orphans — articles with edges but no room assignment
        cursor = await conn.execute("""
            SELECT DISTINCT article_slug FROM kg_edges WHERE kb = ?
        """, (kb_name,))
        all_graph_slugs = {row["article_slug"] for row in await cursor.fetchall()}
        orphans = all_graph_slugs - set(assigned.keys())

        if orphans:
            misc_room_id = await db.upsert_room(kb_name, "_misc", None, len(orphans))
            for slug in orphans:
                await db.add_room_member(misc_room_id, slug, kb_name, 0.5)
                total_memberships += 1
            rooms_created += 1

    finally:
        await conn.close()

    logger.info("Created %d rooms with %d memberships in %s", rooms_created, total_memberships, kb_name)
    return {"kb": kb_name, "rooms_created": rooms_created, "memberships": total_memberships}


# --- Palace map --------------------------------------------------------------

# Per-KB memoization of the assembled map. Cheap to recompute (~1s on a
# warm cache, ~20s cold because of S3) so a short TTL is plenty.
_PALACE_CACHE_TTL = 60.0
_palace_cache: dict[str, tuple[float, dict]] = {}


def invalidate_palace_cache(kb_name: str | None = None) -> None:
    if kb_name is None:
        _palace_cache.clear()
    else:
        _palace_cache.pop(kb_name, None)


async def _generate_wing(kb: str) -> dict:
    """Build the palace-map wing for a single KB."""
    from app.wiki import get_articles

    # get_articles is sync + can touch S3, so push it off the event loop.
    articles = await asyncio.to_thread(get_articles, kb)
    article_map = {a["slug"]: a for a in articles}

    # Fire classifications + rooms in parallel so the two DB round-trips
    # overlap instead of serializing.
    classifications, rooms = await asyncio.gather(
        db.get_classifications(kb),
        db.get_rooms(kb),
    )

    # Build a slug -> hall lookup once so the room-member aggregation
    # below is O(1) per member instead of rescanning classifications.
    slug_to_hall: dict[str, str] = {c["slug"]: c["hall"] for c in classifications}

    hall_data: dict[str, dict] = {}
    for c in classifications:
        hall = c["hall"]
        bucket = hall_data.setdefault(hall, {"count": 0, "top_articles": []})
        bucket["count"] += 1
        if len(bucket["top_articles"]) < 5:
            art = article_map.get(c["slug"], {})
            bucket["top_articles"].append({
                "slug": c["slug"],
                "title": art.get("title", c["slug"]),
            })

    # Fan the per-room member lookups out in parallel.
    member_lists = await asyncio.gather(
        *(db.get_room_members(room["id"]) for room in rooms)
    )

    room_data: dict[str, dict] = {}
    for room, members in zip(rooms, member_lists):
        hall_dist: dict[str, int] = defaultdict(int)
        for m in members:
            hall = slug_to_hall.get(m["slug"])
            if hall:
                hall_dist[hall] += 1
        room_data[room["name"]] = {
            "count": room["article_count"],
            "hall_distribution": dict(hall_dist),
        }

    return {
        "article_count": len(articles),
        "classified_count": len(classifications),
        "halls": hall_data,
        "rooms": room_data,
    }


async def generate_palace_map(kb_name: str | None = None) -> dict:
    """Generate the lightweight palace map index.

    Results are cached in-process per-KB for ``_PALACE_CACHE_TTL`` seconds
    so the /palace page load doesn't re-walk S3 / DynamoDB on every hit.
    """
    kbs = [kb_name] if kb_name else storage.list_kbs()

    now = time.time()
    wings: dict[str, dict] = {}
    misses: list[str] = []
    for kb in kbs:
        cached = _palace_cache.get(kb)
        if cached and (now - cached[0]) < _PALACE_CACHE_TTL:
            wings[kb] = cached[1]
        else:
            misses.append(kb)

    if misses:
        fresh = await asyncio.gather(*(_generate_wing(kb) for kb in misses))
        for kb, wing in zip(misses, fresh):
            _palace_cache[kb] = (now, wing)
            wings[kb] = wing

    return {
        "wings": wings,
        "hall_types": HALL_TYPES,
        "hall_colors": HALL_COLORS,
        "hall_labels": HALL_LABELS,
    }


# --- Palace search signal for hybrid search ----------------------------------

async def search_via_palace(query: str, kb_name: str, limit: int = 15) -> list[dict]:
    """Score articles by palace proximity to the query.

    Used as a 4th RRF signal in hybrid search.
    """
    # Classify the query into a hall
    query_hall, _ = classify_article(query, [], query)

    # Match query terms against room names
    query_lower = query.lower()
    query_words = set(re.findall(r"\w+", query_lower))

    rooms = await db.get_rooms(kb_name)
    matching_rooms: list[int] = []
    for room in rooms:
        room_words = set(re.findall(r"\w+", room["name"].lower()))
        if query_words & room_words:
            matching_rooms.append(room["id"])

    # Score articles
    article_scores: dict[str, float] = defaultdict(float)

    # Hall match: +3 for articles in the same hall as the query
    classifications = await db.get_classifications_by_hall(kb_name, query_hall)
    for c in classifications:
        article_scores[c["slug"]] += 3.0

    # Room match: +5 for articles in matching rooms
    for room_id in matching_rooms:
        members = await db.get_room_members(room_id)
        for m in members:
            article_scores[m["slug"]] += 5.0 * m["relevance"]

    # Sort and return top results
    sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [{"slug": slug, "kb": kb_name, "score": score} for slug, score in sorted_articles]
