"""
Knowledge graph: entity extraction, relationship mapping, and graph traversal.

Uses Minimax LLM for entity/relationship extraction (no SpaCy dependency).
Stores entities and edges in SQLite for fast traversal and D3.js visualization.
"""

import json
import logging
import aiosqlite

from app.config import DB_PATH
from app.llm import llm_chat
from app.wiki import get_articles, get_article

logger = logging.getLogger("kb-service.knowledge_graph")


# --- Entity & relationship extraction via LLM --------------------------------


async def extract_entities(text: str, title: str) -> list[dict]:
    """Use Minimax to extract named entities from article text.

    Returns a list of dicts: [{name: str, type: str}, ...]
    Entity types: tool, language, framework, concept, service, protocol, person, company
    """
    # Truncate text to keep prompt reasonable
    truncated = text[:6000]

    prompt = f"""Analyze this technical article and extract all named entities.

Title: {title}

Text:
{truncated}

Return a JSON array of objects with "name" and "type" fields.
Valid types: tool, language, framework, concept, service, protocol, person, company

Rules:
- Only extract specific, named entities (not generic terms like "database" or "server")
- Normalize names: use the canonical form (e.g., "PostgreSQL" not "postgres", "Kubernetes" not "k8s")
- Include programming languages, frameworks, tools, services, protocols, and key concepts
- Do NOT include common English words or generic programming terms
- Return between 3 and 20 entities
- Return ONLY the JSON array, no other text

Example output:
[{{"name": "PostgreSQL", "type": "tool"}}, {{"name": "Python", "type": "language"}}, {{"name": "REST", "type": "protocol"}}]"""

    response = await llm_chat(
        system_msg="You extract named entities from technical text. Return ONLY a valid JSON array.",
        user_msg=prompt,
        max_tokens=1000,
        temperature=0.1,
    )

    return _parse_json_array(response, expected_keys=["name", "type"])


async def extract_relationships(
    text: str, entities: list[dict]
) -> list[dict]:
    """Use Minimax to find relationships between extracted entities.

    Returns a list of dicts: [{source: str, target: str, relationship: str}, ...]
    Relationship types: uses, implements, compares, extends, deploys-on, integrates,
                        alternative-to, part-of, built-with
    """
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]
    truncated = text[:4000]

    prompt = f"""Given these entities extracted from a technical article, identify relationships between them.

Entities: {json.dumps(entity_names)}

Article text (excerpt):
{truncated}

Return a JSON array of relationship objects with "source", "target", and "relationship" fields.
Valid relationships: uses, implements, compares, extends, deploys-on, integrates, alternative-to, part-of, built-with

Rules:
- Only include relationships that are clearly stated or strongly implied in the text
- Source and target must be exact entity names from the list above
- Return between 1 and 15 relationships
- Return ONLY the JSON array, no other text

Example:
[{{"source": "Django", "target": "Python", "relationship": "built-with"}}, {{"source": "FastAPI", "target": "Django", "relationship": "alternative-to"}}]"""

    response = await llm_chat(
        system_msg="You identify relationships between technical entities. Return ONLY a valid JSON array.",
        user_msg=prompt,
        max_tokens=1000,
        temperature=0.1,
    )

    return _parse_json_array(response, expected_keys=["source", "target", "relationship"])


def _parse_json_array(text: str, expected_keys: list[str]) -> list[dict]:
    """Parse a JSON array from LLM output, handling common formatting issues."""
    # Try to find JSON array in the response
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (the fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [
                item for item in result
                if isinstance(item, dict) and all(k in item for k in expected_keys)
            ]
    except json.JSONDecodeError:
        pass

    # Try to find array within the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return [
                    item for item in result
                    if isinstance(item, dict) and all(k in item for k in expected_keys)
                ]
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON array from LLM response: %s", text[:200])
    return []


# --- Graph building ----------------------------------------------------------

async def build_graph_for_article(kb_name: str, slug: str) -> dict:
    """Extract entities and relationships from an article, store in SQLite.

    Returns a status dict with counts.
    """
    article = get_article(kb_name, slug)
    if not article:
        return {"status": "error", "error": f"Article not found: {kb_name}/{slug}"}

    title = article["title"]
    body = article.get("raw_markdown", "")
    if not body:
        return {"status": "skip", "reason": "empty body"}

    # Extract entities
    try:
        entities = await extract_entities(body, title)
    except Exception as exc:
        logger.warning("Entity extraction failed for %s/%s: %s", kb_name, slug, exc)
        return {"status": "error", "error": f"Entity extraction failed: {exc}"}

    if not entities:
        return {"status": "ok", "entities": 0, "relationships": 0}

    # Extract relationships
    relationships = []
    try:
        relationships = await extract_relationships(body, entities)
    except Exception as exc:
        logger.warning(
            "Relationship extraction failed for %s/%s: %s", kb_name, slug, exc,
        )

    # Store in SQLite
    entity_ids = {}
    db = await _get_kg_db()
    try:
        for entity in entities:
            entity_id = await _upsert_entity(db, entity["name"], entity["type"])
            entity_ids[entity["name"]] = entity_id
            # Increment article count
            await db.execute(
                """UPDATE kg_entities SET article_count = (
                       SELECT COUNT(DISTINCT ke.article_slug)
                       FROM kg_edges ke WHERE ke.source_entity_id = ? OR ke.target_entity_id = ?
                   ) + 1 WHERE id = ?""",
                (entity_id, entity_id, entity_id),
            )

        for rel in relationships:
            source_id = entity_ids.get(rel["source"])
            target_id = entity_ids.get(rel["target"])
            if source_id and target_id:
                await _upsert_edge(
                    db, source_id, target_id,
                    rel["relationship"], slug, kb_name,
                )

        await db.commit()
    finally:
        await db.close()

    logger.info(
        "Graph built for %s/%s: %d entities, %d relationships",
        kb_name, slug, len(entities), len(relationships),
    )
    return {
        "status": "ok",
        "slug": slug,
        "kb": kb_name,
        "entities": len(entities),
        "relationships": len(relationships),
    }


async def build_full_graph(kb_name: str) -> dict:
    """Build the knowledge graph for all articles in a KB."""
    articles = get_articles(kb_name)
    if not articles:
        return {"status": "ok", "kb": kb_name, "processed": 0, "errors": 0}

    processed = 0
    errors = 0
    total_entities = 0
    total_relationships = 0

    for article in articles:
        try:
            result = await build_graph_for_article(kb_name, article["slug"])
            if result.get("status") == "ok":
                processed += 1
                total_entities += result.get("entities", 0)
                total_relationships += result.get("relationships", 0)
            elif result.get("status") == "error":
                errors += 1
        except Exception as exc:
            errors += 1
            logger.warning("Graph build error for %s/%s: %s", kb_name, article["slug"], exc)

    logger.info(
        "Full graph build: kb=%s, processed=%d, errors=%d, entities=%d, relationships=%d",
        kb_name, processed, errors, total_entities, total_relationships,
    )
    return {
        "status": "ok",
        "kb": kb_name,
        "processed": processed,
        "errors": errors,
        "total_entities": total_entities,
        "total_relationships": total_relationships,
    }


# --- Graph queries -----------------------------------------------------------

async def get_related_by_graph(
    slug: str, kb_name: str, depth: int = 2
) -> list[dict]:
    """Traverse the knowledge graph to find articles related to a given article.

    Follows entity connections up to `depth` hops and returns articles
    found along the way, scored by connection strength.
    """
    db = await _get_kg_db()
    try:
        # Find all entities connected to this article's edges
        cursor = await db.execute(
            """SELECT DISTINCT e.id, e.name, e.type
               FROM kg_entities e
               JOIN kg_edges ed ON (e.id = ed.source_entity_id OR e.id = ed.target_entity_id)
               WHERE ed.article_slug = ? AND ed.kb = ?""",
            (slug, kb_name),
        )
        seed_entities = [dict(r) for r in await cursor.fetchall()]

        if not seed_entities:
            return []

        seed_ids = {e["id"] for e in seed_entities}
        visited_articles: dict[str, dict] = {}

        # BFS traversal
        current_ids = seed_ids
        for hop in range(depth):
            if not current_ids:
                break

            placeholders = ",".join("?" * len(current_ids))
            cursor = await db.execute(
                f"""SELECT DISTINCT ed.article_slug, ed.kb,
                           e1.name as source_name, e2.name as target_name,
                           ed.relationship
                    FROM kg_edges ed
                    JOIN kg_entities e1 ON ed.source_entity_id = e1.id
                    JOIN kg_entities e2 ON ed.target_entity_id = e2.id
                    WHERE (ed.source_entity_id IN ({placeholders})
                           OR ed.target_entity_id IN ({placeholders}))
                      AND ed.article_slug != ?""",
                list(current_ids) + list(current_ids) + [slug],
            )
            rows = await cursor.fetchall()

            next_entity_ids = set()
            for row in rows:
                row = dict(row)
                art_slug = row["article_slug"]
                if art_slug not in visited_articles:
                    visited_articles[art_slug] = {
                        "slug": art_slug,
                        "kb": row["kb"],
                        "score": 0,
                        "connections": [],
                        "hop": hop + 1,
                    }
                # Score: closer hops score higher
                visited_articles[art_slug]["score"] += max(1, depth - hop)
                conn = f"{row['source_name']} --{row['relationship']}--> {row['target_name']}"
                if conn not in visited_articles[art_slug]["connections"]:
                    visited_articles[art_slug]["connections"].append(conn)

                # Get entity IDs for next hop
                cursor2 = await db.execute(
                    """SELECT source_entity_id, target_entity_id FROM kg_edges
                       WHERE article_slug = ?""",
                    (art_slug,),
                )
                for edge_row in await cursor2.fetchall():
                    next_entity_ids.add(edge_row[0])
                    next_entity_ids.add(edge_row[1])

            current_ids = next_entity_ids - seed_ids

        result = list(visited_articles.values())
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    finally:
        await db.close()


async def get_entity_articles(entity_name: str) -> list[dict]:
    """Find all articles that mention a given entity."""
    db = await _get_kg_db()
    try:
        cursor = await db.execute(
            """SELECT DISTINCT ed.article_slug, ed.kb, ed.relationship,
                      e1.name as source_name, e2.name as target_name
               FROM kg_edges ed
               JOIN kg_entities e1 ON ed.source_entity_id = e1.id
               JOIN kg_entities e2 ON ed.target_entity_id = e2.id
               WHERE e1.name = ? OR e2.name = ?""",
            (entity_name, entity_name),
        )
        rows = await cursor.fetchall()

        articles: dict[str, dict] = {}
        for row in rows:
            row = dict(row)
            key = f"{row['kb']}/{row['article_slug']}"
            if key not in articles:
                articles[key] = {
                    "slug": row["article_slug"],
                    "kb": row["kb"],
                    "relationships": [],
                }
            articles[key]["relationships"].append({
                "source": row["source_name"],
                "target": row["target_name"],
                "relationship": row["relationship"],
            })

        return list(articles.values())
    finally:
        await db.close()


async def get_graph_data() -> dict:
    """Return the full graph as nodes + edges for D3.js visualization."""
    db = await _get_kg_db()
    try:
        # Nodes: all entities
        cursor = await db.execute(
            "SELECT id, name, type, article_count FROM kg_entities ORDER BY article_count DESC"
        )
        entity_rows = await cursor.fetchall()
        nodes = [
            {
                "id": row["id"],
                "name": row["name"],
                "type": row["type"],
                "article_count": row["article_count"],
            }
            for row in entity_rows
        ]

        # Edges
        cursor = await db.execute(
            """SELECT source_entity_id, target_entity_id, relationship,
                      article_slug, kb
               FROM kg_edges"""
        )
        edge_rows = await cursor.fetchall()
        edges = [
            {
                "source": row["source_entity_id"],
                "target": row["target_entity_id"],
                "relationship": row["relationship"],
                "article_slug": row["article_slug"],
                "kb": row["kb"],
            }
            for row in edge_rows
        ]

        return {"nodes": nodes, "edges": edges}
    finally:
        await db.close()


# --- SQLite helpers ----------------------------------------------------------

async def _get_kg_db() -> aiosqlite.Connection:
    """Open a connection with row factory enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def _upsert_entity(
    db: aiosqlite.Connection, name: str, entity_type: str
) -> int:
    """Insert or get an entity, returning its ID."""
    cursor = await db.execute(
        "SELECT id FROM kg_entities WHERE name = ?", (name,)
    )
    row = await cursor.fetchone()
    if row:
        return row["id"]

    cursor = await db.execute(
        "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, 0)",
        (name, entity_type),
    )
    return cursor.lastrowid


async def _upsert_edge(
    db: aiosqlite.Connection,
    source_id: int,
    target_id: int,
    relationship: str,
    article_slug: str,
    kb: str,
) -> None:
    """Insert an edge if it doesn't already exist for this article."""
    cursor = await db.execute(
        """SELECT id FROM kg_edges
           WHERE source_entity_id = ? AND target_entity_id = ?
             AND relationship = ? AND article_slug = ? AND kb = ?""",
        (source_id, target_id, relationship, article_slug, kb),
    )
    if await cursor.fetchone():
        return

    await db.execute(
        """INSERT INTO kg_edges (source_entity_id, target_entity_id, relationship, article_slug, kb)
           VALUES (?, ?, ?, ?, ?)""",
        (source_id, target_id, relationship, article_slug, kb),
    )
