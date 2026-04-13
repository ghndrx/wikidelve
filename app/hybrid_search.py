"""
Hybrid search: combines FTS5 keyword search, vector similarity, and knowledge
graph expansion using Reciprocal Rank Fusion (RRF) for result merging.
"""

import json
import logging

from app.search import search_fts
from app.embeddings import search_similar
from app.knowledge_graph import get_entity_articles, _get_kg_db
from app.config import LLM_PROVIDER, MINIMAX_API_KEY
from app.llm import llm_chat, _embedding_circuit, EmbeddingUnavailable

logger = logging.getLogger("kb-service.hybrid_search")


async def hybrid_search(
    query: str,
    kb_name: str | None = None,
    limit: int = 15,
) -> list[dict]:
    """Run hybrid search combining FTS5, vector similarity, and graph expansion.

    Each method produces a ranked list. Results are merged using Reciprocal Rank
    Fusion to produce a single relevance-ordered list.
    """
    if not query or not query.strip():
        return []

    result_lists: list[list[dict]] = []

    # 1. In-memory FTS keyword search
    try:
        fts_results = await search_fts(query, limit=20, kb=kb_name)
        if kb_name:
            fts_results = [r for r in fts_results if r.get("kb") == kb_name]
        result_lists.append(fts_results)
        logger.debug("Hybrid FTS: %d results", len(fts_results))
    except Exception as exc:
        logger.warning("Hybrid search FTS failed: %s", exc)
        result_lists.append([])

    # 2. Vector similarity search — skipped entirely while the embedding
    # circuit is open so a provider outage doesn't block every query.
    if _embedding_circuit.is_open:
        logger.debug("Hybrid vector: circuit open, skipping")
        result_lists.append([])
    else:
        try:
            vector_results = await search_similar(query, kb_name, limit=20)
            result_lists.append(vector_results)
            logger.debug("Hybrid vector: %d results", len(vector_results))
        except EmbeddingUnavailable as exc:
            logger.info("Hybrid vector skipped: %s", exc)
            result_lists.append([])
        except Exception as exc:
            logger.warning("Hybrid search vector failed: %s", exc)
            result_lists.append([])

    # 3. Graph expansion
    try:
        graph_results = await search_via_graph(query, kb_name, limit=10)
        result_lists.append(graph_results)
        logger.debug("Hybrid graph: %d results", len(graph_results))
    except Exception as exc:
        logger.warning("Hybrid search graph failed: %s", exc)
        result_lists.append([])

    # 4. Palace proximity (hall + room matching)
    try:
        from app.palace import search_via_palace
        if kb_name:
            palace_results = await search_via_palace(query, kb_name, limit=15)
            result_lists.append(palace_results)
            logger.debug("Hybrid palace: %d results", len(palace_results))
    except Exception as exc:
        logger.warning("Hybrid search palace failed: %s", exc)
        result_lists.append([])

    # 5. Merge with Reciprocal Rank Fusion
    merged = reciprocal_rank_fusion(result_lists)
    return merged[:limit]


def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document d:
        score(d) = sum( 1 / (k + rank_i(d)) ) for each list i where d appears

    k=60 is the standard constant from the original RRF paper (Cormack et al. 2009).
    Higher k values reduce the impact of high rankings.
    """
    scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            # Use slug+kb as unique key
            doc_key = f"{doc.get('kb', 'unknown')}/{doc.get('slug', '')}"
            if not doc.get("slug"):
                continue

            rrf_score = 1.0 / (k + rank + 1)
            scores[doc_key] = scores.get(doc_key, 0.0) + rrf_score

            # Keep the richest version of the document data
            if doc_key not in doc_data or _richness(doc) > _richness(doc_data[doc_key]):
                doc_data[doc_key] = doc

    # Sort by RRF score descending, filter low-relevance noise
    MIN_RRF_SCORE = 0.03
    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    results = []
    for key in ranked_keys:
        score = round(scores[key], 6)
        if score < MIN_RRF_SCORE:
            continue
        doc = doc_data[key].copy()
        doc["rrf_score"] = score
        results.append(doc)

    return results


def _richness(doc: dict) -> int:
    """Score how much metadata a doc dict contains (for choosing the best version)."""
    score = 0
    if doc.get("title"):
        score += 2
    if doc.get("summary"):
        score += 2
    if doc.get("snippet"):
        score += 1
    if doc.get("tags"):
        score += 1
    if doc.get("connections"):
        score += 1
    return score


async def search_via_graph(
    query: str, kb_name: str | None = None, limit: int = 10
) -> list[dict]:
    """Extract entities from the query and find articles via the knowledge graph.

    Uses a lightweight entity extraction approach: try Minimax for short queries,
    then look up matching entities in the graph.
    """
    # Extract potential entity names from the query
    entities = await _extract_query_entities(query)
    if not entities:
        return []

    # Find articles for each entity
    all_articles: dict[str, dict] = {}
    for entity_name in entities:
        try:
            articles = await get_entity_articles(entity_name)
            for article in articles:
                if kb_name and article.get("kb") != kb_name:
                    continue
                key = f"{article['kb']}/{article['slug']}"
                if key not in all_articles:
                    all_articles[key] = {
                        "slug": article["slug"],
                        "kb": article["kb"],
                        "graph_entities": [],
                    }
                all_articles[key]["graph_entities"].append(entity_name)
        except Exception as exc:
            logger.warning("Graph search failed for entity '%s': %s", entity_name, exc)

    # Score by number of matching entities
    results = list(all_articles.values())
    results.sort(key=lambda x: len(x.get("graph_entities", [])), reverse=True)
    return results[:limit]


async def _extract_query_entities(query: str) -> list[str]:
    """Extract entity names from a search query.

    First tries to match against known entities in the database.
    Falls back to Minimax extraction for more complex queries.
    """
    # Step 1: Check for direct matches against known entities
    known_matches = await _match_known_entities(query)
    if known_matches:
        return known_matches

    # Step 2: For longer queries, use LLM to extract entity names
    has_llm = (LLM_PROVIDER == "bedrock") or MINIMAX_API_KEY
    if len(query.split()) >= 3 and has_llm:
        try:
            return await _llm_extract_entities(query)
        except Exception as exc:
            logger.warning("LLM entity extraction failed for query: %s", exc)

    return []


async def _match_known_entities(query: str) -> list[str]:
    """Match query terms against known entities in the database."""
    db = await _get_kg_db()
    try:
        cursor = await db.execute("SELECT name FROM kg_entities")
        rows = await cursor.fetchall()
        entity_names = [row["name"] for row in rows]
    finally:
        await db.close()

    if not entity_names:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())

    matches = []
    for name in entity_names:
        name_lower = name.lower()
        # Exact substring match in query
        if name_lower in query_lower:
            matches.append(name)
        # Word-level match for single-word entities
        elif name_lower in query_words:
            matches.append(name)

    return matches


async def _llm_extract_entities(query: str) -> list[str]:
    """Use the configured LLM to extract entity names from a search query."""
    content = await llm_chat(
        system_msg=(
            "Extract specific technology/tool/framework/language names from the search query. "
            "Return ONLY a JSON array of strings. If no specific entities, return []."
        ),
        user_msg=query,
        max_tokens=200,
        temperature=0.0,
    )

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return [str(item) for item in result if isinstance(item, str)]
    except json.JSONDecodeError:
        pass

    return []
