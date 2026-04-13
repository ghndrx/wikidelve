"""
Vector embedding service using Minimax embo-01.

Provides text embedding, article indexing, and cosine similarity search.
Stores embeddings as JSON arrays in SQLite — no sqlite-vss dependency needed.
With ~667 articles this brute-force approach is fast enough (<100ms).
"""

import logging
import math

from app.llm import llm_embed
from app.vector_store import get_vector_store
from app.wiki import get_articles, get_article

logger = logging.getLogger("kb-service.embeddings")

# Maximum tokens per embedding request — chunk articles longer than this
MAX_CHUNK_CHARS = 8000


# --- Core embedding functions ------------------------------------------------

async def embed_text(text: str, embed_type: str = "db") -> list[float]:
    """Embed a single text string, return the vector.

    embed_type: "db" for storing articles, "query" for search queries.
    """
    vectors = await llm_embed([text[:MAX_CHUNK_CHARS]], embed_type)
    if not vectors:
        raise ValueError("No vectors returned from embedding API")
    return vectors[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts. Returns list of vectors in same order."""
    if not texts:
        return []
    return await llm_embed(texts, embed_type="db")


# --- Article embedding -------------------------------------------------------

def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks that fit within the embedding model's limit.

    Splits on paragraph boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text[:max_chars]]


async def embed_article(kb_name: str, slug: str) -> dict:
    """Read an article, embed it, and store the embedding in SQLite.

    For long articles, chunks are embedded separately and averaged into
    a single vector (mean pooling).

    Returns a status dict.
    """
    article = get_article(kb_name, slug)
    if not article:
        return {"status": "error", "error": f"Article not found: {kb_name}/{slug}"}

    # Build the text to embed: title + summary + body
    parts = [article["title"]]
    if article.get("summary"):
        parts.append(article["summary"])
    body = article.get("raw_markdown", "")
    if body:
        parts.append(body)
    full_text = "\n\n".join(parts)

    # Chunk if needed
    chunks = _chunk_text(full_text)

    if len(chunks) == 1:
        embedding = await embed_text(chunks[0])
    else:
        # Mean pooling across chunks
        chunk_embeddings = await embed_texts(chunks)
        if not chunk_embeddings:
            return {"status": "error", "error": "No embeddings returned for chunks"}
        dim = len(chunk_embeddings[0])
        embedding = [0.0] * dim
        for ce in chunk_embeddings:
            for i in range(dim):
                embedding[i] += ce[i]
        n = len(chunk_embeddings)
        embedding = [v / n for v in embedding]

    embedding = _normalize_vector(embedding)
    await get_vector_store().upsert(kb_name, slug, embedding)

    logger.info("Embedded article: %s/%s (%d chunks)", kb_name, slug, len(chunks))
    return {"status": "ok", "slug": slug, "kb": kb_name, "chunks": len(chunks)}


async def embed_all_articles(kb_name: str) -> dict:
    """Embed every article in a KB. Returns summary stats."""
    articles = get_articles(kb_name)
    if not articles:
        return {"status": "ok", "kb": kb_name, "embedded": 0, "errors": 0}

    embedded = 0
    errors = 0

    import asyncio
    for article in articles:
        try:
            result = await embed_article(kb_name, article["slug"])
            if result.get("status") == "ok":
                embedded += 1
            else:
                errors += 1
                logger.warning(
                    "Failed to embed %s/%s: %s",
                    kb_name, article["slug"], result.get("error"),
                )
        except Exception as exc:
            errors += 1
            logger.warning("Embed error for %s/%s: %s", kb_name, article["slug"], exc)

        # Rate limit: ~10 requests per minute to stay under Minimax limits
        await asyncio.sleep(6)

    logger.info(
        "Embed all complete: kb=%s, embedded=%d, errors=%d",
        kb_name, embedded, errors,
    )
    return {"status": "ok", "kb": kb_name, "embedded": embedded, "errors": errors}


# --- Vector search -----------------------------------------------------------

async def search_similar(
    query: str, kb_name: str | None = None, limit: int = 10
) -> list[dict]:
    """Embed the query and find the most similar articles by cosine similarity.

    If kb_name is None, searches across all KBs.
    Routes through the configured vector store backend.
    """
    if not query or not query.strip():
        return []

    query_vec = await embed_text(query, embed_type="query")
    return await get_vector_store().query(kb_name, query_vec, top_k=limit)


# --- Math helpers ------------------------------------------------------------

def _normalize_vector(vec: list[float]) -> list[float]:
    """L2-normalize a vector."""
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude == 0:
        return vec
    return [v / magnitude for v in vec]


def _dot_product(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors (cosine similarity when both are normalized)."""
    return sum(x * y for x, y in zip(a, b))
