"""
MCP (Model Context Protocol) server for WikiDelve.

Exposes wiki search, article retrieval, KB listing, and research
as MCP tools that can be used by Claude Code and other MCP clients.
"""

import logging
from fastmcp import FastMCP

from app import storage
from app.wiki import get_articles, get_article
from app.search import search_fts

logger = logging.getLogger("kb-service.mcp")

mcp = FastMCP("WikiDelve")


@mcp.tool()
async def search(query: str, limit: int = 10) -> list[dict]:
    """Search the knowledge base for articles matching a query.

    Uses full-text search with BM25 ranking. Returns titles, slugs,
    snippets, and tags for matching articles.
    """
    results = await search_fts(query, limit=min(limit, 30))
    return [
        {
            "title": r.get("title", r.get("slug", "")),
            "slug": r.get("slug", ""),
            "kb": r.get("kb", "personal"),
            "snippet": r.get("snippet", r.get("summary", "")),
            "tags": r.get("tags", []),
        }
        for r in results
    ]


@mcp.tool()
async def get_article_content(kb: str, slug: str) -> dict:
    """Get the full content of a wiki article by KB name and slug.

    Returns the article's title, raw markdown, metadata, and word count.
    """
    article = get_article(kb, slug)
    if not article:
        return {"error": f"Article not found: {kb}/{slug}"}
    return {
        "title": article.get("title", slug),
        "slug": slug,
        "kb": kb,
        "markdown": article.get("raw_markdown", ""),
        "summary": article.get("summary", ""),
        "tags": article.get("tags", []),
        "status": article.get("status", "draft"),
        "updated": article.get("updated", ""),
        "word_count": article.get("word_count", 0),
    }


@mcp.tool()
async def list_knowledge_bases() -> list[dict]:
    """List all knowledge bases with article counts and word totals."""
    kbs = []
    for kb_name in storage.list_kbs():
        articles = get_articles(kb_name)
        kbs.append({
            "name": kb_name,
            "articles": len(articles),
            "words": sum(a.get("word_count", 0) for a in articles),
        })
    return kbs


@mcp.tool()
async def list_articles(kb: str = "personal") -> list[dict]:
    """List all articles in a knowledge base with their metadata."""
    articles = get_articles(kb)
    return [
        {
            "slug": a["slug"],
            "title": a["title"],
            "tags": a.get("tags", []),
            "word_count": a.get("word_count", 0),
            "status": a.get("status", "draft"),
            "updated": a.get("updated", ""),
        }
        for a in articles
    ]


@mcp.tool()
async def research_topic(topic: str, kb: str = "personal") -> dict:
    """Start a research job on a topic. The research runs asynchronously.

    Returns the job ID which can be used to check status. The research
    pipeline searches the web, synthesizes findings, and adds the result
    to the wiki.
    """
    from app import db

    # Validate
    if not topic or len(topic.strip()) < 10:
        return {"error": "Topic must be at least 10 characters"}

    topic = topic.strip()

    # Check cooldown
    existing = await db.check_cooldown(topic)
    if existing:
        return {
            "status": "cooldown",
            "detail": f"Topic was researched recently (job {existing['id']})",
            "existing_job_id": existing["id"],
        }

    # Create job — the worker will pick it up from Redis
    job_id = await db.create_job(topic)

    return {
        "status": "queued",
        "job_id": job_id,
        "topic": topic,
        "kb": kb,
        "detail": f"Research job #{job_id} created. The worker will process it asynchronously.",
    }


# ---------------------------------------------------------------------------
# Extended tools — chat, graph traversal, auto-discovery, version history
# ---------------------------------------------------------------------------
#
# These tools expose the chat / graph / auto-discovery / versioning layers
# to MCP-aware clients (Claude Code, Cursor, etc.) so an external session
# can run a full research/chat cycle against the wiki without going
# through the HTTP API.


@mcp.tool()
async def chat_ask(question: str, kb: str = "personal") -> dict:
    """Ask the wiki a question and get a grounded, cited answer.

    Runs the same RAG pipeline as the web chat: hybrid search → chunking
    → reranking → LLM synthesis with citation rules. Tools are NOT used
    so the answer is purely retrieval-grounded.

    Returns the answer text plus the list of source passages used.
    """
    from app.chat import retrieve_context, build_chat_prompt
    from app.llm import llm_chat

    if not question or not question.strip():
        return {"error": "question is required"}

    passages = await retrieve_context(question, kb=kb, k=8)
    system_msg, user_msg = build_chat_prompt(question, passages, [])
    try:
        answer = await llm_chat(
            system_msg, user_msg, max_tokens=2000, temperature=0.2,
        )
    except Exception as exc:
        return {"error": f"LLM call failed: {exc}"}

    return {
        "kb": kb,
        "question": question,
        "answer": answer,
        "sources": [
            {
                "title": p.get("title"),
                "url": p.get("url"),
                "score": p.get("score"),
            }
            for p in passages
        ],
    }


@mcp.tool()
async def get_graph_neighbors(kb: str, slug: str, depth: int = 2) -> list[dict]:
    """Walk the knowledge graph from an article and return neighbours.

    Returns articles connected via entity relationships within ``depth``
    hops, sorted by connection strength. Useful for "give me everything
    related to this article in the wiki".
    """
    from app.knowledge_graph import get_related_by_graph

    try:
        results = await get_related_by_graph(slug, kb_name=kb, depth=depth)
    except Exception as exc:
        return [{"error": str(exc)}]

    return [
        {
            "slug": r.get("slug"),
            "kb": r.get("kb"),
            "score": r.get("score"),
            "hop": r.get("hop"),
            "connections": r.get("connections", []),
        }
        for r in results
    ]


@mcp.tool()
async def enqueue_auto_discovery(kb: str = "personal") -> dict:
    """Trigger an immediate auto-discovery refill + enqueue cycle.

    Walks every enabled discovery strategy (KG entities, LLM follow-ups,
    chat gaps, contradictions, stale articles, orphan entities, questions,
    blind spots, TOC stubs, broken wikilinks) and queues new candidate
    topics. Then drains the queue into research_task jobs subject to the
    per-KB Serper budget.
    """
    from app.auto_discovery import (
        run_discovery_for_kb, enqueue_next_candidates_for_kb,
    )
    if kb not in storage.list_kbs():
        return {"error": f"Unknown KB: {kb}"}

    # We don't have an arq pool inside the MCP context — try to fetch
    # one if main.py is running, otherwise just run discovery.
    discovery_result = await run_discovery_for_kb(kb)
    enqueue_result: dict = {"skipped": "no_arq_pool_in_mcp_context"}
    try:
        from app.config import REDIS_HOST, REDIS_PORT
        from arq import create_pool
        from arq.connections import RedisSettings
        pool = await create_pool(RedisSettings(host=REDIS_HOST, port=REDIS_PORT))
        try:
            enqueue_result = await enqueue_next_candidates_for_kb(pool, kb)
        finally:
            await pool.close()
    except Exception as exc:
        enqueue_result = {"error": f"arq pool unavailable: {exc}"}

    return {"discovery": discovery_result, "enqueue": enqueue_result}


@mcp.tool()
async def list_topic_candidates(kb: str = "personal", limit: int = 30) -> list[dict]:
    """List pending auto-discovery topic candidates for a KB.

    Each candidate has a topic, source ('kg_entity', 'llm_followup',
    'chat_gap', 'contradiction', 'stale', 'orphan_entity', 'question',
    'blind_spot', 'stub', 'wikilink', 'rss', 'sitemap'), and a score.
    """
    from app import db
    pending = await db.get_pending_candidates(kb, limit=limit)
    return [
        {
            "id": c["id"],
            "topic": c["topic"],
            "source": c.get("source"),
            "source_ref": c.get("source_ref"),
            "score": c.get("score"),
            "created_at": c.get("created_at"),
        }
        for c in pending
    ]


@mcp.tool()
async def get_article_versions(kb: str, slug: str) -> list[dict]:
    """List historical snapshots of an article from ``article_versions``.

    Returns metadata for each version (id, created_at, change_type, hash,
    word count) — fetch a specific version's full content via the
    ``/api/articles/{kb}/{slug}/versions/{id}`` HTTP endpoint if needed.
    """
    from app import db
    versions = await db.get_article_versions(kb, slug, limit=50)
    return [
        {
            "id": v["id"],
            "created_at": v.get("created_at"),
            "change_type": v.get("change_type"),
            "content_hash": v.get("content_hash"),
            "word_count": len((v.get("full_content") or "").split()),
            "job_id": v.get("job_id"),
            "note": v.get("note"),
        }
        for v in versions
    ]
