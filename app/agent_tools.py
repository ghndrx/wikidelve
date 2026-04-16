"""
WikiDelve agent tools — wraps existing WikiDelve functions as LangChain tools
for use with DeepAgents.

Each tool calls WikiDelve's internal functions directly (no HTTP round-trips).
"""

import json
import logging

import httpx
from langchain_core.tools import tool

logger = logging.getLogger("kb-service.agent.tools")


# ---------------------------------------------------------------------------
# Search tools
# ---------------------------------------------------------------------------

@tool
async def search_web(query: str, num_results: int = 8) -> str:
    """Search the web for information on a topic using Google (via Serper API).
    Returns ranked results with titles, URLs, and snippets.
    Use this to gather information from the internet."""
    from app.sources.serper import SerperProvider

    async with httpx.AsyncClient(timeout=15) as client:
        provider = SerperProvider(client)
        results = await provider.search(query, num=num_results)

    if not results:
        return "No search results found. Try a different query or check that SERPER_API_KEY is set."
    return json.dumps(results, indent=2)


@tool
async def search_kb(query: str, kb: str = "personal", limit: int = 10) -> str:
    """Search the WikiDelve knowledge base for existing articles matching a query.
    Uses hybrid search (full-text + vector + knowledge graph).
    Use this to check what's already in the wiki before researching."""
    from app.hybrid_search import hybrid_search

    results = await hybrid_search(query, kb_name=kb, limit=limit)
    if not results:
        return "No matching articles found in the knowledge base."
    # Ensure results are serializable
    safe = []
    for r in results[:limit]:
        if isinstance(r, dict):
            safe.append(r)
        else:
            safe.append({"result": str(r)})
    return json.dumps(safe, indent=2, default=str)


# ---------------------------------------------------------------------------
# Web reading tools
# ---------------------------------------------------------------------------

@tool
async def read_webpage(url: str) -> str:
    """Read and extract the full text content from a webpage URL.
    Use this to get detailed information from a specific source.
    Returns the extracted text content (up to 8000 chars)."""
    from app.browser import read_page_smart

    text = await read_page_smart(url)
    if not text:
        return f"Failed to read content from {url}"
    if not isinstance(text, str):
        text = str(text)
    return text[:8000]


# ---------------------------------------------------------------------------
# Article tools
# ---------------------------------------------------------------------------

@tool
def get_article(kb: str, slug: str) -> str:
    """Get the full content of an existing WikiDelve article by its KB name and slug.
    Returns the article metadata and raw markdown content."""
    from app.wiki import get_article as _get

    article = _get(kb, slug)
    if not article:
        return f"Article not found: {kb}/{slug}"
    return json.dumps({
        "slug": article.get("slug"),
        "title": article.get("title"),
        "summary": article.get("summary"),
        "tags": article.get("tags", []),
        "word_count": article.get("word_count", 0),
        "raw_markdown": article.get("raw_markdown", ""),
    }, indent=2, default=str)


@tool
def list_articles(kb: str = "personal") -> str:
    """List all articles in a knowledge base with their titles and word counts.
    Use this to understand what's already covered in the KB."""
    from app.wiki import get_articles

    articles = get_articles(kb)
    summary = [
        {"slug": a.get("slug"), "title": a.get("title", ""), "words": a.get("word_count", 0)}
        for a in articles
    ]
    return json.dumps(summary, indent=2)


@tool
async def write_article(kb: str, topic: str, content: str, source_type: str = "research") -> str:
    """Write or update a wiki article. The content should be full markdown with YAML frontmatter.
    If a related article exists, it will be updated; otherwise a new one is created.

    The content MUST start with YAML frontmatter:
    ---
    title: Article Title
    tags: [tag1, tag2]
    summary: Brief summary
    source_type: research
    ---

    Followed by the article body in markdown."""
    from app.wiki import create_or_update_article

    slug, change_type = await create_or_update_article(kb, topic, content, source_type=source_type)
    return json.dumps({"slug": slug, "kb": kb, "change_type": change_type, "topic": topic})


# ---------------------------------------------------------------------------
# Quality tools
# ---------------------------------------------------------------------------

@tool
async def check_article_quality(kb: str, slug: str) -> str:
    """Check the quality score of an article (0-100).
    Returns a score breakdown with suggestions for improvement.
    Use this after writing an article to verify quality."""
    from app.quality import score_article_quality

    # score_article_quality is sync — awaiting its dict result
    # raised TypeError and crashed 3/5 pilot runs. Don't await it.
    result = score_article_quality(kb, slug)
    if not result:
        return f"Could not score article: {kb}/{slug}"
    return json.dumps(result, indent=2, default=str)


@tool
async def enrich_article(kb: str, slug: str) -> str:
    """Improve a shallow article by expanding its content with additional detail and structure.
    Use this when an article's quality score is below 60."""
    from app.quality import enrich_article as _enrich

    result = await _enrich(kb, slug)
    return json.dumps(result, indent=2, default=str)


@tool
async def add_crosslinks(kb: str, slug: str) -> str:
    """Add [[Wikilinks]] to related articles within the article content.
    Use this after writing an article to connect it to the rest of the KB."""
    # Function is named add_crosslinks in quality.py — rename on
    # import so the tool name stays user-friendly.
    from app.quality import add_crosslinks as _add_crosslinks

    result = await _add_crosslinks(kb, slug)
    return json.dumps(result, indent=2, default=str)


@tool
async def fact_check_article(kb: str, slug: str) -> str:
    """Run fact-checking on an article. Extracts key claims and verifies them
    via web search. Returns supported, unsupported, and unverifiable claims.
    Use this after writing an article to validate accuracy."""
    from app.quality import fact_check_article as _fc

    result = await _fc(kb, slug)
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Knowledge graph tools
# ---------------------------------------------------------------------------

@tool
async def find_related_articles(kb: str, slug: str) -> str:
    """Find articles related to a given article via knowledge graph traversal.
    Returns articles connected by shared entities and relationships."""
    from app.knowledge_graph import get_related_by_graph

    related = await get_related_by_graph(kb, slug)
    if not related:
        return "No related articles found via knowledge graph."
    return json.dumps(related, indent=2, default=str)


@tool
async def get_scaffold_file(kb: str, slug: str, rel_path: str) -> str:
    """Read a single file from an existing scaffold.

    Used by the extension agent to read styles.css / index.html /
    manifest.json so it can match design tokens and BEM naming
    when adding sibling pages."""
    from app.scaffolds import get_file, get_manifest
    if rel_path == "manifest.json":
        m = get_manifest(kb, slug)
        if not m:
            return json.dumps({"error": f"scaffold not found: {kb}/{slug}"})
        return json.dumps(m, indent=2)
    content = get_file(kb, slug, rel_path)
    if content is None:
        return json.dumps({"error": f"file not found: {rel_path}"})
    return content


@tool
async def add_scaffold_page(kb: str, slug: str, path: str, content: str) -> str:
    """Add ONE sibling page to an existing scaffold.

    Used by the extension agent (second pass). Matches the
    scaffold's existing design tokens — read styles.css with
    get_scaffold_file FIRST so you know what tokens exist.

    Removes the matching planned_extensions entry from the
    manifest on success so the orchestrator's queue stays
    accurate."""
    from app.scaffolds import add_page_to_scaffold
    try:
        manifest = add_page_to_scaffold(kb, slug, path, content)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({
        "kb": kb, "slug": slug, "added": path,
        "files_total": len(manifest.get("files") or []),
        "planned_remaining": len(manifest.get("planned_extensions") or []),
    })


@tool
async def write_scaffold_files(kb: str, manifest: dict, files: list) -> str:
    """Write a plug-and-play scaffold (template package) to storage.

    This is the scaffold agent's primary output tool. Use it INSTEAD
    of write_article when producing a code template the user can
    preview in a sandboxed iframe and copy.

    Args:
        kb: knowledge base name (use the current KB).
        manifest: dict with title, description, scaffold_type,
            framework, preview_type, entrypoint. The entrypoint must
            be one of the file paths in `files`.
        files: list of {"path": str, "content": str} entries. Paths
            are relative to the scaffold root (no leading /, no ..).
            256KB per file max, 2MB total max, 50 files max.

    Returns the resulting scaffold slug on success."""
    from app.scaffolds import create_scaffold

    if not isinstance(manifest, dict):
        return json.dumps({"error": "manifest must be a dict"})
    if not isinstance(files, list):
        return json.dumps({"error": "files must be a list"})
    try:
        slug = create_scaffold(kb, manifest, files)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps({
        "slug": slug, "kb": kb,
        "preview_url": f"/scaffolds/{kb}/{slug}",
        "file_count": len(files),
    })




# ---------------------------------------------------------------------------
# All tools exported
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_web,
    search_kb,
    read_webpage,
    get_article,
    list_articles,
    write_article,
    write_scaffold_files,
    get_scaffold_file,
    add_scaffold_page,
    check_article_quality,
    enrich_article,
    add_crosslinks,
    fact_check_article,
    find_related_articles,
]
