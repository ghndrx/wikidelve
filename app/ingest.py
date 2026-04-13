"""
Document ingestion: download PDFs/ePubs from URLs, extract text, add to wiki.

Supports:
- Direct PDF/ePub URLs
- Open directory listings (crawl for documents)
- Archive.org collections
- GitHub repos with free books/docs
"""

import asyncio
import logging
import re

from app.config import LLM_PROVIDER, MINIMAX_API_KEY, RESEARCH_DIR
from app.llm import llm_chat
from app.browser import download_and_extract
from app.wiki import create_or_update_article

logger = logging.getLogger("kb-service.ingest")

DOWNLOAD_DIR = RESEARCH_DIR / "downloads"


# ─── URL Document Ingestion ─────────────────────────────────────────────────

async def ingest_document_url(url: str, kb_name: str = "personal") -> dict:
    """Download a PDF/ePub from URL, extract text, synthesize, add to wiki."""
    logger.info("Ingesting document: %s", url)

    # Download and extract text
    doc = await download_and_extract(url)
    if not doc:
        return {"error": f"Failed to download or extract: {url}"}

    title = doc["title"]
    content = doc["content"]
    word_count = doc["word_count"]

    logger.info("Extracted %d words from %s", word_count, title)

    # If content is very long, synthesize key points
    if word_count > 5000:
        content = await _synthesize_document(title, content)

    # Add to wiki
    try:
        slug, change_type = await create_or_update_article(kb_name, title, content)
        return {
            "title": title,
            "url": url,
            "word_count": word_count,
            "synthesized_words": len(content.split()),
            "slug": slug,
            "change_type": change_type,
            "filepath": doc.get("filepath", ""),
        }
    except Exception as exc:
        return {"error": f"Failed to add to wiki: {exc}", "title": title}


async def ingest_document_urls(urls: list[str], kb_name: str = "personal") -> list[dict]:
    """Ingest multiple document URLs."""
    results = []
    for url in urls:
        result = await ingest_document_url(url, kb_name)
        results.append(result)
        await asyncio.sleep(2)  # Rate limit
    return results


# ─── Open Directory Crawler ─────────────────────────────────────────────────

async def crawl_open_directory(base_url: str, extensions: list[str] = None, max_files: int = 20) -> list[str]:
    """Crawl an open directory listing for document URLs.

    Returns list of document URLs found.
    """
    if extensions is None:
        extensions = [".pdf", ".epub"]

    logger.info("Crawling open directory: %s", base_url)
    found_urls = []

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(base_url, headers={
                "User-Agent": "WikiDelve/1.0 (knowledge base research bot)",
            })
            if resp.status_code != 200:
                return []

            html = resp.text

            # Find links to documents
            for match in re.findall(r'href=["\']([^"\']+)["\']', html):
                for ext in extensions:
                    if match.lower().endswith(ext):
                        # Resolve relative URLs
                        if match.startswith("http"):
                            doc_url = match
                        elif match.startswith("/"):
                            from urllib.parse import urlparse
                            parsed = urlparse(base_url)
                            doc_url = f"{parsed.scheme}://{parsed.netloc}{match}"
                        else:
                            doc_url = f"{base_url.rstrip('/')}/{match}"

                        found_urls.append(doc_url)
                        if len(found_urls) >= max_files:
                            break

                if len(found_urls) >= max_files:
                    break

    except Exception as exc:
        logger.warning("Failed to crawl %s: %s", base_url, exc)

    logger.info("Found %d documents in %s", len(found_urls), base_url)
    return found_urls


async def ingest_open_directory(base_url: str, kb_name: str = "personal", max_files: int = 10) -> dict:
    """Crawl an open directory and ingest all found documents."""
    urls = await crawl_open_directory(base_url, max_files=max_files)
    if not urls:
        return {"error": "No documents found", "url": base_url}

    results = await ingest_document_urls(urls, kb_name)
    success = sum(1 for r in results if "error" not in r)

    return {
        "url": base_url,
        "documents_found": len(urls),
        "documents_ingested": success,
        "results": results,
    }


# ─── Document Synthesis ─────────────────────────────────────────────────────

async def _synthesize_document(title: str, content: str) -> str:
    """Synthesize a long document into a structured wiki article."""
    has_llm = (LLM_PROVIDER == "bedrock") or MINIMAX_API_KEY
    if not has_llm:
        return content[:8000]

    # Take first 12K chars for synthesis
    excerpt = content[:12000]

    prompt = f"""Synthesize this document into a comprehensive knowledge base article.

DOCUMENT TITLE: {title}

DOCUMENT CONTENT (excerpt):
{excerpt}

Create a well-structured article that:
1. Executive summary (2-3 sentences)
2. Key concepts organized by ## headers
3. Practical takeaways and recommendations
4. Code examples or configurations if relevant
5. Key Takeaways section at the end

Target 1500-2500 words. Extract the KNOWLEDGE, keep it practical."""

    system = "You are a technical writer synthesizing documents for an engineering knowledge base. Extract actionable knowledge, skip filler."

    try:
        return await llm_chat(system, prompt, max_tokens=4000, temperature=0.2)
    except Exception as exc:
        logger.warning("Document synthesis failed: %s", exc)

    return content[:8000]
