"""
Page content extraction: agent-browser CLI (primary) + raw httpx (fallback).

All self-hosted — no third-party services.
  - agent-browser: headless Chrome on your gateway, handles JS-rendered sites
  - httpx fallback: simple HTML fetch + text extraction for static pages
"""

import asyncio
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger("kb-service.browser")

AGENT_BROWSER = "agent-browser"
PAGE_TIMEOUT = 15000  # ms


# ─── agent-browser CLI (primary) ────────────────────────────────────────────

async def _run_cmd(*args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run agent-browser CLI command and return (returncode, stdout, stderr)."""
    cmd = [AGENT_BROWSER, *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 1, "", "agent-browser not found"
    except Exception as exc:
        return 1, "", str(exc)


async def is_available() -> bool:
    """Check if agent-browser CLI is installed and working."""
    code, _, _ = await _run_cmd("--version", timeout=5)
    return code == 0


async def read_page_browser(url: str) -> Optional[dict]:
    """Navigate to a URL with agent-browser and extract full text content."""
    logger.info("agent-browser reading: %s", url)

    # Navigate
    code, _, err = await _run_cmd("goto", url, f"--timeout={PAGE_TIMEOUT}", timeout=20)
    if code != 0:
        logger.warning("agent-browser failed to navigate to %s: %s", url, err[:200])
        return None

    await asyncio.sleep(1)

    # Get title
    code, title, _ = await _run_cmd("title", timeout=10)
    title = title.strip() if code == 0 else ""

    # Extract text via JS
    code, content, _ = await _run_cmd("eval", "document.body.innerText", timeout=15)

    if code != 0 or not content.strip():
        # Fallback: accessibility tree snapshot
        code, content, _ = await _run_cmd("snapshot", timeout=15)
        if code != 0:
            logger.warning("agent-browser: no content from %s", url)
            return None

    text = _clean_page_content(content.strip())

    if len(text) > 15000:
        text = text[:15000] + "\n\n[Truncated]"

    word_count = len(text.split())
    if word_count < 20:
        return None

    return {
        "url": url,
        "title": title,
        "content": text,
        "word_count": word_count,
        "source": "agent_browser",
    }


# ─── httpx fallback (static HTML pages) ─────────────────────────────────────

async def read_page_httpx(url: str) -> Optional[dict]:
    """Simple HTTP fetch + HTML text extraction for static pages."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "WikiDelve/1.0 (research bot)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            })
            if resp.status_code != 200:
                return None

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None

            html = resp.text

            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]*)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else url

            # Extract text content
            text = _html_to_text(html)

            if len(text) > 15000:
                text = text[:15000] + "\n\n[Truncated]"

            word_count = len(text.split())
            if word_count < 30:
                return None

            return {
                "url": url,
                "title": title,
                "content": text,
                "word_count": word_count,
                "source": "httpx",
            }
    except Exception as exc:
        logger.warning("httpx fetch failed for %s: %s", url, exc)
        return None


# ─── Smart reader: agent-browser primary, httpx fallback ────────────────────

async def read_page_smart(url: str) -> Optional[dict]:
    """Try agent-browser first, fall back to httpx for static pages."""
    if await is_available():
        result = await read_page_browser(url)
        if result and result["word_count"] > 50:
            return result

    # Fallback to simple HTTP fetch
    return await read_page_httpx(url)


async def read_pages_smart(urls: list[str], max_concurrent: int = 3) -> list[dict]:
    """Read multiple pages with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def read_with_limit(url: str) -> Optional[dict]:
        async with semaphore:
            return await read_page_smart(url)

    tasks = [read_with_limit(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


# ─── HTML extraction helpers ────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Extract meaningful text from HTML."""
    # Remove script, style, nav, footer
    text = html
    for tag in ("script", "style", "nav", "footer", "header", "aside"):
        text = re.sub(f'<{tag}[\\s\\S]*?</{tag}>', '', text, flags=re.IGNORECASE)

    # Try to find article/main content
    article = re.search(r'<article[\s\S]*?</article>', text, re.IGNORECASE)
    main = re.search(r'<main[\s\S]*?</main>', text, re.IGNORECASE)
    if article:
        text = article.group()
    elif main:
        text = main.group()

    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode entities
    for entity, char in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                          ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(entity, char)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r' \n ', '\n', text)

    return text


# ─── Document Download & Extraction (PDFs, ePubs) ───────────────────────────

DOWNLOAD_DIR = "/kb/research/downloads"

async def download_and_extract(url: str) -> Optional[dict]:
    """Download a PDF or ePub and extract text content.

    Returns dict with: url, title, content, word_count, source, filepath
    Or None on failure.
    """
    from pathlib import Path

    ext = _get_document_ext(url)
    if not ext:
        return None

    logger.info("Downloading document: %s", url)

    # Create download dir
    dl_dir = Path(DOWNLOAD_DIR)
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Download the file
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "WikiDelve/1.0 (research bot)",
            })
            if resp.status_code != 200:
                logger.warning("Download failed %d: %s", resp.status_code, url)
                return None

            # Check content type
            ct = resp.headers.get("content-type", "")
            if ext == ".pdf" and "pdf" not in ct and "octet" not in ct:
                return None
            if ext == ".epub" and "epub" not in ct and "octet" not in ct:
                return None

            # Save file
            safe_name = re.sub(r'[^\w\-.]', '_', url.split('/')[-1])[:80]
            if not safe_name.endswith(ext):
                safe_name += ext
            filepath = dl_dir / safe_name
            filepath.write_bytes(resp.content)
            logger.info("Downloaded %s (%d bytes)", filepath, len(resp.content))

    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return None

    # Extract text based on file type
    text = ""
    title = safe_name

    if ext == ".pdf":
        text, title = await _extract_pdf_text(filepath)
    elif ext == ".epub":
        text, title = await _extract_epub_text(filepath)

    if not text or len(text.split()) < 30:
        logger.warning("Insufficient text extracted from %s", url)
        return None

    # Truncate very long documents
    if len(text) > 30000:
        text = text[:30000] + "\n\n[Document truncated at 30,000 characters]"

    return {
        "url": url,
        "title": title,
        "content": text,
        "word_count": len(text.split()),
        "source": f"downloaded_{ext.lstrip('.')}",
        "filepath": str(filepath),
    }


async def download_documents(urls: list[str], max_concurrent: int = 2) -> list[dict]:
    """Download and extract text from multiple document URLs."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def dl_with_limit(url: str) -> Optional[dict]:
        async with semaphore:
            return await download_and_extract(url)

    tasks = [dl_with_limit(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


def find_document_urls(search_results: list[dict]) -> list[str]:
    """Find PDF and ePub URLs from search results."""
    doc_urls = []
    seen = set()
    for r in search_results:
        url = r.get("url", "")
        if url and url not in seen and _get_document_ext(url):
            doc_urls.append(url)
            seen.add(url)
    return doc_urls


def _get_document_ext(url: str) -> Optional[str]:
    """Check if a URL points to a downloadable document."""
    url_lower = url.lower().split('?')[0].split('#')[0]
    if url_lower.endswith('.pdf'):
        return '.pdf'
    if url_lower.endswith('.epub'):
        return '.epub'
    # Check for common PDF hosting patterns
    if 'arxiv.org/pdf' in url_lower:
        return '.pdf'
    if '/pdf/' in url_lower and not url_lower.endswith('.html'):
        return '.pdf'
    return None


async def _extract_pdf_text(filepath) -> tuple[str, str]:
    """Extract text from a PDF file. Returns (text, title)."""
    try:
        # Try PyMuPDF (fitz) first — fast and reliable
        import fitz
        doc = fitz.open(str(filepath))
        title = doc.metadata.get("title", "") or filepath.stem
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n\n".join(pages)
        return text.strip(), title
    except ImportError:
        pass

    # Fallback: try pdfplumber
    try:
        import pdfplumber
        text_parts = []
        title = filepath.stem
        with pdfplumber.open(str(filepath)) as pdf:
            for page in pdf.pages[:50]:  # max 50 pages
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts).strip(), title
    except ImportError:
        pass

    # Last resort: pdftotext CLI
    try:
        proc = await asyncio.create_subprocess_exec(
            "pdftotext", str(filepath), "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        text = stdout.decode("utf-8", errors="replace").strip()
        return text, filepath.stem
    except Exception:
        pass

    logger.warning("No PDF extraction library available")
    return "", filepath.stem


async def _extract_epub_text(filepath) -> tuple[str, str]:
    """Extract text from an ePub file. Returns (text, title)."""
    try:
        import zipfile
        import xml.etree.ElementTree as ET

        text_parts = []
        title = filepath.stem

        with zipfile.ZipFile(str(filepath), 'r') as epub:
            for name in epub.namelist():
                if name.endswith(('.xhtml', '.html', '.htm')):
                    with epub.open(name) as f:
                        content = f.read().decode('utf-8', errors='replace')
                        # Strip HTML tags
                        clean = re.sub(r'<[^>]+>', ' ', content)
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        if len(clean) > 50:
                            text_parts.append(clean)

            # Try to get title from OPF
            for name in epub.namelist():
                if name.endswith('.opf'):
                    with epub.open(name) as f:
                        try:
                            tree = ET.parse(f)
                            t = tree.find('.//{http://purl.org/dc/elements/1.1/}title')
                            if t is not None and t.text:
                                title = t.text
                        except Exception:
                            pass

        return "\n\n".join(text_parts).strip(), title
    except Exception as exc:
        logger.warning("ePub extraction failed for %s: %s", filepath, exc)
        return "", filepath.stem


def _clean_page_content(text: str) -> str:
    """Remove common web page noise from extracted text."""
    noise_patterns = [
        r"Accept all cookies.*?\n",
        r"Cookie settings.*?\n",
        r"Privacy Policy.*?Terms of Service",
        r"Sign up.*?Log in",
        r"Subscribe to our newsletter.*?\n",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
