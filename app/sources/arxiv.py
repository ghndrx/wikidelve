"""
arXiv provider — academic papers via the public Atom API.

No API key required. Endpoint: ``http://export.arxiv.org/api/query``.

Returns abstracts as the source content. PDF links can be downloaded later
by ``app/browser.py:download_documents`` if the synthesis pass needs full
text. Tier 1 by default — arXiv is the canonical academic source for AI/ML,
math, physics, etc.
"""

import logging
import re

import httpx

logger = logging.getLogger("kb-service.sources.arxiv")

ARXIV_API_URL = "http://export.arxiv.org/api/query"


# arXiv returns Atom XML. Rather than pulling in feedparser as a hard
# dependency just for this provider, we use a small regex extractor.
# The Atom format is very stable; entries are wrapped in <entry>...</entry>.
_ENTRY_RE = re.compile(r"<entry>([\s\S]*?)</entry>")
_TITLE_RE = re.compile(r"<title>([\s\S]*?)</title>")
_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>")
_LINK_RE = re.compile(r'<link [^>]*?href="([^"]+)"[^>]*?rel="alternate"')
_ID_RE = re.compile(r"<id>([\s\S]*?)</id>")


def _clean_xml_text(s: str) -> str:
    """Strip whitespace and decode the small set of XML entities arXiv emits."""
    s = s.strip()
    s = (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )
    # Collapse internal whitespace runs
    s = re.sub(r"\s+", " ", s)
    return s


class ArxivProvider:
    """arXiv search provider (academic papers)."""

    name = "arxiv"
    tier_default = 1
    budget_attribution = False  # Free public API; not metered.

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        """Run an arXiv search and return normalized results.

        ``num`` is capped at 10 — arXiv prefers small result sets and
        synthesis doesn't need 50 abstracts on the same topic.
        """
        if not query or not query.strip():
            return []

        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(min(max(num, 1), 10)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            resp = await self.client.get(ARXIV_API_URL, params=params, timeout=15)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("arXiv API request failed: %s", exc)
            return []

        text = resp.text
        if not text:
            return []

        results: list[dict] = []
        for entry in _ENTRY_RE.findall(text):
            title_m = _TITLE_RE.search(entry)
            summary_m = _SUMMARY_RE.search(entry)
            link_m = _LINK_RE.search(entry)
            id_m = _ID_RE.search(entry)

            title = _clean_xml_text(title_m.group(1)) if title_m else ""
            abstract = _clean_xml_text(summary_m.group(1)) if summary_m else ""
            url = (link_m.group(1) if link_m else (id_m.group(1).strip() if id_m else ""))

            if not title or not url:
                continue
            # Prefix abstract with [arXiv] so the synthesis prompt knows the
            # provenance even after the formatter strips the tier label.
            content = f"[arXiv abstract] {abstract}" if abstract else "(no abstract)"
            results.append({"title": title, "content": content, "url": url})

        return results
