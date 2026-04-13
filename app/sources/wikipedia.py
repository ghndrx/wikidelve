"""
Wikipedia direct provider — bypass Serper for Wikipedia results.

Uses two MediaWiki API endpoints:
  1. ``opensearch`` to find page titles matching the query
  2. ``query?prop=extracts`` to fetch plaintext extracts in one request

No API key required. Tier 1 by default — Wikipedia is generally
authoritative for definitions and overviews.
"""

import logging
from urllib.parse import quote

import httpx

logger = logging.getLogger("kb-service.sources.wikipedia")

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_BASE = "https://en.wikipedia.org/wiki/"


class WikipediaProvider:
    """Wikipedia search + extract provider."""

    name = "wikipedia"
    tier_default = 1
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        # Step 1: opensearch to find candidate page titles
        try:
            search_resp = await self.client.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": str(min(max(num, 1), 10)),
                    "namespace": "0",
                    "format": "json",
                },
                headers={"User-Agent": "wikidelve/1.0 (+https://github.com)"},
                timeout=10,
            )
            search_resp.raise_for_status()
            opensearch = search_resp.json()
        except Exception as exc:
            logger.warning("Wikipedia opensearch failed: %s", exc)
            return []

        # opensearch returns [query, [titles], [descriptions], [urls]]
        if not isinstance(opensearch, list) or len(opensearch) < 4:
            return []
        titles = opensearch[1] if isinstance(opensearch[1], list) else []
        urls = opensearch[3] if isinstance(opensearch[3], list) else []
        if not titles:
            return []

        # Step 2: fetch plaintext extracts for those titles in one request
        try:
            extracts_resp = await self.client.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "prop": "extracts",
                    "exintro": "1",  # Just the lead section
                    "explaintext": "1",
                    "titles": "|".join(titles),
                    "format": "json",
                    "redirects": "1",
                },
                headers={"User-Agent": "wikidelve/1.0 (+https://github.com)"},
                timeout=15,
            )
            extracts_resp.raise_for_status()
            extracts = extracts_resp.json()
        except Exception as exc:
            logger.warning("Wikipedia extracts request failed: %s", exc)
            return []

        pages = extracts.get("query", {}).get("pages", {})
        # Build a title → extract map (titles in extracts may differ from
        # search titles due to redirects/normalization).
        title_to_extract: dict[str, str] = {}
        for page in pages.values():
            t = page.get("title")
            extract = page.get("extract") or ""
            if t and extract:
                title_to_extract[t] = extract

        results: list[dict] = []
        for i, title in enumerate(titles):
            extract = title_to_extract.get(title, "")
            if not extract:
                # Fall back to a 1-line placeholder if extract missing
                extract = "(no extract available)"
            url = urls[i] if i < len(urls) and urls[i] else (
                WIKIPEDIA_PAGE_BASE + quote(title.replace(" ", "_"))
            )
            results.append({
                "title": title,
                "content": f"[Wikipedia] {extract}",
                "url": url,
            })

        return results
