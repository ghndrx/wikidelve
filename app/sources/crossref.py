"""
Crossref provider — academic papers via the public REST API.

No API key required. Endpoint: ``https://api.crossref.org/works``.

Crossref indexes scholarly publications via DOIs. Tier 1 by default —
peer-reviewed papers are about as authoritative as it gets for technical
research. The provider returns the title + abstract (when available)
plus the DOI URL so the deep-read pass can pull the full paper if it's
open access.
"""

import logging

import httpx

logger = logging.getLogger("kb-service.sources.crossref")

CROSSREF_API_URL = "https://api.crossref.org/works"


class CrossrefProvider:
    """Crossref scholarly publication search."""

    name = "crossref"
    tier_default = 1
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        params = {
            "query": query,
            "rows": str(min(max(num, 1), 10)),
            "select": "title,abstract,DOI,URL,author,published-print,container-title",
        }
        try:
            resp = await self.client.get(
                CROSSREF_API_URL,
                params=params,
                headers={
                    "User-Agent": "wikidelve/1.0 (mailto:noreply@wikidelve.local)",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Crossref API request failed: %s", exc)
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Crossref response was not JSON: %s", exc)
            return []

        items = data.get("message", {}).get("items", [])
        results: list[dict] = []
        for item in items:
            titles = item.get("title") or []
            title = titles[0].strip() if titles else ""
            doi = item.get("DOI") or ""
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            if not title or not url:
                continue

            # Crossref abstracts are often wrapped in JATS XML — strip tags.
            raw_abstract = (item.get("abstract") or "").strip()
            if raw_abstract:
                import re as _re
                abstract = _re.sub(r"<[^>]+>", "", raw_abstract).strip()
            else:
                abstract = ""

            authors = item.get("author") or []
            author_names = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in authors[:3]
            ).strip()
            if len(authors) > 3:
                author_names += ", et al."

            container = (item.get("container-title") or [""])[0]
            year = ""
            published = item.get("published-print", {}) or {}
            date_parts = published.get("date-parts") or []
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])

            preface_bits = []
            if author_names:
                preface_bits.append(author_names)
            if container:
                preface_bits.append(container)
            if year:
                preface_bits.append(year)
            preface = " — ".join(preface_bits) if preface_bits else ""

            content = f"[Crossref] {preface}\n{abstract or '(no abstract available)'}"
            results.append({"title": title, "content": content, "url": url})

        return results
