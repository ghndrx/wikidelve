"""
Stack Exchange provider — multi-site search via the public API v2.3.

No API key required for low-volume use (a key just lifts rate limits).
Endpoint: ``https://api.stackexchange.com/2.3/search/advanced``.

Stack Exchange has many sites — Stack Overflow is the most famous, but
``security.stackexchange.com``, ``dba.stackexchange.com``,
``serverfault.com``, ``unix.stackexchange.com`` etc. are all valuable
for technical research. We default to ``stackoverflow`` but the
``site`` kwarg lets callers pick a different one.

Tier 2 by default — Q&A threads with accepted answers are reliable
but not authoritative the way official docs are.
"""

import logging

import httpx

logger = logging.getLogger("kb-service.sources.stackexchange")

STACKEXCHANGE_API_URL = "https://api.stackexchange.com/2.3/search/advanced"


class StackExchangeProvider:
    """Stack Exchange Q&A search."""

    name = "stackexchange"
    tier_default = 2
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient, site: str = "stackoverflow"):
        self.client = client
        self.site = site

    async def search(self, query: str, num: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        params = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": self.site,
            "pagesize": str(min(max(num, 1), 10)),
            "filter": "default",
        }
        try:
            resp = await self.client.get(
                STACKEXCHANGE_API_URL,
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Stack Exchange API request failed: %s", exc)
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Stack Exchange response was not JSON: %s", exc)
            return []

        items = data.get("items", [])
        results: list[dict] = []
        for item in items:
            title = (item.get("title") or "").strip()
            link = item.get("link") or ""
            if not title or not link:
                continue
            score = item.get("score") or 0
            answer_count = item.get("answer_count") or 0
            is_answered = item.get("is_answered", False)
            tags = item.get("tags") or []

            preface_bits = [
                f"{score} votes",
                f"{answer_count} answers",
            ]
            if is_answered:
                preface_bits.append("\u2713 accepted")
            if tags:
                preface_bits.append("[" + ", ".join(tags[:5]) + "]")
            preface = "[" + self.site + " \u2022 " + " \u2022 ".join(preface_bits) + "]"

            content = f"{preface} (open the thread for the question + answers)"
            results.append({"title": title, "content": content, "url": link})

        return results
