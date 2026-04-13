"""
Tavily provider (research/academic-focused web search).

Tavily is NOT in the per-KB Serper budget
(``budget_attribution = False``).
"""

import logging

import httpx

from app.config import TAVILY_API_KEY, TAVILY_URL

logger = logging.getLogger("kb-service.sources.tavily")


class TavilyProvider:
    """Tavily search API provider.

    Tier 2 by default. Used as a complementary source alongside Serper.
    Not budget-tracked because Tavily has its own free tier and isn't
    governed by the wikidelve auto-discovery daily budget.
    """

    name = "tavily"
    tier_default = 2
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        """Execute one Tavily search and return normalized results.

        ``num`` maps to Tavily's ``max_results``.
        """
        if not TAVILY_API_KEY:
            return []

        resp = await self.client.post(
            TAVILY_URL,
            json={
                "query": query,
                "max_results": num,
                "search_depth": "basic",
            },
            headers={
                "Authorization": f"Bearer {TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "url": r.get("url", ""),
            }
            for r in data.get("results", [])
        ]
