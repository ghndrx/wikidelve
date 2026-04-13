"""
Serper.dev provider (Google search via API).

Preserves the per-KB budget instrumentation by reading the ContextVars
that ``run_research`` sets — they are owned by ``app/research`` and are
imported here read-only.
"""

import logging

import httpx

from app import db
from app.config import SERPER_API_KEY, SERPER_URL

logger = logging.getLogger("kb-service.sources.serper")


class SerperProvider:
    """Google Serper API provider.

    Tier 1 by default — Serper surfaces authoritative results from official
    docs, .gov, .edu, etc. ``budget_attribution`` is True so every call is
    logged to ``serper_usage`` against the current KB + job (set via
    ContextVars in ``run_research``).
    """

    name = "serper"
    tier_default = 1
    budget_attribution = True

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 8) -> list[dict]:
        """Execute one Serper search and return normalized results.

        Returns an empty list if SERPER_API_KEY is missing (graceful
        degradation — research can still proceed via Tavily / other providers).
        """
        if not SERPER_API_KEY:
            return []

        resp = await self.client.post(
            SERPER_URL,
            json={"q": query, "num": num},
            headers={"X-API-KEY": SERPER_API_KEY},
        )
        resp.raise_for_status()

        # Budget logging — read ContextVars from research module so the
        # auto-discovery per-KB daily budget keeps working unchanged.
        try:
            from app.research import _current_kb, _current_job_id
            await db.log_serper_call(
                query, num, _current_kb.get(), _current_job_id.get(),
            )
        except Exception as exc:
            logger.warning("Failed to log Serper usage: %s", exc)

        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "content": r.get("snippet", ""),
                "url": r.get("link", ""),
            }
            for r in data.get("organic", [])
        ]
