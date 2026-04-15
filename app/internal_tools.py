"""Internal HTTP endpoints the kimi-bridge sidecar calls back into.

Gated on a shared secret (``KIMI_BRIDGE_SECRET``) so only the sidecar
can reach them. These expose search_web / read_webpage / search_kb to
kimi-cli as externalTools, without giving kimi direct access to our
Serper API key.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException

import httpx

logger = logging.getLogger("kb-service.internal_tools")

router = APIRouter(prefix="/api/internal", tags=["internal"])

_SECRET = os.getenv("KIMI_BRIDGE_SECRET", "").strip()


def _require_secret(x_kimi_bridge_secret: str | None) -> None:
    if not _SECRET:
        # No secret configured → internal endpoints are disabled.
        raise HTTPException(status_code=404, detail="internal tools disabled")
    if (x_kimi_bridge_secret or "").strip() != _SECRET:
        raise HTTPException(status_code=401, detail="bad bridge secret")


@router.post("/search_web")
async def internal_search_web(
    body: dict[str, Any],
    x_kimi_bridge_secret: str | None = Header(default=None, alias="x-kimi-bridge-secret"),
) -> dict:
    _require_secret(x_kimi_bridge_secret)
    query = str(body.get("query", "")).strip()
    num = int(body.get("num_results", 8))
    if not query:
        raise HTTPException(status_code=400, detail="query required")
    from app.sources.serper import SerperProvider
    async with httpx.AsyncClient(timeout=15) as client:
        provider = SerperProvider(client)
        results = await provider.search(query, num=min(max(num, 1), 20))
    return {"query": query, "results": results or []}


@router.post("/read_webpage")
async def internal_read_webpage(
    body: dict[str, Any],
    x_kimi_bridge_secret: str | None = Header(default=None, alias="x-kimi-bridge-secret"),
) -> dict:
    _require_secret(x_kimi_bridge_secret)
    url = str(body.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="url required")
    from app.browser import read_page_smart
    text = await read_page_smart(url)
    if not text:
        return {"url": url, "text": None, "error": "failed to read"}
    text = str(text)
    return {"url": url, "text": text[:12000], "truncated": len(text) > 12000}


@router.post("/search_kb")
async def internal_search_kb(
    body: dict[str, Any],
    x_kimi_bridge_secret: str | None = Header(default=None, alias="x-kimi-bridge-secret"),
) -> dict:
    _require_secret(x_kimi_bridge_secret)
    query = str(body.get("query", "")).strip()
    kb = str(body.get("kb", "personal")).strip() or "personal"
    limit = int(body.get("limit", 10))
    if not query:
        raise HTTPException(status_code=400, detail="query required")
    from app.hybrid_search import hybrid_search
    results = await hybrid_search(query, kb_name=kb, limit=min(max(limit, 1), 30))
    safe = []
    for r in (results or [])[:limit]:
        if isinstance(r, dict):
            safe.append({
                "slug": r.get("slug"), "kb": r.get("kb"),
                "title": r.get("title"), "summary": r.get("summary"),
                "snippet": (r.get("snippet") or "")[:600],
                "score": r.get("score"),
            })
    return {"query": query, "kb": kb, "results": safe}
