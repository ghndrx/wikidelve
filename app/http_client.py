"""
Shared httpx.AsyncClient pool.

A single process-wide httpx client with connection pooling reuses TCP
+ TLS across upstream calls instead of paying the handshake cost on
every request. At our volume (multiple LLM + Serper + S3 round-trips
per research job) this compounds to seconds saved per job.

Usage:

    from app.http_client import get_http_client
    client = get_http_client()
    resp = await client.get("https://example.com", timeout=15)

Callers can pass a per-call ``timeout=`` override to the individual
method (get/post/etc) to override the default 30 s.

The client is lazy-initialised on first access — both the FastAPI app
and the arq worker can import this without coordinating a lifespan
hook. ``close_http_client()`` should be called from the lifespan
shutdown to drain keep-alives gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger("kb-service.http_client")


_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide shared httpx.AsyncClient.

    Lazy-creates on first call. Safe to call from any task; httpx
    handles concurrent access to the client.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
        logger.debug("Initialised shared httpx client pool")
    return _client


async def close_http_client() -> None:
    """Close the shared client and release its connections.

    Called from the FastAPI lifespan shutdown and the worker
    ``on_shutdown`` hook.
    """
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        logger.debug("Closed shared httpx client pool")
    _client = None
