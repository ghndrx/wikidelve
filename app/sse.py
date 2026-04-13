"""
Server-Sent Events helper.

Wraps an async generator into a ``StreamingResponse`` formatted as SSE
chunks (``data: {json}\\n\\n``). Used by the chat endpoint to stream LLM
token deltas to the browser.

Note that ``EventSource`` cannot send custom headers — clients that need
to pass an API_KEY Bearer token should consume the SSE format via
``fetch()`` + ``Response.body.getReader()`` instead. The wire format is
identical either way.

The helper polls ``request.is_disconnected()`` between yields so an
orphan browser tab doesn't keep an upstream LLM stream burning tokens
after the client is gone.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Mapping, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("kb-service.sse")


# Default reconnect hint in ms. Clients that drop will wait this long
# before re-opening; tune via ``retry_ms`` on the call site.
_DEFAULT_RETRY_MS = 3000

# Keep-alive comment sent when no events flow for longer than this.
# Browsers + most proxies will drop an idle SSE stream after ~30s, so
# 15s is the conventional interval.
_KEEPALIVE_INTERVAL = 15.0


def sse_response(
    async_gen: AsyncGenerator[Mapping, None],
    request: Optional[Request] = None,
    retry_ms: int = _DEFAULT_RETRY_MS,
) -> StreamingResponse:
    """Wrap an async generator into a Server-Sent Events StreamingResponse.

    Each yielded mapping should contain ``data`` (any JSON-serializable value)
    and may optionally include:
        - ``event``: SSE event name (for ``addEventListener`` on the client)
        - ``id``: event id for reconnection support
        - ``retry``: per-event reconnect hint; defaults to ``retry_ms``

    When ``request`` is provided the loop exits as soon as the client
    disconnects instead of waiting for the upstream generator to finish
    on its own — this stops LLM token streams from running into the
    void when the browser tab closes mid-response.
    """

    async def stream():
        # Emit an initial retry hint so a client that drops mid-stream
        # reconnects with a predictable delay rather than whatever its
        # default is.
        yield f"retry: {retry_ms}\n\n"

        gen_task: Optional[asyncio.Task] = None
        try:
            aiter = async_gen.__aiter__()
            while True:
                if request is not None and await request.is_disconnected():
                    logger.info("SSE client disconnected, closing stream")
                    break

                gen_task = asyncio.create_task(aiter.__anext__())
                try:
                    # Bound the wait so we can also emit keep-alives and
                    # re-check the disconnect flag on idle streams.
                    msg = await asyncio.wait_for(
                        asyncio.shield(gen_task), timeout=_KEEPALIVE_INTERVAL
                    )
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                except StopAsyncIteration:
                    break

                if not isinstance(msg, Mapping):
                    # Be forgiving: wrap bare strings/objects so callers can
                    # yield raw payloads if they want.
                    msg = {"data": msg}
                if "event" in msg:
                    yield f"event: {msg['event']}\n"
                if "id" in msg:
                    yield f"id: {msg['id']}\n"
                if "retry" in msg:
                    yield f"retry: {msg['retry']}\n"
                yield f"data: {json.dumps(msg.get('data', ''))}\n\n"
        except GeneratorExit:
            # Client disconnected — exit cleanly without logging.
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE stream error")
            try:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            except Exception:
                pass
        finally:
            if gen_task is not None and not gen_task.done():
                gen_task.cancel()
                try:
                    await gen_task
                except (asyncio.CancelledError, Exception):
                    pass
            aclose = getattr(async_gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
