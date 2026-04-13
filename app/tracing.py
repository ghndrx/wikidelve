"""LLM call tracing — captures every llm_chat / llm_embed invocation
as a structured trace span for debugging, regression detection, and
cost attribution.

Design (informed by research jobs #148-155):
    - In-process, zero-dep tracing (no OTel SDK required at runtime)
    - Fields follow OTel GenAI semantic conventions for naming consistency
    - SQLite storage via the existing DB path (llm_traces table)
    - DynamoDB storage via PK=TRACE#{date}, SK=ts#{trace_id}#{span_id}
    - Request-scoped trace_id via contextvar (propagated automatically
      across async boundaries by FastAPI middleware)
    - Batch flush: spans buffered in-memory, flushed every N or T seconds
    - Golden-response playback: serialize trace → fixture → diff in CI

Usage from llm.py:

    with trace_llm_call(provider="minimax", model="text-01", kind="chat") as span:
        span.set_prompt(system_msg, user_msg)
        result = await _minimax_chat(...)
        span.set_completion(result, usage=resp_usage)
    # span is auto-flushed with duration + status

Or from middleware:

    @app.middleware("http")
    async def tracing_middleware(request, call_next):
        token = start_trace(request)
        try:
            return await call_next(request)
        finally:
            end_trace(token)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("kb-service.tracing")

# Per-request trace id propagated via contextvar. Set by middleware,
# read by trace_llm_call() to nest spans under the same root.
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_request_path_var: ContextVar[str | None] = ContextVar("request_path", default=None)


def current_trace_id() -> str | None:
    return _trace_id_var.get()


def start_trace(request=None) -> Any:
    """Set a fresh trace_id for the current async context. Returns a
    reset token so the middleware can clean up."""
    tid = uuid.uuid4().hex
    token = _trace_id_var.set(tid)
    if request:
        _request_path_var.set(getattr(request.url, "path", None))
    return token


def end_trace(token: Any) -> None:
    _trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass
class TraceSpan:
    """One LLM call — mutable during the `with trace_llm_call()` block,
    then frozen and flushed on exit."""

    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    timestamp: str = ""
    name: str = ""
    provider: str = ""
    model: str = ""
    kind: str = ""  # chat | embed | chat_stream | chat_tools
    prompt_hash: str = ""
    prompt_chars: int = 0
    completion_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    status: str = "ok"
    error: str = ""
    request_path: str = ""
    session_id: str = ""

    _start_time: float = field(default=0.0, repr=False)

    def set_prompt(self, system_msg: str, user_msg: str) -> None:
        combined = (system_msg or "") + (user_msg or "")
        self.prompt_chars = len(combined)
        self.prompt_hash = hashlib.sha256(combined.encode("utf-8", errors="replace")).hexdigest()[:16]

    def set_completion(self, text: str, *, usage: dict | None = None) -> None:
        self.completion_chars = len(text or "")
        if usage:
            self.input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("inputTokens") or 0)
            self.output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("outputTokens") or 0)

    def set_error(self, exc: Exception) -> None:
        self.status = "error"
        self.error = str(exc)[:500]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_start_time", None)
        return d


# ---------------------------------------------------------------------------
# Context manager for tracing a single LLM call
# ---------------------------------------------------------------------------


@contextmanager
def trace_llm_call(
    *,
    provider: str,
    model: str,
    kind: str = "chat",
    name: str | None = None,
    session_id: str = "",
):
    """Context manager that emits a TraceSpan on exit.

    Usage:
        with trace_llm_call(provider="minimax", model="text-01", kind="chat") as span:
            span.set_prompt(system, user)
            result = await do_call(...)
            span.set_completion(result, usage=resp.get("usage"))
    """
    tid = _trace_id_var.get() or uuid.uuid4().hex
    span = TraceSpan(
        trace_id=tid,
        timestamp=datetime.now(timezone.utc).isoformat(),
        name=name or f"llm.{kind}",
        provider=provider,
        model=model,
        kind=kind,
        request_path=_request_path_var.get() or "",
        session_id=session_id,
    )
    span._start_time = time.monotonic()
    try:
        yield span
    except Exception as exc:
        span.set_error(exc)
        raise
    finally:
        span.duration_ms = int((time.monotonic() - span._start_time) * 1000)
        _buffer_span(span)


# ---------------------------------------------------------------------------
# In-memory buffer + flush
# ---------------------------------------------------------------------------

_BUFFER: deque[TraceSpan] = deque(maxlen=10_000)
_FLUSH_THRESHOLD = 50


def _buffer_span(span: TraceSpan) -> None:
    """Append a completed span to the in-memory buffer.

    The buffer is bounded (10k max) so a runaway loop can't OOM.
    Flushing to the DB is fire-and-forget from the event loop.
    """
    _BUFFER.append(span)
    if len(_BUFFER) >= _FLUSH_THRESHOLD:
        _schedule_flush()


def _schedule_flush() -> None:
    """Fire-and-forget: drain the buffer to the DB."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_flush_buffer())


async def _flush_buffer() -> None:
    """Write buffered spans to the trace store. Called periodically or
    when the buffer hits the threshold."""
    if not _BUFFER:
        return
    batch = []
    while _BUFFER:
        batch.append(_BUFFER.popleft())
    try:
        from app import db
        for span in batch:
            await db.save_trace_span(span.to_dict())
    except Exception as exc:
        logger.debug("trace flush failed: %s", exc)


async def flush_all() -> None:
    """Drain every buffered span. Call from lifespan shutdown."""
    await _flush_buffer()


# ---------------------------------------------------------------------------
# Query API (for admin dashboard + playback tests)
# ---------------------------------------------------------------------------


async def get_recent_traces(
    limit: int = 50,
    *,
    session_id: str | None = None,
    provider: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return the most recent trace spans, optionally filtered."""
    try:
        from app import db
        return await db.get_trace_spans(
            limit=limit,
            session_id=session_id,
            provider=provider,
            status=status,
        )
    except Exception as exc:
        logger.debug("get_recent_traces failed: %s", exc)
        return []


async def get_trace(trace_id: str) -> list[dict]:
    """Return all spans for a single trace_id."""
    try:
        from app import db
        return await db.get_trace_spans_by_trace_id(trace_id)
    except Exception as exc:
        logger.debug("get_trace failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def reset_buffer() -> None:
    """Test-only: clear the buffer without flushing."""
    _BUFFER.clear()
