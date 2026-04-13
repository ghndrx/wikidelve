"""Unit tests for app.tracing — span creation, context propagation,
buffer management, and DB persistence.
"""

from __future__ import annotations

import pytest

from app import tracing


@pytest.fixture(autouse=True)
def _clean_state():
    tracing.reset_buffer()
    yield
    tracing.reset_buffer()


# ---------------------------------------------------------------------------
# TraceSpan dataclass
# ---------------------------------------------------------------------------


class TestTraceSpan:
    def test_set_prompt_computes_hash_and_chars(self):
        span = tracing.TraceSpan(trace_id="t1")
        span.set_prompt("system prompt", "user message")
        assert span.prompt_chars == len("system prompt") + len("user message")
        assert len(span.prompt_hash) == 16

    def test_set_completion_extracts_usage(self):
        span = tracing.TraceSpan(trace_id="t1")
        span.set_completion("hello world", usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
        })
        assert span.completion_chars == 11
        assert span.input_tokens == 10
        assert span.output_tokens == 5

    def test_set_completion_handles_bedrock_keys(self):
        span = tracing.TraceSpan(trace_id="t1")
        span.set_completion("result", usage={
            "inputTokens": 20,
            "outputTokens": 8,
        })
        assert span.input_tokens == 20
        assert span.output_tokens == 8

    def test_set_error(self):
        span = tracing.TraceSpan(trace_id="t1")
        span.set_error(RuntimeError("minimax 500"))
        assert span.status == "error"
        assert "minimax 500" in span.error

    def test_to_dict_excludes_private(self):
        span = tracing.TraceSpan(trace_id="t1")
        d = span.to_dict()
        assert "_start_time" not in d
        assert "trace_id" in d

    def test_same_prompt_same_hash(self):
        a = tracing.TraceSpan(trace_id="t1")
        a.set_prompt("sys", "usr")
        b = tracing.TraceSpan(trace_id="t2")
        b.set_prompt("sys", "usr")
        assert a.prompt_hash == b.prompt_hash

    def test_different_prompt_different_hash(self):
        a = tracing.TraceSpan(trace_id="t1")
        a.set_prompt("sys", "hello")
        b = tracing.TraceSpan(trace_id="t2")
        b.set_prompt("sys", "world")
        assert a.prompt_hash != b.prompt_hash


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestTraceLlmCall:
    def test_basic_span_creation(self):
        with tracing.trace_llm_call(
            provider="minimax", model="text-01", kind="chat",
        ) as span:
            span.set_prompt("sys", "usr")
            span.set_completion("result", usage={"prompt_tokens": 5, "completion_tokens": 3})

        assert span.provider == "minimax"
        assert span.model == "text-01"
        assert span.kind == "chat"
        assert span.duration_ms >= 0
        assert span.status == "ok"

    def test_error_sets_status(self):
        with pytest.raises(ValueError):
            with tracing.trace_llm_call(
                provider="minimax", model="text-01", kind="chat",
            ) as span:
                raise ValueError("bad input")

        assert span.status == "error"
        assert "bad input" in span.error

    def test_span_buffered(self):
        tracing.reset_buffer()
        with tracing.trace_llm_call(
            provider="minimax", model="text-01", kind="chat",
        ) as span:
            pass

        assert len(tracing._BUFFER) == 1
        assert tracing._BUFFER[0] is span


# ---------------------------------------------------------------------------
# Trace context propagation
# ---------------------------------------------------------------------------


class TestContextPropagation:
    def test_start_and_end_trace(self):
        token = tracing.start_trace()
        tid = tracing.current_trace_id()
        assert tid is not None and len(tid) == 32
        tracing.end_trace(token)
        assert tracing.current_trace_id() is None

    def test_spans_inherit_request_trace_id(self):
        token = tracing.start_trace()
        expected = tracing.current_trace_id()

        with tracing.trace_llm_call(
            provider="test", model="test", kind="chat",
        ) as span:
            pass

        assert span.trace_id == expected
        tracing.end_trace(token)


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------


class TestBuffer:
    def test_buffer_bounded(self):
        tracing.reset_buffer()
        for i in range(11_000):
            tracing._BUFFER.append(tracing.TraceSpan(trace_id=f"t{i}"))
        # maxlen=10_000
        assert len(tracing._BUFFER) == 10_000

    def test_reset_clears(self):
        tracing._BUFFER.append(tracing.TraceSpan(trace_id="t"))
        assert len(tracing._BUFFER) > 0
        tracing.reset_buffer()
        assert len(tracing._BUFFER) == 0


# ---------------------------------------------------------------------------
# DB persistence (requires init_db)
# ---------------------------------------------------------------------------


class TestTracePersistence:
    @pytest.mark.asyncio
    async def test_save_and_query(self, tmp_path, monkeypatch):
        from app import db, config
        db_path = tmp_path / "trace-test.db"
        monkeypatch.setattr(config, "DB_PATH", db_path)
        monkeypatch.setattr(db, "DB_PATH", db_path, raising=False)
        await db.init_db()

        span_dict = tracing.TraceSpan(
            trace_id="abc123",
            provider="minimax",
            model="text-01",
            kind="chat",
            timestamp="2026-04-12T00:00:00Z",
        ).to_dict()

        await db.save_trace_span(span_dict)

        rows = await db.get_trace_spans(limit=10)
        assert len(rows) == 1
        assert rows[0]["trace_id"] == "abc123"
        assert rows[0]["provider"] == "minimax"

    @pytest.mark.asyncio
    async def test_query_by_trace_id(self, tmp_path, monkeypatch):
        from app import db, config
        db_path = tmp_path / "trace-test.db"
        monkeypatch.setattr(config, "DB_PATH", db_path)
        monkeypatch.setattr(db, "DB_PATH", db_path, raising=False)
        await db.init_db()

        for i in range(3):
            span = tracing.TraceSpan(
                trace_id="xyz789",
                span_id=f"span-{i}",
                provider="bedrock",
                model="sonnet",
                kind="chat",
                timestamp="2026-04-12T00:00:00Z",
            ).to_dict()
            await db.save_trace_span(span)

        rows = await db.get_trace_spans_by_trace_id("xyz789")
        assert len(rows) == 3
        assert all(r["trace_id"] == "xyz789" for r in rows)

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, tmp_path, monkeypatch):
        from app import db, config
        db_path = tmp_path / "trace-test.db"
        monkeypatch.setattr(config, "DB_PATH", db_path)
        monkeypatch.setattr(db, "DB_PATH", db_path, raising=False)
        await db.init_db()

        for p in ("minimax", "minimax", "bedrock"):
            await db.save_trace_span(tracing.TraceSpan(
                trace_id="t",
                provider=p,
                model="m",
                kind="chat",
                timestamp="2026-04-12T00:00:00Z",
            ).to_dict())

        minimax_only = await db.get_trace_spans(limit=50, provider="minimax")
        assert len(minimax_only) == 2

    @pytest.mark.asyncio
    async def test_filter_by_status(self, tmp_path, monkeypatch):
        from app import db, config
        db_path = tmp_path / "trace-test.db"
        monkeypatch.setattr(config, "DB_PATH", db_path)
        monkeypatch.setattr(db, "DB_PATH", db_path, raising=False)
        await db.init_db()

        ok_span = tracing.TraceSpan(trace_id="t", provider="p", model="m",
                                     kind="chat", status="ok",
                                     timestamp="2026-04-12T00:00:00Z")
        err_span = tracing.TraceSpan(trace_id="t", provider="p", model="m",
                                      kind="chat", status="error",
                                      timestamp="2026-04-12T00:00:00Z")
        await db.save_trace_span(ok_span.to_dict())
        await db.save_trace_span(err_span.to_dict())

        errors = await db.get_trace_spans(limit=50, status="error")
        assert len(errors) == 1
        assert errors[0]["status"] == "error"
