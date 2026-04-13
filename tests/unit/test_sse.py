"""Unit tests for app.sse."""

import asyncio
import json

import pytest

from app.sse import sse_response


async def _collect(async_gen):
    chunks = []
    async for chunk in async_gen:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
    return chunks


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestSseResponse:
    def test_data_only_chunk(self):
        async def gen():
            yield {"data": "hello"}

        response = sse_response(gen())
        chunks = _run(_collect(response.body_iterator))
        assert any(c == 'data: "hello"\n\n' for c in chunks)

    def test_named_event_and_id(self):
        async def gen():
            yield {"event": "status", "id": "1", "data": {"ok": True}}

        response = sse_response(gen())
        chunks = _run(_collect(response.body_iterator))
        assert "event: status\n" in chunks
        assert "id: 1\n" in chunks
        payload = [c for c in chunks if c.startswith("data:")]
        assert payload
        assert json.loads(payload[0].removeprefix("data: ").strip()) == {"ok": True}

    def test_plain_string_is_wrapped(self):
        async def gen():
            yield "bare-string"

        response = sse_response(gen())
        chunks = _run(_collect(response.body_iterator))
        assert any(c == 'data: "bare-string"\n\n' for c in chunks)

    def test_exception_emits_error_chunk(self):
        async def gen():
            yield {"data": "before"}
            raise RuntimeError("boom")

        response = sse_response(gen())
        chunks = _run(_collect(response.body_iterator))
        assert any('"error"' in c and "boom" in c for c in chunks)

    def test_headers_include_no_cache(self):
        async def gen():
            if False:
                yield {}
            return

        response = sse_response(gen())
        assert response.media_type == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["connection"] == "keep-alive"

    def test_initial_retry_hint_emitted(self):
        async def gen():
            yield {"data": "x"}

        response = sse_response(gen(), retry_ms=2500)
        chunks = _run(_collect(response.body_iterator))
        assert chunks[0] == "retry: 2500\n\n"

    def test_client_disconnect_exits_early(self):
        # Fake Request whose is_disconnected returns True after the
        # first iteration — the loop should exit cleanly with no more
        # data chunks streamed.
        calls = {"n": 0}

        class FakeRequest:
            async def is_disconnected(self):
                calls["n"] += 1
                return calls["n"] >= 2  # clean on first check, disconnected after

        async def gen():
            yield {"data": "first"}
            yield {"data": "second"}
            yield {"data": "third"}

        response = sse_response(gen(), request=FakeRequest())
        chunks = _run(_collect(response.body_iterator))
        data_chunks = [c for c in chunks if c.startswith("data:")]
        # Should stream exactly one data event before the disconnect check
        # trips on the next iteration.
        assert len(data_chunks) == 1
        assert '"first"' in data_chunks[0]
