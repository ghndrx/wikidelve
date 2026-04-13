"""Unit tests for the shared httpx client pool."""

from __future__ import annotations

import httpx
import pytest

from app import http_client


@pytest.fixture(autouse=True)
async def reset_pool():
    """Reset the module-level singleton between tests."""
    await http_client.close_http_client()
    yield
    await http_client.close_http_client()


class TestGetHttpClient:
    def test_returns_async_client(self):
        client = http_client.get_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_returns_same_instance_on_repeat_calls(self):
        a = http_client.get_http_client()
        b = http_client.get_http_client()
        assert a is b

    def test_pool_limits_are_configured(self):
        client = http_client.get_http_client()
        # httpx exposes the limits on the transport pool — we check the
        # ones we actually rely on for the research pipeline's
        # concurrent fan-out.
        limits = client._transport._pool._max_connections
        assert limits == 100

    def test_default_timeout_is_thirty_seconds(self):
        client = http_client.get_http_client()
        # Read timeout, not connect
        assert client.timeout.read == 30.0
        assert client.timeout.connect == 10.0

    def test_follow_redirects_enabled(self):
        client = http_client.get_http_client()
        assert client.follow_redirects is True


class TestCloseHttpClient:
    @pytest.mark.asyncio
    async def test_close_marks_client_closed(self):
        client = http_client.get_http_client()
        assert not client.is_closed
        await http_client.close_http_client()
        # After close, the module-level reference should be cleared so
        # a subsequent get_http_client() rebuilds a fresh pool.
        assert http_client._client is None

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        await http_client.close_http_client()
        # Calling close on an already-closed pool must not raise.
        await http_client.close_http_client()

    @pytest.mark.asyncio
    async def test_get_after_close_rebuilds_pool(self):
        a = http_client.get_http_client()
        await http_client.close_http_client()
        b = http_client.get_http_client()
        assert a is not b
        assert not b.is_closed
