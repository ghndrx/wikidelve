"""Tests for API key authentication middleware in app.main."""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# We must neutralise the app lifespan so TestClient doesn't try to reach
# Redis, SQLite, or S3 during startup/shutdown.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app_instance):
    """Replacement lifespan that skips all infra setup."""
    app_instance.state.redis = MagicMock()
    yield


@pytest.fixture(autouse=True)
def _patch_lifespan():
    """Replace the app's lifespan with a no-op for every test in this module."""
    from app.main import app
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    yield
    app.router.lifespan_context = original


# ---------------------------------------------------------------------------
# Fixtures: two flavours of TestClient — with and without API_KEY
# ---------------------------------------------------------------------------

@pytest.fixture
def client_no_key(monkeypatch):
    """TestClient with API_KEY unset (empty string)."""
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_with_key(monkeypatch):
    """TestClient with API_KEY set to 'test-secret'."""
    monkeypatch.setattr("app.main.API_KEY", "test-secret")
    monkeypatch.setattr("app.config.API_KEY", "test-secret")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ===========================================================================
# 1. No API_KEY set -- all requests pass (no 401)
# ===========================================================================

class TestNoApiKey:
    """When API_KEY is empty, auth middleware is effectively disabled."""

    def test_get_health_no_key(self, client_no_key):
        resp = client_no_key.get("/health")
        assert resp.status_code == 200

    def test_get_articles_no_key(self, client_no_key):
        resp = client_no_key.get("/api/articles")
        assert resp.status_code != 401

    def test_post_reindex_no_key(self, client_no_key):
        """POST should not require auth when API_KEY is unset."""
        resp = client_no_key.post("/api/search/reindex")
        assert resp.status_code != 401

    def test_delete_no_key(self, client_no_key):
        """DELETE should not return 401 when API_KEY is unset."""
        resp = client_no_key.delete("/api/articles/test/nonexistent")
        assert resp.status_code != 401


# ===========================================================================
# 2. API_KEY set -- GET requests always pass without auth
# ===========================================================================

class TestGetAlwaysOpen:
    """GET endpoints never require auth, even when API_KEY is configured."""

    def test_health_no_auth_header(self, client_with_key):
        resp = client_with_key.get("/health")
        assert resp.status_code == 200

    def test_articles_list_no_auth_header(self, client_with_key):
        resp = client_with_key.get("/api/articles")
        assert resp.status_code != 401

    def test_search_no_auth_header(self, client_with_key):
        resp = client_with_key.get("/api/search?q=test")
        assert resp.status_code != 401

    def test_autocomplete_no_auth_header(self, client_with_key):
        resp = client_with_key.get("/api/search/autocomplete?q=t")
        assert resp.status_code != 401


# ===========================================================================
# 3. API_KEY set -- POST without auth -> 401
# ===========================================================================

class TestPostRequiresAuth:
    """POST endpoints must return 401 without a valid Bearer token."""

    def test_post_no_header(self, client_with_key):
        resp = client_with_key.post("/api/search/reindex")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"

    def test_post_wrong_key(self, client_with_key):
        resp = client_with_key.post(
            "/api/search/reindex",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"

    def test_post_correct_key_passes(self, client_with_key):
        resp = client_with_key.post(
            "/api/search/reindex",
            headers={"Authorization": "Bearer test-secret"},
        )
        # Should pass auth — may succeed or error downstream, but never 401
        assert resp.status_code != 401

    def test_post_no_bearer_prefix(self, client_with_key):
        """Authorization header present but missing 'Bearer ' prefix."""
        resp = client_with_key.post(
            "/api/search/reindex",
            headers={"Authorization": "test-secret"},
        )
        assert resp.status_code == 401


# ===========================================================================
# 4. API_KEY set -- DELETE without auth -> 401
# ===========================================================================

class TestDeleteRequiresAuth:
    """DELETE endpoints return 401 without valid auth."""

    def test_delete_no_header(self, client_with_key):
        resp = client_with_key.delete("/api/articles/test/some-slug")
        assert resp.status_code == 401

    def test_delete_wrong_key(self, client_with_key):
        resp = client_with_key.delete(
            "/api/articles/test/some-slug",
            headers={"Authorization": "Bearer bad-key"},
        )
        assert resp.status_code == 401

    def test_delete_correct_key_passes(self, client_with_key):
        """With correct key, should not get 401 (may get 404 for missing article)."""
        resp = client_with_key.delete(
            "/api/articles/test/nonexistent",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert resp.status_code != 401


# ===========================================================================
# 5. Health endpoint always accessible
# ===========================================================================

class TestHealthAlwaysAccessible:
    """The /health endpoint works regardless of auth configuration."""

    def test_health_without_api_key_configured(self, client_no_key):
        resp = client_no_key.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_with_api_key_configured(self, client_with_key):
        resp = client_with_key.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
