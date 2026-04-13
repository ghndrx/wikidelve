"""Tests for HTTP middleware stack in app.main.

Covers: body size limit, security headers, CSP nonce generation.
Auth middleware is covered in test_auth.py.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(app_instance):
    app_instance.state.redis = MagicMock()
    yield


@pytest.fixture(autouse=True)
def _patch_lifespan():
    from app.main import app
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    yield
    app.router.lifespan_context = original


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ===========================================================================
# Body size limit middleware
# ===========================================================================

class TestBodySizeLimit:
    def test_rejects_oversized_content_length(self, client, monkeypatch):
        monkeypatch.setattr("app.main.MAX_REQUEST_BODY_BYTES", 100)
        resp = client.post(
            "/api/search/reindex",
            headers={"Content-Length": "200"},
        )
        assert resp.status_code == 413
        assert resp.json()["detail"] == "Request body too large"

    def test_accepts_within_limit(self, client, monkeypatch):
        monkeypatch.setattr("app.main.MAX_REQUEST_BODY_BYTES", 10_000_000)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_rejects_invalid_content_length(self, client):
        resp = client.post(
            "/api/search/reindex",
            headers={"Content-Length": "not-a-number"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid Content-Length"


# ===========================================================================
# Security headers middleware
# ===========================================================================

class TestSecurityHeaders:
    def test_sets_hsts(self, client):
        resp = client.get("/health")
        assert "max-age=63072000" in resp.headers.get("Strict-Transport-Security", "")

    def test_sets_nosniff(self, client):
        assert client.get("/health").headers["X-Content-Type-Options"] == "nosniff"

    def test_sets_frame_options(self, client):
        assert client.get("/health").headers["X-Frame-Options"] == "SAMEORIGIN"

    def test_sets_referrer_policy(self, client):
        assert "strict-origin" in client.get("/health").headers.get("Referrer-Policy", "")

    def test_sets_permissions_policy(self, client):
        pp = client.get("/health").headers.get("Permissions-Policy", "")
        assert "geolocation=()" in pp

    def test_sets_csp_with_nonce(self, client):
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "nonce-" in csp
        assert "'strict-dynamic'" in csp
        assert "default-src 'self'" in csp

    def test_csp_nonce_changes_per_request(self, client):
        r1 = client.get("/health")
        r2 = client.get("/health")
        csp1 = r1.headers["Content-Security-Policy"]
        csp2 = r2.headers["Content-Security-Policy"]
        assert csp1 != csp2

    def test_server_header_replaced(self, client):
        resp = client.get("/health")
        assert resp.headers.get("Server") == "wikidelve"

    def test_cross_origin_headers(self, client):
        resp = client.get("/health")
        assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
        assert resp.headers.get("Cross-Origin-Resource-Policy") == "same-origin"
