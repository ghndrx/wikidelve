"""Unit tests for the security headers middleware registered in app.main."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Import inside the fixture so the env overrides in tests/conftest.py
    # (DB_BACKEND=sqlite, STORAGE_BACKEND=local) take effect first.
    from app.main import app

    # Stub the lifespan startup so the test client doesn't try to reach
    # Redis / DynamoDB / S3 during collection.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app_):
        app_.state.redis = None
        yield

    app.router.lifespan_context = _noop_lifespan

    with TestClient(app) as c:
        yield c


class TestSecurityHeaders:
    def test_hsts_set(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "Strict-Transport-Security" in resp.headers
        assert "max-age" in resp.headers["Strict-Transport-Security"]

    def test_nosniff(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_frame_options(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"

    def test_referrer_policy(self, client):
        resp = client.get("/health")
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_permissions_policy(self, client):
        resp = client.get("/health")
        pp = resp.headers.get("Permissions-Policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp

    def test_csp_frame_ancestors(self, client):
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "frame-ancestors 'self'" in csp
        assert "default-src 'self'" in csp

    def test_csp_has_nonce(self, client):
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "nonce-" in csp
        assert "strict-dynamic" in csp

    def test_script_src_is_strict(self, client):
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        # Isolate the script-src directive and verify unsafe-inline is
        # NOT allowed there. style-src is intentionally looser.
        script_src = next(
            (d for d in csp.split(";") if d.strip().startswith("script-src")),
            "",
        )
        assert script_src
        assert "'unsafe-inline'" not in script_src
        assert "strict-dynamic" in script_src

    def test_csp_nonce_rotates_per_request(self, client):
        a = client.get("/health").headers.get("Content-Security-Policy", "")
        b = client.get("/health").headers.get("Content-Security-Policy", "")
        import re
        na = re.search(r"nonce-([A-Za-z0-9_\-]+)", a)
        nb = re.search(r"nonce-([A-Za-z0-9_\-]+)", b)
        assert na and nb and na.group(1) != nb.group(1)

    def test_server_header_generic(self, client):
        resp = client.get("/health")
        # Server header should be the generic token, not "uvicorn".
        assert resp.headers.get("Server") == "wikidelve"

    def test_no_powered_by_leak(self, client):
        resp = client.get("/health")
        assert "X-Powered-By" not in resp.headers


class TestTrustedHost:
    def test_unknown_host_rejected(self, client):
        resp = client.get("/health", headers={"Host": "evil.example.com"})
        assert resp.status_code == 400

    def test_localhost_allowed(self, client):
        resp = client.get("/health", headers={"Host": "localhost"})
        assert resp.status_code == 200

    def test_tailnet_host_allowed(self, client):
        resp = client.get(
            "/health", headers={"Host": "wikidelve.example.ts.net"}
        )
        assert resp.status_code == 200


class TestBodySizeLimit:
    def test_small_body_ok(self, client):
        resp = client.post(
            "/api/research",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        # 400 because the handler rejects invalid JSON, but NOT 413.
        assert resp.status_code != 413

    def test_oversized_body_rejected(self, client, monkeypatch):
        from app import main as _main
        monkeypatch.setattr(_main, "MAX_REQUEST_BODY_BYTES", 1024)
        resp = client.post(
            "/api/research",
            content=b"{}",
            headers={"content-type": "application/json", "content-length": "99999"},
        )
        assert resp.status_code == 413

    def test_invalid_content_length_rejected(self, client):
        resp = client.post(
            "/api/research",
            content=b"{}",
            headers={"content-type": "application/json", "content-length": "abc"},
        )
        assert resp.status_code == 400


class TestHealthMethods:
    def test_head_health_allowed(self, client):
        resp = client.head("/health")
        assert resp.status_code == 200
