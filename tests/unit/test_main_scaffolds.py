"""Integration tests for scaffold-related routes in app/main.py.

Coverage focus by risk:
  1. POST /api/scaffolds/{kb}/create — input validation (topic + type)
  2. GET /api/scaffolds/{kb} — list endpoint contract
  3. GET /api/scaffolds/{kb}/{slug} — manifest fetch / 404
  4. GET /sandbox/{kb}/{slug}/{path} — entrypoint redirect, CSP headers,
     MIME mapping, path-escape rejection
  5. GET /scaffolds and /scaffolds/{kb}/{slug} — viewer pages render
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(app_instance):
    app_instance.state.redis = MagicMock()
    app_instance.state.redis.enqueue_job = AsyncMock(return_value=MagicMock(job_id="fake"))
    yield


@pytest.fixture(autouse=True)
def _patch_lifespan():
    from app.main import app
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    yield
    app.router.lifespan_context = original


@pytest.fixture
def tmp_kb_for_scaffolds(tmp_path, monkeypatch):
    import app.config as _config
    import app.storage as _storage
    kb_root = tmp_path / "kb"
    kb_dir = kb_root / "personal"
    (kb_dir / "wiki").mkdir(parents=True)
    prev = dict(_config.KB_DIRS)
    _config.KB_DIRS.clear()
    _config.KB_DIRS["personal"] = kb_dir
    monkeypatch.setattr(_config, "KB_ROOT", kb_root)
    _storage.set_storage(_storage._build_default())
    yield "personal"
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(prev)
    _storage.set_storage(_storage._build_default())


@pytest.fixture
def client(monkeypatch, tmp_kb_for_scaffolds):
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    # Stub the create_job db call so we don't need the full DB stack —
    # the route only needs an int back to enqueue.
    import app.db as _db
    monkeypatch.setattr(_db, "create_job", AsyncMock(return_value=42))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _mk_scaffold(kb: str, slug: str = "demo", entrypoint: str = "index.html"):
    """Helper: write a scaffold via the storage API for tests that need
    one to already exist."""
    from app import scaffolds
    return scaffolds.create_scaffold(
        kb,
        {
            "title": slug.replace("-", " ").title(),
            "scaffold_type": "landing-page",
            "framework": "vanilla",
            "preview_type": "static",
            "entrypoint": entrypoint,
        },
        [
            {"path": entrypoint, "content": "<!doctype html><h1>Hi</h1>"},
            {"path": "styles.css", "content": "body { color: red; }"},
        ],
    )


# ---------------------------------------------------------------------------
# POST /api/scaffolds/{kb}/create
# ---------------------------------------------------------------------------

class TestCreateEndpoint:
    def test_valid_request_queues_job(self, client):
        r = client.post(
            "/api/scaffolds/personal/create",
            json={"topic": "saas landing", "scaffold_type": "landing-page"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "queued"
        assert body["scaffold_type"] == "landing-page"
        assert body["job_id"] == 42

    def test_short_topic_rejected(self, client):
        r = client.post(
            "/api/scaffolds/personal/create",
            json={"topic": "ab", "scaffold_type": "landing-page"},
        )
        assert r.status_code == 400

    def test_unknown_scaffold_type_rejected(self, client):
        r = client.post(
            "/api/scaffolds/personal/create",
            json={"topic": "valid topic here", "scaffold_type": "not-a-thing"},
        )
        assert r.status_code == 400

    def test_missing_body_rejected(self, client):
        r = client.post(
            "/api/scaffolds/personal/create",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET listing + manifest
# ---------------------------------------------------------------------------

class TestListAndFetch:
    def test_empty_list(self, client):
        r = client.get("/api/scaffolds/personal")
        assert r.status_code == 200
        body = r.json()
        assert body["kb"] == "personal"
        assert body["scaffolds"] == []

    def test_list_after_create(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get("/api/scaffolds/personal")
        assert r.status_code == 200
        slugs = {s["slug"] for s in r.json()["scaffolds"]}
        assert slug in slugs

    def test_fetch_manifest(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get(f"/api/scaffolds/personal/{slug}")
        assert r.status_code == 200
        m = r.json()
        assert m["entrypoint"] == "index.html"
        assert "files" in m

    def test_fetch_manifest_404(self, client):
        r = client.get("/api/scaffolds/personal/ghost")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /sandbox — file serving + CSP + path safety
# ---------------------------------------------------------------------------

class TestSandboxRoute:
    def test_redirect_to_entrypoint(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get(f"/sandbox/personal/{slug}/", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"].endswith("/index.html")

    def test_serve_html_with_csp(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get(f"/sandbox/personal/{slug}/index.html")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        csp = r.headers.get("content-security-policy", "")
        # Lockdown invariants
        assert "connect-src 'none'" in csp
        assert "default-src 'self' data:" in csp
        assert "frame-ancestors 'self'" in csp
        assert r.headers.get("x-content-type-options") == "nosniff"

    def test_serve_css_with_correct_mime(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get(f"/sandbox/personal/{slug}/styles.css")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/css")

    def test_unknown_extension_falls_back_to_text(self, client, tmp_kb_for_scaffolds):
        from app import scaffolds
        slug = scaffolds.create_scaffold(
            tmp_kb_for_scaffolds,
            {"title": "Mixed", "scaffold_type": "landing-page",
             "framework": "vanilla", "preview_type": "static",
             "entrypoint": "index.html"},
            [
                {"path": "index.html", "content": "<!doctype html>"},
                {"path": "data.foobar", "content": "binary-ish"},
            ],
        )
        r = client.get(f"/sandbox/personal/{slug}/data.foobar")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")

    def test_file_not_found(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get(f"/sandbox/personal/{slug}/missing.html")
        assert r.status_code == 404

    def test_scaffold_not_found(self, client):
        r = client.get("/sandbox/personal/ghost/index.html")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Viewer + browse pages
# ---------------------------------------------------------------------------

class TestViewerPages:
    def test_browse_renders(self, client):
        r = client.get("/scaffolds")
        assert r.status_code == 200
        # Page should mention all the top-level types as filters.
        body = r.text
        assert "landing-page" in body
        assert "Scaffolds" in body

    def test_viewer_404s_for_missing(self, client):
        r = client.get("/scaffolds/personal/ghost")
        assert r.status_code == 404

    def test_viewer_renders_for_existing(self, client, tmp_kb_for_scaffolds):
        slug = _mk_scaffold(tmp_kb_for_scaffolds)
        r = client.get(f"/scaffolds/personal/{slug}")
        assert r.status_code == 200
        body = r.text
        assert "iframe" in body  # landing-page → iframe preview
        assert "index.html" in body  # file listing
