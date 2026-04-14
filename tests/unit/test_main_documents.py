"""Integration tests for documents-related routes in app/main.py.

The Documents feature ships in chunks; chunk 1 is just the storage
layer + a placeholder browse page. These tests pin the contract for
what's exposed today so chunks 2-4 don't accidentally break it.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_lifespan(app_instance):
    app_instance.state.redis = MagicMock()
    app_instance.state.redis.enqueue_job = AsyncMock()
    yield


@pytest.fixture(autouse=True)
def _patch_lifespan():
    from app.main import app
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    yield
    app.router.lifespan_context = original


@pytest.fixture
def tmp_kb_for_documents(tmp_path, monkeypatch):
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
def client(monkeypatch, tmp_kb_for_documents):
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestDocumentsBrowsePage:
    def test_renders_empty(self, client):
        r = client.get("/documents")
        assert r.status_code == 200
        body = r.text
        assert "Documents" in body
        # Empty count surfaced
        assert "<strong>0</strong>" in body

    def test_renders_with_existing(self, client, tmp_kb_for_documents):
        from app import documents
        documents.create_document(
            tmp_kb_for_documents,
            "Q2 Brief", "A test brief",
        )
        r = client.get("/documents")
        assert r.status_code == 200
        # Count should include the one we just created
        assert "<strong>1</strong>" in r.text

    def test_nav_includes_documents_link(self, client):
        r = client.get("/documents")
        assert r.status_code == 200
        # The shared base.html nav must surface the Documents link in
        # the new "Build" section so it's discoverable from any page.
        assert "Documents" in r.text
        assert 'href="/documents"' in r.text
        assert 'href="/scaffolds"' in r.text
