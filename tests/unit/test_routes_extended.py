"""Extended tests for API route handlers in app.main.

Covers the uncovered route handlers identified by the coverage report.
Uses the same noop lifespan + TestClient pattern as test_routes.py.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Noop lifespan -- same pattern as test_routes.py
# ---------------------------------------------------------------------------

def _make_redis_mock():
    """Create a MagicMock whose awaitable methods (enqueue_job, etc.) are AsyncMock."""
    m = MagicMock()
    m.enqueue_job = AsyncMock()
    return m


@asynccontextmanager
async def _noop_lifespan(app_instance):
    app_instance.state.redis = _make_redis_mock()
    yield


@pytest.fixture(autouse=True)
def _patch_lifespan():
    from app.main import app
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    yield
    app.router.lifespan_context = original


class _FakeTemplate:
    """A fake Jinja2 template that renders a simple HTML page."""
    def render(self, **kwargs):
        return "<html><body>test</body></html>"


@pytest.fixture
def client(monkeypatch):
    """TestClient with API_KEY unset so auth middleware is disabled."""
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ===========================================================================
# Web UI: GET / (home page)
# ===========================================================================

class TestHomePage:
    def test_home_renders(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles") as mock_get, \
             patch("app.main.jinja_env") as mock_env:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_get.return_value = [
                {"slug": "a1", "title": "Article One", "word_count": 200,
                 "tags": ["python", "web"]},
            ]
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/")
        assert resp.status_code == 200

    def test_home_empty_kbs(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.jinja_env") as mock_env:
            mock_storage.list_kbs.return_value = []
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/")
        assert resp.status_code == 200


# ===========================================================================
# Web UI: GET /wiki/{kb_name}/{slug}
# ===========================================================================

class TestViewArticlePage:
    def test_view_article_found(self, client):
        fake_article = {
            "slug": "test-article", "title": "Test Article",
            "html": "<p>content</p>", "word_count": 100, "tags": [],
            "raw_markdown": "# Test",
        }
        with patch("app.main.get_article", return_value=fake_article), \
             patch("app.main.db") as mock_db, \
             patch("app.main.jinja_env") as mock_env:
            mock_db.get_claims_for_article = AsyncMock(return_value=[])
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/wiki/personal/test-article")
        assert resp.status_code == 200

    def test_view_article_not_found_unknown_kb(self, client):
        with patch("app.main.get_article", return_value=None), \
             patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/wiki/unknown-kb/test")
        assert resp.status_code == 404

    def test_view_article_not_found_shows_suggestions(self, client):
        with patch("app.main.get_article", return_value=None), \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.hybrid_search", new_callable=AsyncMock, return_value=[
                 {"slug": "related-article", "title": "Related Article"},
             ]), \
             patch("app.main.jinja_env") as mock_env:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/wiki/personal/missing-article")
        assert resp.status_code == 200

    def test_view_article_with_confidence_high(self, client):
        fake_article = {
            "slug": "test", "title": "Test", "html": "<p>x</p>",
            "word_count": 100, "tags": [], "raw_markdown": "# Test",
        }
        claims = [
            {"status": "verified", "confidence": 0.9},
            {"status": "verified", "confidence": 0.8},
        ]
        with patch("app.main.get_article", return_value=fake_article), \
             patch("app.main.db") as mock_db, \
             patch("app.main.jinja_env") as mock_env:
            mock_db.get_claims_for_article = AsyncMock(return_value=claims)
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/wiki/personal/test")
        assert resp.status_code == 200

    def test_view_article_with_confidence_medium(self, client):
        fake_article = {
            "slug": "test", "title": "Test", "html": "<p>x</p>",
            "word_count": 100, "tags": [], "raw_markdown": "# Test",
        }
        claims = [
            {"status": "verified", "confidence": 0.6},
            {"status": "unverified", "confidence": 0.5},
        ]
        with patch("app.main.get_article", return_value=fake_article), \
             patch("app.main.db") as mock_db, \
             patch("app.main.jinja_env") as mock_env:
            mock_db.get_claims_for_article = AsyncMock(return_value=claims)
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/wiki/personal/test-med")
        assert resp.status_code == 200

    def test_view_article_with_confidence_low(self, client):
        fake_article = {
            "slug": "test", "title": "Test", "html": "<p>x</p>",
            "word_count": 100, "tags": [], "raw_markdown": "# Test",
        }
        claims = [
            {"status": "outdated", "confidence": 0.3},
            {"status": "partially_correct", "confidence": 0.2},
        ]
        with patch("app.main.get_article", return_value=fake_article), \
             patch("app.main.db") as mock_db, \
             patch("app.main.jinja_env") as mock_env:
            mock_db.get_claims_for_article = AsyncMock(return_value=claims)
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/wiki/personal/test-low")
        assert resp.status_code == 200


# ===========================================================================
# GET /research/{filename} (view research)
# ===========================================================================

class TestViewResearch:
    def test_view_research_found(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.parse_frontmatter", return_value=({}, "# Hello\nworld")), \
             patch("app.main.jinja_env") as mock_env:
            mock_storage.read_text.return_value = "---\ntitle: test\n---\n# Hello\nworld"
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/research/output.md")
        assert resp.status_code == 200

    def test_view_research_invalid_filename(self, client):
        resp = client.get("/research/..%2Fetc%2Fpasswd")
        assert resp.status_code in (400, 404)

    def test_view_research_not_found(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.read_text.return_value = None
            resp = client.get("/research/nonexistent.md")
        assert resp.status_code == 404


# ===========================================================================
# POST /api/embeddings/build
# ===========================================================================

class TestBuildEmbeddings:
    def test_build_embeddings_default_kb(self, client):
        with patch("app.main.app") as _:
            pass
        resp = client.post("/api/embeddings/build", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "build_embeddings"

    def test_build_embeddings_with_kb(self, client):
        resp = client.post("/api/embeddings/build", json={"kb": "work"})
        assert resp.status_code == 200
        assert resp.json()["kb"] == "work"

    def test_build_embeddings_no_body(self, client):
        resp = client.post("/api/embeddings/build")
        assert resp.status_code == 200
        assert resp.json()["kb"] == "personal"


# ===========================================================================
# POST /api/graph/build
# ===========================================================================

class TestBuildGraph:
    def test_build_graph_default_kb(self, client):
        resp = client.post("/api/graph/build", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "build_graph"

    def test_build_graph_with_kb(self, client):
        resp = client.post("/api/graph/build", json={"kb": "work"})
        assert resp.status_code == 200
        assert resp.json()["kb"] == "work"


# ===========================================================================
# POST /api/quality/auto-enrich
# ===========================================================================

class TestAutoEnrich:
    def test_auto_enrich(self, client):
        scores = [
            {"slug": "low-article", "score": 30},
            {"slug": "ok-article", "score": 70},
        ]
        with patch("app.main.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=scores)
            # Need to re-patch since asyncio is used elsewhere
            pass
        with patch("app.quality.score_all_articles", return_value=scores):
            resp = client.post("/api/quality/auto-enrich", json={"kb": "personal", "threshold": 60})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"


# ===========================================================================
# POST /api/research (full flow with cooldown, KB creation, etc.)
# ===========================================================================

class TestResearchFullFlow:
    def test_research_valid_topic(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.check_cooldown = AsyncMock(return_value=None)
            mock_db.create_job = AsyncMock(return_value=42)
            resp = client.post("/api/research", json={
                "topic": "How does Kubernetes networking work in detail",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == 42
        assert data["status"] == "queued"

    def test_research_cooldown_hit(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.check_cooldown = AsyncMock(return_value={"id": 99})
            resp = client.post("/api/research", json={
                "topic": "Detailed explanation of Kubernetes networking",
            })
        assert resp.status_code == 429

    def test_research_auto_create_kb(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db, \
             patch("app.config.register_kb") as mock_register:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.check_cooldown = AsyncMock(return_value=None)
            mock_db.create_job = AsyncMock(return_value=55)
            resp = client.post("/api/research", json={
                "topic": "How does Kubernetes networking work in detail",
                "kb": "new-kb",
            })
        assert resp.status_code == 200

    def test_research_with_review_sources(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.check_cooldown = AsyncMock(return_value=None)
            mock_db.create_job = AsyncMock(return_value=60)
            resp = client.post("/api/research", json={
                "topic": "How does Kubernetes networking work in detail",
                "review_sources": True,
            })
        assert resp.status_code == 200
        assert resp.json()["review_sources"] is True

    def test_research_non_dict_body(self, client):
        resp = client.post("/api/research", json="just a string")
        assert resp.status_code == 400


# ===========================================================================
# POST /api/research/local (full flow)
# ===========================================================================

class TestLocalResearchFullFlow:
    def test_local_research_success(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.create_job = AsyncMock(return_value=10)
            resp = client.post("/api/research/local", json={
                "topic": "auth flow analysis",
                "path": "/code/myproject",
                "pattern": "*.py",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == 10
        assert data["status"] == "queued"

    def test_local_research_non_dict_body(self, client):
        resp = client.post("/api/research/local", json="not an object")
        assert resp.status_code == 400


# ===========================================================================
# GET /api/research/jobs
# ===========================================================================

class TestResearchJobs:
    def test_list_jobs(self, client):
        fake_jobs = [
            {"id": 1, "topic": "Test", "status": "complete", "created_at": "2025-01-01",
             "word_count": 500, "sources_count": 3, "added_to_wiki": True,
             "error": None, "content": "article text"},
        ]
        with patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[]):
            mock_db.get_jobs = AsyncMock(return_value=fake_jobs)
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/research/jobs")
        assert resp.status_code == 200

    def test_list_jobs_compact(self, client):
        fake_jobs = [
            {"id": 1, "topic": "Test", "status": "complete", "created_at": "2025-01-01",
             "word_count": 500, "sources_count": 3, "added_to_wiki": False,
             "error": None, "content": "long content"},
        ]
        with patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[]):
            mock_db.get_jobs = AsyncMock(return_value=fake_jobs)
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/research/jobs?compact=true")
        assert resp.status_code == 200
        data = resp.json()
        for row in data:
            assert "content" not in row

    def test_list_jobs_with_status_filter(self, client):
        fake_jobs = [
            {"id": 1, "topic": "T1", "status": "error", "created_at": "2025-01-01",
             "word_count": 0, "sources_count": 0, "added_to_wiki": False,
             "error": "fail", "content": ""},
            {"id": 2, "topic": "T2", "status": "complete", "created_at": "2025-01-01",
             "word_count": 500, "sources_count": 3, "added_to_wiki": True,
             "error": None, "content": "ok"},
        ]
        with patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[]):
            mock_db.get_jobs = AsyncMock(return_value=fake_jobs)
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/research/jobs?status=error")
        assert resp.status_code == 200
        data = resp.json()
        assert all(r["status"] == "error" for r in data)

    def test_list_jobs_with_before_id(self, client):
        fake_jobs = [
            {"id": 10, "topic": "T10", "status": "complete", "created_at": "2025-01-01",
             "word_count": 100, "sources_count": 1, "added_to_wiki": False,
             "error": None, "content": ""},
            {"id": 5, "topic": "T5", "status": "queued", "created_at": "2025-01-01",
             "word_count": 0, "sources_count": 0, "added_to_wiki": False,
             "error": None, "content": ""},
        ]
        with patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[]):
            mock_db.get_jobs = AsyncMock(return_value=fake_jobs)
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/research/jobs?before_id=8")
        assert resp.status_code == 200
        data = resp.json()
        for r in data:
            assert r["id"] < 8


# ===========================================================================
# GET /api/auto-discovery/status
# ===========================================================================

class TestAutoDiscoveryStatus:
    def test_auto_discovery_status(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.auto_discovery.get_status_for_kb", new_callable=AsyncMock,
                   return_value={"kb": "personal", "enabled": True}), \
             patch("app.main.app") as _:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/auto-discovery/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "global_enabled" in data
        assert "kbs" in data


# ===========================================================================
# PUT /api/auto-discovery/config/{kb}
# ===========================================================================

class TestAutoDiscoveryConfig:
    def test_config_success(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.upsert_auto_discovery_config = AsyncMock(return_value={"enabled": True})
            resp = client.put("/api/auto-discovery/config/personal",
                              json={"enabled": True, "daily_budget": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["kb"] == "personal"

    def test_config_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/unknown", json={"enabled": True})
        assert resp.status_code == 404

    def test_config_unknown_fields(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/personal",
                              json={"bogus_field": True})
        assert resp.status_code == 400

    def test_config_invalid_strategy(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/personal",
                              json={"strategy": "invalid_strategy"})
        assert resp.status_code == 400

    def test_config_negative_budget(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/personal",
                              json={"daily_budget": -5})
        assert resp.status_code == 400

    def test_config_invalid_json(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/personal",
                              content=b"not json",
                              headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_config_non_dict_body(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/personal", json="string")
        assert resp.status_code == 400

    def test_config_seed_topics_list(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.upsert_auto_discovery_config = AsyncMock(return_value={})
            resp = client.put("/api/auto-discovery/config/personal",
                              json={"seed_topics": ["topic1", "topic2"]})
        assert resp.status_code == 200

    def test_config_seed_topics_not_list(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/auto-discovery/config/personal",
                              json={"seed_topics": "not a list"})
        assert resp.status_code == 400


# ===========================================================================
# POST /api/auto-discovery/run/{kb}
# ===========================================================================

class TestAutoDiscoveryRun:
    def test_run_success(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post("/api/auto-discovery/run/personal")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_run_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post("/api/auto-discovery/run/unknown")
        assert resp.status_code == 404


# ===========================================================================
# PUT /api/kb-settings/{kb}
# ===========================================================================

class TestKbSettingsPut:
    def test_put_settings_success(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.upsert_kb_settings = AsyncMock(return_value={"persona": "expert"})
            resp = client.put("/api/kb-settings/personal",
                              json={"persona": "expert"})
        assert resp.status_code == 200
        assert resp.json()["config"]["persona"] == "expert"

    def test_put_settings_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/kb-settings/unknown", json={})
        assert resp.status_code == 404

    def test_put_settings_unknown_fields(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/kb-settings/personal",
                              json={"unknown_field": "val"})
        assert resp.status_code == 400

    def test_put_settings_invalid_provider(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/kb-settings/personal",
                              json={"synthesis_provider": "openai"})
        assert resp.status_code == 400

    def test_put_settings_invalid_json(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/kb-settings/personal",
                              content=b"not json",
                              headers={"content-type": "application/json"})
        assert resp.status_code == 400

    def test_put_settings_non_dict(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.put("/api/kb-settings/personal", json=[1, 2, 3])
        assert resp.status_code == 400


# ===========================================================================
# GET /api/research/history
# ===========================================================================

class TestResearchHistory:
    def test_history(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_jobs = AsyncMock(return_value=[
                {"id": 1, "topic": "T", "status": "complete", "created_at": "2025-01-01",
                 "word_count": 100, "sources_count": 2, "added_to_wiki": False,
                 "error": None, "content": ""},
            ])
            mock_db.get_article_updates = AsyncMock(return_value=[])
            with patch("app.main.storage") as mock_storage, \
                 patch("app.main.get_articles", return_value=[]):
                mock_storage.list_kbs.return_value = ["personal"]
                resp = client.get("/api/research/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "article_updates" in data


# ===========================================================================
# GET /api/research/sources/{job_id}
# ===========================================================================

class TestResearchSources:
    def test_sources_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "complete", "topic": "test",
            })
            mock_db.get_sources = AsyncMock(return_value=[
                {"id": 1, "url": "https://example.com", "title": "Example"},
            ])
            resp = client.get("/api/research/sources/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == 1
        assert len(data["sources"]) == 1

    def test_sources_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.get("/api/research/sources/999")
        assert resp.status_code == 404


# ===========================================================================
# PUT /api/research/sources/{job_id}
# ===========================================================================

class TestUpdateSources:
    def test_update_sources_select_all(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "awaiting_review",
            })
            mock_db.select_all_sources = AsyncMock(return_value=5)
            resp = client.put("/api/research/sources/1",
                              json={"select_all": True})
        assert resp.status_code == 200
        assert resp.json()["updated"] == 5

    def test_update_sources_specific_ids(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "awaiting_review",
            })
            mock_db.update_source_selection = AsyncMock(return_value=2)
            resp = client.put("/api/research/sources/1",
                              json={"source_ids": [1, 2], "selected": True})
        assert resp.status_code == 200
        assert resp.json()["updated"] == 2

    def test_update_sources_not_awaiting_review(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "complete",
            })
            resp = client.put("/api/research/sources/1",
                              json={"source_ids": [1]})
        assert resp.status_code == 409

    def test_update_sources_empty_ids(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "awaiting_review",
            })
            resp = client.put("/api/research/sources/1",
                              json={"source_ids": []})
        assert resp.status_code == 400

    def test_update_sources_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.put("/api/research/sources/999", json={"select_all": True})
        assert resp.status_code == 404


# ===========================================================================
# POST /api/research/synthesize/{job_id}
# ===========================================================================

class TestSynthesize:
    def test_synthesize_success(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "awaiting_review", "topic": "test topic",
            })
            resp = client.post("/api/research/synthesize/1", json={"kb": "personal"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued_for_synthesis"

    def test_synthesize_not_awaiting(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "complete", "topic": "test",
            })
            resp = client.post("/api/research/synthesize/1")
        assert resp.status_code == 409

    def test_synthesize_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.post("/api/research/synthesize/999")
        assert resp.status_code == 404


# ===========================================================================
# POST /api/research/retry-errors
# ===========================================================================

class TestRetryErrors:
    def test_retry_errors(self, client):
        errored = [
            {"id": 1, "topic": "Failed topic"},
            {"id": 2, "topic": "Another fail"},
        ]
        with patch("app.main.db") as mock_db:
            mock_db.get_errored_jobs = AsyncMock(return_value=errored)
            mock_db.reset_job_for_retry = AsyncMock()
            resp = client.post("/api/research/retry-errors", json={"limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert data["retried"] == 2

    def test_retry_errors_no_errored(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_errored_jobs = AsyncMock(return_value=[])
            resp = client.post("/api/research/retry-errors")
        assert resp.status_code == 200
        assert resp.json()["retried"] == 0


# ===========================================================================
# POST /api/research/smart-retry/{job_id}
# ===========================================================================

class TestSmartRetry:
    def test_smart_retry_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.post("/api/research/smart-retry/999")
        assert resp.status_code == 404

    def test_smart_retry_not_retryable(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "complete", "topic": "test",
            })
            resp = client.post("/api/research/smart-retry/1")
        assert resp.status_code == 409

    def test_smart_retry_infra_error(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "error", "topic": "test",
                "error": "MINIMAX_API_KEY not set", "kb": "personal",
            })
            resp = client.post("/api/research/smart-retry/1")
        assert resp.status_code == 400

    def test_smart_retry_too_narrow(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.llm.llm_chat", new_callable=AsyncMock,
                   return_value="Broader research topic about Python"):
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "no_results", "topic": "obscure xyz",
                "error": "no results found", "kb": "personal",
            })
            mock_db.create_job = AsyncMock(return_value=2)
            resp = client.post("/api/research/smart-retry/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_job_id"] == 1
        assert data["new_job_id"] == 2

    def test_smart_retry_too_wide(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.llm.llm_chat", new_callable=AsyncMock,
                   return_value="Focused Python web security"):
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "error",
                "topic": "everything about all programming languages and their entire ecosystems",
                "error": "timeout during synthesis", "kb": "personal",
            })
            mock_db.create_job = AsyncMock(return_value=4)
            resp = client.post("/api/research/smart-retry/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["failure_mode"] == "too_wide"

    def test_smart_retry_unknown_error(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "status": "error", "topic": "a valid topic for research",
                "error": "random error", "kb": "personal",
            })
            mock_db.create_job = AsyncMock(return_value=3)
            resp = client.post("/api/research/smart-retry/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rewritten"] is False


# ===========================================================================
# POST /api/quality/pass
# ===========================================================================

class TestQualityPass:
    def test_quality_pass(self, client):
        resp = client.post("/api/quality/pass", json={"kb": "personal", "max_articles": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["max_articles"] == 5


# ===========================================================================
# POST /api/quality/enrich/{kb_name}/{slug}
# ===========================================================================

class TestEnrichArticle:
    def test_enrich(self, client):
        resp = client.post("/api/quality/enrich/personal/test-article")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "enrich"
        assert data["slug"] == "test-article"


# ===========================================================================
# POST /api/quality/crosslink/{kb_name}/{slug}
# ===========================================================================

class TestCrosslinkArticle:
    def test_crosslink(self, client):
        resp = client.post("/api/quality/crosslink/personal/test-article")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "crosslink"


# ===========================================================================
# GET /api/quality/shallow/{kb_name}
# ===========================================================================

class TestShallowArticles:
    def test_shallow(self, client):
        with patch("app.quality.find_shallow_articles", return_value=[
            {"slug": "s1", "score": 20},
        ]):
            resp = client.get("/api/quality/shallow/personal")
        assert resp.status_code == 200


# ===========================================================================
# GET /api/quality/duplicates/{kb_name}
# ===========================================================================

class TestDuplicates:
    def test_duplicates(self, client):
        with patch("app.quality.find_duplicates", return_value=[]):
            resp = client.get("/api/quality/duplicates/personal")
        assert resp.status_code == 200


# ===========================================================================
# GET /api/quality/broken-wikilinks/{kb_name}
# ===========================================================================

class TestBrokenWikilinks:
    def test_broken_wikilinks(self, client):
        with patch("app.quality.find_broken_wikilinks", return_value={"by_target": []}):
            resp = client.get("/api/quality/broken-wikilinks/personal")
        assert resp.status_code == 200


# ===========================================================================
# POST /api/quality/factcheck/{kb_name}/{slug}
# ===========================================================================

class TestFactCheck:
    def test_fact_check(self, client):
        resp = client.post("/api/quality/factcheck/personal/test-article")
        assert resp.status_code == 200
        assert resp.json()["action"] == "fact_check"


# ===========================================================================
# POST /api/quality/freshness/{kb_name}/{slug}
# ===========================================================================

class TestFreshnessAudit:
    def test_freshness_audit(self, client):
        resp = client.post("/api/quality/freshness/personal/test-article",
                           json={"auto_update": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "freshness_audit"
        assert data["auto_update"] is False


# ===========================================================================
# POST /api/quality/freshness-batch
# ===========================================================================

class TestFreshnessBatch:
    def test_freshness_batch(self, client):
        resp = client.post("/api/quality/freshness-batch",
                           json={"kb": "personal", "max_articles": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["max_articles"] == 5


# ===========================================================================
# POST /api/github/check-releases
# ===========================================================================

class TestGithubReleases:
    def test_check_releases(self, client):
        resp = client.post("/api/github/check-releases")
        assert resp.status_code == 200
        assert resp.json()["action"] == "check_releases"

    def test_index_repos(self, client):
        resp = client.post("/api/github/index-repos")
        assert resp.status_code == 200
        assert resp.json()["action"] == "index_repos"


# ===========================================================================
# GET /api/github/tracked
# ===========================================================================

class TestGithubTracked:
    def test_tracked(self, client):
        with patch("app.github_monitor.TRACKED_REPOS", ["repo1"]), \
             patch("app.github_monitor.OWN_REPOS", ["own1"]):
            resp = client.get("/api/github/tracked")
        assert resp.status_code == 200
        data = resp.json()
        assert "tracked" in data
        assert "own" in data


# ===========================================================================
# GET /api/status
# ===========================================================================

class TestStatusRoute:
    def test_status(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[
                 {"slug": "a1", "word_count": 100},
             ]):
            mock_db.get_job_stats = AsyncMock(return_value={
                "total": 10, "complete": 8, "active": 1, "errors": 1,
                "cancelled": 0, "total_words": 5000, "added_to_wiki": 7,
            })
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "grokmaxxing" in data
        assert "jobs" in data
        assert "wiki" in data


# ===========================================================================
# GET /api/quality/scores/{kb_name}
# ===========================================================================

class TestQualityScores:
    def test_quality_scores(self, client):
        scores = [
            {"slug": "s1", "score": 90},
            {"slug": "s2", "score": 50},
            {"slug": "s3", "score": 20},
        ]
        with patch("app.quality.score_all_articles", return_value=scores):
            resp = client.get("/api/quality/scores/personal")
        assert resp.status_code == 200
        data = resp.json()
        assert data["kb"] == "personal"
        assert data["total_articles"] == 3
        assert "distribution" in data


# ===========================================================================
# GET /api/quality/score/{kb_name}/{slug}
# ===========================================================================

class TestQualityScoreSingle:
    def test_score_single(self, client):
        with patch("app.quality.score_article_quality", return_value={"score": 75}):
            resp = client.get("/api/quality/score/personal/test-article")
        assert resp.status_code == 200


# ===========================================================================
# GET /api/quality/wikilinks/{kb_name} and POST
# ===========================================================================

class TestWikilinksCheck:
    def test_check_wikilinks(self, client):
        with patch("app.quality.check_wikilinks", return_value={"broken": []}):
            resp = client.get("/api/quality/wikilinks/personal")
        assert resp.status_code == 200

    def test_fix_wikilinks(self, client):
        with patch("app.quality.fix_wikilinks", return_value={"fixed": 2}):
            resp = client.post("/api/quality/wikilinks/personal",
                               json={"slug": "test", "auto_remove": True})
        assert resp.status_code == 200


# ===========================================================================
# GET /api/articles/{kb_name}/{slug}/versions
# ===========================================================================

class TestArticleVersions:
    def test_versions_list(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_article_versions = AsyncMock(return_value=[
                {"id": 1, "kb": "personal", "article_slug": "test",
                 "change_type": "update", "job_id": None, "content_hash": "abc",
                 "created_at": "2025-01-01", "note": None, "full_content": "content here"},
            ])
            resp = client.get("/api/articles/personal/test/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) == 1
        assert data["versions"][0]["word_count"] == 2  # "content here"

    def test_versions_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/articles/unknown/test/versions")
        assert resp.status_code == 404


# ===========================================================================
# GET /api/articles/{kb_name}/{slug}/versions/{version_id}
# ===========================================================================

class TestArticleVersion:
    def test_version_found(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_article_version_by_id = AsyncMock(return_value={
                "id": 1, "kb": "personal", "article_slug": "test",
                "full_content": "old content",
            })
            resp = client.get("/api/articles/personal/test/versions/1")
        assert resp.status_code == 200

    def test_version_not_found(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_article_version_by_id = AsyncMock(return_value=None)
            resp = client.get("/api/articles/personal/test/versions/999")
        assert resp.status_code == 404

    def test_version_wrong_kb(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal", "work"]
            mock_db.get_article_version_by_id = AsyncMock(return_value={
                "id": 1, "kb": "work", "article_slug": "test",
                "full_content": "x",
            })
            resp = client.get("/api/articles/personal/test/versions/1")
        assert resp.status_code == 404


# ===========================================================================
# GET /api/articles/{kb_name}/{slug}/versions/{version_id}/diff
# ===========================================================================

class TestArticleVersionDiff:
    def test_diff(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db, \
             patch("app.main.get_article", return_value={
                 "slug": "test", "raw_markdown": "# Test\nnew content",
             }), \
             patch("app.main.read_article_text", return_value="# Test\nnew content"):
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_article_version_by_id = AsyncMock(return_value={
                "id": 1, "kb": "personal", "article_slug": "test",
                "full_content": "# Test\nold content",
                "created_at": "2025-01-01",
            })
            resp = client.get("/api/articles/personal/test/versions/1/diff")
        assert resp.status_code == 200
        data = resp.json()
        assert "diff" in data
        assert "additions" in data
        assert "deletions" in data


# ===========================================================================
# GET /api/articles/{kb_name}/{slug}/suggestions
# ===========================================================================

class TestArticleSuggestions:
    def test_suggestions(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_article", return_value={
                 "slug": "test", "title": "Test Article",
             }), \
             patch("app.hybrid_search.hybrid_search", new_callable=AsyncMock,
                   return_value=[
                       {"slug": "related", "kb": "personal", "title": "Related", "rrf_score": 0.5},
                   ]), \
             patch("app.knowledge_graph.get_related_by_graph", new_callable=AsyncMock,
                   return_value=[]), \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_pending_candidates = AsyncMock(return_value=[])
            resp = client.get("/api/articles/personal/test/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data

    def test_suggestions_article_not_found(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_article", return_value=None):
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/articles/personal/nonexistent/suggestions")
        assert resp.status_code == 404

    def test_suggestions_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/articles/unknown/test/suggestions")
        assert resp.status_code == 404


# ===========================================================================
# GET /api/articles/{kb_name}/{slug}/related
# ===========================================================================

class TestRelatedArticles:
    def test_related(self, client):
        with patch("app.main.get_article", return_value={
            "slug": "test", "title": "Test", "tags": ["python"],
            "raw_markdown": "# Test\n[[Other Article]]",
        }), \
             patch("app.main.get_articles", return_value=[
                 {"slug": "other-article", "title": "Other Article",
                  "tags": ["python"], "word_count": 100},
             ]), \
             patch("app.main.read_article_text", return_value=""):
            resp = client.get("/api/articles/personal/test/related")
        assert resp.status_code == 200
        data = resp.json()
        assert "related" in data

    def test_related_not_found(self, client):
        with patch("app.main.get_article", return_value=None):
            resp = client.get("/api/articles/personal/nonexistent/related")
        assert resp.status_code == 404


# ===========================================================================
# POST /api/media/youtube
# ===========================================================================

class TestYoutubeIngest:
    def test_youtube_ingest(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.create_job = AsyncMock(return_value=100)
            resp = client.post("/api/media/youtube",
                               json={"url": "https://youtube.com/watch?v=abc"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    def test_youtube_ingest_urls(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.create_job = AsyncMock(side_effect=[1, 2])
            resp = client.post("/api/media/youtube",
                               json={"urls": ["https://youtube.com/watch?v=a",
                                               "https://youtube.com/watch?v=b"]})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_youtube_no_url(self, client):
        resp = client.post("/api/media/youtube", json={})
        assert resp.status_code == 400

    def test_youtube_invalid_json(self, client):
        resp = client.post("/api/media/youtube",
                           content=b"not json",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400


# ===========================================================================
# POST /api/ingest/document
# ===========================================================================

class TestIngestDocument:
    def test_document_ingest(self, client):
        resp = client.post("/api/ingest/document",
                           json={"url": "https://example.com/book.pdf"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_document_no_url(self, client):
        resp = client.post("/api/ingest/document", json={})
        assert resp.status_code == 400


# ===========================================================================
# POST /api/ingest/directory
# ===========================================================================

class TestIngestDirectory:
    def test_directory_ingest(self, client):
        resp = client.post("/api/ingest/directory",
                           json={"url": "https://example.com/books/",
                                 "max_files": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"

    def test_directory_no_url(self, client):
        resp = client.post("/api/ingest/directory", json={})
        assert resp.status_code == 400

    def test_directory_invalid_json(self, client):
        resp = client.post("/api/ingest/directory",
                           content=b"not json",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400


# ===========================================================================
# Palace routes
# ===========================================================================

class TestPalaceRoutes:
    def test_palace_page(self, client):
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/palace")
        assert resp.status_code == 200

    def test_palace_map(self, client):
        with patch("app.palace.generate_palace_map", new_callable=AsyncMock,
                   return_value={"wings": []}):
            resp = client.get("/api/palace/map")
        assert resp.status_code == 200

    def test_palace_hall(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.main.get_articles", return_value=[
                 {"slug": "a1", "title": "A1", "summary": "s", "word_count": 100, "tags": []},
             ]):
            mock_db.get_classifications_by_hall = AsyncMock(return_value=[
                {"slug": "a1", "kb": "personal", "hall": "tech", "confidence": 0.9},
            ])
            resp = client.get("/api/palace/hall/personal/tech")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hall"] == "tech"
        assert data["count"] == 1

    def test_palace_room(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.main.get_articles", return_value=[]):
            mock_db.get_rooms = AsyncMock(return_value=[
                {"id": 1, "name": "my-room"},
            ])
            mock_db.get_room_members = AsyncMock(return_value=[])
            mock_db.get_classifications = AsyncMock(return_value=[])
            resp = client.get("/api/palace/room/personal/my-room")
        assert resp.status_code == 200

    def test_palace_room_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_rooms = AsyncMock(return_value=[])
            resp = client.get("/api/palace/room/personal/missing-room")
        assert resp.status_code == 404

    def test_palace_article(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_article_classification = AsyncMock(return_value={
                "hall": "tech", "confidence": 0.95,
            })
            mock_db.get_article_rooms = AsyncMock(return_value=[
                {"name": "room1", "relevance": 0.8},
            ])
            resp = client.get("/api/palace/article/personal/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["hall"] == "tech"
        assert len(data["rooms"]) == 1

    def test_palace_classify(self, client):
        resp = client.post("/api/palace/classify", json={"kb": "personal"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_palace_cluster(self, client):
        resp = client.post("/api/palace/cluster", json={"kb": "personal"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"


# ===========================================================================
# GET /gm
# ===========================================================================

class TestGmRoute:
    def test_gm(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job_stats = AsyncMock(return_value={"total_words": 12345})
            resp = client.get("/gm")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "we're so back"


# ===========================================================================
# GET /api/stats
# ===========================================================================

class TestStatsRoute:
    def test_stats(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[
                 {"slug": "a1", "word_count": 500},
             ]), \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_jobs = AsyncMock(return_value=[{"id": 1}])
            resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_articles"] == 1
        assert data["total_words"] == 500


# ===========================================================================
# POST /api/chat/event
# ===========================================================================

class TestChatEvent:
    def test_log_event(self, client):
        with patch("app.main.log_chat_interaction"), \
             patch("app.main.db") as mock_db:
            mock_db.log_chat_event = AsyncMock()
            resp = client.post("/api/chat/event",
                               json={"event": "copy", "session_id": "abc"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged"

    def test_log_event_missing_event(self, client):
        resp = client.post("/api/chat/event", json={"session_id": "abc"})
        assert resp.status_code == 400

    def test_log_event_invalid_json(self, client):
        resp = client.post("/api/chat/event",
                           content=b"not json",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400


# ===========================================================================
# POST /api/articles/refine-titles
# ===========================================================================

class TestRefineTitles:
    def test_refine_titles_dry_run(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main._refine_titles_sync", return_value=(
                 [{"slug": "s1", "old_title": "Old", "new_title": "New"}], 5,
             )):
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post("/api/articles/refine-titles",
                               json={"kb": "personal", "dry_run": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["updated"] == 1

    def test_refine_titles_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post("/api/articles/refine-titles",
                               json={"kb": "unknown"})
        assert resp.status_code == 404


# ===========================================================================
# GET /api/kbs and POST /api/kbs
# ===========================================================================

class TestKbRoutes:
    def test_list_kbs(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[
                 {"slug": "a", "word_count": 100},
             ]):
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/kbs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "personal"

    def test_create_kb(self, client):
        mock_path = MagicMock()
        mock_path.name = "my-new-kb"
        with patch("app.config.register_kb", return_value=mock_path), \
             patch("app.main.build_search_index", new_callable=AsyncMock):
            resp = client.post("/api/kbs", json={"name": "my-new-kb"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

    def test_create_kb_short_name(self, client):
        resp = client.post("/api/kbs", json={"name": "x"})
        assert resp.status_code == 400

    def test_create_kb_invalid_json(self, client):
        resp = client.post("/api/kbs",
                           content=b"not json",
                           headers={"content-type": "application/json"})
        assert resp.status_code == 400


# ===========================================================================
# GET /api/storage/status
# ===========================================================================

class TestStorageStatus:
    def test_storage_status(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.backend_name.return_value = "s3"
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/storage/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["storage_backend"] == "s3"


# ===========================================================================
# GET /metrics
# ===========================================================================

class TestMetricsEndpoint:
    def test_metrics(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.metrics.prometheus_text", return_value="# HELP\n"):
            mock_db.get_job_stats = AsyncMock(return_value={"queued": 0, "running": 0, "error": 0})
            resp = client.get("/metrics")
        assert resp.status_code == 200


# ===========================================================================
# GET /api/usage/all-time
# ===========================================================================

class TestUsageAllTime:
    def test_usage_all_time(self, client):
        with patch("app.metrics.get_all_time_totals", new_callable=AsyncMock,
                   return_value={"total_tokens": 999}):
            resp = client.get("/api/usage/all-time")
        assert resp.status_code == 200


# ===========================================================================
# GET /api/github/releases
# ===========================================================================

class TestGithubLatestReleases:
    def test_latest_releases(self, client):
        with patch("app.github_monitor.check_all_releases", new_callable=AsyncMock,
                   return_value=[{"repo": "test", "tag": "v1.0"}]):
            resp = client.get("/api/github/releases")
        assert resp.status_code == 200


# ===========================================================================
# POST /api/quality/broken-wikilinks/{kb_name}/research
# ===========================================================================

class TestBrokenWikilinksResearch:
    def test_research_broken_wikilinks(self, client):
        report = {
            "by_target": [
                {"slug": "missing-article", "target": "Missing Article Reference"},
            ],
        }
        with patch("app.quality.find_broken_wikilinks", return_value=report), \
             patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage:
            mock_db.check_cooldown = AsyncMock(return_value=None)
            mock_db.create_job = AsyncMock(return_value=42)
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post("/api/quality/broken-wikilinks/personal/research")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["queued"]) == 1

    def test_research_broken_wikilinks_with_filter(self, client):
        report = {
            "by_target": [
                {"slug": "s1", "target": "Target One Reference Topic"},
                {"slug": "s2", "target": "Target Two Reference Topic"},
            ],
        }
        with patch("app.quality.find_broken_wikilinks", return_value=report), \
             patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage:
            mock_db.check_cooldown = AsyncMock(return_value=None)
            mock_db.create_job = AsyncMock(return_value=43)
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post("/api/quality/broken-wikilinks/personal/research",
                               json={"targets": ["s1"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["queued"]) == 1

    def test_research_broken_wikilinks_cooldown(self, client):
        report = {
            "by_target": [
                {"slug": "s1", "target": "Already Researched Topic"},
            ],
        }
        with patch("app.quality.find_broken_wikilinks", return_value=report), \
             patch("app.main.db") as mock_db:
            mock_db.check_cooldown = AsyncMock(return_value={"id": 99})
            resp = client.post("/api/quality/broken-wikilinks/personal/research")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["skipped"]) == 1
        assert len(data["queued"]) == 0


# ===========================================================================
# Web UI pages: /chat, /graph, /api-docs
# ===========================================================================

class TestWebUIPages:
    def test_graph_page(self, client):
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/graph")
        assert resp.status_code == 200

    def test_api_docs_page(self, client):
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/api-docs")
        assert resp.status_code == 200


# ===========================================================================
# GET /research/review/{job_id}
# ===========================================================================

class TestSourceReviewPage:
    def test_review_page(self, client):
        with patch("app.main.db") as mock_db, \
             patch("app.main.storage") as mock_storage, \
             patch("app.main.jinja_env") as mock_env:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "topic": "test", "status": "awaiting_review",
            })
            mock_db.get_sources = AsyncMock(return_value=[])
            mock_storage.list_kbs.return_value = ["personal"]
            mock_env.get_template.return_value = _FakeTemplate()
            resp = client.get("/research/review/1")
        assert resp.status_code == 200

    def test_review_page_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.get("/research/review/999")
        assert resp.status_code == 404


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelperFunctions:
    def test_job_to_api_format(self):
        from app.main import _job_to_api_format
        job = {
            "id": 1, "topic": "Test topic", "status": "complete",
            "created_at": "2025-01-01", "completed_at": "2025-01-02",
            "sources_count": 3, "word_count": 500, "error": None,
            "added_to_wiki": True, "content": "article text",
        }
        with patch("app.main._find_article_slug", return_value=("test-topic", "personal")):
            result = _job_to_api_format(job)
        assert result["id"] == 1
        assert result["status"] == "complete"
        assert result["wiki_slug"] == "test-topic"

    def test_job_to_api_format_no_resolve(self):
        from app.main import _job_to_api_format
        job = {
            "id": 2, "topic": "Another", "status": "queued",
            "created_at": "2025-01-01", "sources_count": 0,
            "word_count": 0, "error": None, "added_to_wiki": False,
            "content": "",
        }
        result = _job_to_api_format(job)
        assert result["wiki_slug"] is None

    def test_build_slug_index(self):
        from app.main import _build_slug_index
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[
                 {"slug": "article-one"},
                 {"slug": "article-two"},
             ]):
            mock_storage.list_kbs.return_value = ["personal"]
            index = _build_slug_index()
        assert "article-one" in index
        assert index["article-one"] == "personal"

    def test_find_article_slug_exact(self):
        from app.main import _find_article_slug
        index = {"test-topic": "personal"}
        slug, kb = _find_article_slug("Test topic", slug_index=index)
        assert slug == "test-topic"
        assert kb == "personal"

    def test_find_article_slug_no_match(self):
        from app.main import _find_article_slug
        index = {"other-slug": "personal"}
        slug, kb = _find_article_slug("Very unique topic name", slug_index=index)
        assert kb is None
