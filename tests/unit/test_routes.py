"""Tests for API route handlers in app.main.

Uses the noop lifespan pattern from test_auth.py so TestClient doesn't
hit Redis, SQLite, or S3 during startup/shutdown.  All database calls
and external services are mocked.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Noop lifespan -- same pattern as test_auth.py
# ---------------------------------------------------------------------------

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
    """TestClient with API_KEY unset so auth middleware is disabled."""
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ===========================================================================
# /health
# ===========================================================================

class TestHealthRoute:
    def test_health_returns_ok(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.backend_name.return_value = "local"
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["storage"] == "local"
        assert "kbs" in data

    def test_health_head_method(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.backend_name.return_value = "local"
            mock_storage.list_kbs.return_value = []
            resp = client.head("/health")
        assert resp.status_code == 200


# ===========================================================================
# /api/articles
# ===========================================================================

class TestArticlesRoute:
    def test_list_all_articles(self, client):
        fake_articles = [
            {"slug": "python-basics", "title": "Python Basics", "word_count": 100,
             "tags": ["python"], "kb": "test"},
        ]
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=fake_articles):
            mock_storage.list_kbs.return_value = ["test"]
            resp = client.get("/api/articles")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["slug"] == "python-basics"

    def test_list_articles_with_kb_filter(self, client):
        fake_articles = [
            {"slug": "a1", "title": "A1", "word_count": 50, "tags": [], "kb": "work"},
        ]
        with patch("app.main.get_articles", return_value=fake_articles):
            resp = client.get("/api/articles?kb=work")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_list_articles_with_source_filter(self, client):
        fake_articles = [
            {"slug": "a1", "title": "A1", "word_count": 50, "tags": [],
             "source_type": "local"},
            {"slug": "a2", "title": "A2", "word_count": 50, "tags": [],
             "source_type": "research"},
        ]
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=fake_articles):
            mock_storage.list_kbs.return_value = ["test"]
            resp = client.get("/api/articles?source=local")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["source_type"] == "local"

    def test_list_articles_empty(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.get_articles", return_value=[]):
            mock_storage.list_kbs.return_value = ["test"]
            resp = client.get("/api/articles")
        assert resp.status_code == 200
        assert resp.json() == []


# ===========================================================================
# /api/articles/{kb_name}/{slug}
# ===========================================================================

class TestArticleDetailRoute:
    def test_get_article_found(self, client):
        fake_article = {
            "slug": "python-basics", "title": "Python Basics",
            "html": "<p>Hello</p>", "word_count": 100, "tags": [],
        }
        with patch("app.main.get_article", return_value=fake_article):
            resp = client.get("/api/articles/test/python-basics")
        assert resp.status_code == 200
        assert resp.json()["slug"] == "python-basics"

    def test_get_article_not_found(self, client):
        with patch("app.main.get_article", return_value=None):
            resp = client.get("/api/articles/test/nonexistent")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Article not found"


# ===========================================================================
# DELETE /api/articles/{kb_name}/{slug}
# ===========================================================================

class TestDeleteArticleRoute:
    def test_delete_article_success(self, client):
        result = {"status": "deleted", "slug": "old-article"}
        with patch("app.main.delete_article", return_value=result), \
             patch("app.main.get_article", return_value=None):
            # Mock out the vector store and KG cleanup
            with patch("app.vector_store.get_vector_store") as mock_vs, \
                 patch("aiosqlite.connect", new_callable=MagicMock):
                mock_vs.return_value.delete = AsyncMock()
                resp = client.delete("/api/articles/test/old-article")
        assert resp.status_code == 200

    def test_delete_article_not_found(self, client):
        with patch("app.main.delete_article", side_effect=FileNotFoundError("not found")):
            resp = client.delete("/api/articles/test/nonexistent")
        assert resp.status_code == 404

    def test_delete_article_bad_request(self, client):
        with patch("app.main.delete_article", side_effect=ValueError("Invalid slug")):
            resp = client.delete("/api/articles/test/bad-slug")
        assert resp.status_code == 400


# ===========================================================================
# DELETE /api/research/{filename}
# ===========================================================================

class TestDeleteResearchRoute:
    def test_delete_research_file_success(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.exists.return_value = True
            mock_storage.delete.return_value = None
            resp = client.delete("/api/research/test-output.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["filename"] == "test-output.md"

    def test_delete_research_file_not_found(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.exists.return_value = False
            resp = client.delete("/api/research/nope.md")
        assert resp.status_code == 404

    def test_delete_research_dotdot_rejected(self, client):
        # Path with ".." in the filename component triggers 400
        resp = client.delete("/api/research/..passwd")
        assert resp.status_code == 400

    def test_delete_research_backslash_rejected(self, client):
        resp = client.delete("/api/research/foo%5Cbar.md")
        assert resp.status_code == 400


# ===========================================================================
# DELETE /api/research/job/{job_id}
# ===========================================================================

class TestDeleteJobRoute:
    def test_delete_job_success(self, client):
        fake_job = {"id": 1, "topic": "Test topic", "status": "complete"}
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=fake_job)
            mock_db.delete_job = AsyncMock(return_value=None)
            resp = client.delete("/api/research/job/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["job_id"] == 1

    def test_delete_job_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.delete("/api/research/job/999")
        assert resp.status_code == 404


# ===========================================================================
# /api/search
# ===========================================================================

class TestSearchRoute:
    def test_search_returns_results(self, client):
        fake_results = [
            {"slug": "python-basics", "title": "Python Basics", "snippet": "..."},
        ]
        with patch("app.main.search_fts", new_callable=AsyncMock, return_value=fake_results):
            resp = client.get("/api/search?q=python")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["slug"] == "python-basics"

    def test_search_empty_query(self, client):
        with patch("app.main.search_fts", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/search?q=")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_no_query_param(self, client):
        with patch("app.main.search_fts", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/search")
        assert resp.status_code == 200


# ===========================================================================
# /api/search/hybrid
# ===========================================================================

class TestHybridSearchRoute:
    def test_hybrid_search_returns_results(self, client):
        fake_results = [{"slug": "rust", "title": "Rust", "rrf_score": 0.5}]
        with patch("app.main.hybrid_search", new_callable=AsyncMock, return_value=fake_results):
            resp = client.get("/api/search/hybrid?q=rust")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_hybrid_search_empty_query_returns_empty(self, client):
        resp = client.get("/api/search/hybrid?q=")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_hybrid_search_with_kb_filter(self, client):
        with patch("app.main.hybrid_search", new_callable=AsyncMock, return_value=[]) as mock_hs:
            resp = client.get("/api/search/hybrid?q=docker&kb=work")
        assert resp.status_code == 200
        mock_hs.assert_called_once_with("docker", kb_name="work", limit=15)


# ===========================================================================
# POST /api/search/reindex
# ===========================================================================

class TestReindexRoute:
    def test_reindex_success(self, client):
        with patch("app.main.build_search_index", new_callable=AsyncMock, return_value=42):
            resp = client.post("/api/search/reindex")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"
        assert data["articles"] == 42


# ===========================================================================
# /api/graph
# ===========================================================================

class TestGraphRoutes:
    def test_graph_data(self, client):
        fake_graph = {"nodes": [{"id": "Python"}], "edges": []}
        with patch("app.main.get_graph_data", new_callable=AsyncMock, return_value=fake_graph):
            resp = client.get("/api/graph/data")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data

    def test_entity_articles(self, client):
        fake_articles = [{"slug": "python-basics", "kb": "test"}]
        with patch("app.main.get_entity_articles", new_callable=AsyncMock, return_value=fake_articles):
            resp = client.get("/api/graph/entity/Python")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "Python"
        assert len(data["articles"]) == 1

    def test_graph_related(self, client):
        fake_related = [{"slug": "rust", "kb": "test", "score": 0.5}]
        with patch("app.main.get_related_by_graph", new_callable=AsyncMock, return_value=fake_related):
            resp = client.get("/api/graph/related/test/python-basics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "python-basics"
        assert data["kb"] == "test"

    def test_graph_related_depth_capped(self, client):
        with patch("app.main.get_related_by_graph", new_callable=AsyncMock, return_value=[]) as mock_gr:
            resp = client.get("/api/graph/related/test/slug?depth=10")
        assert resp.status_code == 200
        # depth should be capped at 3
        mock_gr.assert_called_once_with("slug", "test", depth=3)


# ===========================================================================
# /api/chat
# ===========================================================================

class TestChatRoutes:
    def test_chat_sessions_list(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_sessions = AsyncMock(return_value=[
                {"session_id": "abc", "created_at": "2025-01-01"},
            ])
            resp = client.get("/api/chat/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "abc"

    def test_chat_messages_for_session(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_messages = AsyncMock(return_value=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ])
            resp = client.get("/api/chat/sessions/abc123")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_add_chat_message_success(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.add_chat_message = AsyncMock(return_value={"id": 1, "role": "user", "content": "hi"})
            resp = client.post(
                "/api/chat/sessions/abc123/messages",
                json={"role": "user", "content": "hi"},
            )
        assert resp.status_code == 200

    def test_add_chat_message_empty_content(self, client):
        resp = client.post(
            "/api/chat/sessions/abc123/messages",
            json={"role": "user", "content": ""},
        )
        assert resp.status_code == 400

    def test_add_chat_message_invalid_json(self, client):
        resp = client.post(
            "/api/chat/sessions/abc123/messages",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_delete_chat_session(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.delete_chat_session = AsyncMock(return_value=None)
            resp = client.delete("/api/chat/sessions/abc123")
        assert resp.status_code == 200

    def test_chat_ask_missing_question(self, client):
        resp = client.post(
            "/api/chat/sessions/abc123/ask",
            json={"question": ""},
        )
        assert resp.status_code == 400
        assert "required" in resp.json()["detail"]

    def test_chat_ask_too_short(self, client):
        resp = client.post(
            "/api/chat/sessions/abc123/ask",
            json={"question": "ab"},
        )
        assert resp.status_code == 400
        assert "too short" in resp.json()["detail"]

    def test_chat_ask_too_long(self, client):
        resp = client.post(
            "/api/chat/sessions/abc123/ask",
            json={"question": "x" * 5000},
        )
        assert resp.status_code == 400
        assert "limit" in resp.json()["detail"]

    def test_chat_ask_invalid_kb(self, client):
        resp = client.post(
            "/api/chat/sessions/abc123/ask",
            json={"question": "What is Python?", "kb": ""},
        )
        assert resp.status_code == 400
        assert "kb" in resp.json()["detail"]


# ===========================================================================
# /api/research/status/{job_id}
# ===========================================================================

class TestResearchStatusRoute:
    def test_research_status_found(self, client):
        fake_job = {
            "id": 1, "topic": "Python basics", "status": "complete",
            "word_count": 500, "error": None, "created_at": "2025-01-01",
            "content": "", "sources_count": 3, "added_to_wiki": True,
            "kb": "personal",
        }
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=fake_job)
            resp = client.get("/api/research/status/1")
        assert resp.status_code == 200

    def test_research_status_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.get("/api/research/status/999")
        assert resp.status_code == 404


# ===========================================================================
# POST /api/research
# ===========================================================================

class TestResearchRoute:
    def test_research_invalid_json(self, client):
        resp = client.post(
            "/api/research",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_research_missing_topic(self, client):
        resp = client.post("/api/research", json={"topic": ""})
        assert resp.status_code == 400

    def test_research_topic_too_short(self, client):
        resp = client.post("/api/research", json={"topic": "short"})
        assert resp.status_code == 400
        assert "10 characters" in resp.json()["detail"]


# ===========================================================================
# POST /api/research/cancel/{job_id}
# ===========================================================================

class TestCancelResearchRoute:
    def test_cancel_queued_job(self, client):
        fake_job = {"id": 1, "topic": "Test", "status": "queued"}
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=fake_job)
            mock_db.update_job = AsyncMock(return_value=None)
            resp = client.post("/api/research/cancel/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    def test_cancel_completed_job(self, client):
        fake_job = {"id": 1, "topic": "Test", "status": "complete"}
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=fake_job)
            resp = client.post("/api/research/cancel/1")
        assert resp.status_code == 200
        assert "already completed" in resp.json()["detail"]

    def test_cancel_missing_job(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.post("/api/research/cancel/999")
        assert resp.status_code == 404


# ===========================================================================
# /api/chat/analytics and /api/chat/events
# ===========================================================================

class TestChatAnalyticsRoutes:
    def test_chat_analytics(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_analytics = AsyncMock(return_value={"total_sessions": 5})
            resp = client.get("/api/chat/analytics")
        assert resp.status_code == 200

    def test_chat_events(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_events = AsyncMock(return_value=[])
            resp = client.get("/api/chat/events")
        assert resp.status_code == 200


# ===========================================================================
# POST /api/research/local
# ===========================================================================

class TestLocalResearchRoute:
    def test_local_research_missing_topic(self, client):
        resp = client.post(
            "/api/research/local",
            json={"topic": "ab", "path": "/tmp/code"},
        )
        assert resp.status_code == 400
        assert "min 3 chars" in resp.json()["detail"]

    def test_local_research_missing_path(self, client):
        resp = client.post(
            "/api/research/local",
            json={"topic": "auth flow"},
        )
        assert resp.status_code == 400
        assert "path" in resp.json()["detail"]

    def test_local_research_invalid_json(self, client):
        resp = client.post(
            "/api/research/local",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


# ===========================================================================
# /api/kb-settings/{kb}
# ===========================================================================

class TestKbSettingsRoutes:
    def test_get_kb_settings_found(self, client):
        with patch("app.main.storage") as mock_storage, \
             patch("app.main.db") as mock_db:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_db.get_kb_settings = AsyncMock(return_value={"persona": "expert"})
            resp = client.get("/api/kb-settings/personal")
        assert resp.status_code == 200
        data = resp.json()
        assert data["kb"] == "personal"

    def test_get_kb_settings_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/kb-settings/unknown")
        assert resp.status_code == 404


# ===========================================================================
# /api/usage
# ===========================================================================

class TestUsageRoutes:
    def test_usage_summary(self, client):
        with patch("app.metrics.get_usage_summary", return_value={"total_tokens": 100}):
            resp = client.get("/api/usage/summary")
        assert resp.status_code == 200


# ===========================================================================
# /api/traces
# ===========================================================================

class TestTracesRoutes:
    def test_traces_list(self, client):
        with patch("app.tracing.get_recent_traces", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/traces")
        assert resp.status_code == 200

    def test_trace_detail_found(self, client):
        with patch("app.tracing.get_trace", new_callable=AsyncMock, return_value=[{"span": "root"}]):
            resp = client.get("/api/traces/abc-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == "abc-123"

    def test_trace_detail_not_found(self, client):
        with patch("app.tracing.get_trace", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/traces/missing")
        assert resp.status_code == 404
