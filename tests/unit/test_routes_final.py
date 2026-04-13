"""Final coverage tests for app.main route handlers.

Covers remaining uncovered lines:
- /chat page (suggestions sampling, fallback)
- /admin page
- /api/research/stream/{job_id} SSE endpoint
- /api/articles/{kb}/{slug}/suggestions
- /api/chat/sessions/{id}/ask validation
- /api/articles/refine-titles + _refine_titles_sync
- /api/chat/event
- /api/chat/analytics
- /api/chat/events

Uses the noop lifespan + TestClient pattern.
"""

import json
import pytest
from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Noop lifespan
# ---------------------------------------------------------------------------

def _make_redis_mock():
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
    def render(self, **kwargs):
        return "<html><body>test</body></html>"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.main.API_KEY", "")
    monkeypatch.setattr("app.config.API_KEY", "")
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ===========================================================================
# /chat page - suggestions sampling (lines 499-506)
# ===========================================================================


class TestChatPage:
    def test_chat_page_with_good_articles(self, client):
        """When good_articles exist, random suggestions are drawn."""
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            with patch("app.main.search_kb", return_value=[]):
                with patch("app.main.db") as mock_db:
                    mock_db.get_jobs = AsyncMock(return_value=[])
                    mock_db.get_chat_sessions = AsyncMock(return_value=[])
                    with patch("app.main.storage") as mock_storage:
                        mock_storage.list_kbs.return_value = ["personal"]
                        with patch("app.main.get_articles", return_value=[
                            {"title": "A Very Long Title About Kubernetes Networking",
                             "slug": "kubernetes-networking", "word_count": 500},
                            {"title": "Another Good Title About Rust Memory",
                             "slug": "rust-memory", "word_count": 800},
                        ]):
                            resp = client.get("/chat")
        assert resp.status_code == 200

    def test_chat_page_fallback_suggestions(self, client):
        """When no good_articles exist, fallback suggestions are used."""
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            with patch("app.main.search_kb", return_value=[]):
                with patch("app.main.db") as mock_db:
                    mock_db.get_jobs = AsyncMock(return_value=[])
                    mock_db.get_chat_sessions = AsyncMock(return_value=[])
                    with patch("app.main.storage") as mock_storage:
                        mock_storage.list_kbs.return_value = ["personal"]
                        with patch("app.main.get_articles", return_value=[
                            {"title": "short", "slug": "short", "word_count": 50},
                        ]):
                            resp = client.get("/chat")
        assert resp.status_code == 200


# ===========================================================================
# /admin page (lines 535-572)
# ===========================================================================


class TestAdminPage:
    def test_admin_renders(self, client):
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            with patch("app.main.storage") as mock_storage:
                mock_storage.list_kbs.return_value = ["personal"]
                with patch("app.main.get_articles", return_value=[
                    {"slug": "art1", "title": "Art1", "word_count": 500},
                ]):
                    with patch("app.quality.find_shallow_articles", return_value=[]):
                        with patch("app.quality.find_duplicates", return_value=[]):
                            with patch("app.auto_discovery.get_status_for_kb", new_callable=AsyncMock,
                                       return_value={"kb": "personal", "enabled": False}):
                                with patch("app.main.db") as mock_db:
                                    mock_db.get_kb_settings = AsyncMock(return_value={})
                                    resp = client.get("/admin")
        assert resp.status_code == 200

    def test_admin_handles_discovery_error(self, client):
        with patch("app.main.jinja_env") as mock_env:
            mock_env.get_template.return_value = _FakeTemplate()
            with patch("app.main.storage") as mock_storage:
                mock_storage.list_kbs.return_value = ["personal"]
                with patch("app.main.get_articles", return_value=[]):
                    with patch("app.quality.find_shallow_articles", return_value=[]):
                        with patch("app.quality.find_duplicates", return_value=[]):
                            with patch("app.auto_discovery.get_status_for_kb", new_callable=AsyncMock,
                                       side_effect=RuntimeError("boom")):
                                with patch("app.main.db") as mock_db:
                                    mock_db.get_kb_settings = AsyncMock(return_value={})
                                    resp = client.get("/admin")
        assert resp.status_code == 200


# ===========================================================================
# /api/research/stream/{job_id} SSE (lines 1043-1112)
# ===========================================================================


class TestSSEResearchStream:
    def test_stream_job_not_found(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value=None)
            resp = client.get("/api/research/stream/999")
        assert resp.status_code == 404

    def test_stream_complete_job(self, client):
        """A job already in terminal state emits status + complete events."""
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 1, "topic": "test", "status": "complete",
                "word_count": 500, "sources_count": 3,
                "added_to_wiki": True, "error": None,
            })
            resp = client.get("/api/research/stream/1")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "status" in body
        assert "complete" in body

    def test_stream_error_job(self, client):
        """A job in error state emits done event."""
        with patch("app.main.db") as mock_db:
            mock_db.get_job = AsyncMock(return_value={
                "id": 2, "topic": "broken", "status": "error",
                "word_count": 0, "sources_count": 0,
                "added_to_wiki": False, "error": "timeout",
            })
            resp = client.get("/api/research/stream/2")
        assert resp.status_code == 200
        body = resp.text
        assert "done" in body


# ===========================================================================
# /api/articles/{kb}/{slug}/suggestions (lines 1960-1998)
# ===========================================================================


class TestArticleSuggestions:
    def test_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.get("/api/articles/nonexistent/some-slug/suggestions")
        assert resp.status_code == 404

    def test_article_not_found(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            with patch("app.main.get_article", return_value=None):
                resp = client.get("/api/articles/personal/missing-article/suggestions")
        assert resp.status_code == 404

    def test_suggestions_returned(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            with patch("app.main.get_article", return_value={
                "title": "Test Article", "slug": "test-article",
            }):
                with patch("app.hybrid_search.hybrid_search", new_callable=AsyncMock, return_value=[
                    {"slug": "related", "kb": "personal", "title": "Related",
                     "rrf_score": 0.8},
                ]):
                    with patch("app.knowledge_graph.get_related_by_graph", new_callable=AsyncMock, return_value=[]):
                        with patch("app.main.db") as mock_db:
                            mock_db.get_pending_candidates = AsyncMock(return_value=[])
                            resp = client.get("/api/articles/personal/test-article/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) >= 1

    def test_suggestions_handles_graph_error(self, client):
        """Graph neighbour failures don't break the endpoint."""
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            with patch("app.main.get_article", return_value={
                "title": "Test", "slug": "test",
            }):
                with patch("app.main.hybrid_search", new_callable=AsyncMock, return_value=[]):
                    with patch("app.main.get_related_by_graph", new_callable=AsyncMock,
                               side_effect=RuntimeError("graph down")):
                        with patch("app.main.db") as mock_db:
                            mock_db.get_pending_candidates = AsyncMock(return_value=[])
                            resp = client.get("/api/articles/personal/test/suggestions")
        assert resp.status_code == 200

    def test_suggestions_includes_pending_candidates(self, client):
        """Pending topic candidates matching the slug show up."""
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            with patch("app.main.get_article", return_value={
                "title": "Test Article", "slug": "test-article",
            }):
                with patch("app.main.hybrid_search", new_callable=AsyncMock, return_value=[]):
                    with patch("app.main.get_related_by_graph", new_callable=AsyncMock, return_value=[]):
                        with patch("app.main.db") as mock_db:
                            mock_db.get_pending_candidates = AsyncMock(return_value=[
                                {"topic": "Follow-up Topic", "source": "llm_followup",
                                 "source_ref": "test-article", "score": 1.0},
                            ])
                            resp = client.get("/api/articles/personal/test-article/suggestions")
        assert resp.status_code == 200
        data = resp.json()
        pending = [s for s in data["suggestions"] if s.get("kind") == "pending_topic"]
        assert len(pending) >= 1


# ===========================================================================
# /api/chat/sessions/{id}/ask validation (lines 2382-2518)
# ===========================================================================


class TestChatAskValidation:
    def test_invalid_json(self, client):
        resp = client.post(
            "/api/chat/sessions/test-session/ask",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_missing_question(self, client):
        resp = client.post(
            "/api/chat/sessions/test-session/ask",
            json={"kb": "personal"},
        )
        assert resp.status_code == 400

    def test_short_question(self, client):
        resp = client.post(
            "/api/chat/sessions/test-session/ask",
            json={"question": "hi"},
        )
        assert resp.status_code == 400

    def test_long_question(self, client):
        resp = client.post(
            "/api/chat/sessions/test-session/ask",
            json={"question": "x" * 5000},
        )
        assert resp.status_code == 400

    def test_invalid_kb_empty_string(self, client):
        resp = client.post(
            "/api/chat/sessions/test-session/ask",
            json={"question": "What is Kubernetes?", "kb": ""},
        )
        assert resp.status_code == 400

    def test_body_not_object(self, client):
        resp = client.post(
            "/api/chat/sessions/test-session/ask",
            json="just a string",
        )
        assert resp.status_code == 400


# ===========================================================================
# /api/articles/refine-titles (lines 2606-2630)
# ===========================================================================


class TestRefineTitles:
    def test_unknown_kb(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            resp = client.post(
                "/api/articles/refine-titles",
                json={"kb": "nonexistent", "dry_run": True},
            )
        assert resp.status_code == 404

    def test_dry_run(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            mock_storage.iter_articles.return_value = [
                ("art1", '---\ntitle: "Old Title"\n---\n\n# Better Title\n\nBody text.'),
            ]
            with patch("app.main._refine_titles_sync", return_value=([
                {"slug": "art1", "old_title": "Old Title", "new_title": "Better Title"},
            ], 1)):
                resp = client.post(
                    "/api/articles/refine-titles",
                    json={"kb": "personal", "dry_run": True},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["updated"] == 1

    def test_no_body_defaults(self, client):
        with patch("app.main.storage") as mock_storage:
            mock_storage.list_kbs.return_value = ["personal"]
            with patch("app.main._refine_titles_sync", return_value=([], 0)):
                resp = client.post("/api/articles/refine-titles")
        assert resp.status_code == 200


class TestRefineTitlesSync:
    def test_refine_titles_sync_dry_run(self):
        from app.main import _refine_titles_sync
        with patch("app.main.storage") as mock_storage:
            mock_storage.iter_articles.return_value = [
                ("art1", '---\ntitle: "slug-title"\n---\n\n# Much Better Title\n\nContent here.'),
            ]
            with patch("app.wiki.extract_title", return_value="Much Better Title"):
                changes, total = _refine_titles_sync("personal", dry_run=True)
        assert total == 1
        assert len(changes) == 1
        assert changes[0]["new_title"] == "Much Better Title"

    def test_refine_titles_sync_no_change(self):
        from app.main import _refine_titles_sync
        with patch("app.main.storage") as mock_storage:
            mock_storage.iter_articles.return_value = [
                ("art1", '---\ntitle: "Same Title"\n---\n\n# Same Title\n\nContent.'),
            ]
            with patch("app.wiki.extract_title", return_value="Same Title"):
                changes, total = _refine_titles_sync("personal", dry_run=True)
        assert total == 1
        assert len(changes) == 0


# ===========================================================================
# /api/chat/event (lines 2530-2557)
# ===========================================================================


class TestChatEvent:
    def test_log_event_success(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.log_chat_event = AsyncMock()
            with patch("app.main.log_chat_interaction"):
                resp = client.post(
                    "/api/chat/event",
                    json={"event": "message_sent", "session_id": "s1"},
                )
        assert resp.status_code == 200
        assert resp.json()["status"] == "logged"

    def test_log_event_missing_event_field(self, client):
        resp = client.post("/api/chat/event", json={"session_id": "s1"})
        assert resp.status_code == 400

    def test_log_event_invalid_json(self, client):
        resp = client.post(
            "/api/chat/event",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400


# ===========================================================================
# /api/chat/analytics and /api/chat/events
# ===========================================================================


class TestChatAnalytics:
    def test_analytics(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_analytics = AsyncMock(return_value={"total_sessions": 5})
            resp = client.get("/api/chat/analytics")
        assert resp.status_code == 200

    def test_events(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_events = AsyncMock(return_value=[])
            resp = client.get("/api/chat/events")
        assert resp.status_code == 200

    def test_events_with_filter(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.get_chat_events = AsyncMock(return_value=[])
            resp = client.get("/api/chat/events?event=message_sent&limit=10")
        assert resp.status_code == 200


# ===========================================================================
# /api/chat/sessions/{session_id} DELETE
# ===========================================================================


class TestDeleteChatSession:
    def test_delete_session(self, client):
        with patch("app.main.db") as mock_db:
            mock_db.delete_chat_session = AsyncMock()
            resp = client.delete("/api/chat/sessions/test-session")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
