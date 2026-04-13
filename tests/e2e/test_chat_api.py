"""E2E tests for chat sessions, events, and analytics APIs."""

import pytest
from playwright.sync_api import APIRequestContext

BASE_URL = "http://localhost:8888"


@pytest.fixture(scope="session")
def api(playwright):
    ctx = playwright.request.new_context(base_url=BASE_URL, timeout=60000)
    yield ctx
    ctx.dispose()


class TestChatSessionsAPI:
    """Chat session tests. All use e2e- prefixed IDs and clean up after."""

    def _cleanup_session(self, api, session_id):
        api.delete(f"/api/chat/sessions/{session_id}")

    def test_list_sessions(self, api: APIRequestContext):
        resp = api.get("/api/chat/sessions")
        assert resp.status == 200
        assert isinstance(resp.json(), list)

    def test_create_and_load_session(self, api: APIRequestContext):
        session_id = "e2e-test-session-1"
        # Add a message (auto-creates session)
        resp = api.post(
            f"/api/chat/sessions/{session_id}/messages",
            data={"role": "user", "content": "Hello from E2E test"},
        )
        assert resp.status == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["role"] == "user"

        # Add AI response
        resp = api.post(
            f"/api/chat/sessions/{session_id}/messages",
            data={"role": "ai", "content": "Hi there!"},
        )
        assert resp.status == 200

        # Load messages
        resp = api.get(f"/api/chat/sessions/{session_id}")
        assert resp.status == 200
        msgs = resp.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "ai"
        # Cleanup
        self._cleanup_session(api, session_id)

    def test_session_appears_in_list(self, api: APIRequestContext):
        session_id = "e2e-test-session-2"
        api.post(
            f"/api/chat/sessions/{session_id}/messages",
            data={"role": "user", "content": "Test session listing"},
        )
        sessions = api.get("/api/chat/sessions").json()
        ids = [s["id"] for s in sessions]
        assert session_id in ids
        # Cleanup
        self._cleanup_session(api, session_id)

    def test_delete_session(self, api: APIRequestContext):
        session_id = "e2e-test-delete"
        api.post(
            f"/api/chat/sessions/{session_id}/messages",
            data={"role": "user", "content": "Delete me"},
        )
        resp = api.delete(f"/api/chat/sessions/{session_id}")
        assert resp.status == 200
        msgs = api.get(f"/api/chat/sessions/{session_id}").json()
        assert len(msgs) == 0

    def test_message_requires_content(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/x/messages",
            data={"role": "user"},
        )
        assert resp.status == 400


class TestChatEventsAPI:
    def test_log_event(self, api: APIRequestContext):
        resp = api.post("/api/chat/event", data={
            "event": "e2e_test",
            "session_id": "test",
            "user_input": "hello",
        })
        assert resp.status == 200
        assert resp.json()["status"] == "logged"

    def test_event_requires_event_field(self, api: APIRequestContext):
        resp = api.post("/api/chat/event", data={"user_input": "hello"})
        assert resp.status == 400

    def test_list_events(self, api: APIRequestContext):
        # Log an event first
        api.post("/api/chat/event", data={"event": "e2e_list_test"})
        resp = api.get("/api/chat/events?limit=10")
        assert resp.status == 200
        events = resp.json()
        assert isinstance(events, list)
        assert any(e["event"] == "e2e_list_test" for e in events)

    def test_filter_events_by_type(self, api: APIRequestContext):
        api.post("/api/chat/event", data={"event": "special_type"})
        resp = api.get("/api/chat/events?event=special_type&limit=10")
        assert resp.status == 200
        events = resp.json()
        for e in events:
            assert e["event"] == "special_type"

    def test_analytics(self, api: APIRequestContext):
        resp = api.get("/api/chat/analytics")
        assert resp.status == 200
        data = resp.json()
        assert "total_sessions" in data
        assert "total_messages" in data
        assert "event_counts" in data
        assert "unmatched_commands" in data
        assert "error_patterns" in data


class TestKBManagementAPI:
    def test_list_kbs(self, api: APIRequestContext):
        resp = api.get("/api/kbs")
        assert resp.status == 200
        kbs = resp.json()
        assert isinstance(kbs, list)
        names = [kb["name"] for kb in kbs]
        assert "personal" in names

    def test_create_kb(self, api: APIRequestContext):
        resp = api.post("/api/kbs", data={"name": "e2e-test-kb"})
        assert resp.status == 200
        data = resp.json()
        assert data["name"] == "e2e-test-kb"
        assert data["status"] == "created"

    def test_create_kb_too_short(self, api: APIRequestContext):
        resp = api.post("/api/kbs", data={"name": "x"})
        assert resp.status == 400

    def test_create_kb_idempotent(self, api: APIRequestContext):
        api.post("/api/kbs", data={"name": "e2e-idempotent"})
        resp = api.post("/api/kbs", data={"name": "e2e-idempotent"})
        assert resp.status == 200  # Should not error


class TestTitleRefinement:
    def test_refine_titles_dry_run(self, api: APIRequestContext):
        resp = api.post(
            "/api/articles/refine-titles",
            data={"kb": "personal", "dry_run": True},
        )
        assert resp.status == 200
        data = resp.json()
        assert "total" in data
        assert "updated" in data
        assert "changes" in data
        assert data["dry_run"] is True

    def test_refine_titles_unknown_kb(self, api: APIRequestContext):
        resp = api.post(
            "/api/articles/refine-titles",
            data={"kb": "nonexistent-kb-xyz"},
        )
        assert resp.status == 404


class TestResearchRedirect:
    def test_research_redirects_to_search(self, api: APIRequestContext):
        resp = api.get("/research", max_redirects=0)
        assert resp.status == 302
        assert "/search" in resp.headers.get("location", "")
