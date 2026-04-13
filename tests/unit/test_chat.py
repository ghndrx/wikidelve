"""Unit tests for chat sessions and events."""

import pytest
from unittest.mock import patch

_test_db_path = None


@pytest.fixture(autouse=True)
def _set_db_path(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            yield


from app import db


@pytest.fixture
async def init_db():
    await db.init_db()
    yield


class TestChatSessions:
    @pytest.mark.asyncio
    async def test_create_session(self, init_db):
        session = await db.create_chat_session("test-123", "My Chat")
        assert session["id"] == "test-123"
        assert session["title"] == "My Chat"

    @pytest.mark.asyncio
    async def test_list_sessions(self, init_db):
        await db.create_chat_session("s1", "First")
        await db.create_chat_session("s2", "Second")
        sessions = await db.get_chat_sessions(limit=10)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_add_message(self, init_db):
        await db.create_chat_session("s1", "Chat")
        msg = await db.add_chat_message("s1", "user", "Hello world")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello world"
        assert msg["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_get_messages(self, init_db):
        await db.create_chat_session("s1", "Chat")
        await db.add_chat_message("s1", "user", "Hello")
        await db.add_chat_message("s1", "ai", "Hi there!")
        msgs = await db.get_chat_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "ai"

    @pytest.mark.asyncio
    async def test_auto_create_session(self, init_db):
        # Adding a message to non-existent session auto-creates it
        msg = await db.add_chat_message("auto-1", "user", "First message here")
        assert msg["session_id"] == "auto-1"
        sessions = await db.get_chat_sessions()
        assert any(s["id"] == "auto-1" for s in sessions)
        # Title should be set from first message
        session = next(s for s in sessions if s["id"] == "auto-1")
        assert session["title"] == "First message here"

    @pytest.mark.asyncio
    async def test_delete_session(self, init_db):
        await db.create_chat_session("del-1", "Delete me")
        await db.add_chat_message("del-1", "user", "Hello")
        await db.delete_chat_session("del-1")
        sessions = await db.get_chat_sessions()
        assert not any(s["id"] == "del-1" for s in sessions)
        msgs = await db.get_chat_messages("del-1")
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_session_ordering(self, init_db):
        await db.create_chat_session("old", "Old Chat")
        await db.create_chat_session("new", "New Chat")
        # Add message to old session to bump updated_at
        import asyncio
        await asyncio.sleep(0.01)
        await db.add_chat_message("old", "user", "Updated")
        sessions = await db.get_chat_sessions()
        assert sessions[0]["id"] == "old"  # Most recently updated


class TestChatEvents:
    @pytest.mark.asyncio
    async def test_log_event(self, init_db):
        await db.log_chat_event(
            event="user_message",
            session_id="s1",
            user_input="test query",
        )
        events = await db.get_chat_events(limit=10)
        assert len(events) >= 1
        assert events[0]["event"] == "user_message"
        assert events[0]["user_input"] == "test query"

    @pytest.mark.asyncio
    async def test_event_filtering(self, init_db):
        await db.log_chat_event(event="user_message", user_input="hello")
        await db.log_chat_event(event="command_parsed", command="stats")
        await db.log_chat_event(event="user_message", user_input="world")

        all_events = await db.get_chat_events(limit=10)
        assert len(all_events) == 3

        msg_events = await db.get_chat_events(limit=10, event_type="user_message")
        assert len(msg_events) == 2

    @pytest.mark.asyncio
    async def test_analytics(self, init_db):
        await db.log_chat_event(event="user_message", user_input="test")
        await db.log_chat_event(event="command_parsed", command="stats")
        await db.log_chat_event(event="command_unmatched", user_input="foo bar")
        await db.log_chat_event(event="command_unmatched", user_input="foo bar")

        analytics = await db.get_chat_analytics()
        assert analytics["event_counts"]["user_message"] == 1
        assert analytics["event_counts"]["command_parsed"] == 1
        assert analytics["event_counts"]["command_unmatched"] == 2
        assert len(analytics["unmatched_commands"]) >= 1
        assert analytics["unmatched_commands"][0]["input"] == "foo bar"
        assert analytics["unmatched_commands"][0]["count"] == 2

    @pytest.mark.asyncio
    async def test_log_event_with_error(self, init_db):
        await db.log_chat_event(
            event="research_error",
            user_input="bad query",
            error="API timeout",
        )
        events = await db.get_chat_events(limit=10)
        assert events[0]["error"] == "API timeout"
