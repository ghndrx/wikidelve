"""Chat session + message parity — create/get/append/delete."""

from __future__ import annotations

import pytest


class TestChatSession:
    @pytest.mark.asyncio
    async def test_create_returns_session_dict(self, parity_db):
        result = await parity_db.create_chat_session("sess-1", title="First")
        assert result is not None
        assert result.get("id") == "sess-1" or result.get("session_id") == "sess-1"

    @pytest.mark.asyncio
    async def test_get_sessions_empty(self, parity_db):
        sessions = await parity_db.get_chat_sessions(limit=10)
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_sessions_returns_created(self, parity_db):
        await parity_db.create_chat_session("sess-a", title="Alpha")
        await parity_db.create_chat_session("sess-b", title="Beta")
        sessions = await parity_db.get_chat_sessions(limit=10)
        ids = {s.get("id") or s.get("session_id") for s in sessions}
        assert {"sess-a", "sess-b"}.issubset(ids)

    @pytest.mark.asyncio
    async def test_get_sessions_limit(self, parity_db):
        for i in range(5):
            await parity_db.create_chat_session(f"sess-{i}", title=f"T{i}")
        sessions = await parity_db.get_chat_sessions(limit=3)
        assert len(sessions) == 3


class TestChatMessages:
    @pytest.mark.asyncio
    async def test_empty_session_has_no_messages(self, parity_db):
        await parity_db.create_chat_session("empty-sess", title="empty")
        msgs = await parity_db.get_chat_messages("empty-sess")
        assert msgs == []

    @pytest.mark.asyncio
    async def test_add_message_persists(self, parity_db):
        await parity_db.create_chat_session("msg-sess", title="msg")
        await parity_db.add_chat_message("msg-sess", "user", "hello")
        msgs = await parity_db.get_chat_messages("msg-sess")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_messages_ordered_by_insertion(self, parity_db):
        await parity_db.create_chat_session("order-sess", title="order")
        await parity_db.add_chat_message("order-sess", "user", "one")
        await parity_db.add_chat_message("order-sess", "assistant", "two")
        await parity_db.add_chat_message("order-sess", "user", "three")
        msgs = await parity_db.get_chat_messages("order-sess")
        assert [m["content"] for m in msgs] == ["one", "two", "three"]


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_removes_session(self, parity_db):
        await parity_db.create_chat_session("del-sess", title="del")
        await parity_db.delete_chat_session("del-sess")
        sessions = await parity_db.get_chat_sessions(limit=10)
        ids = {s.get("id") or s.get("session_id") for s in sessions}
        assert "del-sess" not in ids

    @pytest.mark.asyncio
    async def test_delete_missing_is_graceful(self, parity_db):
        # Neither backend raises for a missing session id.
        try:
            await parity_db.delete_chat_session("never-existed")
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"delete_chat_session on missing id raised: {exc}")
