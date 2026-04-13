"""Unit tests for app.db_dynamo – DynamoDB backend using moto mock."""

import os
import asyncio

import boto3
import pytest
from moto import mock_aws
from unittest.mock import patch

# Force config values before importing the module under test
os.environ.setdefault("DB_DYNAMO_TABLE", "wikidelve-test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TABLE_NAME = "wikidelve-test"


def _create_table(dynamodb):
    """Create the DynamoDB table with the schema expected by db_dynamo."""
    dynamodb.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
            {"AttributeName": "GSI2PK", "AttributeType": "S"},
            {"AttributeName": "GSI2SK", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "GSI2",
                "KeySchema": [
                    {"AttributeName": "GSI2PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI2SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    """Set dummy AWS credentials for moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("DB_DYNAMO_TABLE", TABLE_NAME)


@pytest.fixture()
def dynamo_backend(monkeypatch):
    """Provide a moto-mocked DynamoDB with the table created, and reset
    the module-level _table singleton between tests."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        _create_table(dynamodb)

        # Patch the config values that _get_table reads
        import app.db_dynamo as ddb
        ddb._table = None  # reset singleton
        monkeypatch.setattr(ddb, "TABLE_NAME", TABLE_NAME)
        monkeypatch.setattr(ddb, "AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setattr(ddb, "AWS_SECRET_ACCESS_KEY", "testing")
        monkeypatch.setattr(ddb, "AWS_SESSION_TOKEN", "testing")
        monkeypatch.setattr(ddb, "BEDROCK_REGION", "us-east-1")

        yield ddb

        ddb._table = None  # clean up singleton


def _run_sync(coro):
    """Helper to run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    async def test_clean_item_converts_decimals(self, dynamo_backend):
        from decimal import Decimal
        ddb = dynamo_backend
        item = {"count": Decimal("42"), "score": Decimal("3.14"), "name": "test"}
        cleaned = ddb._clean_item(item)
        assert cleaned["count"] == 42
        assert isinstance(cleaned["count"], int)
        assert abs(cleaned["score"] - 3.14) < 0.001
        assert isinstance(cleaned["score"], float)
        assert cleaned["name"] == "test"

    async def test_to_decimal_converts_float(self, dynamo_backend):
        from decimal import Decimal
        ddb = dynamo_backend
        assert isinstance(ddb._to_decimal(3.14), Decimal)
        assert ddb._to_decimal(42) == 42
        assert ddb._to_decimal("hello") == "hello"

    async def test_strip_dynamo_keys(self, dynamo_backend):
        ddb = dynamo_backend
        item = {"PK": "x", "SK": "y", "GSI1PK": "a", "GSI1SK": "b",
                "GSI2PK": "c", "GSI2SK": "d", "name": "keep"}
        result = ddb._strip_dynamo_keys(item)
        assert "PK" not in result
        assert "SK" not in result
        assert "GSI1PK" not in result
        assert result["name"] == "keep"


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    async def test_init_db_success(self, dynamo_backend):
        ddb = dynamo_backend
        # Should not raise when table exists
        await ddb.init_db()

    async def test_init_db_missing_table(self, dynamo_backend, monkeypatch):
        ddb = dynamo_backend
        monkeypatch.setattr(ddb, "TABLE_NAME", "nonexistent-table")
        ddb._table = None
        with pytest.raises(RuntimeError, match="does not exist"):
            await ddb.init_db()


# ---------------------------------------------------------------------------
# Research jobs CRUD
# ---------------------------------------------------------------------------

class TestJobs:
    async def test_create_and_get_job(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Test Topic")
        assert isinstance(job_id, int)
        assert job_id >= 1

        job = await ddb.get_job(job_id)
        assert job is not None
        assert job["topic"] == "Test Topic"
        assert job["status"] == "queued"
        assert job["id"] == job_id

    async def test_get_nonexistent_job(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_job(99999) is None

    async def test_update_job(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Update Me")
        await ddb.update_job(job_id, status="complete", word_count=500)

        job = await ddb.get_job(job_id)
        assert job["status"] == "complete"
        assert job["word_count"] == 500

    async def test_update_job_no_fields(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("No update")
        # Should not raise
        await ddb.update_job(job_id)

    async def test_delete_job(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Delete Me")
        assert await ddb.get_job(job_id) is not None
        await ddb.delete_job(job_id)
        assert await ddb.get_job(job_id) is None

    async def test_get_jobs_returns_list(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.create_job("Job A")
        await ddb.create_job("Job B")
        jobs = await ddb.get_jobs(limit=10)
        assert len(jobs) >= 2
        topics = [j["topic"] for j in jobs]
        assert "Job A" in topics
        assert "Job B" in topics

    async def test_get_jobs_compact(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Compact")
        await ddb.update_job(job_id, content="Very long article content...")
        jobs = await ddb.get_jobs(limit=10, compact=True)
        assert len(jobs) >= 1
        # In compact mode, content should not be present (projection excludes it)
        found = [j for j in jobs if j["topic"] == "Compact"]
        assert len(found) == 1
        assert "content" not in found[0]

    async def test_get_jobs_limit(self, dynamo_backend):
        ddb = dynamo_backend
        for i in range(5):
            await ddb.create_job(f"Limited {i}")
        jobs = await ddb.get_jobs(limit=3)
        assert len(jobs) == 3

    async def test_get_stuck_jobs(self, dynamo_backend):
        ddb = dynamo_backend
        j1 = await ddb.create_job("Stuck")
        await ddb.update_job(j1, status="searching")
        j2 = await ddb.create_job("Done")
        await ddb.update_job(j2, status="complete")

        stuck = await ddb.get_stuck_jobs()
        stuck_ids = [j["id"] for j in stuck]
        assert j1 in stuck_ids
        assert j2 not in stuck_ids

    async def test_get_errored_jobs(self, dynamo_backend):
        ddb = dynamo_backend
        j1 = await ddb.create_job("Error")
        await ddb.update_job(j1, status="error", error="boom")
        j2 = await ddb.create_job("OK")
        await ddb.update_job(j2, status="complete")

        errored = await ddb.get_errored_jobs()
        errored_ids = [j["id"] for j in errored]
        assert j1 in errored_ids
        assert j2 not in errored_ids

    async def test_reset_job_for_retry(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Retry me")
        await ddb.update_job(job_id, status="error", error="fail")
        await ddb.reset_job_for_retry(job_id)
        job = await ddb.get_job(job_id)
        assert job["status"] == "queued"
        assert job["error"] is None

    async def test_get_job_stats(self, dynamo_backend):
        ddb = dynamo_backend
        j1 = await ddb.create_job("Stats 1")
        await ddb.update_job(j1, status="complete", word_count=100)
        j2 = await ddb.create_job("Stats 2")
        await ddb.update_job(j2, status="error")
        j3 = await ddb.create_job("Stats 3")
        # j3 stays queued

        stats = await ddb.get_job_stats()
        assert stats["total"] >= 3
        assert stats["complete"] >= 1
        assert stats["errors"] >= 1
        assert stats["active"] >= 1

    async def test_check_cooldown_returns_none_when_no_match(self, dynamo_backend):
        ddb = dynamo_backend
        result = await ddb.check_cooldown("Never researched")
        assert result is None

    async def test_check_cooldown_returns_recent_job(self, dynamo_backend):
        ddb = dynamo_backend
        j = await ddb.create_job("Cooldown Topic")
        await ddb.update_job(j, status="complete")
        result = await ddb.check_cooldown("Cooldown Topic")
        assert result is not None
        assert result["topic"] == "Cooldown Topic"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class TestSources:
    async def test_save_and_get_sources(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Sources")
        sources = [
            {"url": "https://a.com", "title": "A", "content": "aaa", "tier": 1},
            {"url": "https://b.com", "title": "B", "content": "bbb", "tier": 2},
        ]
        await ddb.save_sources(job_id, sources, round_num=1)
        result = await ddb.get_sources(job_id)
        assert len(result) == 2
        urls = {r["url"] for r in result}
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    async def test_save_empty_sources(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Empty")
        await ddb.save_sources(job_id, [], round_num=1)
        assert await ddb.get_sources(job_id) == []

    async def test_update_source_selection(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Selection")
        sources = [{"url": "https://x.com", "title": "X", "content": "x"}]
        await ddb.save_sources(job_id, sources, round_num=1)

        all_src = await ddb.get_sources(job_id)
        src_id = all_src[0]["id"]

        # Deselect
        count = await ddb.update_source_selection(job_id, [src_id], selected=False)
        assert count == 1

        selected = await ddb.get_selected_sources(job_id)
        assert len(selected) == 0

        # Re-select
        await ddb.update_source_selection(job_id, [src_id], selected=True)
        selected = await ddb.get_selected_sources(job_id)
        assert len(selected) == 1

    async def test_update_source_selection_empty_list(self, dynamo_backend):
        ddb = dynamo_backend
        count = await ddb.update_source_selection(999, [], selected=True)
        assert count == 0

    async def test_select_all_sources(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Select all")
        sources = [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
        ]
        await ddb.save_sources(job_id, sources, round_num=1)

        # Deselect all
        count = await ddb.select_all_sources(job_id, selected=False)
        assert count == 2
        selected = await ddb.get_selected_sources(job_id)
        assert len(selected) == 0

        # Select all back
        await ddb.select_all_sources(job_id, selected=True)
        selected = await ddb.get_selected_sources(job_id)
        assert len(selected) == 2

    async def test_get_selected_sources_sorted(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Sort")
        sources = [
            {"url": "https://a.com", "title": "A", "tier": 3},
            {"url": "https://b.com", "title": "B", "tier": 1},
        ]
        await ddb.save_sources(job_id, sources, round_num=1)
        selected = await ddb.get_selected_sources(job_id)
        # Tier 1 should come before tier 3
        assert int(selected[0]["tier"]) <= int(selected[1]["tier"])

    async def test_delete_job_also_deletes_sources(self, dynamo_backend):
        ddb = dynamo_backend
        job_id = await ddb.create_job("Delete sources")
        await ddb.save_sources(job_id, [{"url": "u", "title": "t"}], round_num=1)
        assert len(await ddb.get_sources(job_id)) == 1
        await ddb.delete_job(job_id)
        assert len(await ddb.get_sources(job_id)) == 0


# ---------------------------------------------------------------------------
# Article updates
# ---------------------------------------------------------------------------

class TestArticleUpdates:
    async def test_log_and_get_updates(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.log_article_update("my-article", "testkb", job_id=1, change_type="created")
        await ddb.log_article_update("other-article", "testkb", job_id=None, change_type="updated")
        updates = await ddb.get_article_updates(limit=10)
        assert len(updates) >= 2
        slugs = {u["article_slug"] for u in updates}
        assert "my-article" in slugs


# ---------------------------------------------------------------------------
# Classifications
# ---------------------------------------------------------------------------

class TestClassifications:
    async def test_upsert_and_get_classification(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_classification("my-slug", "kb1", "infrastructure", 0.95)
        result = await ddb.get_article_classification("my-slug", "kb1")
        assert result is not None
        assert result["hall"] == "infrastructure"
        assert float(result["confidence"]) == pytest.approx(0.95, abs=0.01)

    async def test_get_classifications_by_kb(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_classification("a", "kb1", "hall-a", 0.9)
        await ddb.upsert_classification("b", "kb1", "hall-b", 0.8)
        await ddb.upsert_classification("c", "kb2", "hall-c", 0.7)

        kb1_classes = await ddb.get_classifications(kb="kb1")
        assert len(kb1_classes) == 2

        all_classes = await ddb.get_classifications(kb=None)
        assert len(all_classes) >= 3

    async def test_get_classifications_by_hall(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_classification("x", "kb1", "infra", 0.9)
        await ddb.upsert_classification("y", "kb1", "infra", 0.7)
        await ddb.upsert_classification("z", "kb1", "devtools", 0.8)

        infra = await ddb.get_classifications_by_hall("kb1", "infra")
        assert len(infra) == 2
        # Should be sorted by confidence descending
        assert float(infra[0]["confidence"]) >= float(infra[1]["confidence"])

    async def test_get_article_classification_missing(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_article_classification("nope", "kb1") is None

    async def test_upsert_overwrites(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_classification("s", "kb", "old-hall", 0.5)
        await ddb.upsert_classification("s", "kb", "new-hall", 0.9)
        result = await ddb.get_article_classification("s", "kb")
        assert result["hall"] == "new-hall"


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------

class TestChatSessions:
    async def test_create_and_list_sessions(self, dynamo_backend):
        ddb = dynamo_backend
        s = await ddb.create_chat_session("sess-1", "My Chat")
        assert s["id"] == "sess-1"
        assert s["title"] == "My Chat"

        sessions = await ddb.get_chat_sessions()
        assert len(sessions) >= 1
        assert any(s["id"] == "sess-1" for s in sessions)

    async def test_add_and_get_messages(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.create_chat_session("sess-2")
        await ddb.add_chat_message("sess-2", "user", "Hello")
        await ddb.add_chat_message("sess-2", "assistant", "Hi there")

        msgs = await ddb.get_chat_messages("sess-2")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    async def test_add_message_creates_session_if_missing(self, dynamo_backend):
        ddb = dynamo_backend
        msg = await ddb.add_chat_message("auto-sess", "user", "Auto-created session")
        assert msg["session_id"] == "auto-sess"

        sessions = await ddb.get_chat_sessions()
        assert any(s["id"] == "auto-sess" for s in sessions)

    async def test_add_message_updates_title_from_new_chat(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.create_chat_session("title-sess", "New Chat")
        await ddb.add_chat_message("title-sess", "user", "What is Docker?")

        sessions = await ddb.get_chat_sessions()
        sess = [s for s in sessions if s["id"] == "title-sess"][0]
        assert sess["title"] == "What is Docker?"

    async def test_delete_chat_session(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.create_chat_session("del-sess")
        await ddb.add_chat_message("del-sess", "user", "bye")
        await ddb.delete_chat_session("del-sess")

        msgs = await ddb.get_chat_messages("del-sess")
        assert len(msgs) == 0


# ---------------------------------------------------------------------------
# Chat events
# ---------------------------------------------------------------------------

class TestChatEvents:
    async def test_log_and_get_events(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.log_chat_event("command_matched", session_id="s1",
                               user_input="research Docker", command="research")
        await ddb.log_chat_event("command_unmatched", session_id="s1",
                               user_input="foobar")

        events = await ddb.get_chat_events(limit=10)
        assert len(events) >= 2

    async def test_get_chat_events_filter_by_type(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.log_chat_event("command_matched")
        await ddb.log_chat_event("command_unmatched")

        matched = await ddb.get_chat_events(limit=10, event_type="command_matched")
        for e in matched:
            assert e["event"] == "command_matched"

    async def test_get_chat_analytics(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.create_chat_session("analytics-1")
        await ddb.log_chat_event("command_matched")
        await ddb.log_chat_event("command_unmatched", user_input="unknown cmd")

        analytics = await ddb.get_chat_analytics()
        assert "total_sessions" in analytics
        assert "event_counts" in analytics
        assert analytics["total_sessions"] >= 1


# ---------------------------------------------------------------------------
# Palace rooms
# ---------------------------------------------------------------------------

class TestRooms:
    async def test_upsert_and_get_rooms(self, dynamo_backend):
        ddb = dynamo_backend
        room_id = await ddb.upsert_room("kb1", "Infrastructure", anchor_entity_id=None, article_count=5)
        assert isinstance(room_id, int)

        rooms = await ddb.get_rooms("kb1")
        assert len(rooms) >= 1
        assert any(r["name"] == "Infrastructure" for r in rooms)

    async def test_upsert_room_updates_existing(self, dynamo_backend):
        ddb = dynamo_backend
        r1 = await ddb.upsert_room("kb1", "DevTools", anchor_entity_id=None, article_count=3)
        r2 = await ddb.upsert_room("kb1", "DevTools", anchor_entity_id=1, article_count=7)
        assert r1 == r2  # Same room ID

    async def test_add_and_get_room_members(self, dynamo_backend):
        ddb = dynamo_backend
        room_id = await ddb.upsert_room("kb1", "Test Room", None, 1)
        await ddb.add_room_member(room_id, "article-a", "kb1", 0.9)
        await ddb.add_room_member(room_id, "article-b", "kb1", 0.7)

        members = await ddb.get_room_members(room_id)
        assert len(members) == 2
        # Sorted by relevance descending
        assert float(members[0]["relevance"]) >= float(members[1]["relevance"])

    async def test_get_article_rooms(self, dynamo_backend):
        ddb = dynamo_backend
        r1 = await ddb.upsert_room("kb1", "Room A", None, 1)
        await ddb.add_room_member(r1, "my-article", "kb1", 0.8)
        r2 = await ddb.upsert_room("kb1", "Room B", None, 2)
        await ddb.add_room_member(r2, "my-article", "kb1", 0.5)

        rooms = await ddb.get_article_rooms("my-article", "kb1")
        assert len(rooms) == 2
        names = {r["name"] for r in rooms}
        assert "Room A" in names
        assert "Room B" in names

    async def test_clear_rooms(self, dynamo_backend):
        ddb = dynamo_backend
        r = await ddb.upsert_room("kb1", "Clearable", None, 2)
        await ddb.add_room_member(r, "x", "kb1")

        await ddb.clear_rooms("kb1")
        rooms = await ddb.get_rooms("kb1")
        assert len(rooms) == 0


# ---------------------------------------------------------------------------
# Serper usage
# ---------------------------------------------------------------------------

class TestSerperUsage:
    async def test_log_and_count_serper_calls(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.log_serper_call("query1", 10, kb="kb1", job_id=1)
        await ddb.log_serper_call("query2", 5, kb="kb1", job_id=2)

        count = await ddb.serper_calls_today("kb1")
        assert count == 2

    async def test_serper_calls_different_kb(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.log_serper_call("q", 10, kb="kb1", job_id=1)
        await ddb.log_serper_call("q", 10, kb="kb2", job_id=2)

        assert await ddb.serper_calls_today("kb1") == 1
        assert await ddb.serper_calls_today("kb2") == 1


# ---------------------------------------------------------------------------
# Topic candidates
# ---------------------------------------------------------------------------

class TestTopicCandidates:
    async def test_insert_and_get_pending(self, dynamo_backend):
        ddb = dynamo_backend
        ok = await ddb.insert_topic_candidate("kb1", "Docker 101", "llm", None, score=0.8)
        assert ok is True

        pending = await ddb.get_pending_candidates("kb1", limit=10)
        assert len(pending) >= 1
        assert any(c["topic"] == "Docker 101" for c in pending)

    async def test_insert_duplicate_topic_returns_false(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.insert_topic_candidate("kb1", "Unique Topic", "llm", None)
        ok = await ddb.insert_topic_candidate("kb1", "Unique Topic", "llm", None)
        assert ok is False

    async def test_mark_candidate_enqueued(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.insert_topic_candidate("kb1", "Enqueue Me", "llm", None)
        pending = await ddb.get_pending_candidates("kb1", limit=10)
        cand = [c for c in pending if c["topic"] == "Enqueue Me"][0]

        await ddb.mark_candidate_enqueued(cand["id"], job_id=42)
        # Should no longer appear as pending
        remaining = await ddb.get_pending_candidates("kb1", limit=10)
        assert not any(c["topic"] == "Enqueue Me" for c in remaining)

    async def test_mark_candidate_skipped(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.insert_topic_candidate("kb1", "Skip Me", "llm", None)
        pending = await ddb.get_pending_candidates("kb1", limit=10)
        cand = [c for c in pending if c["topic"] == "Skip Me"][0]

        await ddb.mark_candidate_skipped(cand["id"], "duplicate")
        remaining = await ddb.get_pending_candidates("kb1", limit=10)
        assert not any(c["topic"] == "Skip Me" for c in remaining)

    async def test_count_pending_candidates(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.insert_topic_candidate("kb1", "Count A", "llm", None)
        await ddb.insert_topic_candidate("kb1", "Count B", "llm", None)
        count = await ddb.count_pending_candidates("kb1")
        assert count == 2

    async def test_get_pending_candidates_zero_limit(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.insert_topic_candidate("kb1", "Zero limit", "llm", None)
        assert await ddb.get_pending_candidates("kb1", limit=0) == []


# ---------------------------------------------------------------------------
# Auto-discovery config
# ---------------------------------------------------------------------------

class TestAutoDiscoveryConfig:
    async def test_upsert_and_get_config(self, dynamo_backend):
        ddb = dynamo_backend
        cfg = await ddb.upsert_auto_discovery_config("kb1", enabled=True, daily_budget=100)
        assert cfg["enabled"] == 1
        assert cfg["daily_budget"] == 100

        fetched = await ddb.get_auto_discovery_config("kb1")
        assert fetched is not None
        assert fetched["enabled"] == 1

    async def test_get_nonexistent_config(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_auto_discovery_config("nope") is None

    async def test_list_enabled_configs(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_auto_discovery_config("enabled-kb", enabled=True)
        await ddb.upsert_auto_discovery_config("disabled-kb", enabled=False)

        enabled = await ddb.list_enabled_auto_discovery_configs()
        kbs = [c["kb"] for c in enabled]
        assert "enabled-kb" in kbs
        assert "disabled-kb" not in kbs

    async def test_upsert_merges_with_defaults(self, dynamo_backend):
        ddb = dynamo_backend
        cfg = await ddb.upsert_auto_discovery_config("kb1", strategy="llm_only")
        # Should have default values for fields not specified
        assert cfg["daily_budget"] == 500  # default
        assert cfg["strategy"] == "llm_only"


# ---------------------------------------------------------------------------
# Article versions
# ---------------------------------------------------------------------------

class TestArticleVersions:
    async def test_save_and_get_versions(self, dynamo_backend):
        ddb = dynamo_backend
        v1 = await ddb.save_article_version("kb1", "my-art", "# Hello\nWorld")
        v2 = await ddb.save_article_version("kb1", "my-art", "# Hello\nWorld v2")

        versions = await ddb.get_article_versions("kb1", "my-art")
        assert len(versions) == 2
        # Newest first
        assert versions[0]["id"] == v2
        assert versions[1]["id"] == v1

    async def test_get_version_by_id(self, dynamo_backend):
        ddb = dynamo_backend
        vid = await ddb.save_article_version("kb1", "slug", "content")
        result = await ddb.get_article_version_by_id(vid)
        assert result is not None
        assert result["full_content"] == "content"

    async def test_get_version_by_id_missing(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_article_version_by_id(999999) is None

    async def test_save_version_with_metadata(self, dynamo_backend):
        ddb = dynamo_backend
        vid = await ddb.save_article_version(
            "kb1", "slug", "body", job_id=42, change_type="created", note="initial"
        )
        result = await ddb.get_article_version_by_id(vid)
        assert result["job_id"] == 42
        assert result["change_type"] == "created"
        assert result["note"] == "initial"

    async def test_content_hash_computed(self, dynamo_backend):
        ddb = dynamo_backend
        vid = await ddb.save_article_version("kb1", "slug", "content")
        result = await ddb.get_article_version_by_id(vid)
        assert "content_hash" in result
        assert len(result["content_hash"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Article claims
# ---------------------------------------------------------------------------

class TestClaims:
    async def test_save_and_get_claims(self, dynamo_backend):
        ddb = dynamo_backend
        cid = await ddb.save_claim("kb1", "my-art", "Python is fast", claim_type="performance")
        assert isinstance(cid, int)

        claims = await ddb.get_claims_for_article("kb1", "my-art")
        assert len(claims) >= 1
        assert any(c["claim_text"] == "Python is fast" for c in claims)

    async def test_save_claim_upserts_on_duplicate(self, dynamo_backend):
        ddb = dynamo_backend
        c1 = await ddb.save_claim("kb1", "art", "Same claim", confidence=0.5)
        c2 = await ddb.save_claim("kb1", "art", "Same claim", confidence=0.9)
        # Same claim text means same hash, should upsert
        assert c1 == c2

        claims = await ddb.get_claims_for_article("kb1", "art")
        matching = [c for c in claims if c["claim_text"] == "Same claim"]
        assert len(matching) == 1

    async def test_update_claim_status(self, dynamo_backend):
        ddb = dynamo_backend
        cid = await ddb.save_claim("kb1", "art", "Updateable claim")
        await ddb.update_claim_status(cid, "verified", 0.95)
        # Verify via get_claims_for_article
        claims = await ddb.get_claims_for_article("kb1", "art")
        claim = [c for c in claims if c["id"] == cid][0]
        assert claim["status"] == "verified"

    async def test_get_stale_claims(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.save_claim("kb1", "art", "Stale claim")
        stale = await ddb.get_stale_claims(days=0)
        # All claims with last_checked_at=None should be stale
        assert len(stale) >= 1


# ---------------------------------------------------------------------------
# KB settings
# ---------------------------------------------------------------------------

class TestKBSettings:
    async def test_upsert_and_get_settings(self, dynamo_backend):
        ddb = dynamo_backend
        result = await ddb.upsert_kb_settings("kb1", synthesis_provider="openai", synthesis_model="gpt-4")
        assert result["synthesis_provider"] == "openai"
        assert result["synthesis_model"] == "gpt-4"

        fetched = await ddb.get_kb_settings("kb1")
        assert fetched["synthesis_provider"] == "openai"

    async def test_get_nonexistent_settings(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_kb_settings("nope") is None

    async def test_upsert_strips_empty_strings(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_kb_settings("kb1", synthesis_provider="openai")
        await ddb.upsert_kb_settings("kb1", synthesis_provider="  ")
        fetched = await ddb.get_kb_settings("kb1")
        assert fetched["synthesis_provider"] is None

    async def test_upsert_ignores_unknown_fields(self, dynamo_backend):
        ddb = dynamo_backend
        result = await ddb.upsert_kb_settings("kb1", unknown_field="ignored", synthesis_provider="test")
        assert "unknown_field" not in result
        assert result["synthesis_provider"] == "test"


# ---------------------------------------------------------------------------
# Article metadata index
# ---------------------------------------------------------------------------

class TestArticleMeta:
    async def test_upsert_and_get_meta(self, dynamo_backend):
        ddb = dynamo_backend
        meta = {"title": "Docker Guide", "summary": "Containers", "word_count": 500}
        await ddb.upsert_article_meta("kb1", "docker-guide", meta)

        result = await ddb.get_article_meta("kb1", "docker-guide")
        assert result is not None
        assert result["title"] == "Docker Guide"
        assert result["word_count"] == 500

    async def test_get_missing_meta(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_article_meta("kb1", "nope") is None

    async def test_list_article_metas(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_article_meta("kb1", "a", {"title": "A", "updated": "2024-01-02"})
        await ddb.upsert_article_meta("kb1", "b", {"title": "B", "updated": "2024-01-01"})

        metas = await ddb.list_article_metas("kb1")
        assert len(metas) == 2
        # Newest first
        assert metas[0]["slug"] == "a"

    async def test_delete_article_meta(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_article_meta("kb1", "del-me", {"title": "Delete"})
        assert await ddb.get_article_meta("kb1", "del-me") is not None
        await ddb.delete_article_meta("kb1", "del-me")
        assert await ddb.get_article_meta("kb1", "del-me") is None

    async def test_article_meta_count(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_article_meta("kb1", "x", {"title": "X"})
        await ddb.upsert_article_meta("kb1", "y", {"title": "Y"})
        count = await ddb.article_meta_count("kb1")
        assert count == 2

    async def test_upsert_meta_skips_empty_kb_or_slug(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.upsert_article_meta("", "slug", {"title": "X"})
        await ddb.upsert_article_meta("kb1", "", {"title": "X"})
        # Should not have inserted anything
        assert await ddb.article_meta_count("") == 0

    async def test_delete_meta_skips_empty_kb_or_slug(self, dynamo_backend):
        ddb = dynamo_backend
        # Should not raise
        await ddb.delete_article_meta("", "slug")
        await ddb.delete_article_meta("kb1", "")


# ---------------------------------------------------------------------------
# LLM usage totals
# ---------------------------------------------------------------------------

class TestLLMUsage:
    async def test_record_and_get_usage(self, dynamo_backend):
        ddb = dynamo_backend
        await ddb.record_llm_usage_total("openai", "gpt-4", "synthesis",
                                       input_tokens=1000, output_tokens=500)
        await ddb.record_llm_usage_total("openai", "gpt-4", "synthesis",
                                       input_tokens=2000, output_tokens=300)

        totals = await ddb.get_llm_usage_totals()
        assert len(totals) >= 1
        row = [t for t in totals if t["model"] == "gpt-4"][0]
        assert row["calls"] == 2
        assert row["input_tokens"] == 3000
        assert row["output_tokens"] == 800

    async def test_get_empty_usage(self, dynamo_backend):
        ddb = dynamo_backend
        assert await ddb.get_llm_usage_totals() == []


# ---------------------------------------------------------------------------
# Auto-increment counter
# ---------------------------------------------------------------------------

class TestCounter:
    async def test_next_id_increments(self, dynamo_backend):
        ddb = dynamo_backend
        id1 = await ddb._next_id("test_entity")
        id2 = await ddb._next_id("test_entity")
        assert id2 == id1 + 1

    async def test_next_id_separate_entities(self, dynamo_backend):
        ddb = dynamo_backend
        a1 = await ddb._next_id("entity_a")
        b1 = await ddb._next_id("entity_b")
        a2 = await ddb._next_id("entity_a")
        assert a2 == a1 + 1
        # entity_b counter is independent
        assert b1 == 1
