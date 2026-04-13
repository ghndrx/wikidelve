"""Extended unit tests for app/db.py — targeting uncovered lines.

Covers:
  - init_db migrations: wiki_slug, wiki_kb, claims_json, selected (307-323)
  - update_job: disallowed fields (361-364)
  - get_stuck_jobs, get_errored_jobs, reset_job_for_retry (429-464)
  - save_sources (513)
  - palace classifications, rooms, room members, clear_rooms (633-782)
  - get_pending_candidates with limit <= 0 (1026)
  - save_article_version, get_article_versions, get_article_at_timestamp (1199-1210)
  - get_stale_claims (1289-1301)
  - get_article_meta, list_article_metas, delete_article_meta, article_meta_count stubs (1395-1407)
  - save_trace_span, get_trace_spans, get_trace_spans_by_trace_id (1465-1547)
  - record_llm_usage_total, get_llm_usage_totals (1504-1547)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _set_db_path(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "test_db_ext.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            yield


from app import db


@pytest.fixture
async def init_db():
    await db.init_db()
    yield


# ===========================================================================
# init_db migrations (lines 307-323)
# ===========================================================================


class TestInitDBMigrations:
    @pytest.mark.asyncio
    async def test_idempotent_init(self, init_db):
        """Calling init_db twice should not fail (migrations are idempotent)."""
        await db.init_db()
        conn = await db._get_db()
        try:
            cursor = await conn.execute("PRAGMA table_info(research_jobs)")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "wiki_slug" in cols
            assert "wiki_kb" in cols
            assert "claims_json" in cols
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_selected_column_exists_on_sources(self, init_db):
        conn = await db._get_db()
        try:
            cursor = await conn.execute("PRAGMA table_info(research_sources)")
            cols = {row["name"] for row in await cursor.fetchall()}
            assert "selected" in cols
        finally:
            await conn.close()


# ===========================================================================
# update_job — disallowed fields (lines 361-364)
# ===========================================================================


class TestUpdateJobValidation:
    @pytest.mark.asyncio
    async def test_disallowed_field_raises(self, init_db):
        job_id = await db.create_job("validation test")
        with pytest.raises(ValueError, match="disallowed"):
            await db.update_job(job_id, status="complete", bogus_field="evil")

    @pytest.mark.asyncio
    async def test_empty_fields_noop(self, init_db):
        job_id = await db.create_job("noop test")
        await db.update_job(job_id)  # no fields → should return without error
        job = await db.get_job(job_id)
        assert job["status"] == "queued"


# ===========================================================================
# get_stuck_jobs (lines 429-437)
# ===========================================================================


class TestGetStuckJobs:
    @pytest.mark.asyncio
    async def test_finds_stuck_jobs(self, init_db):
        job1 = await db.create_job("Stuck in searching")
        await db.update_job(job1, status="searching")
        job2 = await db.create_job("Complete job")
        await db.update_job(job2, status="complete")
        job3 = await db.create_job("Stuck in synthesizing")
        await db.update_job(job3, status="synthesizing")

        stuck = await db.get_stuck_jobs()
        stuck_ids = {j["id"] for j in stuck}
        assert job1 in stuck_ids
        assert job3 in stuck_ids
        assert job2 not in stuck_ids

    @pytest.mark.asyncio
    async def test_no_stuck_jobs_returns_empty(self, init_db):
        job_id = await db.create_job("Complete")
        await db.update_job(job_id, status="complete")
        stuck = await db.get_stuck_jobs()
        assert stuck == []


# ===========================================================================
# get_errored_jobs + reset_job_for_retry (lines 440-464)
# ===========================================================================


class TestErroredJobsAndRetry:
    @pytest.mark.asyncio
    async def test_get_errored_jobs(self, init_db):
        j1 = await db.create_job("Errored job")
        await db.update_job(j1, status="error", error="timeout")
        j2 = await db.create_job("Good job")
        await db.update_job(j2, status="complete")

        errored = await db.get_errored_jobs(limit=10)
        assert len(errored) == 1
        assert errored[0]["id"] == j1
        assert errored[0]["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_reset_job_for_retry(self, init_db):
        j1 = await db.create_job("Retry me")
        await db.update_job(j1, status="error", error="failed")

        await db.reset_job_for_retry(j1)
        job = await db.get_job(j1)
        assert job["status"] == "queued"
        assert job["error"] is None

    @pytest.mark.asyncio
    async def test_reset_only_affects_errored_jobs(self, init_db):
        j1 = await db.create_job("Not errored")
        await db.update_job(j1, status="complete")

        await db.reset_job_for_retry(j1)
        job = await db.get_job(j1)
        assert job["status"] == "complete"  # unchanged


# ===========================================================================
# save_sources with empty list (line 513)
# ===========================================================================


class TestSaveSourcesEdge:
    @pytest.mark.asyncio
    async def test_empty_sources_noop(self, init_db):
        job_id = await db.create_job("No sources")
        await db.save_sources(job_id, [], round_num=1)  # should not raise
        sources = await db.get_sources(job_id)
        assert sources == []


# ===========================================================================
# Palace classifications (lines 633-687)
# ===========================================================================


class TestPalaceClassifications:
    @pytest.mark.asyncio
    async def test_upsert_and_get_classification(self, init_db):
        await db.upsert_classification("k8s-basics", "personal", "infrastructure", 0.95)
        result = await db.get_article_classification("k8s-basics", "personal")
        assert result is not None
        assert result["hall"] == "infrastructure"
        assert result["confidence"] == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, init_db):
        await db.upsert_classification("art1", "personal", "old-hall", 0.5)
        await db.upsert_classification("art1", "personal", "new-hall", 0.9)
        result = await db.get_article_classification("art1", "personal")
        assert result["hall"] == "new-hall"
        assert result["confidence"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_get_classifications_all(self, init_db):
        await db.upsert_classification("a", "kb1", "h1", 0.8)
        await db.upsert_classification("b", "kb2", "h2", 0.7)
        all_cls = await db.get_classifications()
        assert len(all_cls) == 2

    @pytest.mark.asyncio
    async def test_get_classifications_by_kb(self, init_db):
        await db.upsert_classification("a", "kb1", "h1", 0.8)
        await db.upsert_classification("b", "kb2", "h2", 0.7)
        cls = await db.get_classifications(kb="kb1")
        assert len(cls) == 1
        assert cls[0]["slug"] == "a"

    @pytest.mark.asyncio
    async def test_get_classifications_by_hall(self, init_db):
        await db.upsert_classification("a", "personal", "infra", 0.9)
        await db.upsert_classification("b", "personal", "infra", 0.8)
        await db.upsert_classification("c", "personal", "ml", 0.7)
        results = await db.get_classifications_by_hall("personal", "infra")
        assert len(results) == 2
        # Ordered by confidence DESC
        assert results[0]["confidence"] >= results[1]["confidence"]

    @pytest.mark.asyncio
    async def test_get_article_classification_missing(self, init_db):
        result = await db.get_article_classification("nonexistent", "personal")
        assert result is None


# ===========================================================================
# Palace rooms (lines 691-782)
# ===========================================================================


class TestPalaceRooms:
    @pytest.mark.asyncio
    async def test_upsert_room_and_get(self, init_db):
        room_id = await db.upsert_room("personal", "Container Room", None, 5)
        assert isinstance(room_id, int)
        rooms = await db.get_rooms("personal")
        assert len(rooms) == 1
        assert rooms[0]["name"] == "Container Room"

    @pytest.mark.asyncio
    async def test_add_room_member_and_get(self, init_db):
        room_id = await db.upsert_room("personal", "Room A", None, 2)
        await db.add_room_member(room_id, "k8s-basics", "personal", 0.9)
        await db.add_room_member(room_id, "docker", "personal", 0.7)
        members = await db.get_room_members(room_id)
        assert len(members) == 2
        # Ordered by relevance DESC
        assert members[0]["relevance"] >= members[1]["relevance"]

    @pytest.mark.asyncio
    async def test_get_article_rooms(self, init_db):
        r1 = await db.upsert_room("personal", "Room A", None, 1)
        r2 = await db.upsert_room("personal", "Room B", None, 1)
        await db.add_room_member(r1, "art1", "personal", 0.9)
        await db.add_room_member(r2, "art1", "personal", 0.6)
        rooms = await db.get_article_rooms("art1", "personal")
        assert len(rooms) == 2
        assert rooms[0]["relevance"] >= rooms[1]["relevance"]

    @pytest.mark.asyncio
    async def test_clear_rooms(self, init_db):
        r1 = await db.upsert_room("personal", "Room X", None, 1)
        await db.add_room_member(r1, "art1", "personal")
        await db.clear_rooms("personal")
        rooms = await db.get_rooms("personal")
        assert rooms == []
        members = await db.get_room_members(r1)
        assert members == []

    @pytest.mark.asyncio
    async def test_clear_rooms_no_rooms_noop(self, init_db):
        """Clearing rooms when none exist should not raise."""
        await db.clear_rooms("empty_kb")
        rooms = await db.get_rooms("empty_kb")
        assert rooms == []


# ===========================================================================
# get_pending_candidates — limit <= 0 (line 1026)
# ===========================================================================


class TestPendingCandidatesEdge:
    @pytest.mark.asyncio
    async def test_zero_limit_returns_empty(self, init_db):
        await db.insert_topic_candidate("personal", "topic1", "gap", None, 1.0)
        result = await db.get_pending_candidates("personal", limit=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_negative_limit_returns_empty(self, init_db):
        result = await db.get_pending_candidates("personal", limit=-5)
        assert result == []


# ===========================================================================
# Article versions (lines 1199-1210)
# ===========================================================================


class TestArticleVersions:
    @pytest.mark.asyncio
    async def test_save_and_get_versions(self, init_db):
        v1 = await db.save_article_version(
            "personal", "k8s", "# v1 content", change_type="created",
        )
        v2 = await db.save_article_version(
            "personal", "k8s", "# v2 content", change_type="updated",
        )
        assert v1 > 0
        assert v2 > v1

        versions = await db.get_article_versions("personal", "k8s")
        assert len(versions) == 2
        # newest first
        assert versions[0]["id"] == v2

    @pytest.mark.asyncio
    async def test_get_article_at_timestamp(self, init_db):
        await db.save_article_version("personal", "art1", "old content")
        # Use a far-future timestamp to capture the version we just saved
        result = await db.get_article_at_timestamp("personal", "art1", "2099-01-01T00:00:00")
        assert result is not None
        assert result["full_content"] == "old content"

    @pytest.mark.asyncio
    async def test_get_article_at_timestamp_before_any(self, init_db):
        await db.save_article_version("personal", "art1", "content")
        result = await db.get_article_at_timestamp("personal", "art1", "2000-01-01T00:00:00")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_article_version_by_id(self, init_db):
        vid = await db.save_article_version("personal", "art1", "content here")
        result = await db.get_article_version_by_id(vid)
        assert result is not None
        assert result["full_content"] == "content here"

    @pytest.mark.asyncio
    async def test_get_article_version_by_id_missing(self, init_db):
        result = await db.get_article_version_by_id(99999)
        assert result is None


# ===========================================================================
# get_stale_claims (lines 1289-1301)
# ===========================================================================


class TestStaleClaims:
    @pytest.mark.asyncio
    async def test_stale_claims_returns_null_last_checked(self, init_db):
        await db.save_claim("personal", "art1", "The sky is blue.")
        stale = await db.get_stale_claims(days=90)
        assert len(stale) >= 1
        assert stale[0]["claim_text"] == "The sky is blue."

    @pytest.mark.asyncio
    async def test_recently_checked_claims_not_stale(self, init_db):
        claim_id = await db.save_claim("personal", "art1", "Fresh claim")
        await db.update_claim_status(claim_id, "verified", 0.9)
        stale = await db.get_stale_claims(days=90)
        # The claim we just checked should not be stale
        stale_ids = {s["id"] for s in stale}
        assert claim_id not in stale_ids


# ===========================================================================
# Article meta stubs (lines 1395-1407)
# ===========================================================================


class TestArticleMetaStubs:
    @pytest.mark.asyncio
    async def test_get_article_meta_returns_none(self):
        result = await db.get_article_meta("personal", "slug")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_article_metas_returns_empty(self):
        result = await db.list_article_metas("personal")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_article_meta_noop(self):
        await db.delete_article_meta("personal", "slug")  # should not raise

    @pytest.mark.asyncio
    async def test_article_meta_count_returns_zero(self):
        result = await db.article_meta_count("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_upsert_article_meta_returns_none(self):
        result = await db.upsert_article_meta("personal", "slug", {"title": "T"})
        assert result is None


# ===========================================================================
# Trace spans (lines 1465-1547)
# ===========================================================================


class TestTraceSpans:
    @pytest.mark.asyncio
    async def test_save_and_get_trace_spans(self, init_db):
        span = {
            "trace_id": "tr-1",
            "span_id": "sp-1",
            "timestamp": "2026-04-12T00:00:00",
            "name": "llm.chat",
            "provider": "anthropic",
            "model": "claude-3",
            "kind": "chat",
            "input_tokens": 100,
            "output_tokens": 50,
            "duration_ms": 500,
            "status": "ok",
            "session_id": "sess-1",
        }
        await db.save_trace_span(span)
        spans = await db.get_trace_spans(limit=10)
        assert len(spans) >= 1
        assert spans[0]["trace_id"] == "tr-1"

    @pytest.mark.asyncio
    async def test_get_trace_spans_filtered(self, init_db):
        await db.save_trace_span({
            "trace_id": "tr-a", "span_id": "sp-a",
            "timestamp": "2026-01-01", "provider": "anthropic",
            "model": "claude-3", "status": "ok", "session_id": "sess-1",
        })
        await db.save_trace_span({
            "trace_id": "tr-b", "span_id": "sp-b",
            "timestamp": "2026-01-02", "provider": "openai",
            "model": "gpt-4", "status": "error", "session_id": "sess-2",
        })

        # Filter by provider
        spans = await db.get_trace_spans(limit=10, provider="anthropic")
        assert all(s["provider"] == "anthropic" for s in spans)

        # Filter by status
        spans = await db.get_trace_spans(limit=10, status="error")
        assert all(s["status"] == "error" for s in spans)

        # Filter by session_id
        spans = await db.get_trace_spans(limit=10, session_id="sess-1")
        assert all(s["session_id"] == "sess-1" for s in spans)

    @pytest.mark.asyncio
    async def test_get_trace_spans_by_trace_id(self, init_db):
        await db.save_trace_span({
            "trace_id": "tr-same", "span_id": "sp-1", "timestamp": "2026-01-01",
        })
        await db.save_trace_span({
            "trace_id": "tr-same", "span_id": "sp-2", "timestamp": "2026-01-01",
        })
        await db.save_trace_span({
            "trace_id": "tr-other", "span_id": "sp-3", "timestamp": "2026-01-01",
        })
        spans = await db.get_trace_spans_by_trace_id("tr-same")
        assert len(spans) == 2
        assert all(s["trace_id"] == "tr-same" for s in spans)


# ===========================================================================
# LLM usage totals (lines 1504-1547)
# ===========================================================================


class TestLlmUsageTotals:
    @pytest.mark.asyncio
    async def test_record_and_get_usage(self, init_db):
        await db.record_llm_usage_total(
            "anthropic", "claude-3", "chat",
            input_tokens=100, output_tokens=50,
        )
        totals = await db.get_llm_usage_totals()
        assert len(totals) >= 1
        row = totals[0]
        assert row["provider"] == "anthropic"
        assert row["calls"] == 1
        assert row["input_tokens"] == 100

    @pytest.mark.asyncio
    async def test_accumulates_on_conflict(self, init_db):
        await db.record_llm_usage_total(
            "openai", "gpt-4", "chat", input_tokens=100, output_tokens=50,
        )
        await db.record_llm_usage_total(
            "openai", "gpt-4", "chat", input_tokens=200, output_tokens=100,
        )
        totals = await db.get_llm_usage_totals()
        openai_row = next(t for t in totals if t["provider"] == "openai")
        assert openai_row["calls"] == 2
        assert openai_row["input_tokens"] == 300
        assert openai_row["output_tokens"] == 150


# ===========================================================================
# get_jobs compact mode
# ===========================================================================


class TestGetJobsCompact:
    @pytest.mark.asyncio
    async def test_compact_excludes_content(self, init_db):
        j1 = await db.create_job("Compact test")
        await db.update_job(j1, status="complete", content="big markdown blob")
        jobs = await db.get_jobs(limit=10, compact=True)
        assert len(jobs) >= 1
        # compact mode should not include 'content' key
        assert "content" not in jobs[0]

    @pytest.mark.asyncio
    async def test_full_includes_content(self, init_db):
        j1 = await db.create_job("Full test")
        await db.update_job(j1, status="complete", content="big markdown blob")
        jobs = await db.get_jobs(limit=10, compact=False)
        assert len(jobs) >= 1
        assert jobs[0]["content"] == "big markdown blob"


# ===========================================================================
# Topic candidates and discovery config
# ===========================================================================


class TestTopicCandidates:
    @pytest.mark.asyncio
    async def test_insert_and_count(self, init_db):
        inserted = await db.insert_topic_candidate("personal", "Rust async", "gap", None, 1.0)
        assert inserted is True
        count = await db.count_pending_candidates("personal")
        assert count == 1

    @pytest.mark.asyncio
    async def test_duplicate_rejected(self, init_db):
        await db.insert_topic_candidate("personal", "Rust async", "gap", None, 1.0)
        inserted = await db.insert_topic_candidate("personal", "Rust async", "gap", None, 2.0)
        assert inserted is False

    @pytest.mark.asyncio
    async def test_mark_enqueued(self, init_db):
        await db.insert_topic_candidate("personal", "Topic X", "gap", None, 1.0)
        pending = await db.get_pending_candidates("personal", limit=10)
        assert len(pending) == 1
        await db.mark_candidate_enqueued(pending[0]["id"], job_id=42)
        pending = await db.get_pending_candidates("personal", limit=10)
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_mark_skipped(self, init_db):
        await db.insert_topic_candidate("personal", "Skip Me", "gap", None, 1.0)
        pending = await db.get_pending_candidates("personal", limit=10)
        await db.mark_candidate_skipped(pending[0]["id"], "already exists")
        pending_after = await db.get_pending_candidates("personal", limit=10)
        assert len(pending_after) == 0


class TestAutoDiscoveryConfig:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, init_db):
        result = await db.upsert_auto_discovery_config(
            "personal", enabled=True, daily_budget=100, strategy="hybrid",
        )
        assert result["enabled"] == 1
        assert result["daily_budget"] == 100

        config = await db.get_auto_discovery_config("personal")
        assert config is not None
        assert config["enabled"] == 1

    @pytest.mark.asyncio
    async def test_list_enabled(self, init_db):
        await db.upsert_auto_discovery_config("kb1", enabled=True)
        await db.upsert_auto_discovery_config("kb2", enabled=False)
        configs = await db.list_enabled_auto_discovery_configs()
        assert len(configs) == 1
        assert configs[0]["kb"] == "kb1"


# ===========================================================================
# KB settings
# ===========================================================================


class TestKbSettings:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, init_db):
        result = await db.upsert_kb_settings("personal", synthesis_provider="anthropic")
        assert result["synthesis_provider"] == "anthropic"

        settings = await db.get_kb_settings("personal")
        assert settings is not None
        assert settings["synthesis_provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_empty_string_becomes_null(self, init_db):
        await db.upsert_kb_settings("personal", synthesis_provider="anthropic")
        await db.upsert_kb_settings("personal", synthesis_provider="")
        settings = await db.get_kb_settings("personal")
        assert settings["synthesis_provider"] is None

    @pytest.mark.asyncio
    async def test_get_missing_kb_returns_none(self, init_db):
        result = await db.get_kb_settings("nonexistent")
        assert result is None


# ===========================================================================
# Claims
# ===========================================================================


class TestClaims:
    @pytest.mark.asyncio
    async def test_save_and_get_claims(self, init_db):
        cid = await db.save_claim("personal", "art1", "Claim text here")
        assert cid > 0
        claims = await db.get_claims_for_article("personal", "art1")
        assert len(claims) == 1
        assert claims[0]["claim_text"] == "Claim text here"

    @pytest.mark.asyncio
    async def test_save_claim_dedupes(self, init_db):
        c1 = await db.save_claim("personal", "art1", "Same claim")
        c2 = await db.save_claim("personal", "art1", "Same claim", confidence=0.9)
        assert c1 == c2
        claims = await db.get_claims_for_article("personal", "art1")
        assert len(claims) == 1
        assert claims[0]["confidence"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_update_claim_status(self, init_db):
        cid = await db.save_claim("personal", "art1", "To verify")
        await db.update_claim_status(cid, "verified", 0.95)
        claims = await db.get_claims_for_article("personal", "art1")
        assert claims[0]["status"] == "verified"
        assert claims[0]["confidence"] == pytest.approx(0.95)
        assert claims[0]["last_checked_at"] is not None
