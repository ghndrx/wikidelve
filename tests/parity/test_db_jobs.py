"""Job table parity: every db.{create,update,get,delete}_job path
runs against sqlite and dynamodb-moto.

The trailer at the end of `app/db.py` re-exports db_dynamo.* when
DB_BACKEND=dynamodb — the `parity_db` fixture in conftest flips
that env var and re-imports the module, so a single test body
proves both backends return the same shape.
"""

from __future__ import annotations

import pytest


class TestCreateAndGetJob:
    @pytest.mark.asyncio
    async def test_create_returns_int(self, parity_db):
        job_id = await parity_db.create_job("topic one")
        assert isinstance(job_id, int)
        assert job_id > 0

    @pytest.mark.asyncio
    async def test_ids_are_unique(self, parity_db):
        a = await parity_db.create_job("A")
        b = await parity_db.create_job("B")
        c = await parity_db.create_job("C")
        assert len({a, b, c}) == 3

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, parity_db):
        assert await parity_db.get_job(99999) is None

    @pytest.mark.asyncio
    async def test_get_returns_row_with_expected_fields(self, parity_db):
        jid = await parity_db.create_job("fields test")
        row = await parity_db.get_job(jid)
        assert row is not None
        assert row["id"] == jid
        assert row["topic"] == "fields test"
        assert row["status"] == "queued"
        assert "created_at" in row

    @pytest.mark.asyncio
    async def test_created_at_is_iso_timestamp(self, parity_db):
        jid = await parity_db.create_job("timestamp test")
        row = await parity_db.get_job(jid)
        # ISO-8601 string, both backends should store it the same way
        assert isinstance(row["created_at"], str)
        assert "T" in row["created_at"]


class TestUpdateJob:
    @pytest.mark.asyncio
    async def test_update_status_persists(self, parity_db):
        jid = await parity_db.create_job("status test")
        await parity_db.update_job(jid, status="running")
        row = await parity_db.get_job(jid)
        assert row["status"] == "running"

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, parity_db):
        jid = await parity_db.create_job("multi field")
        await parity_db.update_job(
            jid,
            status="complete",
            sources_count=5,
            word_count=1234,
        )
        row = await parity_db.get_job(jid)
        assert row["status"] == "complete"
        assert row["sources_count"] == 5
        assert row["word_count"] == 1234

    @pytest.mark.asyncio
    async def test_update_error_field(self, parity_db):
        jid = await parity_db.create_job("error test")
        await parity_db.update_job(jid, status="error", error="kaboom")
        row = await parity_db.get_job(jid)
        assert row["status"] == "error"
        assert row["error"] == "kaboom"

    @pytest.mark.asyncio
    async def test_update_nonexistent_is_graceful(self, parity_db):
        # SQLite silently no-ops; DynamoDB creates the item (UpdateItem
        # is upsert semantics). Both accepted as "graceful" — neither
        # raises, and both leave the system in a consistent state.
        # If this ever breaks, it's a real regression.
        try:
            await parity_db.update_job(99999, status="running")
        except Exception as exc:  # pragma: no cover - pinning docs the diff
            pytest.fail(f"update_job on missing id should not raise: {exc}")


class TestGetJobs:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self, parity_db):
        rows = await parity_db.get_jobs(limit=10)
        assert rows == []

    @pytest.mark.asyncio
    async def test_returns_all_written_under_limit(self, parity_db):
        for t in ("alpha", "beta", "gamma"):
            await parity_db.create_job(t)
        rows = await parity_db.get_jobs(limit=10)
        topics = {r["topic"] for r in rows}
        assert topics == {"alpha", "beta", "gamma"}

    @pytest.mark.asyncio
    async def test_limit_is_honored(self, parity_db):
        for i in range(5):
            await parity_db.create_job(f"topic-{i}")
        rows = await parity_db.get_jobs(limit=3)
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_reverse_chronological_order(self, parity_db):
        ids = []
        for t in ("first", "second", "third"):
            ids.append(await parity_db.create_job(t))
        rows = await parity_db.get_jobs(limit=10)
        # Newest first — last created should be at index 0
        assert rows[0]["topic"] == "third"
        assert rows[-1]["topic"] == "first"

    @pytest.mark.asyncio
    async def test_compact_mode_excludes_content(self, parity_db):
        jid = await parity_db.create_job("compact test")
        # Populate a large content field.
        await parity_db.update_job(jid, content="x" * 50_000)

        compact_rows = await parity_db.get_jobs(limit=10, compact=True)
        full_rows = await parity_db.get_jobs(limit=10, compact=False)

        assert len(compact_rows) == 1
        assert len(full_rows) == 1

        # Compact mode contract: `content` key is either absent or
        # falsy. Non-compact mode must return the full 50_000-char
        # body. This is the read-path optimization that keeps the
        # /api/research/jobs endpoint cheap.
        compact_content = compact_rows[0].get("content")
        full_content = full_rows[0].get("content")

        assert not compact_content, (
            f"compact mode must not return the `content` column, got "
            f"{len(compact_content) if compact_content else 0} chars"
        )
        assert full_content and len(full_content) == 50_000, (
            f"non-compact mode must return full content, got "
            f"{len(full_content) if full_content else 0} chars"
        )


class TestDeleteJob:
    @pytest.mark.asyncio
    async def test_delete_removes_row(self, parity_db):
        jid = await parity_db.create_job("deletable")
        await parity_db.delete_job(jid)
        assert await parity_db.get_job(jid) is None

    @pytest.mark.asyncio
    async def test_delete_missing_is_noop(self, parity_db):
        # Neither backend raises for a missing id.
        await parity_db.delete_job(99999)


class TestGetJobStats:
    """Both backends agree on the stats dict shape:
      total, complete, errors, active, cancelled, total_words, added_to_wiki
    Note: 'active' buckets queued + searching + synthesizing + variants.
    'errors' (plural) is the canonical key — not 'error'.
    """

    @pytest.mark.asyncio
    async def test_empty_stats(self, parity_db):
        stats = await parity_db.get_job_stats()
        # All counter fields should be 0 on an empty table.
        for key in ("total", "complete", "errors", "active", "cancelled"):
            assert (stats.get(key) or 0) == 0

    @pytest.mark.asyncio
    async def test_stats_shape_is_stable(self, parity_db):
        # Schema contract — these keys must always exist on both
        # backends so main.py can render them without KeyErrors.
        stats = await parity_db.get_job_stats()
        for key in (
            "total", "complete", "errors", "active",
            "cancelled", "total_words", "added_to_wiki",
        ):
            assert key in stats, f"missing key {key}: {stats.keys()}"

    @pytest.mark.asyncio
    async def test_stats_reflect_counts(self, parity_db):
        # 2 queued (→ 'active'), 1 complete, 1 error
        for _ in range(2):
            await parity_db.create_job("q")
        c = await parity_db.create_job("c")
        await parity_db.update_job(c, status="complete")
        e = await parity_db.create_job("e")
        await parity_db.update_job(e, status="error")

        stats = await parity_db.get_job_stats()
        assert stats.get("total", 0) == 4
        assert stats.get("active", 0) == 2
        assert stats.get("complete", 0) == 1
        assert stats.get("errors", 0) == 1


class TestCheckCooldown:
    @pytest.mark.asyncio
    async def test_no_prior_returns_none(self, parity_db):
        result = await parity_db.check_cooldown("brand new topic")
        assert result is None

    @pytest.mark.asyncio
    async def test_recent_job_returns_existing(self, parity_db):
        jid = await parity_db.create_job("cooldown topic")
        await parity_db.update_job(jid, status="complete")
        result = await parity_db.check_cooldown("cooldown topic")
        # Either the same id or a dict with `id` key — both backends
        # return a row-shaped dict.
        assert result is not None
        assert result.get("id") == jid
