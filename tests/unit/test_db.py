"""Unit tests for app/db.py — SQLite database operations."""

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

# Must set DB_PATH before importing db module
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
    """Initialize the test database."""
    await db.init_db()
    yield
    # Cleanup handled by tmp_path


class TestInitDB:
    @pytest.mark.asyncio
    async def test_creates_tables(self, init_db):
        conn = await db._get_db()
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row["name"] async for row in cursor}
            assert "research_jobs" in tables
            assert "research_sources" in tables
            assert "article_updates" in tables
            assert "article_embeddings" in tables
            assert "kg_entities" in tables
            assert "kg_edges" in tables
        finally:
            await conn.close()


class TestJobs:
    @pytest.mark.asyncio
    async def test_create_and_get_job(self, init_db):
        job_id = await db.create_job("Test topic for research")
        assert isinstance(job_id, int)
        assert job_id > 0

        job = await db.get_job(job_id)
        assert job is not None
        assert job["topic"] == "Test topic for research"
        assert job["status"] == "queued"

    @pytest.mark.asyncio
    async def test_update_job(self, init_db):
        job_id = await db.create_job("Update test")
        await db.update_job(job_id, status="searching", sources_count=5)

        job = await db.get_job(job_id)
        assert job["status"] == "searching"
        assert job["sources_count"] == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, init_db):
        job = await db.get_job(99999)
        assert job is None

    @pytest.mark.asyncio
    async def test_get_jobs_list(self, init_db):
        await db.create_job("Topic A")
        await db.create_job("Topic B")
        await db.create_job("Topic C")

        jobs = await db.get_jobs(limit=10)
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_job_stats(self, init_db):
        job_id = await db.create_job("Stats test")
        await db.update_job(job_id, status="complete", word_count=500)

        stats = await db.get_job_stats()
        assert stats["total"] >= 1
        assert stats["complete"] >= 1

    @pytest.mark.asyncio
    async def test_delete_job(self, init_db):
        job_id = await db.create_job("Delete me")
        await db.delete_job(job_id)
        job = await db.get_job(job_id)
        assert job is None


class TestCooldown:
    @pytest.mark.asyncio
    async def test_no_cooldown_new_topic(self, init_db):
        result = await db.check_cooldown("Brand new topic")
        assert result is None

    @pytest.mark.asyncio
    async def test_cooldown_recent_topic(self, init_db):
        job_id = await db.create_job("Recent topic test")
        await db.update_job(job_id, status="complete")
        result = await db.check_cooldown("Recent topic test")
        assert result is not None
        assert result["topic"] == "Recent topic test"


class TestArticleUpdates:
    @pytest.mark.asyncio
    async def test_log_article_update(self, init_db):
        # article_updates has FK to research_jobs, so create a job first
        job_id = await db.create_job("Test topic")
        await db.log_article_update("test-slug", "personal", job_id, "created")
        updates = await db.get_article_updates(limit=10)
        assert len(updates) >= 1
        assert updates[0]["article_slug"] == "test-slug"

    @pytest.mark.asyncio
    async def test_get_article_updates(self, init_db):
        job_id = await db.create_job("Test topic")
        await db.log_article_update("slug-a", "personal", job_id, "created")
        await db.log_article_update("slug-b", "personal", job_id, "updated")

        updates = await db.get_article_updates(limit=10)
        assert len(updates) >= 2
