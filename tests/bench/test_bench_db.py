"""Latency benchmarks for database operations across sqlite + moto-DynamoDB."""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    """Helper to run an async func inside a sync benchmark callback."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def seeded_db(bench_db):
    """Seed the database with 200 jobs and return (db_mod, backend, n)."""
    db_mod, backend = bench_db
    n = 200

    async def _seed():
        for i in range(n):
            jid = await db_mod.create_job(f"bench topic {i}")
            if i % 3 == 0:
                await db_mod.update_job(jid, status="complete", word_count=i * 100)
            elif i % 3 == 1:
                await db_mod.update_job(jid, status="error", error="timeout")

    _run(_seed())
    return db_mod, backend, n


@pytest.mark.bench
class TestGetJobs:
    def test_get_jobs_compact_50(self, seeded_db, benchmark):
        db_mod, backend, n = seeded_db

        def run():
            return _run(db_mod.get_jobs(limit=50, compact=True))

        result = benchmark(run)
        assert len(result) == 50

    def test_get_jobs_full_50(self, seeded_db, benchmark):
        db_mod, backend, n = seeded_db

        def run():
            return _run(db_mod.get_jobs(limit=50, compact=False))

        result = benchmark(run)
        assert len(result) == 50


@pytest.mark.bench
class TestGetJobStats:
    def test_get_job_stats(self, seeded_db, benchmark):
        db_mod, backend, n = seeded_db

        def run():
            return _run(db_mod.get_job_stats())

        result = benchmark(run)
        assert result["total"] == n


@pytest.mark.bench
class TestCreateJob:
    def test_create_job_single(self, bench_db, benchmark):
        db_mod, backend = bench_db
        counter = {"n": 0}

        def run():
            counter["n"] += 1
            return _run(db_mod.create_job(f"bench-create-{counter['n']}"))

        result = benchmark(run)
        assert isinstance(result, int)


@pytest.mark.bench
class TestRecordUsage:
    def test_record_usage_single(self, bench_db, benchmark):
        db_mod, backend = bench_db

        def run():
            return _run(db_mod.record_llm_usage_total(
                "minimax", "text-01", "chat",
                input_tokens=100, output_tokens=50,
            ))

        benchmark(run)
