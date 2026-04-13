"""Smoke test: verify parity_db fixture wires both backends correctly.

Kept as a separate file from the real suites so that `pytest -x` pins
any fixture bugs to one obvious file, not the first real test that
happens to hit them. Once P2 is fully green this file stays as a
regression guard — if the conftest starts reusing stale imports or
moto state leaks between variants, this fires first.
"""

from __future__ import annotations

import pytest


class TestParityDbFixtureBoots:
    @pytest.mark.asyncio
    async def test_create_job_roundtrips(self, parity_db):
        job_id = await parity_db.create_job("smoke test topic")
        assert isinstance(job_id, int) and job_id > 0
        row = await parity_db.get_job(job_id)
        assert row is not None
        assert row["topic"] == "smoke test topic"
        assert row["status"] == "queued"

    @pytest.mark.asyncio
    async def test_get_jobs_returns_what_we_wrote(self, parity_db):
        await parity_db.create_job("alpha")
        await parity_db.create_job("beta")
        rows = await parity_db.get_jobs(limit=10)
        topics = {r["topic"] for r in rows}
        assert {"alpha", "beta"}.issubset(topics)

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_list(self, parity_db):
        rows = await parity_db.get_jobs(limit=10)
        assert rows == []
