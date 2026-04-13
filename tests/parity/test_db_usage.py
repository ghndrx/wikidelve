"""LLM usage total parity — the all-time counter table.

The DynamoDB path uses an atomic `ADD` expression to bump counters;
the SQLite path uses `INSERT ... ON CONFLICT DO UPDATE`. Both must
produce identical final counts under sequential and (simulated)
concurrent workloads.
"""

from __future__ import annotations

import asyncio

import pytest


class TestRecordLlmUsage:
    @pytest.mark.asyncio
    async def test_empty_totals_empty(self, parity_db):
        rows = await parity_db.get_llm_usage_totals()
        assert rows == []

    @pytest.mark.asyncio
    async def test_single_record_appears(self, parity_db):
        await parity_db.record_llm_usage_total(
            "minimax", "text-01", "chat",
            input_tokens=100, output_tokens=50,
        )
        rows = await parity_db.get_llm_usage_totals()
        assert len(rows) == 1
        r = rows[0]
        assert r["provider"] == "minimax"
        assert r["model"] == "text-01"
        assert r["kind"] == "chat"
        assert r["calls"] == 1
        assert r["input_tokens"] == 100
        assert r["output_tokens"] == 50

    @pytest.mark.asyncio
    async def test_repeated_record_accumulates(self, parity_db):
        for _ in range(5):
            await parity_db.record_llm_usage_total(
                "minimax", "text-01", "chat",
                input_tokens=10, output_tokens=5,
            )
        rows = await parity_db.get_llm_usage_totals()
        assert len(rows) == 1
        assert rows[0]["calls"] == 5
        assert rows[0]["input_tokens"] == 50
        assert rows[0]["output_tokens"] == 25

    @pytest.mark.asyncio
    async def test_distinct_keys_separate(self, parity_db):
        await parity_db.record_llm_usage_total(
            "minimax", "text-01", "chat", input_tokens=10, output_tokens=5,
        )
        await parity_db.record_llm_usage_total(
            "bedrock", "sonnet", "chat", input_tokens=100, output_tokens=50,
        )
        await parity_db.record_llm_usage_total(
            "minimax", "embo-01", "embed", input_tokens=200, output_tokens=0,
        )
        rows = await parity_db.get_llm_usage_totals()
        assert len(rows) == 3
        by_model = {(r["provider"], r["model"], r["kind"]): r for r in rows}
        assert ("minimax", "text-01", "chat") in by_model
        assert ("bedrock", "sonnet", "chat") in by_model
        assert ("minimax", "embo-01", "embed") in by_model

    @pytest.mark.asyncio
    async def test_concurrent_increments_are_consistent(self, parity_db):
        """50 parallel increments must equal 50 on both backends.

        SQLite gets serialized via its busy_timeout + WAL.
        DynamoDB gets serialized via the atomic ADD UpdateExpression.
        A race on either would show up as < 50.
        """
        async def bump():
            await parity_db.record_llm_usage_total(
                "minimax", "text-01", "chat",
                input_tokens=1, output_tokens=1,
            )

        await asyncio.gather(*(bump() for _ in range(50)))
        rows = await parity_db.get_llm_usage_totals()
        assert len(rows) == 1
        assert rows[0]["calls"] == 50
        assert rows[0]["input_tokens"] == 50
        assert rows[0]["output_tokens"] == 50
