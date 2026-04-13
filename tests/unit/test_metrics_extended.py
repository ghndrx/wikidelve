"""Extended tests for app.metrics -- covers _prune_old_days edge cases,
_schedule_persist_total, get_all_time_totals, and _estimate_cost branches.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import metrics


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset_all()
    yield
    metrics.reset_all()


# ---------------------------------------------------------------------------
# _prune_old_days
# ---------------------------------------------------------------------------


class TestPruneOldDays:
    def test_removes_days_older_than_retention(self):
        """Days older than USAGE_RETENTION_DAYS are pruned."""
        old_day = "2020-01-01"
        metrics._usage_by_day[old_day] = {("p", "m", "k"): {"calls": 1, "input": 0, "output": 0}}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        metrics._prune_old_days(today)
        assert old_day not in metrics._usage_by_day

    def test_keeps_recent_days(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        metrics._usage_by_day[today] = {("p", "m", "k"): {"calls": 1, "input": 0, "output": 0}}
        metrics._prune_old_days(today)
        assert today in metrics._usage_by_day

    def test_removes_malformed_day_keys(self):
        """Invalid date strings are pruned."""
        metrics._usage_by_day["not-a-date"] = {("p", "m", "k"): {"calls": 1, "input": 0, "output": 0}}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        metrics._prune_old_days(today)
        assert "not-a-date" not in metrics._usage_by_day


# ---------------------------------------------------------------------------
# _schedule_persist_total
# ---------------------------------------------------------------------------


class TestSchedulePersistTotal:
    def test_schedules_in_running_loop(self):
        """When a running loop exists, a task is created."""
        async def _run():
            with patch("app.db.record_llm_usage_total", new_callable=AsyncMock):
                metrics._schedule_persist_total("prov", "mod", "chat", 10, 5)
                await asyncio.sleep(0.05)

        asyncio.run(_run())

    def test_runs_without_loop(self):
        """When no running loop exists, asyncio.run is used."""
        with patch("app.db.record_llm_usage_total", new_callable=AsyncMock):
            metrics._schedule_persist_total("prov", "mod", "chat", 10, 5)

    def test_swallows_exception_without_loop(self):
        """Errors in the no-loop branch are swallowed."""
        with patch("app.db.record_llm_usage_total", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            # Should not raise
            metrics._schedule_persist_total("prov", "mod", "chat", 10, 5)


class TestGetAllTimeTotals:
    @pytest.mark.asyncio
    async def test_returns_rows_and_totals(self):
        with patch("app.db.get_llm_usage_totals", new_callable=AsyncMock, return_value=[
            {
                "provider": "minimax", "model": "text-01", "kind": "chat",
                "calls": 10, "input_tokens": 1000, "output_tokens": 500,
                "first_seen": "2026-01-01", "last_seen": "2026-04-01",
            },
        ]):
            result = await metrics.get_all_time_totals()
        assert len(result["rows"]) == 1
        assert result["totals"]["calls"] == 10
        assert result["totals"]["input"] == 1000
        assert result["totals"]["output"] == 500
        assert result["totals"]["cost_usd"] >= 0

    @pytest.mark.asyncio
    async def test_returns_empty_on_db_error(self):
        with patch("app.db.get_llm_usage_totals", new_callable=AsyncMock, side_effect=RuntimeError("db down")):
            result = await metrics.get_all_time_totals()
        assert result["rows"] == []
        assert result["totals"]["calls"] == 0

    @pytest.mark.asyncio
    async def test_handles_null_values(self):
        """Rows with None/0 values are handled gracefully."""
        with patch("app.db.get_llm_usage_totals", new_callable=AsyncMock, return_value=[
            {
                "provider": "x", "model": "y", "kind": "embed",
                "calls": None, "input_tokens": None, "output_tokens": None,
                "first_seen": None, "last_seen": None,
            },
        ]):
            result = await metrics.get_all_time_totals()
        assert result["totals"]["calls"] == 0


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_known_provider_returns_positive_cost(self):
        cost = metrics._estimate_cost("minimax", "chat", 1_000_000, 1_000_000)
        assert cost > 0

    def test_unknown_provider_returns_zero(self):
        cost = metrics._estimate_cost("unknown_provider", "chat", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_embed_pricing(self):
        cost = metrics._estimate_cost("minimax", "embed", 1_000_000, 0)
        assert cost > 0

    def test_zero_tokens_returns_zero(self):
        cost = metrics._estimate_cost("minimax", "chat", 0, 0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# get_usage_summary edge cases
# ---------------------------------------------------------------------------


class TestGetUsageSummaryEdge:
    def test_empty_summary(self):
        summary = metrics.get_usage_summary()
        assert summary["totals_week"]["calls"] == 0
        assert summary["today_breakdown"] == []
        assert summary["per_day"] == []

    def test_multiple_days(self):
        """Records across different provider keys aggregate correctly."""
        metrics.record_llm_call(
            provider="minimax", model="text-01", kind="chat",
            input_tokens=100, output_tokens=50,
        )
        metrics.record_llm_call(
            provider="bedrock", model="sonnet", kind="chat",
            input_tokens=200, output_tokens=100,
        )
        summary = metrics.get_usage_summary()
        assert summary["totals_week"]["calls"] == 2
        assert len(summary["today_breakdown"]) == 2
