"""Extended tests for app/auto_discovery.py -- covers mining strategies,
run_discovery_for_kb dispatch, enqueue loop, and get_status_for_kb.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "ext_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            with patch("app.auto_discovery.DB_PATH", _test_db_path):
                yield


# ---------------------------------------------------------------------------
# _parse_seed_topics
# ---------------------------------------------------------------------------


class TestParseSeedTopics:
    def test_none_returns_none(self):
        from app.auto_discovery import _parse_seed_topics
        assert _parse_seed_topics(None) is None

    def test_empty_string_returns_none(self):
        from app.auto_discovery import _parse_seed_topics
        assert _parse_seed_topics("") is None

    def test_invalid_json_returns_none(self):
        from app.auto_discovery import _parse_seed_topics
        assert _parse_seed_topics("not json") is None

    def test_non_list_returns_none(self):
        from app.auto_discovery import _parse_seed_topics
        assert _parse_seed_topics('{"key": "value"}') is None

    def test_valid_list(self):
        from app.auto_discovery import _parse_seed_topics
        result = _parse_seed_topics('["rust", "go", ""]')
        assert result == ["rust", "go"]

    def test_empty_list_returns_none(self):
        from app.auto_discovery import _parse_seed_topics
        assert _parse_seed_topics('["", "  "]') is None


# ---------------------------------------------------------------------------
# _resolve_seed_slugs
# ---------------------------------------------------------------------------


class TestResolveSeedSlugs:
    def test_none_input_returns_none(self):
        from app.auto_discovery import _resolve_seed_slugs
        assert _resolve_seed_slugs("personal", None) is None

    def test_empty_list_returns_none(self):
        from app.auto_discovery import _resolve_seed_slugs
        assert _resolve_seed_slugs("personal", []) is None

    def test_blank_entries_skipped(self):
        from app.auto_discovery import _resolve_seed_slugs
        with patch("app.auto_discovery.find_related_article", return_value=None):
            with patch("app.auto_discovery.get_articles", return_value=[]):
                result = _resolve_seed_slugs("personal", ["", "  "])
        assert result == []


# ---------------------------------------------------------------------------
# discover_from_stale
# ---------------------------------------------------------------------------


class TestDiscoverFromStale:
    @pytest.mark.asyncio
    async def test_no_articles_returns_zero(self):
        from app.auto_discovery import discover_from_stale
        with patch("app.auto_discovery.get_articles", return_value=[]):
            result = await discover_from_stale("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_recent_articles(self):
        from app.auto_discovery import discover_from_stale
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "recent", "title": "Recent", "tags": ["ai"], "updated": "2026-04-01"},
        ]):
            result = await discover_from_stale("personal", days=90)
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_non_velocity_tags(self):
        from app.auto_discovery import discover_from_stale
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%d")
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "old", "title": "Old Cooking", "tags": ["cooking"], "updated": old_date},
        ]):
            result = await discover_from_stale("personal", days=90)
        assert result == 0

    @pytest.mark.asyncio
    async def test_inserts_old_velocity_article(self):
        from app.auto_discovery import discover_from_stale
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%d")
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "old-k8s", "title": "Old K8s Guide", "tags": ["kubernetes"], "updated": old_date},
        ]):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.check_cooldown = AsyncMock(return_value=False)
                mock_db.insert_topic_candidate = AsyncMock(return_value=True)
                result = await discover_from_stale("personal", days=90)
        assert result == 1


# ---------------------------------------------------------------------------
# discover_from_questions
# ---------------------------------------------------------------------------


class TestDiscoverFromQuestions:
    @pytest.mark.asyncio
    async def test_no_articles_returns_zero(self):
        from app.auto_discovery import discover_from_questions
        with patch("app.auto_discovery.get_articles", return_value=[]):
            result = await discover_from_questions("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_extracts_question_from_body(self):
        from app.auto_discovery import discover_from_questions
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "test-art", "title": "Test Art"},
        ]):
            with patch("app.auto_discovery.get_article", return_value={
                "raw_markdown": "## Overview\n\n" + "Lorem ipsum dolor sit amet. " * 10 + "\n\nHow does Kubernetes handle network policies in multi-cluster deployments?\n\nSome more text here to pad the body.",
            }):
                with patch("app.auto_discovery.find_related_article", return_value=None):
                    with patch("app.auto_discovery.db") as mock_db:
                        mock_db.check_cooldown = AsyncMock(return_value=False)
                        mock_db.insert_topic_candidate = AsyncMock(return_value=True)
                        result = await discover_from_questions("personal")
        assert result >= 1

    @pytest.mark.asyncio
    async def test_skips_short_body(self):
        from app.auto_discovery import discover_from_questions
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "short", "title": "Short"},
        ]):
            with patch("app.auto_discovery.get_article", return_value={
                "raw_markdown": "Short.",
            }):
                result = await discover_from_questions("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_rhetorical_questions(self):
        from app.auto_discovery import discover_from_questions
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "art", "title": "Art"},
        ]):
            with patch("app.auto_discovery.get_article", return_value={
                "raw_markdown": "x" * 250 + "\n\nBut why does this matter in practice?\n\n",
            }):
                with patch("app.auto_discovery.find_related_article", return_value=None):
                    with patch("app.auto_discovery.db") as mock_db:
                        mock_db.check_cooldown = AsyncMock(return_value=False)
                        mock_db.insert_topic_candidate = AsyncMock(return_value=True)
                        result = await discover_from_questions("personal")
        # "But why..." should be skipped because it starts with "But "
        assert result == 0


# ---------------------------------------------------------------------------
# discover_from_stubs
# ---------------------------------------------------------------------------


class TestDiscoverFromStubs:
    @pytest.mark.asyncio
    async def test_no_articles_returns_zero(self):
        from app.auto_discovery import discover_from_stubs
        with patch("app.auto_discovery.get_articles", return_value=[]):
            result = await discover_from_stubs("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_finds_stub_bullets(self):
        from app.auto_discovery import discover_from_stubs
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "rust-art", "title": "Rust Concurrency"},
        ]):
            with patch("app.auto_discovery.get_article", return_value={
                "title": "Rust Concurrency",
                "raw_markdown": "## Topics\n\n- Async cancellation patterns\n- Memory safety guarantees\n\nSome text.",
            }):
                with patch("app.auto_discovery.find_related_article", return_value=None):
                    with patch("app.auto_discovery.db") as mock_db:
                        mock_db.check_cooldown = AsyncMock(return_value=False)
                        mock_db.insert_topic_candidate = AsyncMock(return_value=True)
                        result = await discover_from_stubs("personal")
        assert result >= 1

    @pytest.mark.asyncio
    async def test_skips_prose_bullets(self):
        from app.auto_discovery import discover_from_stubs
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "art", "title": "Art"},
        ]):
            with patch("app.auto_discovery.get_article", return_value={
                "title": "Art",
                "raw_markdown": "## Section\n\n- This is a full sentence with a period at the end.\n",
            }):
                with patch("app.auto_discovery.find_related_article", return_value=None):
                    with patch("app.auto_discovery.db") as mock_db:
                        mock_db.check_cooldown = AsyncMock(return_value=False)
                        mock_db.insert_topic_candidate = AsyncMock(return_value=True)
                        result = await discover_from_stubs("personal")
        # bullets with periods are skipped
        assert result == 0


# ---------------------------------------------------------------------------
# discover_from_broken_wikilinks
# ---------------------------------------------------------------------------


class TestDiscoverFromBrokenWikilinks:
    @pytest.mark.asyncio
    async def test_no_articles_returns_zero(self):
        from app.auto_discovery import discover_from_broken_wikilinks
        with patch("app.auto_discovery.get_articles", return_value=[]):
            result = await discover_from_broken_wikilinks("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_broken_link_inserted(self):
        from app.auto_discovery import discover_from_broken_wikilinks
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "source-art"},
        ]):
            with patch("app.auto_discovery.storage") as mock_storage:
                mock_storage.read_text.return_value = "See [[Nonexistent Topic]] for more."
                with patch("app.auto_discovery.find_related_article", return_value=None):
                    with patch("app.auto_discovery.db") as mock_db:
                        mock_db.check_cooldown = AsyncMock(return_value=False)
                        mock_db.insert_topic_candidate = AsyncMock(return_value=True)
                        result = await discover_from_broken_wikilinks("personal")
        assert result == 1

    @pytest.mark.asyncio
    async def test_existing_slug_not_inserted(self):
        from app.auto_discovery import discover_from_broken_wikilinks
        with patch("app.auto_discovery.get_articles", return_value=[
            {"slug": "source-art"},
            {"slug": "existing-topic"},
        ]):
            with patch("app.auto_discovery.storage") as mock_storage:
                mock_storage.read_text.return_value = "See [[Existing Topic]] for more."
                result = await discover_from_broken_wikilinks("personal")
        assert result == 0


# ---------------------------------------------------------------------------
# run_discovery_for_kb dispatch
# ---------------------------------------------------------------------------


class TestRunDiscoveryForKb:
    @pytest.mark.asyncio
    async def test_disabled_returns_skipped(self):
        from app.auto_discovery import run_discovery_for_kb
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value=None)
            result = await run_discovery_for_kb("personal")
        assert result["skipped"] == "disabled"

    @pytest.mark.asyncio
    async def test_single_mining_strategy(self):
        from app.auto_discovery import run_discovery_for_kb
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value={
                "enabled": True, "strategy": "stale", "seed_topics": None, "llm_sample": 5,
            })
            mock_db.count_pending_candidates = AsyncMock(return_value=0)
            with patch("app.auto_discovery.discover_from_stale", new_callable=AsyncMock) as mock_stale:
                mock_stale.return_value = 3
                result = await run_discovery_for_kb("personal")
        assert result["counts"]["stale"] == 3

    @pytest.mark.asyncio
    async def test_strategy_all_runs_everything(self):
        from app.auto_discovery import run_discovery_for_kb
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value={
                "enabled": True, "strategy": "all", "seed_topics": None, "llm_sample": 5,
            })
            mock_db.count_pending_candidates = AsyncMock(return_value=0)
            with patch("app.auto_discovery.discover_from_kg_entities", new_callable=AsyncMock, return_value=1):
                with patch("app.auto_discovery.discover_from_llm_followups", new_callable=AsyncMock, return_value=2):
                    with patch("app.auto_discovery.discover_from_contradictions", new_callable=AsyncMock, return_value=0):
                        with patch("app.auto_discovery.discover_from_stale", new_callable=AsyncMock, return_value=0):
                            with patch("app.auto_discovery.discover_from_orphan_mentions", new_callable=AsyncMock, return_value=0):
                                with patch("app.auto_discovery.discover_from_questions", new_callable=AsyncMock, return_value=0):
                                    with patch("app.auto_discovery.cluster_research_history", new_callable=AsyncMock, return_value=0):
                                        with patch("app.auto_discovery.discover_from_stubs", new_callable=AsyncMock, return_value=0):
                                            with patch("app.auto_discovery.discover_from_broken_wikilinks", new_callable=AsyncMock, return_value=0):
                                                result = await run_discovery_for_kb("personal")
        assert "kg_entities" in result["counts"]
        assert "llm_followup" in result["counts"]
        assert "contradiction" in result["counts"]

    @pytest.mark.asyncio
    async def test_strategy_exception_caught(self):
        from app.auto_discovery import run_discovery_for_kb
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value={
                "enabled": True, "strategy": "stale", "seed_topics": None, "llm_sample": 5,
            })
            mock_db.count_pending_candidates = AsyncMock(return_value=0)
            with patch("app.auto_discovery.discover_from_stale", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
                result = await run_discovery_for_kb("personal")
        assert result["counts"]["stale"] == 0


# ---------------------------------------------------------------------------
# run_discovery_all
# ---------------------------------------------------------------------------


class TestRunDiscoveryAll:
    @pytest.mark.asyncio
    async def test_global_disabled(self):
        from app.auto_discovery import run_discovery_all
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", False):
            result = await run_discovery_all()
        assert result["skipped"] == "global_disabled"

    @pytest.mark.asyncio
    async def test_runs_for_each_enabled_kb(self):
        from app.auto_discovery import run_discovery_all
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.list_enabled_auto_discovery_configs = AsyncMock(return_value=[
                    {"kb": "kb1"}, {"kb": "kb2"},
                ])
                with patch("app.auto_discovery.run_discovery_for_kb", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = {"kb": "x", "counts": {}}
                    result = await run_discovery_all()
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_error_in_one_kb_doesnt_break_others(self):
        from app.auto_discovery import run_discovery_all
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.list_enabled_auto_discovery_configs = AsyncMock(return_value=[
                    {"kb": "bad"}, {"kb": "good"},
                ])
                with patch("app.auto_discovery.run_discovery_for_kb", new_callable=AsyncMock) as mock_run:
                    mock_run.side_effect = [RuntimeError("fail"), {"kb": "good", "counts": {}}]
                    result = await run_discovery_all()
        assert len(result["results"]) == 2
        assert "error" in result["results"][0]


# ---------------------------------------------------------------------------
# enqueue_next_candidates_for_kb
# ---------------------------------------------------------------------------


class TestEnqueueNextCandidatesForKb:
    @pytest.mark.asyncio
    async def test_global_disabled(self):
        from app.auto_discovery import enqueue_next_candidates_for_kb
        pool = MagicMock()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", False):
            result = await enqueue_next_candidates_for_kb(pool, "personal")
        assert result["skipped"] == "global_disabled"

    @pytest.mark.asyncio
    async def test_kb_disabled(self):
        from app.auto_discovery import enqueue_next_candidates_for_kb
        pool = MagicMock()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.get_auto_discovery_config = AsyncMock(return_value=None)
                result = await enqueue_next_candidates_for_kb(pool, "personal")
        assert result["skipped"] == "disabled"

    @pytest.mark.asyncio
    async def test_budget_exhausted(self):
        from app.auto_discovery import enqueue_next_candidates_for_kb
        pool = MagicMock()
        pool.set = AsyncMock()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.get_auto_discovery_config = AsyncMock(return_value={
                    "enabled": True, "daily_budget": 10, "max_per_hour": 5,
                })
                mock_db.serper_calls_today = AsyncMock(return_value=10)
                result = await enqueue_next_candidates_for_kb(pool, "personal")
        assert result["skipped"] == "budget"

    @pytest.mark.asyncio
    async def test_no_candidates(self):
        from app.auto_discovery import enqueue_next_candidates_for_kb
        pool = MagicMock()
        pool.set = AsyncMock()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.get_auto_discovery_config = AsyncMock(return_value={
                    "enabled": True, "daily_budget": 100, "max_per_hour": 5,
                })
                mock_db.serper_calls_today = AsyncMock(return_value=0)
                mock_db.get_pending_candidates = AsyncMock(return_value=[])
                result = await enqueue_next_candidates_for_kb(pool, "personal")
        assert result["enqueued"] == 0
        assert result["reason"] == "no_candidates"

    @pytest.mark.asyncio
    async def test_enqueues_candidates(self):
        from app.auto_discovery import enqueue_next_candidates_for_kb
        pool = MagicMock()
        pool.set = AsyncMock()
        pool.enqueue_job = AsyncMock()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.get_auto_discovery_config = AsyncMock(return_value={
                    "enabled": True, "daily_budget": 100, "max_per_hour": 5,
                })
                mock_db.serper_calls_today = AsyncMock(return_value=0)
                mock_db.get_pending_candidates = AsyncMock(return_value=[
                    {"id": 1, "topic": "Rust async patterns"},
                    {"id": 2, "topic": "Go generics"},
                ])
                mock_db.check_cooldown = AsyncMock(return_value=False)
                mock_db.create_job = AsyncMock(return_value=42)
                mock_db.mark_candidate_enqueued = AsyncMock()
                with patch("app.auto_discovery.find_related_article", return_value=None):
                    result = await enqueue_next_candidates_for_kb(pool, "personal")
        assert result["enqueued"] == 2

    @pytest.mark.asyncio
    async def test_skips_cooldown_and_existing(self):
        from app.auto_discovery import enqueue_next_candidates_for_kb
        pool = MagicMock()
        pool.set = AsyncMock()
        pool.enqueue_job = AsyncMock()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            with patch("app.auto_discovery.db") as mock_db:
                mock_db.get_auto_discovery_config = AsyncMock(return_value={
                    "enabled": True, "daily_budget": 100, "max_per_hour": 5,
                })
                mock_db.serper_calls_today = AsyncMock(return_value=0)
                mock_db.get_pending_candidates = AsyncMock(return_value=[
                    {"id": 1, "topic": "Cooldown topic"},
                    {"id": 2, "topic": "Existing topic"},
                ])
                mock_db.check_cooldown = AsyncMock(side_effect=[True, False])
                mock_db.mark_candidate_skipped = AsyncMock()
                with patch("app.auto_discovery.find_related_article") as mock_find:
                    mock_find.return_value = {"slug": "existing-topic"}
                    mock_db.mark_candidate_skipped = AsyncMock()
                    result = await enqueue_next_candidates_for_kb(pool, "personal")
        assert result["skipped_cooldown"] == 1
        assert result["skipped_exists"] == 1


# ---------------------------------------------------------------------------
# get_status_for_kb
# ---------------------------------------------------------------------------


class TestGetStatusForKb:
    @pytest.mark.asyncio
    async def test_returns_status(self):
        from app.auto_discovery import get_status_for_kb
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value={
                "enabled": True, "daily_budget": 50, "max_per_hour": 3,
                "strategy": "hybrid", "seed_topics": '["rust"]', "llm_sample": 5,
            })
            mock_db.serper_calls_today = AsyncMock(return_value=10)
            mock_db.count_pending_candidates = AsyncMock(return_value=5)
            result = await get_status_for_kb("personal")
        assert result["enabled"] is True
        assert result["used_today"] == 10
        assert result["pending_candidates"] == 5

    @pytest.mark.asyncio
    async def test_with_arq_pool_last_run(self):
        from app.auto_discovery import get_status_for_kb
        pool = MagicMock()
        pool.get = AsyncMock(return_value='{"enqueued": 2, "at": "2026-04-12T00:00:00"}')
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value={
                "enabled": True, "daily_budget": 50, "max_per_hour": 3,
                "strategy": "hybrid", "seed_topics": None, "llm_sample": 5,
            })
            mock_db.serper_calls_today = AsyncMock(return_value=0)
            mock_db.count_pending_candidates = AsyncMock(return_value=0)
            result = await get_status_for_kb("personal", arq_pool=pool)
        assert result["last_run"]["enqueued"] == 2

    @pytest.mark.asyncio
    async def test_no_config(self):
        from app.auto_discovery import get_status_for_kb
        with patch("app.auto_discovery.db") as mock_db:
            mock_db.get_auto_discovery_config = AsyncMock(return_value=None)
            mock_db.serper_calls_today = AsyncMock(return_value=0)
            mock_db.count_pending_candidates = AsyncMock(return_value=0)
            result = await get_status_for_kb("personal")
        assert result["enabled"] is False
