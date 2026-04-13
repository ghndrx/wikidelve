"""Unit tests for app.palace.classify_article and palace infrastructure."""

import time
from collections import defaultdict
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.palace import (
    classify_article,
    HALL_TYPES,
    HALL_COLORS,
    HALL_LABELS,
    HALL_RULES,
    invalidate_palace_cache,
    _palace_cache,
    _PALACE_CACHE_TTL,
    generate_palace_map,
    search_via_palace,
    classify_all_articles,
)


class TestClassifyArticle:
    def test_troubleshooting_by_title(self):
        hall, conf = classify_article(
            "Debug kubelet crash loop",
            ["debug"],
            "## Error\n\nTraceback (most recent call last):\n...\nroot cause: bad config",
        )
        assert hall == "troubleshooting"
        assert 0 < conf <= 1

    def test_how_to_by_body_markers(self):
        body = (
            "## Setup\n\n"
            "```bash\napt install redis-server\n```\n\n"
            "Step 1 — first do this, then that.\n"
        )
        hall, _ = classify_article("Install Redis", ["setup"], body)
        assert hall == "how-to"

    def test_comparison_by_title(self):
        hall, _ = classify_article(
            "Postgres vs MySQL for OLTP",
            [],
            "| Feature | Postgres | MySQL |\n|---|---|---|\npros and cons below",
        )
        assert hall == "comparison"

    def test_architecture_by_body(self):
        body = (
            "## Design\n\nThe service is composed of multiple components using a distributed, "
            "event-driven architecture with a service mesh."
        )
        hall, _ = classify_article("System overview", ["architecture"], body)
        assert hall == "architecture"

    def test_release_notes_by_title(self):
        hall, _ = classify_article(
            "v2.0 Release notes",
            ["changelog"],
            "## Breaking changes\n\n- removed deprecated field\n",
        )
        assert hall == "release-notes"

    def test_deep_dive_long_article_fallback(self):
        body = (
            "## Part 1\nlots of detail...\n"
            "## Part 2\nmore detail...\n"
            "## Part 3\neven more...\n"
            "## Part 4\nfinal part...\n"
        ) + ("word " * 1500)
        hall, _ = classify_article("Deep internals of foo", [], body, word_count=1800)
        assert hall == "deep-dive"

    def test_default_fallback_is_reference(self):
        hall, _ = classify_article("Random topic", [], "plain body")
        assert hall == "reference"

    def test_confidence_in_unit_range(self):
        for title in ("Install Redis", "Postgres vs MySQL", "Random topic"):
            _, conf = classify_article(title, [], "")
            assert 0.0 <= conf <= 1.0

    def test_returned_hall_is_valid(self):
        hall, _ = classify_article("Anything", [], "")
        assert hall in HALL_TYPES

    # --- Additional coverage ---

    def test_integration_by_title(self):
        hall, _ = classify_article(
            "Webhook Integration with Slack",
            ["integration"],
            "webhook trigger middleware adapter",
        )
        assert hall == "integration"

    def test_reference_by_title(self):
        hall, _ = classify_article(
            "API Reference for Auth Service",
            ["reference", "api"],
            "## API\n\n## Methods\n\n## Parameters\n",
        )
        assert hall == "reference"

    def test_comparison_tags_only(self):
        """Tags alone can classify as comparison."""
        hall, _ = classify_article("Some article", ["vs", "alternative"], "")
        assert hall == "comparison"

    def test_body_pattern_caps_at_three_hits(self):
        """Body pattern match count is capped at 3 per pattern."""
        body = "error: x\nerror: y\nerror: z\nerror: w\nerror: v\n"
        hall1, conf1 = classify_article("title", [], body)
        # 5 hits on "error:" but capped at 3 => 9 points from that pattern
        body2 = "error: x\nerror: y\nerror: z\n"
        hall2, conf2 = classify_article("title", [], body2)
        # Same score because of cap
        assert conf1 == conf2

    def test_empty_tags_handled(self):
        """None/empty tags shouldn't crash."""
        hall, conf = classify_article("Install Docker", None, "```bash\napt install docker\n```")
        assert hall in HALL_TYPES
        assert 0.0 <= conf <= 1.0

    def test_body_truncated_to_10k(self):
        """Only first 10k chars of body are scanned."""
        # Put the trigger word beyond 10k
        body = "x " * 6000 + "error: something"
        hall, _ = classify_article("Neutral title", [], body)
        # The "error:" is at position ~12000, beyond the 10k cutoff
        assert hall == "reference"  # falls back since trigger is beyond scan range

    def test_header_patterns_contribute(self):
        body = "## Common Issues\n\nSome text\n## Fix\n\nMore text\n"
        hall, _ = classify_article("Guide", [], body)
        assert hall == "troubleshooting"

    def test_deep_dive_not_triggered_for_short_articles(self):
        """Deep-dive bonus only kicks in for articles >1200 words with >=4 h2s."""
        body = "## A\n## B\n## C\n## D\n"
        hall, _ = classify_article("Neutral", [], body, word_count=500)
        # Not enough words for the deep-dive bonus
        assert hall != "deep-dive" or hall == "reference"

    def test_all_hall_types_have_colors_and_labels(self):
        for ht in HALL_TYPES:
            assert ht in HALL_COLORS, f"Missing color for {ht}"
            assert ht in HALL_LABELS, f"Missing label for {ht}"
            assert ht in HALL_RULES, f"Missing rules for {ht}"


class TestInvalidatePalaceCache:
    def test_invalidate_specific_kb(self):
        _palace_cache["test-kb"] = (time.time(), {"data": 1})
        _palace_cache["other-kb"] = (time.time(), {"data": 2})
        invalidate_palace_cache("test-kb")
        assert "test-kb" not in _palace_cache
        assert "other-kb" in _palace_cache
        _palace_cache.clear()

    def test_invalidate_all(self):
        _palace_cache["a"] = (time.time(), {})
        _palace_cache["b"] = (time.time(), {})
        invalidate_palace_cache(None)
        assert len(_palace_cache) == 0

    def test_invalidate_nonexistent_kb_no_error(self):
        _palace_cache.clear()
        invalidate_palace_cache("nonexistent")  # should not raise


class TestGeneratePalaceMap:
    async def test_returns_hall_metadata(self):
        with patch("app.palace.storage") as mock_storage:
            mock_storage.list_kbs.return_value = []
            result = await generate_palace_map()
        assert "hall_types" in result
        assert "hall_colors" in result
        assert "hall_labels" in result
        assert result["hall_types"] == HALL_TYPES

    async def test_uses_cache_when_fresh(self):
        _palace_cache["cached-kb"] = (time.time(), {"cached": True})
        try:
            result = await generate_palace_map("cached-kb")
            assert result["wings"]["cached-kb"] == {"cached": True}
        finally:
            _palace_cache.clear()

    async def test_regenerates_when_cache_expired(self):
        _palace_cache["old-kb"] = (time.time() - _PALACE_CACHE_TTL - 10, {"old": True})
        mock_wing = {"article_count": 0, "classified_count": 0, "halls": {}, "rooms": {}}
        with patch("app.palace._generate_wing", new=AsyncMock(return_value=mock_wing)):
            result = await generate_palace_map("old-kb")
        assert result["wings"]["old-kb"]["article_count"] == 0
        _palace_cache.clear()


class TestClassifyAllArticles:
    async def test_classifies_and_stores(self):
        mock_articles = [
            {"slug": "install-redis", "title": "Install Redis", "tags": ["setup"],
             "raw_markdown": "```bash\napt install redis\n```", "body": "", "word_count": 100},
            {"slug": "debug-crash", "title": "Debug crash loop", "tags": ["debug"],
             "raw_markdown": "error: traceback", "body": "", "word_count": 50},
        ]
        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.upsert_classification = AsyncMock()
            result = await classify_all_articles("personal")
        assert result["classified"] == 2
        assert result["kb"] == "personal"
        assert mock_db.upsert_classification.call_count == 2


class TestSearchViaPalace:
    async def test_scores_articles_by_hall_and_room(self):
        mock_rooms = [
            {"id": 1, "name": "kubernetes containers"},
            {"id": 2, "name": "database storage"},
        ]
        mock_classifications = [
            {"slug": "k8s-setup"},
        ]
        mock_members = [
            [{"slug": "k8s-setup", "relevance": 1.0}],
            [{"slug": "pg-setup", "relevance": 0.5}],
        ]
        with patch("app.palace.db") as mock_db:
            mock_db.get_rooms = AsyncMock(return_value=mock_rooms)
            mock_db.get_classifications_by_hall = AsyncMock(return_value=mock_classifications)
            mock_db.get_room_members = AsyncMock(side_effect=mock_members)
            results = await search_via_palace("kubernetes setup", "personal")

        slugs = {r["slug"] for r in results}
        assert "k8s-setup" in slugs

    async def test_empty_query_returns_results(self):
        with patch("app.palace.db") as mock_db:
            mock_db.get_rooms = AsyncMock(return_value=[])
            mock_db.get_classifications_by_hall = AsyncMock(return_value=[])
            results = await search_via_palace("", "personal")
        assert results == []

    async def test_respects_limit(self):
        mock_classifications = [{"slug": f"art-{i}"} for i in range(20)]
        with patch("app.palace.db") as mock_db:
            mock_db.get_rooms = AsyncMock(return_value=[])
            mock_db.get_classifications_by_hall = AsyncMock(return_value=mock_classifications)
            results = await search_via_palace("test", "personal", limit=5)
        assert len(results) <= 5
