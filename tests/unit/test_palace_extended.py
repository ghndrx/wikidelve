"""Extended tests for app.palace — covering uncovered lines."""

import asyncio
import time
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.palace import (
    classify_all_articles,
    classify_article,
    cluster_rooms,
    _generate_wing,
    generate_palace_map,
    invalidate_palace_cache,
    search_via_palace,
    _palace_cache,
    _PALACE_CACHE_TTL,
    HALL_TYPES,
)


# ===========================================================================
# 1. classify_all_articles — line 203 (skip articles without slug)
# ===========================================================================

class TestClassifyAllArticlesExtended:

    async def test_skips_articles_without_slug(self):
        """Line 203: articles missing a slug are skipped."""
        mock_articles = [
            {"slug": "", "title": "No slug", "tags": [], "raw_markdown": "", "word_count": 0},
            {"slug": "valid-slug", "title": "Install Redis", "tags": ["setup"],
             "raw_markdown": "```bash\napt install redis\n```", "word_count": 100},
        ]
        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.upsert_classification = AsyncMock()
            result = await classify_all_articles("personal")
        assert result["classified"] == 1
        assert mock_db.upsert_classification.call_count == 1

    async def test_skips_articles_with_none_slug(self):
        """Edge case: slug is None (falsy)."""
        mock_articles = [
            {"slug": None, "title": "None slug", "tags": [], "raw_markdown": "", "word_count": 0},
        ]
        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.upsert_classification = AsyncMock()
            result = await classify_all_articles("test-kb")
        assert result["classified"] == 0
        mock_db.upsert_classification.assert_not_awaited()

    async def test_empty_articles_list(self):
        mock_articles = []
        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.upsert_classification = AsyncMock()
            result = await classify_all_articles("empty-kb")
        assert result["classified"] == 0
        assert result["halls"] == {}

    async def test_hall_counts_populated(self):
        mock_articles = [
            {"slug": "a", "title": "Debug crash", "tags": ["debug"],
             "raw_markdown": "error: traceback", "word_count": 50},
            {"slug": "b", "title": "Debug another crash", "tags": ["debug"],
             "raw_markdown": "error: traceback root cause", "word_count": 60},
            {"slug": "c", "title": "Install Redis", "tags": ["setup"],
             "raw_markdown": "```bash\napt install redis\n```", "word_count": 100},
        ]
        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.upsert_classification = AsyncMock()
            result = await classify_all_articles("test-kb")
        assert result["classified"] == 3
        assert sum(result["halls"].values()) == 3


# ===========================================================================
# 2. cluster_rooms — lines 220-290
# ===========================================================================

class TestClusterRooms:

    def _make_mock_conn(self, entities, entity_articles, all_graph_slugs):
        """Build a mock DB connection with execute responses."""
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()

        call_count = {"n": 0}

        async def mock_execute(sql, params=None):
            cursor = AsyncMock()
            idx = call_count["n"]
            call_count["n"] += 1

            if "kg_entities" in sql and "GROUP BY" in sql:
                cursor.fetchall = AsyncMock(return_value=entities)
            elif "DISTINCT ke.article_slug" in sql and "source_entity_id" in sql:
                # Per-entity article lookup
                entity_id = params[0] if params else None
                slugs = entity_articles.get(entity_id, [])
                cursor.fetchall = AsyncMock(
                    return_value=[{"article_slug": s} for s in slugs]
                )
            elif "DISTINCT article_slug FROM kg_edges" in sql:
                cursor.fetchall = AsyncMock(
                    return_value=[{"article_slug": s} for s in all_graph_slugs]
                )
            else:
                cursor.fetchall = AsyncMock(return_value=[])
            return cursor

        mock_conn.execute = mock_execute
        return mock_conn

    async def test_creates_rooms_from_entities(self):
        entities = [
            {"id": 1, "name": "Kubernetes", "type": "technology", "article_count": 3},
        ]
        entity_articles = {1: ["k8s-setup", "k8s-deploy", "k8s-debug"]}
        all_graph_slugs = ["k8s-setup", "k8s-deploy", "k8s-debug"]

        mock_conn = self._make_mock_conn(entities, entity_articles, all_graph_slugs)

        with patch("app.palace.db") as mock_db:
            mock_db._get_db = AsyncMock(return_value=mock_conn)
            mock_db.clear_rooms = AsyncMock()
            mock_db.upsert_room = AsyncMock(return_value=1)
            mock_db.add_room_member = AsyncMock()

            result = await cluster_rooms("test-kb")

        assert result["kb"] == "test-kb"
        assert result["rooms_created"] >= 1
        assert result["memberships"] >= 3
        mock_db.clear_rooms.assert_awaited_once_with("test-kb")

    async def test_creates_misc_room_for_orphans(self):
        """Lines 279-284: orphan articles get a _misc room."""
        entities = [
            {"id": 1, "name": "Docker", "type": "tech", "article_count": 2},
        ]
        entity_articles = {1: ["docker-a", "docker-b"]}
        # orphan-c is in graph edges but not assigned to any entity
        all_graph_slugs = ["docker-a", "docker-b", "orphan-c"]

        mock_conn = self._make_mock_conn(entities, entity_articles, all_graph_slugs)

        with patch("app.palace.db") as mock_db:
            mock_db._get_db = AsyncMock(return_value=mock_conn)
            mock_db.clear_rooms = AsyncMock()
            mock_db.upsert_room = AsyncMock(side_effect=[10, 20])
            mock_db.add_room_member = AsyncMock()

            result = await cluster_rooms("test-kb")

        # Should create 2 rooms: one for Docker entity, one _misc for orphan
        assert result["rooms_created"] == 2
        # Docker room: 2 members + misc room: 1 orphan member
        assert result["memberships"] == 3

    async def test_skips_entities_with_no_articles(self):
        """Line 252-253: skip entities with empty article slugs."""
        entities = [
            {"id": 1, "name": "Empty", "type": "tech", "article_count": 2},
        ]
        entity_articles = {1: []}  # No articles returned
        all_graph_slugs = []

        mock_conn = self._make_mock_conn(entities, entity_articles, all_graph_slugs)

        with patch("app.palace.db") as mock_db:
            mock_db._get_db = AsyncMock(return_value=mock_conn)
            mock_db.clear_rooms = AsyncMock()
            mock_db.upsert_room = AsyncMock(return_value=1)
            mock_db.add_room_member = AsyncMock()

            result = await cluster_rooms("test-kb")

        assert result["rooms_created"] == 0
        assert result["memberships"] == 0

    async def test_skips_heavily_assigned_entities(self):
        """Lines 256-258: skip entity if >60% of articles already have 2+ rooms."""
        # Two entities share the same articles. The second entity should be
        # skipped because by the time we get to it all articles already have
        # assignments.
        entities = [
            {"id": 1, "name": "Alpha", "type": "tech", "article_count": 2},
            {"id": 2, "name": "Beta", "type": "tech", "article_count": 2},
            {"id": 3, "name": "Gamma", "type": "tech", "article_count": 2},
        ]
        # All three entities reference the same two articles
        entity_articles = {
            1: ["art-a", "art-b"],
            2: ["art-a", "art-b"],
            3: ["art-a", "art-b"],
        }
        all_graph_slugs = ["art-a", "art-b"]

        mock_conn = self._make_mock_conn(entities, entity_articles, all_graph_slugs)

        with patch("app.palace.db") as mock_db:
            mock_db._get_db = AsyncMock(return_value=mock_conn)
            mock_db.clear_rooms = AsyncMock()
            mock_db.upsert_room = AsyncMock(side_effect=list(range(1, 10)))
            mock_db.add_room_member = AsyncMock()

            result = await cluster_rooms("test-kb")

        # Entity 1 creates a room (0 prior assignments).
        # Entity 2 creates a room (1 prior assignment each -> 0% heavily assigned).
        # Entity 3 is skipped (both articles have 2 assignments -> 100% > 60%).
        assert result["rooms_created"] == 2

    async def test_relevance_decreases_with_assignments(self):
        """Lines 264-267: relevance = 1/(1+prior_assignments)."""
        entities = [
            {"id": 1, "name": "First", "type": "tech", "article_count": 1},
            {"id": 2, "name": "Second", "type": "tech", "article_count": 1},
        ]
        entity_articles = {1: ["art-x"], 2: ["art-x"]}
        all_graph_slugs = ["art-x"]

        mock_conn = self._make_mock_conn(entities, entity_articles, all_graph_slugs)

        with patch("app.palace.db") as mock_db:
            mock_db._get_db = AsyncMock(return_value=mock_conn)
            mock_db.clear_rooms = AsyncMock()
            mock_db.upsert_room = AsyncMock(side_effect=[10, 20])
            mock_db.add_room_member = AsyncMock()

            await cluster_rooms("test-kb")

        calls = mock_db.add_room_member.call_args_list
        # First assignment: relevance = 1.0 / (1+0) = 1.0
        assert calls[0].args == (10, "art-x", "test-kb", 1.0)
        # Second assignment: relevance = 1.0 / (1+1) = 0.5
        assert calls[1].args == (20, "art-x", "test-kb", 0.5)

    async def test_conn_closed_on_success(self):
        """Line 287: conn.close() called in finally block."""
        entities = []
        mock_conn = self._make_mock_conn(entities, {}, [])

        with patch("app.palace.db") as mock_db:
            mock_db._get_db = AsyncMock(return_value=mock_conn)
            mock_db.clear_rooms = AsyncMock()

            await cluster_rooms("test-kb")

        mock_conn.close.assert_awaited_once()


# ===========================================================================
# 3. _generate_wing — lines 310-356
# ===========================================================================

class TestGenerateWing:

    async def test_builds_wing_structure(self):
        mock_articles = [
            {"slug": "k8s-setup", "title": "K8s Setup"},
            {"slug": "pg-guide", "title": "Postgres Guide"},
        ]
        mock_classifications = [
            {"slug": "k8s-setup", "hall": "how-to", "confidence": 0.8},
            {"slug": "pg-guide", "hall": "reference", "confidence": 0.6},
        ]
        mock_rooms = [
            {"id": 1, "name": "kubernetes", "article_count": 1},
        ]
        mock_members = [
            [{"slug": "k8s-setup", "relevance": 1.0}],
        ]

        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.get_classifications = AsyncMock(return_value=mock_classifications)
            mock_db.get_rooms = AsyncMock(return_value=mock_rooms)
            mock_db.get_room_members = AsyncMock(side_effect=mock_members)

            result = await _generate_wing("test-kb")

        assert result["article_count"] == 2
        assert result["classified_count"] == 2
        assert "how-to" in result["halls"]
        assert result["halls"]["how-to"]["count"] == 1
        assert len(result["halls"]["how-to"]["top_articles"]) == 1
        assert "kubernetes" in result["rooms"]
        assert result["rooms"]["kubernetes"]["count"] == 1

    async def test_hall_distribution_in_rooms(self):
        """Lines 346-354: room hall_distribution built from slug_to_hall."""
        mock_articles = [
            {"slug": "a", "title": "A"},
            {"slug": "b", "title": "B"},
        ]
        mock_classifications = [
            {"slug": "a", "hall": "how-to", "confidence": 0.8},
            {"slug": "b", "hall": "how-to", "confidence": 0.7},
        ]
        mock_rooms = [
            {"id": 1, "name": "topic1", "article_count": 2},
        ]
        mock_members = [
            [{"slug": "a", "relevance": 1.0}, {"slug": "b", "relevance": 0.5}],
        ]

        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.get_classifications = AsyncMock(return_value=mock_classifications)
            mock_db.get_rooms = AsyncMock(return_value=mock_rooms)
            mock_db.get_room_members = AsyncMock(side_effect=mock_members)

            result = await _generate_wing("test-kb")

        assert result["rooms"]["topic1"]["hall_distribution"]["how-to"] == 2

    async def test_top_articles_capped_at_5(self):
        """Lines 332-337: only up to 5 top articles per hall."""
        mock_articles = [
            {"slug": f"art-{i}", "title": f"Article {i}"} for i in range(8)
        ]
        mock_classifications = [
            {"slug": f"art-{i}", "hall": "reference", "confidence": 0.5}
            for i in range(8)
        ]
        mock_rooms = []

        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.get_classifications = AsyncMock(return_value=mock_classifications)
            mock_db.get_rooms = AsyncMock(return_value=mock_rooms)
            mock_db.get_room_members = AsyncMock(return_value=[])

            result = await _generate_wing("test-kb")

        assert result["halls"]["reference"]["count"] == 8
        assert len(result["halls"]["reference"]["top_articles"]) == 5

    async def test_room_member_without_classification(self):
        """Lines 348-349: room member slug not in slug_to_hall is skipped in dist."""
        mock_articles = [{"slug": "unclassified", "title": "U"}]
        mock_classifications = []  # No classifications
        mock_rooms = [{"id": 1, "name": "room1", "article_count": 1}]
        mock_members = [[{"slug": "unclassified", "relevance": 1.0}]]

        with patch("app.wiki.get_articles", return_value=mock_articles), \
             patch("app.palace.db") as mock_db:
            mock_db.get_classifications = AsyncMock(return_value=mock_classifications)
            mock_db.get_rooms = AsyncMock(return_value=mock_rooms)
            mock_db.get_room_members = AsyncMock(side_effect=mock_members)

            result = await _generate_wing("test-kb")

        assert result["rooms"]["room1"]["hall_distribution"] == {}


# ===========================================================================
# 4. generate_palace_map — caching + miss paths
# ===========================================================================

class TestGeneratePalaceMapExtended:

    async def test_multiple_kbs_some_cached_some_miss(self):
        """Mixed cache hit/miss scenario."""
        _palace_cache.clear()
        _palace_cache["kb-a"] = (time.time(), {"cached": True})

        mock_wing = {"article_count": 0, "classified_count": 0, "halls": {}, "rooms": {}}
        with patch("app.palace.storage") as mock_storage, \
             patch("app.palace._generate_wing", new=AsyncMock(return_value=mock_wing)):
            mock_storage.list_kbs.return_value = ["kb-a", "kb-b"]
            result = await generate_palace_map()

        assert result["wings"]["kb-a"] == {"cached": True}
        assert result["wings"]["kb-b"]["article_count"] == 0
        _palace_cache.clear()

    async def test_no_kb_name_lists_all(self):
        """When kb_name is None, list_kbs is called."""
        _palace_cache.clear()
        with patch("app.palace.storage") as mock_storage:
            mock_storage.list_kbs.return_value = []
            result = await generate_palace_map(None)
        assert result["wings"] == {}
        assert result["hall_types"] == HALL_TYPES
        _palace_cache.clear()
