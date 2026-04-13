"""Extended tests for app.knowledge_graph – covering graph query functions
(get_related_by_graph, get_entity_articles, get_graph_data) and SQLite
helpers (_get_kg_db, _upsert_entity, _upsert_edge) at lines 272-475."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import aiosqlite
import pytest

from app.knowledge_graph import (
    get_related_by_graph,
    get_entity_articles,
    get_graph_data,
    _get_kg_db,
    _upsert_entity,
    _upsert_edge,
)

# ---------------------------------------------------------------------------
# Schema used by the knowledge graph tables
# ---------------------------------------------------------------------------

KG_SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL UNIQUE,
    type           TEXT NOT NULL,
    aliases        TEXT,
    article_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id   INTEGER,
    target_entity_id   INTEGER,
    relationship       TEXT NOT NULL,
    article_slug       TEXT,
    kb                 TEXT,
    FOREIGN KEY (source_entity_id) REFERENCES kg_entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES kg_entities(id)
);
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db_path(tmp_path):
    """Return a temporary path for the knowledge graph SQLite DB."""
    return tmp_path / "kg_test.db"


@pytest.fixture()
def patch_db_path(tmp_db_path):
    """Patch DB_PATH so _get_kg_db uses the temp database."""
    with patch("app.knowledge_graph.DB_PATH", tmp_db_path):
        yield tmp_db_path


@pytest.fixture()
async def kg_db(patch_db_path):
    """Provide an initialised knowledge graph database (tables created)."""
    db = await _get_kg_db()
    await db.executescript(KG_SCHEMA)
    await db.commit()
    await db.close()
    return patch_db_path


# Removed sync run() helper - using async tests instead


# ---------------------------------------------------------------------------
# Helpers to seed data
# ---------------------------------------------------------------------------

async def _seed_entities_and_edges(db_path):
    """Seed a small graph:
    Python --built-with--> Django (article: django-intro)
    Python --uses--------> FastAPI (article: fastapi-guide)
    Django --alternative--> FastAPI (article: comparison)
    Redis  --integrates---> Django (article: django-intro)
    """
    with patch("app.knowledge_graph.DB_PATH", db_path):
        db = await _get_kg_db()
        await db.executescript(KG_SCHEMA)

        python_id = await _upsert_entity(db, "Python", "language")
        django_id = await _upsert_entity(db, "Django", "framework")
        fastapi_id = await _upsert_entity(db, "FastAPI", "framework")
        redis_id = await _upsert_entity(db, "Redis", "tool")

        await _upsert_edge(db, python_id, django_id, "built-with", "django-intro", "kb1")
        await _upsert_edge(db, python_id, fastapi_id, "uses", "fastapi-guide", "kb1")
        await _upsert_edge(db, django_id, fastapi_id, "alternative-to", "comparison", "kb1")
        await _upsert_edge(db, redis_id, django_id, "integrates", "django-intro", "kb1")

        await db.commit()
        await db.close()

        return {
            "python": python_id,
            "django": django_id,
            "fastapi": fastapi_id,
            "redis": redis_id,
        }


# ---------------------------------------------------------------------------
# _get_kg_db
# ---------------------------------------------------------------------------

class TestGetKgDb:
    async def test_returns_connection_with_row_factory(self, patch_db_path):
        async def _test():
            db = await _get_kg_db()
            try:
                assert db.row_factory is aiosqlite.Row
            finally:
                await db.close()
        await _test()

    async def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "kg.db"
        with patch("app.knowledge_graph.DB_PATH", deep_path):
            async def _test():
                db = await _get_kg_db()
                await db.close()
            await _test()
        assert deep_path.parent.exists()


# ---------------------------------------------------------------------------
# _upsert_entity
# ---------------------------------------------------------------------------

class TestUpsertEntity:
    async def test_inserts_new_entity(self, kg_db):
        async def _test():
            with patch("app.knowledge_graph.DB_PATH", kg_db):
                db = await _get_kg_db()
                try:
                    eid = await _upsert_entity(db, "Go", "language")
                    assert isinstance(eid, int)
                    assert eid >= 1
                    # Verify it was inserted
                    cursor = await db.execute("SELECT name, type FROM kg_entities WHERE id = ?", (eid,))
                    row = await cursor.fetchone()
                    assert row["name"] == "Go"
                    assert row["type"] == "language"
                finally:
                    await db.close()
        await _test()

    async def test_returns_existing_id_on_duplicate(self, kg_db):
        async def _test():
            with patch("app.knowledge_graph.DB_PATH", kg_db):
                db = await _get_kg_db()
                try:
                    id1 = await _upsert_entity(db, "Rust", "language")
                    id2 = await _upsert_entity(db, "Rust", "language")
                    assert id1 == id2
                finally:
                    await db.close()
        await _test()


# ---------------------------------------------------------------------------
# _upsert_edge
# ---------------------------------------------------------------------------

class TestUpsertEdge:
    async def test_inserts_new_edge(self, kg_db):
        async def _test():
            with patch("app.knowledge_graph.DB_PATH", kg_db):
                db = await _get_kg_db()
                try:
                    e1 = await _upsert_entity(db, "A", "tool")
                    e2 = await _upsert_entity(db, "B", "tool")
                    await _upsert_edge(db, e1, e2, "uses", "article-1", "kb1")
                    await db.commit()

                    cursor = await db.execute("SELECT COUNT(*) FROM kg_edges")
                    count = (await cursor.fetchone())[0]
                    assert count == 1
                finally:
                    await db.close()
        await _test()

    async def test_does_not_duplicate_edge(self, kg_db):
        async def _test():
            with patch("app.knowledge_graph.DB_PATH", kg_db):
                db = await _get_kg_db()
                try:
                    e1 = await _upsert_entity(db, "X", "tool")
                    e2 = await _upsert_entity(db, "Y", "tool")
                    await _upsert_edge(db, e1, e2, "uses", "art", "kb")
                    await _upsert_edge(db, e1, e2, "uses", "art", "kb")  # duplicate
                    await db.commit()

                    cursor = await db.execute("SELECT COUNT(*) FROM kg_edges")
                    count = (await cursor.fetchone())[0]
                    assert count == 1
                finally:
                    await db.close()
        await _test()

    async def test_different_article_creates_separate_edge(self, kg_db):
        async def _test():
            with patch("app.knowledge_graph.DB_PATH", kg_db):
                db = await _get_kg_db()
                try:
                    e1 = await _upsert_entity(db, "M", "tool")
                    e2 = await _upsert_entity(db, "N", "tool")
                    await _upsert_edge(db, e1, e2, "uses", "art-1", "kb")
                    await _upsert_edge(db, e1, e2, "uses", "art-2", "kb")
                    await db.commit()

                    cursor = await db.execute("SELECT COUNT(*) FROM kg_edges")
                    count = (await cursor.fetchone())[0]
                    assert count == 2
                finally:
                    await db.close()
        await _test()


# ---------------------------------------------------------------------------
# get_graph_data
# ---------------------------------------------------------------------------

class TestGetGraphData:
    async def test_empty_graph(self, kg_db):
        with patch("app.knowledge_graph.DB_PATH", kg_db):
            data = await get_graph_data()
        assert data == {"nodes": [], "edges": []}

    async def test_returns_nodes_and_edges(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            data = await get_graph_data()

        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 4

        node_names = {n["name"] for n in data["nodes"]}
        assert "Python" in node_names
        assert "Django" in node_names
        assert "FastAPI" in node_names
        assert "Redis" in node_names

        # Each node has the expected fields
        for node in data["nodes"]:
            assert "id" in node
            assert "name" in node
            assert "type" in node
            assert "article_count" in node

        # Each edge has the expected fields
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "relationship" in edge
            assert "article_slug" in edge
            assert "kb" in edge


# ---------------------------------------------------------------------------
# get_entity_articles
# ---------------------------------------------------------------------------

class TestGetEntityArticles:
    async def test_no_matching_entity(self, kg_db):
        with patch("app.knowledge_graph.DB_PATH", kg_db):
            result = await get_entity_articles("NonExistent")
        assert result == []

    async def test_finds_articles_for_entity(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            result = await get_entity_articles("Python")

        # Python appears in django-intro (built-with) and fastapi-guide (uses)
        slugs = {a["slug"] for a in result}
        assert "django-intro" in slugs
        assert "fastapi-guide" in slugs

        # Each article entry has relationships
        for article in result:
            assert "relationships" in article
            assert len(article["relationships"]) >= 1
            for rel in article["relationships"]:
                assert "source" in rel
                assert "target" in rel
                assert "relationship" in rel

    async def test_finds_articles_where_entity_is_target(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            # Django is a target in "built-with" from Python
            result = await get_entity_articles("Django")

        slugs = {a["slug"] for a in result}
        assert "django-intro" in slugs


# ---------------------------------------------------------------------------
# get_related_by_graph
# ---------------------------------------------------------------------------

class TestGetRelatedByGraph:
    async def test_no_edges_returns_empty(self, kg_db):
        with patch("app.knowledge_graph.DB_PATH", kg_db):
            result = await get_related_by_graph("no-article", "kb1")
        assert result == []

    async def test_finds_related_articles(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            # Starting from django-intro, should find fastapi-guide and comparison
            result = await get_related_by_graph("django-intro", "kb1", depth=2)

        slugs = {r["slug"] for r in result}
        # Should find at least the articles connected via shared entities
        assert len(slugs) >= 1
        # django-intro itself should NOT be in the results
        assert "django-intro" not in slugs

    async def test_results_have_score_and_connections(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            result = await get_related_by_graph("django-intro", "kb1", depth=2)

        for item in result:
            assert "slug" in item
            assert "kb" in item
            assert "score" in item
            assert "connections" in item
            assert "hop" in item
            assert item["score"] > 0

    async def test_results_sorted_by_score_descending(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            result = await get_related_by_graph("django-intro", "kb1", depth=2)

        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]["score"] >= result[i + 1]["score"]

    async def test_depth_1_limits_hops(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            result = await get_related_by_graph("django-intro", "kb1", depth=1)

        for item in result:
            assert item["hop"] == 1

    async def test_unknown_article_returns_empty(self, patch_db_path):
        await _seed_entities_and_edges(patch_db_path)
        with patch("app.knowledge_graph.DB_PATH", patch_db_path):
            result = await get_related_by_graph("nonexistent", "kb1", depth=2)
        assert result == []
