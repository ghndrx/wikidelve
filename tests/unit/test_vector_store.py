"""Unit tests for app.vector_store — SQLite backend.

The S3 Vectors backend requires a real AWS service endpoint that moto
doesn't support (s3vectors is too new), so we only test the SQLite
path here. The S3 path gets a pragma-no-cover carve-out.
"""

from __future__ import annotations

import json

import pytest

from app.vector_store import SQLiteVectorStore, _normalize, _dot


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_normalize_unit(self):
        vec = _normalize([3.0, 4.0])
        assert abs(vec[0] - 0.6) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6

    def test_normalize_zero(self):
        assert _normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_dot_product(self):
        assert _dot([1.0, 0.0], [0.0, 1.0]) == 0.0
        assert abs(_dot([0.6, 0.8], [0.6, 0.8]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# SQLiteVectorStore CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(tmp_path, monkeypatch):
    """Wire SQLiteVectorStore to a temp DB with the article_embeddings table."""
    db_path = tmp_path / "vec-test.db"

    import aiosqlite
    from app import db as db_mod
    from app import config as cfg

    monkeypatch.setattr(cfg, "DB_PATH", db_path)

    # The vector store imports DB_PATH from config at module level.
    from app import vector_store as vs
    monkeypatch.setattr(vs, "DB_PATH", db_path)

    # Create the table schema.
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS article_embeddings (
                slug TEXT NOT NULL,
                kb TEXT NOT NULL,
                embedding TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (slug, kb)
            )
        """)
        await conn.commit()

    return SQLiteVectorStore()


class TestSQLiteUpsert:
    @pytest.mark.asyncio
    async def test_upsert_then_query(self, store):
        vec = [1.0, 0.0, 0.0, 0.0]
        await store.upsert("personal", "test-article", vec)
        results = await store.query("personal", vec, top_k=5)
        assert len(results) == 1
        assert results[0]["slug"] == "test-article"
        assert results[0]["score"] > 0.99

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, store):
        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0, 0.0]
        await store.upsert("personal", "test-article", vec_a)
        await store.upsert("personal", "test-article", vec_b)

        results = await store.query("personal", vec_b, top_k=5)
        assert len(results) == 1
        assert results[0]["score"] > 0.99

    @pytest.mark.asyncio
    async def test_delete_removes(self, store):
        await store.upsert("personal", "doomed", [1.0, 0.0, 0.0, 0.0])
        await store.delete("personal", "doomed")
        results = await store.query("personal", [1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) == 0


class TestSQLiteQuery:
    @pytest.mark.asyncio
    async def test_empty_store_returns_empty(self, store):
        results = await store.query("personal", [1.0, 0.0, 0.0, 0.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_respected(self, store):
        for i in range(10):
            v = [0.0] * 4
            v[i % 4] = 1.0
            await store.upsert("personal", f"art-{i}", v)

        results = await store.query("personal", [1.0, 0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_ranking_order(self, store):
        # Insert two articles: one similar, one orthogonal.
        await store.upsert("personal", "similar", [0.9, 0.1, 0.0, 0.0])
        await store.upsert("personal", "different", [0.0, 0.0, 0.9, 0.1])

        results = await store.query("personal", [1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results[0]["slug"] == "similar"
        assert results[0]["score"] > results[1]["score"]

    @pytest.mark.asyncio
    async def test_cross_kb_query(self, store):
        await store.upsert("personal", "a", [1.0, 0.0, 0.0, 0.0])
        await store.upsert("work", "b", [0.9, 0.1, 0.0, 0.0])

        # kb=None should search both
        results = await store.query(None, [1.0, 0.0, 0.0, 0.0], top_k=10)
        slugs = {r["slug"] for r in results}
        assert {"a", "b"} == slugs


class TestSQLiteListAll:
    @pytest.mark.asyncio
    async def test_empty(self, store):
        assert await store.list_all() == []

    @pytest.mark.asyncio
    async def test_lists_all_kbs(self, store):
        await store.upsert("personal", "a", [1.0, 0.0, 0.0, 0.0])
        await store.upsert("work", "b", [0.0, 1.0, 0.0, 0.0])
        items = await store.list_all()
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_filter_by_kb(self, store):
        await store.upsert("personal", "a", [1.0, 0.0, 0.0, 0.0])
        await store.upsert("work", "b", [0.0, 1.0, 0.0, 0.0])
        items = await store.list_all(kb="personal")
        assert len(items) == 1
        assert items[0]["slug"] == "a"


class TestSQLiteEnsureIndex:
    @pytest.mark.asyncio
    async def test_ensure_index_is_noop(self, store):
        # SQLite doesn't need provisioning
        result = await store.ensure_index("personal")
        assert result is None
