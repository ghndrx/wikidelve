"""Extended unit tests for app.vector_store — vector operations, search, upsert, error handling."""

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.vector_store import (
    S3VectorStore,
    SQLiteVectorStore,
    _dot,
    _normalize,
    get_vector_store,
    set_vector_store,
)


# ── Helper function tests ────────────────────────────────────────────────────


class TestNormalize:
    def test_unit_vector_unchanged(self):
        vec = [1.0, 0.0, 0.0]
        result = _normalize(vec)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_normalizes_to_unit_length(self):
        vec = [3.0, 4.0]
        result = _normalize(vec)
        mag = math.sqrt(sum(v * v for v in result))
        assert mag == pytest.approx(1.0)
        assert result == pytest.approx([0.6, 0.8])

    def test_zero_vector_returns_zero(self):
        vec = [0.0, 0.0, 0.0]
        result = _normalize(vec)
        assert result == [0.0, 0.0, 0.0]

    def test_negative_values(self):
        vec = [-3.0, 4.0]
        result = _normalize(vec)
        mag = math.sqrt(sum(v * v for v in result))
        assert mag == pytest.approx(1.0)

    def test_single_element(self):
        vec = [5.0]
        result = _normalize(vec)
        assert result == pytest.approx([1.0])


class TestDot:
    def test_orthogonal_vectors(self):
        assert _dot([1, 0], [0, 1]) == 0.0

    def test_parallel_vectors(self):
        assert _dot([1, 0], [1, 0]) == 1.0

    def test_antiparallel_vectors(self):
        assert _dot([1, 0], [-1, 0]) == -1.0

    def test_general_case(self):
        assert _dot([1, 2, 3], [4, 5, 6]) == 32.0

    def test_empty_vectors(self):
        assert _dot([], []) == 0.0


# ── SQLiteVectorStore tests ──────────────────────────────────────────────────


class TestSQLiteVectorStore:
    @pytest.fixture
    def store(self):
        return SQLiteVectorStore()

    @pytest.fixture
    def mock_db_path(self, tmp_path):
        db_path = tmp_path / "test.db"
        return db_path

    @patch("app.vector_store.DB_PATH")
    async def test_upsert_and_query(self, mock_db_path, tmp_path):
        """Test the full upsert-then-query cycle using a real SQLite DB."""
        import aiosqlite

        db_path = tmp_path / "test_vectors.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        # Create table
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL,
                    kb TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        vec = [1.0, 0.0, 0.0]
        await store.upsert("testkb", "test-article", vec)

        results = await store.query("testkb", [1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0]["slug"] == "test-article"
        assert results[0]["kb"] == "testkb"
        assert results[0]["score"] == pytest.approx(1.0)

    @patch("app.vector_store.DB_PATH")
    async def test_upsert_normalizes_vector(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_norm.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        await store.upsert("kb", "s", [3.0, 4.0])

        async with aiosqlite.connect(str(db_path)) as db:
            cur = await db.execute("SELECT embedding FROM article_embeddings WHERE slug='s'")
            row = await cur.fetchone()
            stored = json.loads(row[0])
            mag = math.sqrt(sum(v * v for v in stored))
            assert mag == pytest.approx(1.0)

    @patch("app.vector_store.DB_PATH")
    async def test_delete(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_del.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        await store.upsert("kb", "to-delete", [1.0, 0.0])
        await store.delete("kb", "to-delete")

        results = await store.query("kb", [1.0, 0.0])
        assert len(results) == 0

    @patch("app.vector_store.DB_PATH")
    async def test_query_empty_db(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_empty.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        results = await store.query("kb", [1.0, 0.0])
        assert results == []

    @patch("app.vector_store.DB_PATH")
    async def test_query_cross_kb(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_cross.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        await store.upsert("kb1", "a1", [1.0, 0.0])
        await store.upsert("kb2", "a2", [0.0, 1.0])

        # kb=None should return both
        results = await store.query(None, [1.0, 0.0])
        assert len(results) == 2

    @patch("app.vector_store.DB_PATH")
    async def test_query_top_k(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_topk.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        for i in range(5):
            await store.upsert("kb", f"art-{i}", [float(i), 1.0])

        results = await store.query("kb", [4.0, 1.0], top_k=2)
        assert len(results) == 2
        # Most similar should be first
        assert results[0]["score"] >= results[1]["score"]

    @patch("app.vector_store.DB_PATH")
    async def test_list_all(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_list.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        await store.upsert("kb1", "a", [1.0])
        await store.upsert("kb2", "b", [1.0])

        all_items = await store.list_all()
        assert len(all_items) == 2

        kb1_items = await store.list_all("kb1")
        assert len(kb1_items) == 1
        assert kb1_items[0]["slug"] == "a"

    @patch("app.vector_store.DB_PATH")
    async def test_upsert_overwrites(self, mock_db_path, tmp_path):
        import aiosqlite

        db_path = tmp_path / "test_overwrite.db"
        mock_db_path.__str__ = lambda self: str(db_path)
        mock_db_path.parent = db_path.parent

        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS article_embeddings (
                    slug TEXT NOT NULL, kb TEXT NOT NULL,
                    embedding TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (slug, kb)
                )
            """)
            await db.commit()

        store = SQLiteVectorStore()
        await store.upsert("kb", "s", [1.0, 0.0])
        await store.upsert("kb", "s", [0.0, 1.0])

        results = await store.query("kb", [0.0, 1.0])
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(1.0)

    async def test_ensure_index_noop(self):
        store = SQLiteVectorStore()
        result = await store.ensure_index("any-kb")
        assert result is None


# ── S3VectorStore tests ──────────────────────────────────────────────────────


class TestS3VectorStore:
    def test_init_requires_bucket(self):
        with pytest.raises(ValueError, match="S3_VECTORS_BUCKET"):
            S3VectorStore("")

    def test_index_name_equals_kb(self):
        store = S3VectorStore("my-bucket")
        assert store._index_name("personal") == "personal"

    @patch("app.vector_store._s3vectors_client")
    async def test_ensure_index_success(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.get_index.return_value = {"indexName": "kb1"}
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        await store.ensure_index("kb1")
        mock_client.get_index.assert_called_once_with(
            vectorBucketName="bucket", indexName="kb1"
        )

    @patch("app.vector_store._s3vectors_client")
    async def test_ensure_index_missing_raises(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.get_index.side_effect = Exception("not found")
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        with pytest.raises(RuntimeError, match="missing"):
            await store.ensure_index("missing-kb")

    @patch("app.vector_store._s3vectors_client")
    async def test_upsert(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        await store.upsert("kb1", "my-slug", [3.0, 4.0])

        mock_client.put_vectors.assert_called_once()
        call_kwargs = mock_client.put_vectors.call_args[1]
        assert call_kwargs["vectorBucketName"] == "bucket"
        assert call_kwargs["indexName"] == "kb1"
        vec_data = call_kwargs["vectors"][0]
        assert vec_data["key"] == "my-slug"
        # Verify the vector was normalized
        f32 = vec_data["data"]["float32"]
        mag = math.sqrt(sum(v * v for v in f32))
        assert mag == pytest.approx(1.0)

    @patch("app.vector_store._s3vectors_client")
    async def test_delete(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        await store.delete("kb1", "slug-to-delete")

        mock_client.delete_vectors.assert_called_once_with(
            vectorBucketName="bucket",
            indexName="kb1",
            keys=["slug-to-delete"],
        )

    @patch("app.vector_store._s3vectors_client")
    async def test_query_single_kb(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "vectors": [
                {
                    "key": "art1",
                    "distance": 0.1,
                    "metadata": {"slug": "art1", "kb": "kb1"},
                },
                {
                    "key": "art2",
                    "distance": 0.3,
                    "metadata": {"slug": "art2", "kb": "kb1"},
                },
            ]
        }
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.query("kb1", [1.0, 0.0], top_k=5)

        assert len(results) == 2
        assert results[0]["slug"] == "art1"
        assert results[0]["score"] == pytest.approx(0.9)
        assert results[1]["score"] == pytest.approx(0.7)
        # Sorted desc by score
        assert results[0]["score"] >= results[1]["score"]

    @patch("app.vector_store._s3vectors_client")
    async def test_query_cross_kb(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = {
            "indexes": [{"indexName": "kb1"}, {"indexName": "kb2"}]
        }
        mock_client.query_vectors.side_effect = [
            {"vectors": [{"key": "a", "distance": 0.1, "metadata": {"slug": "a", "kb": "kb1"}}]},
            {"vectors": [{"key": "b", "distance": 0.2, "metadata": {"slug": "b", "kb": "kb2"}}]},
        ]
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.query(None, [1.0, 0.0], top_k=10)

        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    @patch("app.vector_store._s3vectors_client")
    async def test_query_handles_failure(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.query_vectors.side_effect = Exception("network error")
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.query("kb1", [1.0, 0.0])
        assert results == []

    @patch("app.vector_store._s3vectors_client")
    async def test_query_missing_metadata(self, mock_client_fn):
        """When metadata is missing, falls back to key/index name."""
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "vectors": [
                {"key": "art1", "distance": 0.1, "metadata": None},
            ]
        }
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.query("kb1", [1.0, 0.0])
        assert results[0]["slug"] == "art1"
        assert results[0]["kb"] == "kb1"

    @patch("app.vector_store._s3vectors_client")
    async def test_query_missing_distance(self, mock_client_fn):
        """When distance is None, score defaults to 0."""
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "vectors": [
                {"key": "art1", "distance": None, "metadata": {"slug": "art1", "kb": "kb1"}},
            ]
        }
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.query("kb1", [1.0, 0.0])
        assert results[0]["score"] == 0.0

    @patch("app.vector_store._s3vectors_client")
    async def test_list_all_single_kb(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.list_vectors.return_value = {
            "vectors": [
                {"key": "a", "metadata": {"slug": "a", "kb": "kb1"}},
                {"key": "b", "metadata": {"slug": "b", "kb": "kb1"}},
            ]
        }
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.list_all("kb1")
        assert len(results) == 2

    @patch("app.vector_store._s3vectors_client")
    async def test_list_all_paginated(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.list_vectors.side_effect = [
            {
                "vectors": [{"key": "a", "metadata": {"slug": "a", "kb": "kb1"}}],
                "nextToken": "page2",
            },
            {
                "vectors": [{"key": "b", "metadata": {"slug": "b", "kb": "kb1"}}],
            },
        ]
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.list_all("kb1")
        assert len(results) == 2
        assert mock_client.list_vectors.call_count == 2

    @patch("app.vector_store._s3vectors_client")
    async def test_list_all_cross_kb(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = {
            "indexes": [{"indexName": "kb1"}, {"indexName": "kb2"}]
        }
        mock_client.list_vectors.side_effect = [
            {"vectors": [{"key": "a", "metadata": {"slug": "a", "kb": "kb1"}}]},
            {"vectors": [{"key": "b", "metadata": {"slug": "b", "kb": "kb2"}}]},
        ]
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.list_all(None)
        assert len(results) == 2

    @patch("app.vector_store._s3vectors_client")
    async def test_list_all_handles_error(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.list_vectors.side_effect = Exception("access denied")
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        results = await store.list_all("kb1")
        assert results == []

    @patch("app.vector_store._s3vectors_client")
    async def test_list_indexes_handles_error(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client.list_indexes.side_effect = Exception("service error")
        mock_client_fn.return_value = mock_client

        store = S3VectorStore("bucket")
        indexes = await store._list_indexes()
        assert indexes == []


# ── Module singleton tests ───────────────────────────────────────────────────


class TestModuleSingleton:
    def test_set_and_get_vector_store(self):
        mock_store = MagicMock()
        set_vector_store(mock_store)
        assert get_vector_store() is mock_store
        # Reset
        set_vector_store(None)

    @patch("app.vector_store._BACKEND", "sqlite")
    def test_default_sqlite(self):
        from app.vector_store import _build_default
        store = _build_default()
        assert isinstance(store, SQLiteVectorStore)

    @patch("app.vector_store._VECTORS_BUCKET", "my-bucket")
    @patch("app.vector_store._BACKEND", "s3")
    def test_default_s3(self):
        from app.vector_store import _build_default
        store = _build_default()
        assert isinstance(store, S3VectorStore)
        assert store.bucket == "my-bucket"

    @patch("app.vector_store._VECTORS_BUCKET", "")
    @patch("app.vector_store._BACKEND", "s3")
    def test_s3_without_bucket_raises(self):
        from app.vector_store import _build_default
        with pytest.raises(RuntimeError, match="S3_VECTORS_BUCKET"):
            _build_default()
