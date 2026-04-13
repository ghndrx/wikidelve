"""Unit tests for app.embeddings — text embedding, chunking, search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import embeddings


# ---------------------------------------------------------------------------
# _chunk_text (pure helper)
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert embeddings._chunk_text("short text", max_chars=100) == ["short text"]

    def test_splits_on_paragraph_boundaries(self):
        text = "para one\n\npara two\n\npara three"
        chunks = embeddings._chunk_text(text, max_chars=20)
        assert len(chunks) >= 2
        assert all(len(c) <= 20 for c in chunks)

    def test_huge_paragraph_hard_splits(self):
        text = "x" * 200
        chunks = embeddings._chunk_text(text, max_chars=50)
        assert len(chunks) >= 1

    def test_empty_text(self):
        assert embeddings._chunk_text("", max_chars=100) == [""]


# ---------------------------------------------------------------------------
# _normalize_vector + _dot_product (math helpers)
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_normalize_unit_vector(self):
        vec = embeddings._normalize_vector([3.0, 4.0])
        assert abs(vec[0] - 0.6) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6

    def test_normalize_zero_vector_unchanged(self):
        vec = embeddings._normalize_vector([0.0, 0.0, 0.0])
        assert vec == [0.0, 0.0, 0.0]

    def test_dot_product_orthogonal(self):
        assert embeddings._dot_product([1, 0], [0, 1]) == 0.0

    def test_dot_product_parallel(self):
        assert embeddings._dot_product([1, 0], [1, 0]) == 1.0


# ---------------------------------------------------------------------------
# embed_text / embed_texts (mock llm_embed)
# ---------------------------------------------------------------------------


class TestEmbedText:
    @pytest.mark.asyncio
    async def test_embed_text_returns_vector(self, monkeypatch):
        fake_vec = [0.1] * 1024
        monkeypatch.setattr(
            embeddings, "llm_embed",
            AsyncMock(return_value=[fake_vec]),
        )
        result = await embeddings.embed_text("hello")
        assert result == fake_vec

    @pytest.mark.asyncio
    async def test_embed_text_truncates_long_input(self, monkeypatch):
        capture = {}

        async def spy(texts, embed_type):
            capture["texts"] = texts
            return [[0.1] * 1024]

        monkeypatch.setattr(embeddings, "llm_embed", spy)
        await embeddings.embed_text("x" * 20000)
        assert len(capture["texts"][0]) <= embeddings.MAX_CHUNK_CHARS

    @pytest.mark.asyncio
    async def test_embed_texts_empty_returns_empty(self, monkeypatch):
        result = await embeddings.embed_texts([])
        assert result == []


# ---------------------------------------------------------------------------
# embed_article (full pipeline mock)
# ---------------------------------------------------------------------------


class TestEmbedArticle:
    @pytest.mark.asyncio
    async def test_article_not_found_returns_error(self, monkeypatch):
        monkeypatch.setattr(embeddings, "get_article", lambda kb, slug: None)
        result = await embeddings.embed_article("personal", "ghost")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_happy_path_single_chunk(self, monkeypatch):
        fake_article = {
            "title": "Test",
            "summary": "summary",
            "raw_markdown": "body text",
        }
        monkeypatch.setattr(embeddings, "get_article", lambda kb, slug: fake_article)
        monkeypatch.setattr(
            embeddings, "llm_embed",
            AsyncMock(return_value=[[0.5] * 1024]),
        )
        mock_store = MagicMock()
        mock_store.upsert = AsyncMock()
        monkeypatch.setattr(embeddings, "get_vector_store", lambda: mock_store)

        result = await embeddings.embed_article("personal", "test-slug")
        assert result["status"] == "ok"
        assert result["chunks"] == 1
        mock_store.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_chunk_mean_pools(self, monkeypatch):
        long_body = ("x" * 5000 + "\n\n") * 5  # ~25KB, forces multiple 8KB chunks
        fake_article = {
            "title": "Long",
            "summary": "",
            "raw_markdown": long_body,
        }
        monkeypatch.setattr(embeddings, "get_article", lambda kb, slug: fake_article)

        call_count = {"n": 0}

        async def fake_embed(texts, embed_type="db"):
            call_count["n"] += 1
            return [[float(call_count["n"])] * 4 for _ in texts]

        monkeypatch.setattr(embeddings, "llm_embed", fake_embed)
        mock_store = MagicMock()
        mock_store.upsert = AsyncMock()
        monkeypatch.setattr(embeddings, "get_vector_store", lambda: mock_store)

        result = await embeddings.embed_article("personal", "long-article")
        assert result["status"] == "ok"
        assert result["chunks"] > 1
        # The vector stored should be the mean, then normalized
        stored_vec = mock_store.upsert.call_args[0][2]
        assert len(stored_vec) == 4


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------


class TestSearchSimilar:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        assert await embeddings.search_similar("") == []
        assert await embeddings.search_similar("   ") == []

    @pytest.mark.asyncio
    async def test_routes_through_vector_store(self, monkeypatch):
        monkeypatch.setattr(
            embeddings, "llm_embed",
            AsyncMock(return_value=[[0.1] * 1024]),
        )
        fake_results = [{"slug": "a", "score": 0.9}]
        mock_store = MagicMock()
        mock_store.query = AsyncMock(return_value=fake_results)
        monkeypatch.setattr(embeddings, "get_vector_store", lambda: mock_store)

        results = await embeddings.search_similar("test query", kb_name="personal", limit=5)
        assert results == fake_results
        mock_store.query.assert_awaited_once_with("personal", pytest.approx([0.1] * 1024, abs=0.01), top_k=5)
