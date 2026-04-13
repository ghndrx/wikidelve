"""Extended unit tests for app/embeddings.py — targeting uncovered lines.

Covers:
  - embed_text: ValueError when no vectors returned (line 31)
  - embed_article: empty chunk_embeddings from embed_texts (line 100)
  - embed_all_articles: full pipeline including errors and rate limiting (lines 118-148)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import embeddings


# ===========================================================================
# embed_text — ValueError on empty vectors (line 31)
# ===========================================================================


class TestEmbedTextExtended:
    @pytest.mark.asyncio
    async def test_raises_when_no_vectors(self, monkeypatch):
        """embed_text should raise ValueError when llm_embed returns []."""
        monkeypatch.setattr(embeddings, "llm_embed", AsyncMock(return_value=[]))
        with pytest.raises(ValueError, match="No vectors returned"):
            await embeddings.embed_text("hello")

    @pytest.mark.asyncio
    async def test_raises_when_none_vectors(self, monkeypatch):
        """embed_text should raise ValueError when llm_embed returns None-ish empty."""
        monkeypatch.setattr(embeddings, "llm_embed", AsyncMock(return_value=None))
        with pytest.raises((ValueError, TypeError)):
            await embeddings.embed_text("hello")


# ===========================================================================
# embed_article — empty chunk_embeddings (line 100)
# ===========================================================================


class TestEmbedArticleExtended:
    @pytest.mark.asyncio
    async def test_multi_chunk_no_embeddings_returns_error(self, monkeypatch):
        """If embed_texts returns [] for a multi-chunk article, return error."""
        long_body = ("x" * 5000 + "\n\n") * 5
        fake_article = {
            "title": "Long",
            "summary": "",
            "raw_markdown": long_body,
        }
        monkeypatch.setattr(embeddings, "get_article", lambda kb, slug: fake_article)
        monkeypatch.setattr(embeddings, "llm_embed", AsyncMock(return_value=[]))

        result = await embeddings.embed_article("personal", "long-article")
        assert result["status"] == "error"
        assert "no embeddings" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_article_with_no_body(self, monkeypatch):
        """Article with only title should still embed fine."""
        fake_article = {
            "title": "Title Only",
            "summary": None,
            "raw_markdown": "",
        }
        monkeypatch.setattr(embeddings, "get_article", lambda kb, slug: fake_article)
        monkeypatch.setattr(
            embeddings, "llm_embed",
            AsyncMock(return_value=[[0.5, 0.5]]),
        )
        mock_store = MagicMock()
        mock_store.upsert = AsyncMock()
        monkeypatch.setattr(embeddings, "get_vector_store", lambda: mock_store)

        result = await embeddings.embed_article("personal", "title-only")
        assert result["status"] == "ok"


# ===========================================================================
# embed_all_articles (lines 118-148)
# ===========================================================================


class TestEmbedAllArticles:
    @pytest.mark.asyncio
    async def test_empty_kb_returns_zero(self, monkeypatch):
        """An empty KB should return embedded=0, errors=0."""
        monkeypatch.setattr(embeddings, "get_articles", lambda kb: [])
        result = await embeddings.embed_all_articles("empty-kb")
        assert result["status"] == "ok"
        assert result["embedded"] == 0
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_all_articles_embedded_successfully(self, monkeypatch):
        """All articles embed successfully."""
        articles = [{"slug": "a1"}, {"slug": "a2"}]
        monkeypatch.setattr(embeddings, "get_articles", lambda kb: articles)

        async def fake_embed_article(kb, slug):
            return {"status": "ok", "slug": slug}

        monkeypatch.setattr(embeddings, "embed_article", fake_embed_article)
        # Speed up by mocking sleep
        import asyncio
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await embeddings.embed_all_articles("personal")
        assert result["embedded"] == 2
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_some_articles_fail(self, monkeypatch):
        """Some articles fail to embed — should count errors."""
        articles = [{"slug": "good"}, {"slug": "bad"}, {"slug": "also-good"}]
        monkeypatch.setattr(embeddings, "get_articles", lambda kb: articles)

        async def fake_embed_article(kb, slug):
            if slug == "bad":
                return {"status": "error", "error": "API failure"}
            return {"status": "ok", "slug": slug}

        monkeypatch.setattr(embeddings, "embed_article", fake_embed_article)
        import asyncio
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await embeddings.embed_all_articles("personal")
        assert result["embedded"] == 2
        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_embed_article_exception_counted_as_error(self, monkeypatch):
        """If embed_article raises, count it as an error."""
        articles = [{"slug": "exploder"}]
        monkeypatch.setattr(embeddings, "get_articles", lambda kb: articles)

        async def exploding_embed(kb, slug):
            raise RuntimeError("network timeout")

        monkeypatch.setattr(embeddings, "embed_article", exploding_embed)
        import asyncio
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        result = await embeddings.embed_all_articles("personal")
        assert result["embedded"] == 0
        assert result["errors"] == 1
