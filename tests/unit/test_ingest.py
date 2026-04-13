"""Tests for app.ingest — document ingestion pipeline."""

import httpx
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import app.ingest as ingest_mod

# crawl_open_directory references bare `httpx` but the module never imports it.
# Inject so the function can resolve the name at runtime.
if not hasattr(ingest_mod, "httpx"):
    ingest_mod.httpx = httpx

from app.ingest import (
    ingest_document_url,
    ingest_document_urls,
    crawl_open_directory,
    ingest_open_directory,
    _synthesize_document,
)


# ──────────────────────────────────────────────────────────────────────────────
# ingest_document_url
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestDocumentUrl:
    @patch("app.ingest.create_or_update_article", new_callable=AsyncMock)
    @patch("app.ingest.download_and_extract", new_callable=AsyncMock)
    async def test_successful_ingest(self, mock_download, mock_create):
        mock_download.return_value = {
            "title": "Test Paper",
            "content": "Paper content goes here. " * 100,
            "word_count": 400,
            "filepath": "/tmp/test.pdf",
        }
        mock_create.return_value = ("test-paper", "created")

        result = await ingest_document_url("http://example.com/paper.pdf", kb_name="research")
        assert result["title"] == "Test Paper"
        assert result["slug"] == "test-paper"
        assert result["change_type"] == "created"
        mock_create.assert_awaited_once()

    @patch("app.ingest.download_and_extract", new_callable=AsyncMock, return_value=None)
    async def test_download_failure(self, mock_download):
        result = await ingest_document_url("http://example.com/bad.pdf")
        assert "error" in result

    @patch("app.ingest.create_or_update_article", new_callable=AsyncMock)
    @patch("app.ingest.download_and_extract", new_callable=AsyncMock)
    async def test_wiki_creation_failure(self, mock_download, mock_create):
        mock_download.return_value = {
            "title": "Paper",
            "content": "Content",
            "word_count": 100,
        }
        mock_create.side_effect = RuntimeError("DB write failed")
        result = await ingest_document_url("http://example.com/paper.pdf")
        assert "error" in result

    @patch("app.ingest._synthesize_document", new_callable=AsyncMock)
    @patch("app.ingest.create_or_update_article", new_callable=AsyncMock, return_value=("slug", "created"))
    @patch("app.ingest.download_and_extract", new_callable=AsyncMock)
    async def test_long_document_gets_synthesized(self, mock_download, mock_create, mock_synth):
        mock_download.return_value = {
            "title": "Long Paper",
            "content": "word " * 6000,
            "word_count": 6000,
        }
        mock_synth.return_value = "Synthesized summary."
        result = await ingest_document_url("http://example.com/paper.pdf")
        mock_synth.assert_awaited_once()
        assert "error" not in result

    @patch("app.ingest.create_or_update_article", new_callable=AsyncMock, return_value=("slug", "created"))
    @patch("app.ingest.download_and_extract", new_callable=AsyncMock)
    async def test_short_document_not_synthesized(self, mock_download, mock_create):
        mock_download.return_value = {
            "title": "Short Paper",
            "content": "word " * 100,
            "word_count": 100,
        }
        with patch("app.ingest._synthesize_document", new_callable=AsyncMock) as mock_synth:
            await ingest_document_url("http://example.com/paper.pdf")
            mock_synth.assert_not_awaited()

    @patch("app.ingest.create_or_update_article", new_callable=AsyncMock, return_value=("slug", "created"))
    @patch("app.ingest.download_and_extract", new_callable=AsyncMock)
    async def test_result_includes_word_counts(self, mock_download, mock_create):
        mock_download.return_value = {
            "title": "Paper",
            "content": "word " * 200,
            "word_count": 200,
            "filepath": "/tmp/paper.pdf",
        }
        result = await ingest_document_url("http://example.com/paper.pdf")
        assert result["word_count"] == 200
        assert "synthesized_words" in result


# ──────────────────────────────────────────────────────────────────────────────
# ingest_document_urls
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestDocumentUrls:
    @patch("app.ingest.ingest_document_url", new_callable=AsyncMock)
    @patch("app.ingest.asyncio.sleep", new_callable=AsyncMock)
    async def test_processes_all_urls(self, mock_sleep, mock_ingest):
        mock_ingest.side_effect = [
            {"title": "A", "slug": "a"},
            {"title": "B", "slug": "b"},
            {"error": "failed"},
        ]
        results = await ingest_document_urls([
            "http://a.pdf", "http://b.pdf", "http://c.pdf"
        ])
        assert len(results) == 3
        assert mock_ingest.await_count == 3

    @patch("app.ingest.ingest_document_url", new_callable=AsyncMock)
    @patch("app.ingest.asyncio.sleep", new_callable=AsyncMock)
    async def test_empty_urls(self, mock_sleep, mock_ingest):
        results = await ingest_document_urls([])
        assert results == []
        mock_ingest.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────────────
# crawl_open_directory
# ──────────────────────────────────────────────────────────────────────────────


class TestCrawlOpenDirectory:
    @patch("app.ingest.httpx.AsyncClient")
    async def test_finds_pdf_links(self, mock_client_cls):
        html = '''
        <html><body>
        <a href="paper1.pdf">Paper 1</a>
        <a href="paper2.pdf">Paper 2</a>
        <a href="readme.html">README</a>
        <a href="book.epub">Book</a>
        </body></html>
        '''
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://example.com/docs/")
        assert len(urls) == 3  # 2 PDFs + 1 epub
        assert any("paper1.pdf" in u for u in urls)
        assert any("book.epub" in u for u in urls)

    @patch("app.ingest.httpx.AsyncClient")
    async def test_resolves_relative_urls(self, mock_client_cls):
        html = '<html><body><a href="sub/paper.pdf">Paper</a></body></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://example.com/docs/")
        assert urls[0] == "http://example.com/docs/sub/paper.pdf"

    @patch("app.ingest.httpx.AsyncClient")
    async def test_resolves_absolute_path_urls(self, mock_client_cls):
        html = '<html><body><a href="/files/paper.pdf">Paper</a></body></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://example.com/docs/")
        assert urls[0] == "http://example.com/files/paper.pdf"

    @patch("app.ingest.httpx.AsyncClient")
    async def test_max_files_limit(self, mock_client_cls):
        html = '<html><body>'
        for i in range(50):
            html += f'<a href="paper{i}.pdf">Paper {i}</a>'
        html += '</body></html>'

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://example.com/docs/", max_files=5)
        assert len(urls) == 5

    @patch("app.ingest.httpx.AsyncClient")
    async def test_http_failure_returns_empty(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://example.com/docs/")
        assert urls == []

    @patch("app.ingest.httpx.AsyncClient")
    async def test_network_error_returns_empty(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://down.com/")
        assert urls == []

    @patch("app.ingest.httpx.AsyncClient")
    async def test_custom_extensions(self, mock_client_cls):
        html = '<html><body><a href="file.pdf">PDF</a><a href="file.txt">TXT</a></body></html>'
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        urls = await crawl_open_directory("http://example.com/", extensions=[".txt"])
        assert len(urls) == 1
        assert "file.txt" in urls[0]


# ──────────────────────────────────────────────────────────────────────────────
# ingest_open_directory
# ──────────────────────────────────────────────────────────────────────────────


class TestIngestOpenDirectory:
    @patch("app.ingest.ingest_document_urls", new_callable=AsyncMock)
    @patch("app.ingest.crawl_open_directory", new_callable=AsyncMock)
    async def test_successful_crawl_and_ingest(self, mock_crawl, mock_ingest):
        mock_crawl.return_value = ["http://a.pdf", "http://b.pdf"]
        mock_ingest.return_value = [
            {"title": "A", "slug": "a"},
            {"error": "failed"},
        ]
        result = await ingest_open_directory("http://example.com/docs/")
        assert result["documents_found"] == 2
        assert result["documents_ingested"] == 1

    @patch("app.ingest.crawl_open_directory", new_callable=AsyncMock, return_value=[])
    async def test_no_documents_found(self, mock_crawl):
        result = await ingest_open_directory("http://example.com/empty/")
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────────────
# _synthesize_document
# ──────────────────────────────────────────────────────────────────────────────


class TestSynthesizeDocument:
    @patch("app.ingest.MINIMAX_API_KEY", None)
    @patch("app.ingest.LLM_PROVIDER", "none")
    async def test_no_llm_truncates(self):
        content = "x" * 10000
        result = await _synthesize_document("Title", content)
        assert len(result) == 8000

    @patch("app.ingest.llm_chat", new_callable=AsyncMock)
    @patch("app.ingest.MINIMAX_API_KEY", "fake-key")
    async def test_llm_synthesis(self, mock_llm):
        mock_llm.return_value = "Synthesized content here."
        result = await _synthesize_document("Title", "long content " * 1000)
        assert result == "Synthesized content here."
        mock_llm.assert_awaited_once()

    @patch("app.ingest.llm_chat", new_callable=AsyncMock)
    @patch("app.ingest.MINIMAX_API_KEY", "fake-key")
    async def test_llm_failure_falls_back_to_truncation(self, mock_llm):
        mock_llm.side_effect = RuntimeError("LLM down")
        content = "x" * 10000
        result = await _synthesize_document("Title", content)
        assert len(result) == 8000

    @patch("app.ingest.llm_chat", new_callable=AsyncMock)
    @patch("app.ingest.LLM_PROVIDER", "bedrock")
    async def test_bedrock_provider_triggers_synthesis(self, mock_llm):
        mock_llm.return_value = "Bedrock synthesized."
        result = await _synthesize_document("Title", "content " * 1000)
        assert result == "Bedrock synthesized."
