"""Tests for app.browser — page content extraction via agent-browser CLI and httpx."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.browser import (
    _run_cmd,
    is_available,
    read_page_browser,
    read_page_httpx,
    read_page_smart,
    read_pages_smart,
    _html_to_text,
    _clean_page_content,
    _get_document_ext,
    find_document_urls,
    download_and_extract,
)


# ──────────────────────────────────────────────────────────────────────────────
# _run_cmd
# ──────────────────────────────────────────────────────────────────────────────


class TestRunCmd:
    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_success(self, mock_exec):
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"output", b""))
        mock_exec.return_value = proc

        code, stdout, stderr = await _run_cmd("--version")
        assert code == 0
        assert stdout == "output"
        assert stderr == ""

    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_nonzero_return(self, mock_exec):
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b"", b"error msg"))
        mock_exec.return_value = proc

        code, stdout, stderr = await _run_cmd("goto", "http://example.com")
        assert code == 1
        assert stderr == "error msg"

    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_timeout(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.kill = MagicMock()
        mock_exec.return_value = proc

        code, stdout, stderr = await _run_cmd("goto", "http://slow.com", timeout=1)
        assert code == 1
        assert "timed out" in stderr.lower()

    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_file_not_found(self, mock_exec):
        mock_exec.side_effect = FileNotFoundError()
        code, stdout, stderr = await _run_cmd("--version")
        assert code == 1
        assert "not found" in stderr.lower()

    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_generic_exception(self, mock_exec):
        mock_exec.side_effect = OSError("permission denied")
        code, stdout, stderr = await _run_cmd("--version")
        assert code == 1
        assert "permission denied" in stderr.lower()


# ──────────────────────────────────────────────────────────────────────────────
# is_available
# ──────────────────────────────────────────────────────────────────────────────


class TestIsAvailable:
    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_available(self, mock_cmd):
        mock_cmd.return_value = (0, "1.0.0", "")
        assert await is_available() is True

    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_not_available(self, mock_cmd):
        mock_cmd.return_value = (1, "", "not found")
        assert await is_available() is False


# ──────────────────────────────────────────────────────────────────────────────
# read_page_browser
# ──────────────────────────────────────────────────────────────────────────────


class TestReadPageBrowser:
    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_successful_read(self, mock_cmd):
        """Full happy path: navigate, get title, eval innerText."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args[0] == "goto":
                return (0, "", "")
            elif args[0] == "title":
                return (0, "Test Page Title", "")
            elif args[0] == "eval":
                # Return enough words (>20) to pass the word_count check
                return (0, "This is the page content. " * 10, "")
            return (0, "", "")

        mock_cmd.side_effect = side_effect
        result = await read_page_browser("http://example.com")
        assert result is not None
        assert result["title"] == "Test Page Title"
        assert result["source"] == "agent_browser"
        assert result["url"] == "http://example.com"
        assert result["word_count"] >= 20

    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_navigation_failure(self, mock_cmd):
        mock_cmd.return_value = (1, "", "navigation failed")
        result = await read_page_browser("http://bad.com")
        assert result is None

    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_fallback_to_snapshot(self, mock_cmd):
        """When eval fails, should try snapshot fallback."""
        async def side_effect(*args, **kwargs):
            if args[0] == "goto":
                return (0, "", "")
            elif args[0] == "title":
                return (0, "Title", "")
            elif args[0] == "eval":
                return (1, "", "eval failed")
            elif args[0] == "snapshot":
                return (0, "Snapshot content with many words. " * 10, "")
            return (0, "", "")

        mock_cmd.side_effect = side_effect
        result = await read_page_browser("http://example.com")
        assert result is not None
        assert result["word_count"] >= 20

    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_too_few_words_returns_none(self, mock_cmd):
        async def side_effect(*args, **kwargs):
            if args[0] == "goto":
                return (0, "", "")
            elif args[0] == "title":
                return (0, "Title", "")
            elif args[0] == "eval":
                return (0, "short text", "")
            return (0, "", "")

        mock_cmd.side_effect = side_effect
        result = await read_page_browser("http://example.com")
        assert result is None

    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_long_content_truncated(self, mock_cmd):
        async def side_effect(*args, **kwargs):
            if args[0] == "goto":
                return (0, "", "")
            elif args[0] == "title":
                return (0, "Title", "")
            elif args[0] == "eval":
                return (0, "word " * 20000, "")
            return (0, "", "")

        mock_cmd.side_effect = side_effect
        result = await read_page_browser("http://example.com")
        assert result is not None
        assert "[Truncated]" in result["content"]


# ──────────────────────────────────────────────────────────────────────────────
# read_page_httpx
# ──────────────────────────────────────────────────────────────────────────────


class TestReadPageHttpx:
    @patch("httpx.AsyncClient")
    async def test_successful_html_read(self, mock_client_cls):
        html = (
            "<html><head><title>Test Page</title></head>"
            "<body><main><p>" + "Some content here. " * 20 + "</p></main></body></html>"
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com")
        assert result is not None
        assert result["title"] == "Test Page"
        assert result["source"] == "httpx"
        assert result["word_count"] >= 30

    @patch("httpx.AsyncClient")
    async def test_non_200_returns_none(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com/missing")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_non_html_content_type_returns_none(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com/api")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_too_few_words_returns_none(self, mock_client_cls):
        html = "<html><head><title>T</title></head><body><p>Short.</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_request_exception_returns_none(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://down.com")
        assert result is None


# ──────────────────────────────────────────────────────────────────────────────
# read_page_smart / read_pages_smart
# ──────────────────────────────────────────────────────────────────────────────


class TestReadPageSmart:
    @patch("app.browser.read_page_httpx", new_callable=AsyncMock)
    @patch("app.browser.read_page_browser", new_callable=AsyncMock)
    @patch("app.browser.is_available", new_callable=AsyncMock, return_value=True)
    async def test_uses_browser_when_available(self, mock_avail, mock_browser, mock_httpx):
        mock_browser.return_value = {
            "url": "http://example.com", "title": "T", "content": "c",
            "word_count": 100, "source": "agent_browser",
        }
        result = await read_page_smart("http://example.com")
        assert result["source"] == "agent_browser"
        mock_httpx.assert_not_awaited()

    @patch("app.browser.read_page_httpx", new_callable=AsyncMock)
    @patch("app.browser.is_available", new_callable=AsyncMock, return_value=False)
    async def test_falls_back_to_httpx(self, mock_avail, mock_httpx):
        mock_httpx.return_value = {
            "url": "http://example.com", "title": "T", "content": "c",
            "word_count": 100, "source": "httpx",
        }
        result = await read_page_smart("http://example.com")
        assert result["source"] == "httpx"

    @patch("app.browser.read_page_httpx", new_callable=AsyncMock)
    @patch("app.browser.read_page_browser", new_callable=AsyncMock, return_value=None)
    @patch("app.browser.is_available", new_callable=AsyncMock, return_value=True)
    async def test_falls_back_when_browser_returns_none(self, mock_avail, mock_browser, mock_httpx):
        mock_httpx.return_value = {
            "url": "http://example.com", "title": "T", "content": "c",
            "word_count": 100, "source": "httpx",
        }
        result = await read_page_smart("http://example.com")
        assert result["source"] == "httpx"

    @patch("app.browser.read_page_httpx", new_callable=AsyncMock)
    @patch("app.browser.read_page_browser", new_callable=AsyncMock)
    @patch("app.browser.is_available", new_callable=AsyncMock, return_value=True)
    async def test_falls_back_when_browser_word_count_low(self, mock_avail, mock_browser, mock_httpx):
        mock_browser.return_value = {
            "url": "http://example.com", "title": "T", "content": "short",
            "word_count": 30, "source": "agent_browser",
        }
        mock_httpx.return_value = {
            "url": "http://example.com", "title": "T", "content": "longer content",
            "word_count": 100, "source": "httpx",
        }
        result = await read_page_smart("http://example.com")
        assert result["source"] == "httpx"


class TestReadPagesSmart:
    @patch("app.browser.read_page_smart", new_callable=AsyncMock)
    async def test_multiple_urls(self, mock_smart):
        mock_smart.side_effect = [
            {"url": "http://a.com", "title": "A", "content": "c", "word_count": 100, "source": "httpx"},
            None,  # one failure
            {"url": "http://c.com", "title": "C", "content": "c", "word_count": 200, "source": "httpx"},
        ]
        results = await read_pages_smart(["http://a.com", "http://b.com", "http://c.com"])
        assert len(results) == 2

    @patch("app.browser.read_page_smart", new_callable=AsyncMock, return_value=None)
    async def test_all_failures(self, mock_smart):
        results = await read_pages_smart(["http://a.com", "http://b.com"])
        assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# HTML / content sanitization helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestHtmlToText:
    def test_strips_tags(self):
        result = _html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_removes_script_tags(self):
        html = "<div>text<script>var x = 1;</script>more</div>"
        result = _html_to_text(html)
        assert "var x" not in result
        assert "text" in result
        assert "more" in result

    def test_removes_style_tags(self):
        html = "<div>text<style>.foo { color: red; }</style>more</div>"
        result = _html_to_text(html)
        assert "color" not in result

    def test_removes_nav_footer_header_aside(self):
        html = (
            "<nav>navigation</nav><main>content</main>"
            "<footer>footer text</footer><aside>sidebar</aside>"
        )
        result = _html_to_text(html)
        assert "content" in result
        # nav/footer/aside stripped before main extraction
        assert "navigation" not in result

    def test_prefers_article_content(self):
        html = "<div>noise</div><article><p>the real article content</p></article><div>more noise</div>"
        result = _html_to_text(html)
        assert "real article content" in result

    def test_prefers_main_content(self):
        html = "<div>noise</div><main><p>main content here</p></main><div>noise</div>"
        result = _html_to_text(html)
        assert "main content here" in result

    def test_decodes_html_entities(self):
        html = "<p>A &amp; B &lt; C &gt; D &quot;E&quot; F&#39;s</p>"
        result = _html_to_text(html)
        assert "A & B" in result
        assert "< C >" in result

    def test_empty_html(self):
        result = _html_to_text("")
        assert result == ""


class TestCleanPageContent:
    def test_removes_cookie_banners(self):
        text = "Accept all cookies\nReal content here"
        result = _clean_page_content(text)
        assert "cookie" not in result.lower()
        assert "Real content" in result

    def test_removes_signup_prompts(self):
        text = "Sign up for free Log in\nActual content"
        result = _clean_page_content(text)
        assert "Sign up" not in result

    def test_collapses_newlines(self):
        text = "Line one\n\n\n\n\nLine two"
        result = _clean_page_content(text)
        assert "\n\n\n" not in result

    def test_strips_whitespace(self):
        result = _clean_page_content("  hello world  ")
        assert result == "hello world"


# ──────────────────────────────────────────────────────────────────────────────
# Document URL detection
# ──────────────────────────────────────────────────────────────────────────────


class TestGetDocumentExt:
    def test_pdf_extension(self):
        assert _get_document_ext("http://example.com/paper.pdf") == ".pdf"

    def test_epub_extension(self):
        assert _get_document_ext("http://example.com/book.epub") == ".epub"

    def test_pdf_case_insensitive(self):
        assert _get_document_ext("http://example.com/paper.PDF") == ".pdf"

    def test_pdf_with_query_string(self):
        assert _get_document_ext("http://example.com/paper.pdf?dl=1") == ".pdf"

    def test_pdf_with_fragment(self):
        assert _get_document_ext("http://example.com/paper.pdf#page=5") == ".pdf"

    def test_arxiv_pdf(self):
        assert _get_document_ext("https://arxiv.org/pdf/2301.12345") == ".pdf"

    def test_generic_pdf_path(self):
        assert _get_document_ext("https://example.com/pdf/document") == ".pdf"

    def test_html_url_returns_none(self):
        assert _get_document_ext("http://example.com/page.html") is None

    def test_no_extension_returns_none(self):
        assert _get_document_ext("http://example.com/page") is None

    def test_empty_url(self):
        assert _get_document_ext("") is None


class TestFindDocumentUrls:
    def test_finds_pdf_urls(self):
        results = [
            {"url": "http://example.com/paper.pdf"},
            {"url": "http://example.com/page.html"},
            {"url": "http://example.com/book.epub"},
        ]
        found = find_document_urls(results)
        assert len(found) == 2
        assert "http://example.com/paper.pdf" in found
        assert "http://example.com/book.epub" in found

    def test_deduplicates(self):
        results = [
            {"url": "http://example.com/paper.pdf"},
            {"url": "http://example.com/paper.pdf"},
        ]
        found = find_document_urls(results)
        assert len(found) == 1

    def test_skips_empty_urls(self):
        results = [{"url": ""}, {"other": "data"}]
        found = find_document_urls(results)
        assert found == []


# ──────────────────────────────────────────────────────────────────────────────
# download_and_extract
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadAndExtract:
    async def test_non_document_url_returns_none(self):
        result = await download_and_extract("http://example.com/page.html")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_download_failure_returns_none(self, mock_client_cls, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")):
            result = await download_and_extract("http://example.com/paper.pdf")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_wrong_content_type_returns_none(self, mock_client_cls, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")):
            result = await download_and_extract("http://example.com/paper.pdf")
        assert result is None
