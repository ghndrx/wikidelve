"""Extended tests for app.browser — covering download_and_extract, _extract_pdf_text,
_extract_epub_text, download_documents, and remaining edge cases."""

import asyncio
import io
import zipfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock, PropertyMock

from app.browser import (
    _run_cmd,
    read_page_browser,
    read_page_httpx,
    download_and_extract,
    download_documents,
    _extract_pdf_text,
    _extract_epub_text,
    _clean_page_content,
    _html_to_text,
)


# ──────────────────────────────────────────────────────────────────────────────
# _run_cmd — timeout branch where proc.kill() raises
# ──────────────────────────────────────────────────────────────────────────────


class TestRunCmdExtended:
    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_timeout_kill_raises(self, mock_exec):
        """Line 44: proc.kill() raises an exception — should still return timeout."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.kill = MagicMock(side_effect=OSError("already dead"))
        mock_exec.return_value = proc

        code, stdout, stderr = await _run_cmd("goto", "http://slow.com", timeout=1)
        assert code == 1
        assert "timed out" in stderr.lower()

    @patch("asyncio.create_subprocess_exec", new_callable=AsyncMock)
    async def test_none_returncode_becomes_zero(self, mock_exec):
        """When proc.returncode is None, _run_cmd converts it to 0."""
        proc = AsyncMock()
        proc.returncode = None
        proc.communicate = AsyncMock(return_value=(b"ok", b""))
        mock_exec.return_value = proc

        code, stdout, stderr = await _run_cmd("--version")
        assert code == 0
        assert stdout == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# read_page_browser — snapshot fallback also fails (lines 82-83)
# ──────────────────────────────────────────────────────────────────────────────


class TestReadPageBrowserExtended:
    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_eval_and_snapshot_both_fail(self, mock_cmd):
        """Lines 82-83: eval returns empty, snapshot also fails -> None."""
        async def side_effect(*args, **kwargs):
            if args[0] == "goto":
                return (0, "", "")
            elif args[0] == "title":
                return (0, "Title", "")
            elif args[0] == "eval":
                return (0, "", "")  # empty content
            elif args[0] == "snapshot":
                return (1, "", "snapshot failed")
            return (0, "", "")

        mock_cmd.side_effect = side_effect
        result = await read_page_browser("http://example.com")
        assert result is None

    @patch("app.browser._run_cmd", new_callable=AsyncMock)
    async def test_title_failure_uses_empty_string(self, mock_cmd):
        """Line 73: when title cmd fails, title should be empty string."""
        async def side_effect(*args, **kwargs):
            if args[0] == "goto":
                return (0, "", "")
            elif args[0] == "title":
                return (1, "", "title cmd error")
            elif args[0] == "eval":
                return (0, "word " * 30, "")
            return (0, "", "")

        mock_cmd.side_effect = side_effect
        result = await read_page_browser("http://example.com")
        assert result is not None
        assert result["title"] == ""


# ──────────────────────────────────────────────────────────────────────────────
# read_page_httpx — truncation (line 130)
# ──────────────────────────────────────────────────────────────────────────────


class TestReadPageHttpxExtended:
    @patch("httpx.AsyncClient")
    async def test_long_content_truncated(self, mock_client_cls):
        """Line 129-130: text > 15000 chars gets truncated."""
        long_body = "word " * 5000  # >15000 chars
        html = f"<html><head><title>Long</title></head><body><main><p>{long_body}</p></main></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com/long")
        assert result is not None
        assert "[Truncated]" in result["content"]

    @patch("httpx.AsyncClient")
    async def test_text_plain_content_type(self, mock_client_cls):
        """text/plain content type should also be accepted."""
        html = "<html><body><p>" + "Some content here. " * 20 + "</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com/text")
        assert result is not None

    @patch("httpx.AsyncClient")
    async def test_no_title_tag_uses_url(self, mock_client_cls):
        """When no <title> tag found, URL is used as title."""
        html = "<html><body><main><p>" + "lots of words here " * 20 + "</p></main></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.text = html

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await read_page_httpx("http://example.com/notitle")
        assert result is not None
        assert result["title"] == "http://example.com/notitle"


# ──────────────────────────────────────────────────────────────────────────────
# download_and_extract — full flow (lines 210-281)
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadAndExtractExtended:
    @patch("httpx.AsyncClient")
    async def test_successful_pdf_download(self, mock_client_cls, tmp_path):
        """Full successful PDF download+extract flow."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"fake pdf content"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        extracted_text = "word " * 50
        with (
            patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")),
            patch("app.browser._extract_pdf_text", new_callable=AsyncMock, return_value=(extracted_text, "My PDF")),
        ):
            result = await download_and_extract("http://example.com/paper.pdf")

        assert result is not None
        assert result["title"] == "My PDF"
        assert result["source"] == "downloaded_pdf"
        assert "filepath" in result
        assert result["word_count"] >= 30

    @patch("httpx.AsyncClient")
    async def test_successful_epub_download(self, mock_client_cls, tmp_path):
        """Full successful ePub download+extract flow."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/epub+zip"}
        mock_resp.content = b"fake epub content"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        extracted_text = "word " * 50
        with (
            patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")),
            patch("app.browser._extract_epub_text", new_callable=AsyncMock, return_value=(extracted_text, "My Book")),
        ):
            result = await download_and_extract("http://example.com/book.epub")

        assert result is not None
        assert result["title"] == "My Book"
        assert result["source"] == "downloaded_epub"

    @patch("httpx.AsyncClient")
    async def test_epub_wrong_content_type_returns_none(self, mock_client_cls, tmp_path):
        """Line 242-243: epub with wrong content type returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")):
            result = await download_and_extract("http://example.com/book.epub")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_insufficient_text_returns_none(self, mock_client_cls, tmp_path):
        """Line 266-268: extracted text with < 30 words returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"fake pdf"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with (
            patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")),
            patch("app.browser._extract_pdf_text", new_callable=AsyncMock, return_value=("short", "T")),
        ):
            result = await download_and_extract("http://example.com/paper.pdf")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_long_document_truncated(self, mock_client_cls, tmp_path):
        """Line 271-272: documents > 30000 chars get truncated."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"pdf bytes"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        long_text = "word " * 10000  # > 30000 chars
        with (
            patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")),
            patch("app.browser._extract_pdf_text", new_callable=AsyncMock, return_value=(long_text, "Big PDF")),
        ):
            result = await download_and_extract("http://example.com/big.pdf")

        assert result is not None
        assert "[Document truncated" in result["content"]

    @patch("httpx.AsyncClient")
    async def test_download_exception_returns_none(self, mock_client_cls, tmp_path):
        """Line 253-255: network exception during download returns None."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection reset"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")):
            result = await download_and_extract("http://example.com/paper.pdf")
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_safe_name_appends_extension(self, mock_client_cls, tmp_path):
        """Line 247-248: URL without extension gets ext appended to safe_name."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"pdf bytes"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        extracted_text = "word " * 50
        with (
            patch("app.browser.DOWNLOAD_DIR", str(tmp_path / "downloads")),
            patch("app.browser._extract_pdf_text", new_callable=AsyncMock, return_value=(extracted_text, "T")),
        ):
            result = await download_and_extract("https://arxiv.org/pdf/2301.12345")

        assert result is not None
        assert result["filepath"].endswith(".pdf")


# ──────────────────────────────────────────────────────────────────────────────
# download_documents — concurrent download (lines 284-294)
# ──────────────────────────────────────────────────────────────────────────────


class TestDownloadDocuments:
    @patch("app.browser.download_and_extract", new_callable=AsyncMock)
    async def test_concurrent_downloads(self, mock_dl):
        mock_dl.side_effect = [
            {"url": "http://a.com/a.pdf", "title": "A", "content": "c", "word_count": 100, "source": "downloaded_pdf"},
            None,  # one failure
            {"url": "http://c.com/c.pdf", "title": "C", "content": "c", "word_count": 200, "source": "downloaded_pdf"},
        ]
        results = await download_documents(["http://a.com/a.pdf", "http://b.com/b.pdf", "http://c.com/c.pdf"])
        assert len(results) == 2

    @patch("app.browser.download_and_extract", new_callable=AsyncMock, return_value=None)
    async def test_all_downloads_fail(self, mock_dl):
        results = await download_documents(["http://a.com/a.pdf", "http://b.com/b.pdf"])
        assert results == []

    @patch("app.browser.download_and_extract", new_callable=AsyncMock)
    async def test_exception_in_download_filtered_out(self, mock_dl):
        """Exceptions from gather are filtered out (not dicts)."""
        mock_dl.side_effect = [
            {"url": "http://a.com/a.pdf", "title": "A", "content": "c", "word_count": 100, "source": "downloaded_pdf"},
            Exception("network error"),
        ]
        results = await download_documents(["http://a.com/a.pdf", "http://b.com/b.pdf"])
        assert len(results) == 1


# ──────────────────────────────────────────────────────────────────────────────
# _extract_pdf_text — all three extraction paths (lines 326-368)
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractPdfText:
    async def test_fitz_extraction(self, tmp_path):
        """Lines 326-336: PyMuPDF (fitz) extraction path."""
        filepath = tmp_path / "test.pdf"
        filepath.write_bytes(b"fake")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 content"

        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Fitz PDF Title"}
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            text, title = await _extract_pdf_text(filepath)

        assert text == "Page 1 content"
        assert title == "Fitz PDF Title"

    async def test_fitz_no_title_uses_stem(self, tmp_path):
        """When fitz metadata has no title, filepath.stem is used."""
        filepath = tmp_path / "my_document.pdf"
        filepath.write_bytes(b"fake")

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page text"

        mock_doc = MagicMock()
        mock_doc.metadata = {"title": ""}
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.close = MagicMock()

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            text, title = await _extract_pdf_text(filepath)

        assert title == "my_document"

    async def test_pdfplumber_fallback(self, tmp_path):
        """Lines 341-350: pdfplumber fallback when fitz not available."""
        filepath = tmp_path / "test.pdf"
        filepath.write_bytes(b"fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Plumber page text"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        import sys
        # Setting a module to None in sys.modules makes `import X` raise ImportError
        saved_fitz = sys.modules.get("fitz", "MISSING")
        saved_plumber = sys.modules.get("pdfplumber", "MISSING")
        sys.modules["fitz"] = None
        sys.modules["pdfplumber"] = mock_pdfplumber
        try:
            text, title = await _extract_pdf_text(filepath)
        finally:
            if saved_fitz == "MISSING":
                sys.modules.pop("fitz", None)
            else:
                sys.modules["fitz"] = saved_fitz
            if saved_plumber == "MISSING":
                sys.modules.pop("pdfplumber", None)
            else:
                sys.modules["pdfplumber"] = saved_plumber

        assert "Plumber page text" in text

    async def test_pdftotext_cli_fallback(self, tmp_path):
        """Lines 355-364: pdftotext CLI fallback when no Python libs available."""
        filepath = tmp_path / "test.pdf"
        filepath.write_bytes(b"fake")

        # Make both fitz and pdfplumber fail
        with (
            patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError()) if name in ("fitz", "pdfplumber") else __import__(name, *a, **kw)),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"CLI extracted text", b""))
            mock_exec.return_value = proc

            text, title = await _extract_pdf_text(filepath)

        assert text == "CLI extracted text"
        assert title == "test"

    async def test_all_extractors_fail(self, tmp_path):
        """Lines 367-368: when all extractors fail, returns empty string."""
        filepath = tmp_path / "test.pdf"
        filepath.write_bytes(b"fake")

        with (
            patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError()) if name in ("fitz", "pdfplumber") else __import__(name, *a, **kw)),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=Exception("pdftotext not found")),
        ):
            text, title = await _extract_pdf_text(filepath)

        assert text == ""
        assert title == "test"


# ──────────────────────────────────────────────────────────────────────────────
# _extract_epub_text (lines 371-406)
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractEpubText:
    async def test_valid_epub_extraction(self, tmp_path):
        """Lines 373-403: extract text from a valid ePub zip."""
        filepath = tmp_path / "book.epub"

        # Build a minimal epub zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "chapter1.xhtml",
                "<html><body><p>" + "This is chapter one content. " * 5 + "</p></body></html>"
            )
            zf.writestr(
                "content.opf",
                '<?xml version="1.0"?>'
                '<package xmlns="http://www.idpf.org/2007/opf">'
                '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                '<dc:title>My ePub Title</dc:title>'
                '</metadata></package>'
            )
        filepath.write_bytes(buf.getvalue())

        text, title = await _extract_epub_text(filepath)
        assert "chapter one content" in text
        assert title == "My ePub Title"

    async def test_epub_without_opf_title(self, tmp_path):
        """When .opf has no title, filepath.stem is used."""
        filepath = tmp_path / "untitled.epub"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "chapter1.xhtml",
                "<html><body><p>" + "Some epub content here. " * 5 + "</p></body></html>"
            )
        filepath.write_bytes(buf.getvalue())

        text, title = await _extract_epub_text(filepath)
        assert "epub content" in text
        assert title == "untitled"

    async def test_epub_exception_returns_empty(self, tmp_path):
        """Lines 404-406: exception during extraction returns empty."""
        filepath = tmp_path / "bad.epub"
        filepath.write_bytes(b"not a zip file")

        text, title = await _extract_epub_text(filepath)
        assert text == ""
        assert title == "bad"

    async def test_epub_short_content_skipped(self, tmp_path):
        """Files with < 50 chars of clean text are skipped."""
        filepath = tmp_path / "tiny.epub"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("chapter1.xhtml", "<html><body><p>Hi</p></body></html>")
        filepath.write_bytes(buf.getvalue())

        text, title = await _extract_epub_text(filepath)
        # "Hi" is < 50 chars so text_parts should be empty
        assert text == ""

    async def test_epub_opf_parse_error_graceful(self, tmp_path):
        """OPF file that fails to parse doesn't crash."""
        filepath = tmp_path / "badopf.epub"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "chapter1.xhtml",
                "<html><body><p>" + "Valid content in chapter. " * 5 + "</p></body></html>"
            )
            zf.writestr("metadata.opf", "not valid xml at all <><><<<")
        filepath.write_bytes(buf.getvalue())

        text, title = await _extract_epub_text(filepath)
        assert "Valid content" in text
        # Title falls back to stem since OPF parse failed
        assert title == "badopf"
