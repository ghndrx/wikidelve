"""Unit tests for app.media — YouTube transcript extraction, VTT parsing,
transcript synthesis, and the full ingest pipeline.

All subprocess calls and external HTTP are mocked.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import media


# ---------------------------------------------------------------------------
# _parse_vtt
# ---------------------------------------------------------------------------


class TestParseVtt:
    def test_strips_webvtt_header(self):
        vtt = "WEBVTT\n\nhello world"
        assert media._parse_vtt(vtt) == "hello world"

    def test_strips_timestamps(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello there\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "General Kenobi\n"
        )
        assert media._parse_vtt(vtt) == "Hello there General Kenobi"

    def test_strips_kind_and_language_headers(self):
        vtt = "WEBVTT\nKind: captions\nLanguage: en\n\nsome text"
        assert media._parse_vtt(vtt) == "some text"

    def test_strips_numeric_cue_ids(self):
        vtt = "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nword\n"
        assert media._parse_vtt(vtt) == "word"

    def test_strips_html_tags(self):
        vtt = "WEBVTT\n\n<b>bold</b> and <i>italic</i>"
        assert media._parse_vtt(vtt) == "bold and italic"

    def test_deduplicates_lines(self):
        vtt = "WEBVTT\n\nhello\nhello\nworld\n"
        assert media._parse_vtt(vtt) == "hello world"

    def test_strips_positioning(self):
        vtt = "WEBVTT\n\nalign:start position:10%\nactual text\n"
        assert media._parse_vtt(vtt) == "actual text"

    def test_empty_input(self):
        assert media._parse_vtt("") == ""

    def test_only_headers(self):
        vtt = "WEBVTT\nKind: captions\nLanguage: en\n"
        assert media._parse_vtt(vtt) == ""


# ---------------------------------------------------------------------------
# get_youtube_transcript
# ---------------------------------------------------------------------------


def _make_proc_mock(returncode=0, stdout=b"", stderr=b""):
    """Create a mock asyncio.Process."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


class TestGetYoutubeTranscript:
    @pytest.mark.asyncio
    async def test_yt_dlp_metadata_failure_returns_none(self):
        proc = _make_proc_mock(returncode=1, stderr=b"error happened")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await media.get_youtube_transcript("https://youtu.be/abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_yt_dlp_timeout_returns_none(self):
        async def slow_communicate():
            raise asyncio.TimeoutError()

        proc = AsyncMock()
        proc.communicate = slow_communicate
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await media.get_youtube_transcript("https://youtu.be/abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_yt_dlp_json_parse_error_returns_none(self):
        proc = _make_proc_mock(returncode=0, stdout=b"not json")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await media.get_youtube_transcript("https://youtu.be/abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_subtitles_falls_back_to_description(self):
        info = {
            "title": "Cool Video",
            "channel": "TestChannel",
            "duration": 120,
            "description": "A" * 250,  # >200 chars
        }
        meta_proc = _make_proc_mock(
            returncode=0, stdout=json.dumps(info).encode(),
        )
        # _extract_subtitles will also call create_subprocess_exec
        # but won't find any VTT files. We simulate by making it
        # return None.
        with patch.object(media, "_extract_subtitles", return_value=None):
            with patch("asyncio.create_subprocess_exec", return_value=meta_proc):
                result = await media.get_youtube_transcript("https://youtu.be/abc")

        assert result is not None
        assert result["title"] == "Cool Video"
        assert "[Video description" in result["transcript"]
        assert result["channel"] == "TestChannel"
        assert result["duration"] == 120

    @pytest.mark.asyncio
    async def test_no_subtitles_short_description_returns_none(self):
        info = {
            "title": "Short Vid",
            "channel": "C",
            "duration": 10,
            "description": "brief",
        }
        meta_proc = _make_proc_mock(
            returncode=0, stdout=json.dumps(info).encode(),
        )
        with patch.object(media, "_extract_subtitles", return_value=None):
            with patch("asyncio.create_subprocess_exec", return_value=meta_proc):
                result = await media.get_youtube_transcript("https://youtu.be/abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_transcript_extraction(self):
        info = {
            "title": "Great Talk",
            "channel": "Conf",
            "duration": 600,
            "description": "desc",
        }
        meta_proc = _make_proc_mock(
            returncode=0, stdout=json.dumps(info).encode(),
        )
        with patch.object(media, "_extract_subtitles", return_value="Hello world content"):
            with patch("asyncio.create_subprocess_exec", return_value=meta_proc):
                result = await media.get_youtube_transcript("https://youtu.be/abc")

        assert result is not None
        assert result["title"] == "Great Talk"
        assert result["transcript"] == "Hello world content"
        assert result["word_count"] == 3

    @pytest.mark.asyncio
    async def test_missing_metadata_uses_defaults(self):
        info = {}  # No title, channel, duration, description
        meta_proc = _make_proc_mock(
            returncode=0, stdout=json.dumps(info).encode(),
        )
        with patch.object(media, "_extract_subtitles", return_value="transcript text"):
            with patch("asyncio.create_subprocess_exec", return_value=meta_proc):
                result = await media.get_youtube_transcript("https://youtu.be/abc")

        assert result["title"] == "Unknown Video"
        assert result["channel"] == "Unknown"
        assert result["duration"] == 0


# ---------------------------------------------------------------------------
# _extract_subtitles
# ---------------------------------------------------------------------------


class TestExtractSubtitles:
    @pytest.mark.asyncio
    async def test_subprocess_failure_returns_none(self):
        proc = _make_proc_mock(returncode=1, stderr=b"fail")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await media._extract_subtitles("https://youtu.be/abc", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_subprocess_timeout_returns_none(self):
        async def slow():
            raise asyncio.TimeoutError()

        proc = AsyncMock()
        proc.communicate = slow
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            result = await media._extract_subtitles("https://youtu.be/abc", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_no_vtt_files_returns_none(self, tmp_path):
        proc = _make_proc_mock(returncode=0)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            # The tempdir will be empty (no .vtt files written)
            result = await media._extract_subtitles("https://youtu.be/abc", {})
        assert result is None

    @pytest.mark.asyncio
    async def test_vtt_file_is_parsed(self, tmp_path):
        vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world\n"

        proc = _make_proc_mock(returncode=0)

        # We need to intercept the tempdir to write a VTT file into it
        import tempfile
        real_tmpdir = tempfile.mkdtemp()
        from pathlib import Path
        (Path(real_tmpdir) / "abc.en.vtt").write_text(vtt_content)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            with patch("tempfile.TemporaryDirectory") as mock_td:
                mock_td.return_value.__enter__ = MagicMock(return_value=real_tmpdir)
                mock_td.return_value.__exit__ = MagicMock(return_value=False)
                result = await media._extract_subtitles("https://youtu.be/abc", {})

        assert result == "Hello world"

        # Cleanup
        import shutil
        shutil.rmtree(real_tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# synthesize_transcript
# ---------------------------------------------------------------------------


class TestSynthesizeTranscript:
    @pytest.mark.asyncio
    async def test_no_llm_returns_truncated_transcript(self, monkeypatch):
        monkeypatch.setattr(media, "LLM_PROVIDER", "none")
        monkeypatch.setattr(media, "MINIMAX_API_KEY", "")

        result = await media.synthesize_transcript(
            "Title", "Channel", "x" * 10000, "http://example.com",
        )
        assert len(result) <= 5000

    @pytest.mark.asyncio
    async def test_calls_llm_when_provider_set(self, monkeypatch):
        monkeypatch.setattr(media, "LLM_PROVIDER", "bedrock")
        monkeypatch.setattr(media, "MINIMAX_API_KEY", "")

        fake_llm = AsyncMock(return_value="# Synthesized Article\n\nBody")
        monkeypatch.setattr(media, "llm_chat", fake_llm)

        result = await media.synthesize_transcript(
            "Title", "Channel", "Some transcript", "http://example.com",
        )
        assert result == "# Synthesized Article\n\nBody"
        fake_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_calls_llm_when_minimax_key_set(self, monkeypatch):
        monkeypatch.setattr(media, "LLM_PROVIDER", "none")
        monkeypatch.setattr(media, "MINIMAX_API_KEY", "sk-test123")

        fake_llm = AsyncMock(return_value="# Output")
        monkeypatch.setattr(media, "llm_chat", fake_llm)

        result = await media.synthesize_transcript(
            "Title", "Channel", "transcript", "http://example.com",
        )
        assert result == "# Output"

    @pytest.mark.asyncio
    async def test_truncates_long_transcripts(self, monkeypatch):
        monkeypatch.setattr(media, "LLM_PROVIDER", "bedrock")
        monkeypatch.setattr(media, "MINIMAX_API_KEY", "")

        calls = []

        async def capture_llm(system, prompt, **kw):
            calls.append(prompt)
            return "# Article"

        monkeypatch.setattr(media, "llm_chat", capture_llm)

        long_transcript = "word " * 5000  # Way over 12000 chars
        await media.synthesize_transcript(
            "Title", "Channel", long_transcript, "http://example.com",
        )

        # Verify transcript was truncated in the prompt
        assert "[Transcript truncated]" in calls[0]

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(media, "LLM_PROVIDER", "bedrock")
        monkeypatch.setattr(media, "MINIMAX_API_KEY", "")

        monkeypatch.setattr(
            media, "llm_chat",
            AsyncMock(side_effect=Exception("API error")),
        )

        result = await media.synthesize_transcript(
            "My Video", "Ch", "some content", "http://example.com",
        )
        assert "## Transcript Summary" in result
        assert "My Video" in result
        assert "some content" in result


# ---------------------------------------------------------------------------
# ingest_youtube — full pipeline
# ---------------------------------------------------------------------------


class TestIngestYoutube:
    @pytest.mark.asyncio
    async def test_no_transcript_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            media, "get_youtube_transcript", AsyncMock(return_value=None),
        )
        fake_update = AsyncMock()
        monkeypatch.setattr(media.db, "update_job", fake_update)

        result = await media.ingest_youtube("https://youtu.be/fail", job_id=42)

        assert result["error"] == "No transcript available"
        # Should have set job status to error
        fake_update.assert_any_await(42, status="error", error="No transcript available")

    @pytest.mark.asyncio
    async def test_no_transcript_without_job_id(self, monkeypatch):
        monkeypatch.setattr(
            media, "get_youtube_transcript", AsyncMock(return_value=None),
        )
        result = await media.ingest_youtube("https://youtu.be/fail")
        assert result["error"] == "No transcript available"

    @pytest.mark.asyncio
    async def test_happy_path_writes_files_and_updates_wiki(self, monkeypatch):
        video_data = {
            "title": "Test Video",
            "channel": "TestChan",
            "duration": 300,
            "transcript": "This is the transcript",
            "url": "https://youtu.be/ok",
            "word_count": 5,
        }
        monkeypatch.setattr(
            media, "get_youtube_transcript",
            AsyncMock(return_value=video_data),
        )
        monkeypatch.setattr(
            media, "synthesize_transcript",
            AsyncMock(return_value="# Synthesized\n\nArticle body"),
        )

        # storage and create_or_update_article are imported lazily inside
        # ingest_youtube, so we patch them at the module they live in.
        fake_storage = MagicMock()
        fake_storage.write_text = MagicMock()
        import app.storage as _storage_mod
        monkeypatch.setattr(_storage_mod, "write_text", fake_storage.write_text)

        import app.config as _config_mod
        monkeypatch.setattr(_config_mod, "RESEARCH_KB", "test-kb")

        fake_create = AsyncMock(return_value=("test-video", "created"))
        import app.wiki as _wiki_mod
        monkeypatch.setattr(_wiki_mod, "create_or_update_article", fake_create)

        fake_update = AsyncMock()
        monkeypatch.setattr(media.db, "update_job", fake_update)

        result = await media.ingest_youtube("https://youtu.be/ok", job_id=10)

        assert result["title"] == "Test Video"
        assert result["slug"] == "test-video"
        assert result["change_type"] == "created"
        assert result["channel"] == "TestChan"

        # Verify job status updates
        fake_update.assert_any_await(10, status="extracting_transcript")
        fake_update.assert_any_await(10, status="synthesizing", sources_count=1)

    @pytest.mark.asyncio
    async def test_wiki_merge_failure_returns_error_info(self, monkeypatch):
        video_data = {
            "title": "Wiki Fail",
            "channel": "Ch",
            "duration": 60,
            "transcript": "text",
            "url": "https://youtu.be/wf",
            "word_count": 1,
        }
        monkeypatch.setattr(
            media, "get_youtube_transcript",
            AsyncMock(return_value=video_data),
        )
        monkeypatch.setattr(
            media, "synthesize_transcript",
            AsyncMock(return_value="# Content"),
        )

        import app.storage as _storage_mod
        monkeypatch.setattr(_storage_mod, "write_text", MagicMock())
        import app.config as _config_mod
        monkeypatch.setattr(_config_mod, "RESEARCH_KB", "test-kb")
        import app.wiki as _wiki_mod
        monkeypatch.setattr(
            _wiki_mod, "create_or_update_article",
            AsyncMock(side_effect=Exception("merge failed")),
        )
        monkeypatch.setattr(media.db, "update_job", AsyncMock())

        result = await media.ingest_youtube("https://youtu.be/wf", job_id=5)

        assert "wiki_error" in result
        assert "merge failed" in result["wiki_error"]
        assert result["title"] == "Wiki Fail"


# ---------------------------------------------------------------------------
# ingest_youtube_batch
# ---------------------------------------------------------------------------


class TestIngestYoutubeBatch:
    @pytest.mark.asyncio
    async def test_processes_all_urls(self, monkeypatch):
        call_count = 0

        async def fake_ingest(url, job_id=None):
            nonlocal call_count
            call_count += 1
            return {"url": url, "status": "ok"}

        monkeypatch.setattr(media, "ingest_youtube", fake_ingest)

        results = await media.ingest_youtube_batch([
            "https://youtu.be/a",
            "https://youtu.be/b",
            "https://youtu.be/c",
        ])

        assert len(results) == 3
        assert call_count == 3
        assert results[0]["url"] == "https://youtu.be/a"
        assert results[2]["url"] == "https://youtu.be/c"

    @pytest.mark.asyncio
    async def test_empty_list(self, monkeypatch):
        monkeypatch.setattr(media, "ingest_youtube", AsyncMock())
        results = await media.ingest_youtube_batch([])
        assert results == []
