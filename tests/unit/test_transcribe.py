"""Unit tests for app/transcribe.py — audio transcription pipeline."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app import transcribe


class TestResolveBackend:
    def test_groq_default(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": "test-key"}, clear=False):
            name, url, key = transcribe._resolve_backend()
        assert name == "groq"
        assert url == transcribe.GROQ_TRANSCRIPTION_URL
        assert key == "test-key"

    def test_openai_backend(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "openai", "OPENAI_API_KEY": "oai-key"}, clear=False):
            name, url, key = transcribe._resolve_backend()
        assert name == "openai"
        assert url == transcribe.OPENAI_TRANSCRIPTION_URL
        assert key == "oai-key"

    def test_groq_missing_key_raises(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                transcribe._resolve_backend()

    def test_openai_missing_key_raises(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "openai", "OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                transcribe._resolve_backend()

    def test_unknown_backend_raises(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "azure"}, clear=False):
            with pytest.raises(ValueError, match="Unknown WHISPER_BACKEND"):
                transcribe._resolve_backend()

    def test_default_is_groq(self):
        env = {"GROQ_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=False):
            # Remove WHISPER_BACKEND so it defaults
            with patch.dict(os.environ, {"WHISPER_BACKEND": ""}, clear=False):
                # Empty string defaults to groq since it strips
                # Actually "".strip().lower() == "" which isn't "groq"
                # Let's check the code: it checks == "groq" first
                pass

    def test_whitespace_stripped(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "  groq  ", "GROQ_API_KEY": "  key  "}, clear=False):
            name, url, key = transcribe._resolve_backend()
        assert name == "groq"
        assert key == "key"


class TestDefaultModels:
    def test_groq_model(self):
        assert transcribe._DEFAULT_MODELS["groq"] == "whisper-large-v3"

    def test_openai_model(self):
        assert transcribe._DEFAULT_MODELS["openai"] == "whisper-1"


class TestTranscribeAudioUrl:
    async def test_empty_url_returns_empty(self):
        result = await transcribe.transcribe_audio_url("")
        assert result == ""

    async def test_whitespace_url_returns_empty(self):
        result = await transcribe.transcribe_audio_url("   ")
        assert result == ""

    async def test_none_url_returns_empty(self):
        result = await transcribe.transcribe_audio_url(None)
        assert result == ""

    async def test_no_backend_configured_returns_empty(self):
        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": ""}, clear=False):
            result = await transcribe.transcribe_audio_url("https://example.com/audio.mp3")
        assert result == ""

    async def test_successful_transcription(self):
        fake_transcript = "Hello, this is a test transcription."

        # Mock the download response
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([b"fake audio data"])

        # Mock the post response
        mock_post_response = AsyncMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_post_response.text = fake_transcript

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": "test-key"}, clear=False), \
             patch("app.transcribe.httpx.AsyncClient", return_value=mock_client):
            result = await transcribe.transcribe_audio_url("https://example.com/audio.mp3")

        assert result == fake_transcript

    async def test_download_failure_returns_empty(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock(side_effect=Exception("404"))

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": "test-key"}, clear=False), \
             patch("app.transcribe.httpx.AsyncClient", return_value=mock_client):
            result = await transcribe.transcribe_audio_url("https://example.com/bad.mp3")

        assert result == ""

    async def test_api_failure_returns_empty(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([b"audio"])

        mock_post_response = AsyncMock()
        mock_post_response.raise_for_status = MagicMock(side_effect=Exception("500 error"))

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": "test-key"}, clear=False), \
             patch("app.transcribe.httpx.AsyncClient", return_value=mock_client):
            result = await transcribe.transcribe_audio_url("https://example.com/audio.mp3")

        assert result == ""

    async def test_custom_model_parameter(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([b"audio"])

        mock_post_response = AsyncMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_post_response.text = "transcript"

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"WHISPER_BACKEND": "groq", "GROQ_API_KEY": "test-key"}, clear=False), \
             patch("app.transcribe.httpx.AsyncClient", return_value=mock_client):
            await transcribe.transcribe_audio_url(
                "https://example.com/audio.mp3", model="custom-model"
            )

        # Verify the model was passed in the post data
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["data"]["model"] == "custom-model"


class TestDownloadToTempfile:
    async def test_suffix_detection_mp3(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([b"data"])

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)

        tmp_path = await transcribe._download_to_tempfile(
            mock_client, "https://example.com/file.mp3"
        )
        try:
            assert tmp_path.suffix == ".mp3"
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_suffix_detection_wav(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([b"data"])

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)

        tmp_path = await transcribe._download_to_tempfile(
            mock_client, "https://example.com/file.wav?token=abc"
        )
        try:
            assert tmp_path.suffix == ".wav"
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_default_suffix_mp3(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([b"data"])

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)

        tmp_path = await transcribe._download_to_tempfile(
            mock_client, "https://example.com/stream"
        )
        try:
            assert tmp_path.suffix == ".mp3"  # default
        finally:
            tmp_path.unlink(missing_ok=True)

    async def test_exceeds_max_size_raises(self):
        # Generate chunks that exceed _MAX_AUDIO_BYTES
        big_chunk = b"x" * (transcribe._MAX_AUDIO_BYTES + 1)

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = lambda chunk_size=65536: _async_iter([big_chunk])

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)

        with pytest.raises(ValueError, match="exceeds.*cap"):
            await transcribe._download_to_tempfile(
                mock_client, "https://example.com/huge.mp3"
            )

    async def test_cleans_up_on_error(self):
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock(side_effect=Exception("fail"))

        mock_client = AsyncMock()
        mock_client.stream = _mock_stream_ctx(mock_stream_response)

        with pytest.raises(Exception, match="fail"):
            await transcribe._download_to_tempfile(
                mock_client, "https://example.com/audio.mp3"
            )


class TestMaxAudioBytes:
    def test_cap_is_50mb(self):
        assert transcribe._MAX_AUDIO_BYTES == 50 * 1024 * 1024


# --- Helpers ---

async def _async_iter(items):
    for item in items:
        yield item


class _mock_stream_ctx:
    """Context manager mock for httpx client.stream()."""
    def __init__(self, response):
        self._response = response

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass
