"""
Audio transcription pipeline.

Routes audio URLs through a Whisper-compatible HTTP API and returns the
plain-text transcript. Two backends are supported, selected via the
``WHISPER_BACKEND`` env var:

  - ``groq`` (default): Groq's hosted Whisper-large-v3. Free tier;
    requires ``GROQ_API_KEY``. Endpoint:
    ``https://api.groq.com/openai/v1/audio/transcriptions``.
  - ``openai``: OpenAI's whisper-1 endpoint. Requires ``OPENAI_API_KEY``.
    Endpoint: ``https://api.openai.com/v1/audio/transcriptions``.

Both backends accept multipart/form-data uploads of an audio file. We
download the source URL into a temp file, post it to the chosen backend,
and return the transcript text.

Used by ``app/sources/podcast.py`` to transcribe podcast episodes and by
``app/media.py`` (when the YouTube caption path fails) to handle
videos with no captions.
"""

import logging
import os
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger("kb-service.transcribe")

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"

# Default model IDs per backend
_DEFAULT_MODELS = {
    "groq": "whisper-large-v3",
    "openai": "whisper-1",
}

# Hard cap on file size we'll download (50 MB) — protects against
# accidentally trying to transcribe a multi-hour video.
_MAX_AUDIO_BYTES = 50 * 1024 * 1024


def _resolve_backend() -> tuple[str, str, str]:
    """Return ``(backend_name, api_url, api_key)`` from env vars.

    Raises ``ValueError`` if no usable backend is configured.
    """
    backend = os.getenv("WHISPER_BACKEND", "groq").strip().lower()
    if backend == "groq":
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError("GROQ_API_KEY not set; required for WHISPER_BACKEND=groq")
        return ("groq", GROQ_TRANSCRIPTION_URL, key)
    if backend == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY not set; required for WHISPER_BACKEND=openai")
        return ("openai", OPENAI_TRANSCRIPTION_URL, key)
    raise ValueError(f"Unknown WHISPER_BACKEND: {backend!r}; use 'groq' or 'openai'")


async def _download_to_tempfile(
    client: httpx.AsyncClient, url: str,
) -> Path:
    """Stream a URL to a temp file. Caller is responsible for unlinking it."""
    # Pick a sensible suffix from the URL so the API can detect format.
    suffix = ".mp3"
    for ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mp4"):
        if ext in url.lower():
            suffix = ext
            break

    fd, tmp_path_str = tempfile.mkstemp(prefix="wikidelve_audio_", suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    bytes_written = 0
    try:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as fh:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > _MAX_AUDIO_BYTES:
                        raise ValueError(
                            f"Audio file exceeds {_MAX_AUDIO_BYTES // (1024 * 1024)}MB cap"
                        )
                    fh.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path


async def transcribe_audio_url(url: str, model: str | None = None) -> str:
    """Download an audio URL and return the transcript text.

    Returns the empty string on any failure (logged). Callers should
    treat empty output as "transcription unavailable" and fall back
    accordingly.
    """
    if not url or not url.strip():
        return ""

    try:
        backend, api_url, api_key = _resolve_backend()
    except ValueError as exc:
        logger.warning("Whisper backend not configured: %s", exc)
        return ""

    chosen_model = model or _DEFAULT_MODELS.get(backend, "whisper-1")

    tmp_path: Path | None = None
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            tmp_path = await _download_to_tempfile(client, url)

            with tmp_path.open("rb") as fh:
                files = {"file": (tmp_path.name, fh, "application/octet-stream")}
                data = {"model": chosen_model, "response_format": "text"}
                resp = await client.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data=data,
                )

        resp.raise_for_status()
        # Both Groq + OpenAI return plain text when response_format=text
        text = (resp.text or "").strip()
        if not text:
            logger.warning("Whisper backend returned empty transcript for %s", url)
        return text
    except Exception as exc:
        logger.warning("Audio transcription failed for %s: %s", url, exc)
        return ""
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
