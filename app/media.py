"""
Media ingestion: YouTube videos and podcasts → wiki articles.

Pipeline:
  1. yt-dlp extracts subtitles/captions (or downloads audio)
  2. If no captions: whisper transcribes audio (future — needs GPU)
  3. Minimax summarizes transcript into structured knowledge
  4. Auto-adds to wiki via create_or_update_article

For now: caption-based extraction (free, fast, no GPU needed).
Whisper transcription can be added when GPU hardware is available.
"""

import asyncio
import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import (
    LLM_PROVIDER,
    MINIMAX_API_KEY,
)
from app.llm import llm_chat
from app import db

logger = logging.getLogger("kb-service.media")


# ─── YouTube Transcript Extraction ──────────────────────────────────────────

async def get_youtube_transcript(url: str) -> Optional[dict]:
    """Extract transcript from a YouTube video using yt-dlp.

    Tries: auto-generated captions → manual subtitles → description.
    Returns dict with: title, transcript, duration, channel, url
    """
    logger.info("Extracting transcript: %s", url)

    # Get video metadata + subtitles
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--sub-langs", "en.*,en",
            "--write-auto-subs",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            logger.warning("yt-dlp metadata failed: %s", stderr.decode()[:200])
            return None

        info = json.loads(stdout.decode())
    except asyncio.TimeoutError:
        logger.warning("yt-dlp timeout for %s", url)
        return None
    except Exception as exc:
        logger.warning("yt-dlp failed for %s: %s", url, exc)
        return None

    title = info.get("title", "Unknown Video")
    channel = info.get("channel", info.get("uploader", "Unknown"))
    duration = info.get("duration", 0)
    description = info.get("description", "")

    # Try to get subtitles
    transcript = await _extract_subtitles(url, info)

    if not transcript:
        # Fallback to description if substantial
        if description and len(description) > 200:
            transcript = f"[Video description — no captions available]\n\n{description}"
        else:
            logger.warning("No transcript available for %s", url)
            return None

    return {
        "title": title,
        "channel": channel,
        "duration": duration,
        "transcript": transcript,
        "url": url,
        "word_count": len(transcript.split()),
    }


async def _extract_subtitles(url: str, info: dict) -> Optional[str]:
    """Download and parse subtitles from a YouTube video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            proc = await asyncio.create_subprocess_exec(
                "yt-dlp",
                "--no-download",
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs", "en.*,en",
                "--sub-format", "vtt",
                "--convert-subs", "vtt",
                "-o", f"{tmpdir}/%(id)s.%(ext)s",
                url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=60)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Subtitle download failed: %s", exc)
            return None

        # Find VTT file
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            return None

        # Parse VTT
        vtt_text = vtt_files[0].read_text(errors="replace")
        return _parse_vtt(vtt_text)


def _parse_vtt(vtt_content: str) -> str:
    """Parse WebVTT subtitle file into clean text."""
    lines = []
    seen = set()

    for line in vtt_content.split("\n"):
        # Skip VTT headers, timestamps, positioning
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r'^\d{2}:\d{2}', line):
            continue
        if re.match(r'^[\d\-]+$', line):
            continue
        if "align:" in line or "position:" in line:
            continue

        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', line)
        clean = clean.strip()

        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)

    return " ".join(lines)


# ─── Transcript Synthesis ───────────────────────────────────────────────────

async def synthesize_transcript(title: str, channel: str, transcript: str, url: str) -> str:
    """Synthesize a transcript into a structured wiki article via the configured LLM."""
    has_llm = (LLM_PROVIDER == "bedrock") or MINIMAX_API_KEY
    if not has_llm:
        return transcript[:5000]

    # Chunk transcript if too long
    max_transcript = 12000
    if len(transcript) > max_transcript:
        transcript = transcript[:max_transcript] + "\n\n[Transcript truncated]"

    prompt = f"""Synthesize this video/podcast transcript into a well-structured knowledge base article.

VIDEO: {title}
CHANNEL: {channel}
URL: {url}

TRANSCRIPT:
{transcript}

Create a comprehensive article that:
1. Starts with a one-paragraph summary of the key topics discussed
2. Organizes main topics under ## headers
3. Extracts specific technical claims, recommendations, and insights
4. Includes any mentioned tools, versions, URLs, or resources
5. Notes any opinions or predictions clearly as such
6. Adds a "Key Takeaways" section at the end with bullet points
7. Credits the source: "Source: [{title}]({url}) by {channel}"

Write in encyclopedic voice, not conversational. Extract the KNOWLEDGE, not the conversation."""

    system = "You are a knowledge extraction specialist. Convert video/podcast transcripts into structured, factual wiki articles. Focus on extracting actionable technical knowledge."

    try:
        return await llm_chat(system, prompt, max_tokens=4000, temperature=0.2)
    except Exception as exc:
        logger.warning("Transcript synthesis failed: %s", exc)

    # Fallback: return raw transcript with basic formatting
    return f"## Transcript Summary\n\nSource: [{title}]({url}) by {channel}\n\n{transcript[:5000]}"


# ─── Full Pipeline ──────────────────────────────────────────────────────────

async def ingest_youtube(url: str, job_id: Optional[int] = None) -> dict:
    """Full pipeline: YouTube URL → transcript → synthesis → wiki article."""
    if job_id:
        await db.update_job(job_id, status="extracting_transcript")

    # Extract transcript
    video = await get_youtube_transcript(url)
    if not video:
        if job_id:
            await db.update_job(job_id, status="error", error="No transcript available")
        return {"error": "No transcript available", "url": url}

    if job_id:
        await db.update_job(job_id, status="synthesizing", sources_count=1)

    # Synthesize
    content = await synthesize_transcript(
        video["title"], video["channel"], video["transcript"], url
    )

    # Save raw transcript + synthesized output via storage
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe_title = re.sub(r'[^\w\s-]', '', video["title"])[:60].strip()
    from app import storage
    from app.config import RESEARCH_KB
    storage.write_text(
        RESEARCH_KB,
        f"media/{today}-{safe_title.replace(' ', '-')}.txt",
        f"Title: {video['title']}\nChannel: {video['channel']}\nURL: {url}\n\n{video['transcript']}",
    )
    storage.write_text(
        RESEARCH_KB,
        f"media-{today}-{safe_title.replace(' ', '-')}.md",
        f"# {video['title']}\n\n{content}\n",
    )

    if job_id:
        await db.update_job(
            job_id,
            status="complete",
            word_count=len(content.split()),
            content=content,
        )

    # Auto-add to wiki
    from app.wiki import create_or_update_article
    try:
        slug, change_type = await create_or_update_article("personal", video["title"], content, source_type="media")
        if job_id:
            await db.update_job(job_id, added_to_wiki=1)
        return {
            "title": video["title"],
            "channel": video["channel"],
            "duration": video["duration"],
            "transcript_words": video["word_count"],
            "article_words": len(content.split()),
            "slug": slug,
            "change_type": change_type,
            "url": url,
        }
    except Exception as exc:
        logger.warning("Failed to add media article to wiki: %s", exc)
        return {
            "title": video["title"],
            "channel": video["channel"],
            "article_words": len(content.split()),
            "url": url,
            "wiki_error": str(exc),
        }


async def ingest_youtube_batch(urls: list[str]) -> list[dict]:
    """Ingest multiple YouTube videos."""
    results = []
    for url in urls:
        result = await ingest_youtube(url)
        results.append(result)
    return results
