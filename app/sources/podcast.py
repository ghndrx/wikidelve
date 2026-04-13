"""
Podcast feed ingestion — RSS discovery → audio URL → Whisper transcription.

Configured via the ``PODCAST_WATCHES`` env var, JSON-encoded:

    PODCAST_WATCHES='[
      {"kb": "personal", "url": "https://changelog.com/podcast/feed"},
      {"kb": "personal", "url": "https://podcastfeeds.nbcnews.com/HL4TzgYC"}
    ]'

The cron task walks each feed once a week. For every new episode (one
not already mirrored as a wiki article), it enqueues an
``media_audio_task`` job to download the MP3, transcribe it via
Whisper, and synthesize the transcript into an article.

Re-uses the lightweight RSS parser from ``app/sources/rss.py`` so we
don't add a feedparser dependency for podcast feeds.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from app.config import ARQ_QUEUE_NAME
from app.sources.rss import _parse_feed
from app.wiki import find_related_article

logger = logging.getLogger("kb-service.sources.podcast")

# Pull the audio URL out of an <enclosure url="..." /> tag — RSS 2.0
# podcast feeds use this; some Atom feeds use <link rel="enclosure">
# instead, which the regex also handles.
_ENCLOSURE_RE = re.compile(
    r'<(?:enclosure|link[^>]*?rel="enclosure")[^>]*?(?:url|href)="([^"]+)"',
    re.IGNORECASE,
)
# Cap on episodes processed per feed per cron run.
_MAX_EPISODES_PER_FEED = 5


def _load_podcast_watches() -> list[dict]:
    raw = os.getenv("PODCAST_WATCHES", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("PODCAST_WATCHES is not valid JSON: %s", exc)
        return []
    if not isinstance(parsed, list):
        return []
    return [w for w in parsed if isinstance(w, dict) and w.get("kb") and w.get("url")]


def _extract_audio_urls_from_feed(feed_text: str) -> dict[str, str]:
    """Map ``item.link`` → ``audio enclosure URL`` for every podcast item.

    The lightweight ``_parse_feed`` helper from ``app/sources/rss`` only
    extracts title/url/summary, but for podcasts we also need the
    enclosure URL. We re-walk the raw item blocks here to grab it,
    keyed by the item's web link so the caller can join with the
    parsed entries.
    """
    if not feed_text:
        return {}

    # Find each <item> block and pull out (link, enclosure_url).
    item_re = re.compile(r"<item>([\s\S]*?)</item>", re.IGNORECASE)
    link_re = re.compile(r"<link[^>]*>([\s\S]*?)</link>", re.IGNORECASE)
    out: dict[str, str] = {}
    for match in item_re.findall(feed_text):
        link_m = link_re.search(match)
        link = (link_m.group(1).strip() if link_m else "")
        if not link:
            continue
        enc_m = _ENCLOSURE_RE.search(match)
        if not enc_m:
            continue
        out[link] = enc_m.group(1).strip()
    return out


async def run_podcast_discovery(arq_pool: Any = None) -> dict:
    """Walk every configured podcast feed and enqueue new episodes for
    transcription.

    Existing episodes (matched against the wiki via ``find_related_article``)
    are skipped. New episodes are enqueued onto the ``media_audio_task``
    arq job so the worker downloads + transcribes them off the cron path.
    """
    watches = _load_podcast_watches()
    if not watches:
        logger.info("Podcast discovery: no PODCAST_WATCHES configured")
        return {"feeds": 0, "queued": 0}

    feeds_processed = 0
    feeds_failed = 0
    enqueued = 0
    skipped_existing = 0

    async with httpx.AsyncClient(
        timeout=30,
        headers={"User-Agent": "wikidelve/1.0 podcast-discovery"},
        follow_redirects=True,
    ) as client:
        for watch in watches:
            kb = watch["kb"]
            feed_url = watch["url"]
            try:
                resp = await client.get(feed_url)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("Podcast feed %s failed: %s", feed_url, exc)
                feeds_failed += 1
                continue

            feeds_processed += 1
            entries = _parse_feed(resp.text)[:_MAX_EPISODES_PER_FEED]
            audio_urls = _extract_audio_urls_from_feed(resp.text)

            for entry in entries:
                title = entry.get("title", "").strip()
                page_url = entry.get("url", "").strip()
                if not title or not page_url:
                    continue

                # Skip episodes we've already mirrored
                try:
                    if find_related_article(kb, title):
                        skipped_existing += 1
                        continue
                except Exception:
                    pass

                audio_url = audio_urls.get(page_url) or ""
                if not audio_url:
                    logger.debug("No enclosure for %s", page_url)
                    continue

                if arq_pool is None:
                    # Cron path: assume the worker has been started and
                    # we have a pool. Without one we can only log.
                    logger.info(
                        "Podcast new episode (no arq pool): %s", title[:80],
                    )
                    continue

                try:
                    await arq_pool.enqueue_job(
                        "media_audio_task",
                        audio_url,
                        title,
                        _queue_name=ARQ_QUEUE_NAME,
                    )
                    enqueued += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to enqueue podcast episode %s: %s",
                        title[:60], exc,
                    )

    logger.info(
        "Podcast discovery: %d feeds (%d failed), %d new episodes queued, "
        "%d already in wiki",
        feeds_processed, feeds_failed, enqueued, skipped_existing,
    )
    return {
        "feeds": feeds_processed,
        "feeds_failed": feeds_failed,
        "queued": enqueued,
        "skipped_existing": skipped_existing,
    }
