"""
RSS / Atom feed cron — daily passive discovery.

Unlike the search providers (Serper, arXiv, HN, Wikipedia) which run at
research time, the RSS path is a **background discovery task**:

  1. Cron walks the configured per-KB feeds (env var ``RSS_WATCHES``).
  2. For each feed, fetches headlines + summaries.
  3. For each entry: try to match the title against an existing wiki
     article via ``find_related_article``. If matched, just log it as a
     freshness signal — the stale-article rerun strategy will pick it up.
  4. Unmatched entries become ``topic_candidate`` rows with
     ``source='rss'`` so the auto-discovery enqueue loop will research
     them when budget allows.

Configuration via env var, JSON-encoded:

    RSS_WATCHES='[
      {"kb": "personal", "url": "https://lobste.rs/rss"},
      {"kb": "personal", "url": "https://blog.rust-lang.org/feed.xml"}
    ]'

This mirrors the existing ``LOCAL_RESEARCH_WATCHES`` env-var pattern.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

from app import db
from app.wiki import find_related_article

logger = logging.getLogger("kb-service.sources.rss")

# Atom and RSS use different element names but the same idea.
# We extract <item>/<entry> blocks then pull title + link + summary out.
_RSS_ITEM_RE = re.compile(r"<item>([\s\S]*?)</item>", re.IGNORECASE)
_ATOM_ENTRY_RE = re.compile(r"<entry>([\s\S]*?)</entry>", re.IGNORECASE)
_TITLE_RE = re.compile(r"<title[^>]*>([\s\S]*?)</title>", re.IGNORECASE)
_LINK_HREF_RE = re.compile(r'<link [^>]*?href="([^"]+)"', re.IGNORECASE)
_LINK_TEXT_RE = re.compile(r"<link[^>]*>([\s\S]*?)</link>", re.IGNORECASE)
_DESC_RE = re.compile(
    r"<(?:description|summary|content)[^>]*>([\s\S]*?)</(?:description|summary|content)>",
    re.IGNORECASE,
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_CDATA_RE = re.compile(r"<!\[CDATA\[([\s\S]*?)\]\]>")
_HARD_CAP_PER_FEED = 30  # Don't process more than this many entries per feed per run.


def _decode_xml(s: str) -> str:
    """Decode CDATA wrappers + a small set of XML entities."""
    if not s:
        return ""
    # CDATA: keep inner text
    s = _CDATA_RE.sub(lambda m: m.group(1), s)
    # Strip any inline HTML tags from descriptions
    s = _TAG_STRIP_RE.sub("", s)
    s = (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&#39;", "'")
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_feed(text: str) -> list[dict]:
    """Best-effort feed parser. Returns list of {title, url, summary}.

    Handles RSS 2.0 (<item>) and Atom (<entry>). Tolerant of malformed
    feeds — bad entries are skipped, the rest still parse.
    """
    if not text:
        return []

    # Pick the entry container that matches the feed type
    items = _RSS_ITEM_RE.findall(text)
    if not items:
        items = _ATOM_ENTRY_RE.findall(text)

    out: list[dict] = []
    for item in items[:_HARD_CAP_PER_FEED]:
        title_m = _TITLE_RE.search(item)
        title = _decode_xml(title_m.group(1)) if title_m else ""
        if not title:
            continue

        # Atom: <link href="..."/> ; RSS: <link>https://...</link>
        url = ""
        link_href = _LINK_HREF_RE.search(item)
        if link_href:
            url = link_href.group(1).strip()
        else:
            link_text = _LINK_TEXT_RE.search(item)
            if link_text:
                url = _decode_xml(link_text.group(1)).strip()
        if not url:
            continue

        summary_m = _DESC_RE.search(item)
        summary = _decode_xml(summary_m.group(1)) if summary_m else ""

        out.append({"title": title, "url": url, "summary": summary[:600]})
    return out


def _load_rss_watches() -> list[dict]:
    """Read RSS_WATCHES env var. Returns [] if unset or malformed."""
    raw = os.getenv("RSS_WATCHES", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("RSS_WATCHES is not valid JSON: %s", exc)
        return []
    if not isinstance(parsed, list):
        logger.warning("RSS_WATCHES must be a JSON list of {kb, url} objects")
        return []
    return [w for w in parsed if isinstance(w, dict) and w.get("kb") and w.get("url")]


async def run_rss_discovery() -> dict:
    """One-shot RSS discovery pass over all configured feeds.

    Used by ``app/worker.py:rss_discovery_cron_task`` and the manual
    /admin trigger. Returns counts for the admin dashboard.
    """
    watches = _load_rss_watches()
    if not watches:
        logger.info("RSS discovery: no RSS_WATCHES configured")
        return {"feeds": 0, "matched": 0, "candidates": 0}

    matched_count = 0
    candidate_count = 0
    feeds_processed = 0
    feeds_failed = 0

    async with httpx.AsyncClient(
        timeout=20,
        headers={"User-Agent": "wikidelve/1.0 RSS-discovery"},
    ) as client:
        for watch in watches:
            kb = watch["kb"]
            url = watch["url"]
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                entries = _parse_feed(resp.text)
            except Exception as exc:
                logger.warning("RSS feed %s failed: %s", url, exc)
                feeds_failed += 1
                continue

            feeds_processed += 1
            for entry in entries:
                # Step 1: try to match an existing article — that's a
                # freshness signal, not a new candidate.
                try:
                    existing = find_related_article(kb, entry["title"])
                except Exception:
                    existing = None
                if existing:
                    matched_count += 1
                    logger.debug(
                        "RSS match: '%s' → existing %s/%s",
                        entry["title"][:60], kb, existing["slug"],
                    )
                    continue

                # Step 2: cooldown check so we don't queue topics we
                # already researched recently.
                try:
                    cooldown = await db.check_cooldown(entry["title"])
                except Exception:
                    cooldown = None
                if cooldown:
                    continue

                # Step 3: insert as candidate. Dedupes via UNIQUE constraint.
                try:
                    inserted = await db.insert_topic_candidate(
                        kb, entry["title"][:200],
                        "rss", url, 1.0,
                    )
                    if inserted:
                        candidate_count += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to insert RSS candidate '%s': %s",
                        entry["title"][:60], exc,
                    )

    logger.info(
        "RSS discovery: %d feeds (%d failed), %d existing matches, %d new candidates",
        feeds_processed, feeds_failed, matched_count, candidate_count,
    )
    return {
        "feeds": feeds_processed,
        "feeds_failed": feeds_failed,
        "matched": matched_count,
        "candidates": candidate_count,
    }
