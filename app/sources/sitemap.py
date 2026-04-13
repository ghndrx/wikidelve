"""
Sitemap crawler — weekly passive discovery.

Walks ``sitemap.xml`` for trusted authoritative domains and queues any
URL we haven't seen yet as a research candidate. Configured via the
``SITEMAP_WATCHES`` env var (JSON, mirrors ``RSS_WATCHES`` shape):

    SITEMAP_WATCHES='[
      {"kb": "personal", "url": "https://kubernetes.io/docs/sitemap.xml", "label": "k8s docs"},
      {"kb": "personal", "url": "https://react.dev/sitemap.xml", "label": "React docs"}
    ]'

For each URL in the sitemap that isn't already in ``research_sources``,
we insert a row into ``topic_candidates`` with ``source='sitemap'``. The
auto-discovery enqueue loop then deep-reads + researches them when budget
allows.

Sitemap index files (sitemaps that point at other sitemaps) are followed
one level deep — that's enough for the common case (kubernetes.io,
react.dev, developer.mozilla.org) without recursing forever on giant
sites.
"""

from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import urlparse

import aiosqlite
import httpx

from app import db
from app.config import DB_PATH

logger = logging.getLogger("kb-service.sources.sitemap")


_LOC_RE = re.compile(r"<loc>([\s\S]*?)</loc>", re.IGNORECASE)
_HARD_CAP_PER_SITEMAP = 200  # Don't add more than this many candidates per cron run.
_SITEMAP_INDEX_DEPTH = 1  # Follow sitemap-of-sitemaps one level deep.


def _load_sitemap_watches() -> list[dict]:
    raw = os.getenv("SITEMAP_WATCHES", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("SITEMAP_WATCHES is not valid JSON: %s", exc)
        return []
    if not isinstance(parsed, list):
        logger.warning("SITEMAP_WATCHES must be a JSON list of {kb, url} objects")
        return []
    return [w for w in parsed if isinstance(w, dict) and w.get("kb") and w.get("url")]


async def _fetch_sitemap_locs(
    client: httpx.AsyncClient, url: str, depth: int = 0,
) -> list[str]:
    """Fetch a sitemap.xml and return all <loc> entries.

    If the sitemap is a sitemap index (entries point at other sitemaps),
    follow them one level deep. Returns a flat list of leaf URLs.
    """
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Sitemap fetch failed for %s: %s", url, exc)
        return []

    locs = [m.strip() for m in _LOC_RE.findall(resp.text) if m.strip()]
    if not locs:
        return []

    # Heuristic: if every loc ends in .xml or contains "sitemap", treat
    # this as a sitemap index and recurse one level.
    looks_like_index = (
        depth < _SITEMAP_INDEX_DEPTH
        and all(
            l.lower().endswith(".xml") or "sitemap" in l.lower()
            for l in locs[:10]
        )
    )
    if not looks_like_index:
        return locs

    # Follow each sub-sitemap (cap to 5 to bound work)
    leaf_urls: list[str] = []
    for sub_url in locs[:5]:
        sub = await _fetch_sitemap_locs(client, sub_url, depth=depth + 1)
        leaf_urls.extend(sub)
    return leaf_urls


async def _existing_source_urls() -> set[str]:
    """Load every URL we've already seen via past research_sources."""
    aconn = await aiosqlite.connect(str(DB_PATH))
    aconn.row_factory = aiosqlite.Row
    try:
        cursor = await aconn.execute("SELECT DISTINCT url FROM research_sources")
        rows = await cursor.fetchall()
        return {r["url"].strip().rstrip("/").lower() for r in rows if r["url"]}
    finally:
        await aconn.close()


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _url_to_topic_label(url: str) -> str:
    """Turn a URL into a short, human-readable topic label."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return parsed.netloc
    # Take the last 1-2 path segments, replace separators with spaces
    parts = [p for p in path.split("/") if p][-2:]
    label = " / ".join(p.replace("-", " ").replace("_", " ") for p in parts)
    return f"{parsed.netloc}: {label}"[:200]


async def run_sitemap_discovery() -> dict:
    """One-shot sitemap discovery pass over all configured domains."""
    watches = _load_sitemap_watches()
    if not watches:
        logger.info("Sitemap discovery: no SITEMAP_WATCHES configured")
        return {"sitemaps": 0, "candidates": 0}

    seen_urls = await _existing_source_urls()
    sitemaps_processed = 0
    sitemaps_failed = 0
    total_candidates = 0

    async with httpx.AsyncClient(
        timeout=30,
        headers={"User-Agent": "wikidelve/1.0 sitemap-discovery"},
        follow_redirects=True,
    ) as client:
        for watch in watches:
            kb = watch["kb"]
            sitemap_url = watch["url"]
            label = watch.get("label") or urlparse(sitemap_url).netloc

            try:
                leaf_urls = await _fetch_sitemap_locs(client, sitemap_url)
            except Exception as exc:
                logger.warning("Sitemap walk failed for %s: %s", sitemap_url, exc)
                sitemaps_failed += 1
                continue

            sitemaps_processed += 1
            new_for_this_sitemap = 0
            for leaf in leaf_urls[:_HARD_CAP_PER_SITEMAP]:
                if _normalize_url(leaf) in seen_urls:
                    continue
                topic_label = _url_to_topic_label(leaf)
                # Cooldown check (in case the topic was researched recently)
                try:
                    cooldown = await db.check_cooldown(topic_label)
                except Exception:
                    cooldown = None
                if cooldown:
                    continue
                try:
                    inserted = await db.insert_topic_candidate(
                        kb, topic_label, "sitemap", leaf, 0.5,
                    )
                    if inserted:
                        new_for_this_sitemap += 1
                        seen_urls.add(_normalize_url(leaf))
                except Exception as exc:
                    logger.warning(
                        "Failed to insert sitemap candidate %s: %s", leaf, exc,
                    )

            total_candidates += new_for_this_sitemap
            logger.info(
                "Sitemap [%s] %s: %d new candidates from %d leaf URLs",
                kb, label, new_for_this_sitemap, len(leaf_urls),
            )

    return {
        "sitemaps": sitemaps_processed,
        "sitemaps_failed": sitemaps_failed,
        "candidates": total_candidates,
    }
