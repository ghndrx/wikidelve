"""
Wiki article CRUD: read, create, update/merge, frontmatter parsing.
"""

import asyncio
import json
import re
import yaml
import logging
import markdown as md_lib
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

from app import storage

logger = logging.getLogger("kb-service.wiki")


# --- Type-safety helpers ----------------------------------------------------

def _to_str(val: object, default: str = "") -> str:
    """Coerce any YAML value to a plain string safely."""
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return default
    return str(val)


def _to_str_list(val: object) -> list[str]:
    """Coerce any YAML value to a flat list of strings.

    Handles:
      tags: [a, b]          -> list
      tags: "a, b"          -> comma-separated string
      tags: 42              -> single-element list
      tags: [[nested]]      -> flattened with dicts/lists removed
      tags: null / missing  -> empty list
    """
    if val is None:
        return []
    if isinstance(val, str):
        return [t.strip() for t in val.split(",") if t.strip()]
    if isinstance(val, (int, float, bool)):
        return [str(val)]
    if isinstance(val, list):
        out: list[str] = []
        for item in val:
            if isinstance(item, (dict, list)):
                continue
            if item is not None:
                out.append(str(item).strip())
        return [t for t in out if t]
    return []


def _wikilink_to_slug(title: str) -> str:
    """Convert a wikilink title to a URL-safe slug.

    Strips non-alphanumeric chars (except spaces/hyphens) before slugifying,
    so [[GPU Cloud Platforms: RunPod vs Lambda Labs]] becomes
    gpu-cloud-platforms-runpod-vs-lambda-labs instead of keeping colons.
    """
    safe = "".join(c if c.isalnum() or c in " -" else "" for c in title)
    return safe.lower().strip().replace(" ", "-")


def _safe_read(kb: str, rel_path: str) -> str | None:
    """Read a KB-relative path via the storage backend, returning None on
    miss or any I/O / encoding error. (Sync — used inside async handlers
    that already tolerate brief I/O blocks.)"""
    try:
        return storage.read_text(kb, rel_path)
    except Exception as exc:
        logger.warning("Failed to read %s/%s: %s", kb, rel_path, exc)
        return None


def read_article_text(kb_name: str, slug: str) -> str | None:
    """Convenience: read the raw markdown for an article (no parsing)."""
    return _safe_read(kb_name, f"wiki/{slug}.md")


def write_article_text(kb_name: str, slug: str, content: str) -> None:
    """Convenience: write the raw markdown for an article."""
    storage.write_text(kb_name, f"wiki/{slug}.md", content)


def _parse_yaml_lenient(raw: str) -> dict | None:
    """Parse YAML with lenient handling for unquoted colons in values."""
    import re as _re
    lines = []
    for line in raw.strip().split("\n"):
        m = _re.match(r"^(\w[\w\s]*?):\s+(.+)$", line)
        if m and ":" in m.group(2) and not m.group(2).startswith(("[", '"', "'")):
            lines.append(f'{m.group(1)}: "{m.group(2)}"')
        else:
            lines.append(line)
    try:
        result = yaml.safe_load("\n".join(lines))
        return result if isinstance(result, dict) else None
    except yaml.YAMLError:
        return None


# --- Frontmatter parsing ---------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from a markdown file.

    Handles corrupted files with multiple frontmatter blocks by merging
    all YAML blocks into one dict and returning the clean body.
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1])
        if not isinstance(meta, dict):
            return {}, text
    except yaml.YAMLError:
        # Retry with lenient parsing — quote unquoted values containing colons
        meta = _parse_yaml_lenient(parts[1])
        if not isinstance(meta, dict):
            return {}, text

    body = parts[2].strip()

    # Repair: if the body starts with another frontmatter block, it's corrupted.
    # Merge all frontmatter blocks into one and extract the real body.
    while body.startswith("---"):
        inner_parts = body.split("---", 2)
        if len(inner_parts) < 3:
            break
        try:
            extra_meta = yaml.safe_load(inner_parts[1])
            if isinstance(extra_meta, dict):
                # Merge — existing keys in meta take precedence for 'updated',
                # but extra_meta fills in missing keys (like 'title')
                for k, v in extra_meta.items():
                    if k not in meta:
                        meta[k] = v
            body = inner_parts[2].strip()
        except yaml.YAMLError:
            break

    return meta, body


_TECH_TAG_SET = frozenset({
    "go", "python", "typescript", "bash", "docker", "kubernetes", "terraform",
    "rust", "java", "javascript", "c++", "c#", "swift", "kotlin", "scala",
    "ruby", "php", "sql", "yaml", "json", "html", "css",
})


def _normalize_article_meta(meta: dict, slug: str, body: str, kb_name: str) -> dict:
    """Build a safe, fully-typed article dict from raw YAML frontmatter.

    Includes cheap derived signals (h2_count, link_count, code_count,
    has_tech_tag) so quality scans can read them from the metadata
    index instead of re-fetching every article body from S3.
    """
    body_text = body or ""
    tag_list = _to_str_list(meta.get("tags"))
    h2_count = len(re.findall(r"^## ", body_text, re.MULTILINE)) if body_text else 0
    link_count = len(re.findall(r"\[.*?\]\(.*?\)", body_text)) if body_text else 0
    code_count = body_text.count("```") // 2 if body_text else 0
    wikilink_count = len(re.findall(r"\[\[.*?\]\]", body_text)) if body_text else 0
    has_tech = any((t or "").lower() in _TECH_TAG_SET for t in tag_list)
    return {
        "slug": slug,
        "title": _to_str(meta.get("title")) or slug.replace("-", " ").title(),
        "summary": _to_str(meta.get("summary")),
        "tags": tag_list,
        "status": _to_str(meta.get("status"), "draft"),
        "confidence": _to_str(meta.get("confidence"), "medium"),
        "updated": _to_str(meta.get("updated")),
        "source_type": _to_str(meta.get("source_type"), "manual"),
        "source_files": _to_str_list(meta.get("source_files")),
        "word_count": len(body_text.split()) if body_text else 0,
        "h2_count": h2_count,
        "link_count": link_count,
        "code_count": code_count,
        "wikilink_count": wikilink_count,
        "has_tech_tag": has_tech,
        "kb": kb_name,
    }


# --- Read operations --------------------------------------------------------

_ARTICLES_CACHE: dict[str, tuple[float, list[dict]]] = {}
_ARTICLES_TTL = 120.0


def get_articles(kb_name: str) -> list[dict]:
    """Get all articles from a KB with frontmatter.

    Bulk-loaded via storage. Results are cached in-process for
    ``_ARTICLES_TTL`` seconds so API routes that hit this repeatedly
    (admin panel, /api/kbs, graph, etc.) don't pay the S3 round-trip
    every time. Invalidated by ``invalidate_articles_cache`` when an
    article is created / updated / deleted.

    When ``ARTICLE_META_INDEX=true`` is set AND the DynamoDB index has
    at least one row for the KB, list reads come from the index in a
    single ``query()`` call. If the index is empty (first deploy, not
    yet backfilled) we fall back to the slow S3 scan and backfill the
    index in the background.
    """
    import os as _os
    import time
    now = time.time()
    cached = _ARTICLES_CACHE.get(kb_name)
    if cached and (now - cached[0]) < _ARTICLES_TTL:
        return cached[1]

    use_index = _os.getenv("ARTICLE_META_INDEX", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if use_index:
        try:
            rows = _run_sync(_load_metas_via_index(kb_name))
        except Exception as exc:
            logger.debug("meta-index: list failed, falling back to S3: %s", exc)
            rows = None
        if rows:
            _ARTICLES_CACHE[kb_name] = (now, rows)
            return rows

    articles: list[dict] = []
    for slug, text in storage.iter_articles(kb_name, subdir="wiki"):
        meta, body = parse_frontmatter(text)
        articles.append(_normalize_article_meta(meta, slug, body, kb_name))
    _ARTICLES_CACHE[kb_name] = (now, articles)
    return articles


async def _load_metas_via_index(kb_name: str) -> list[dict] | None:
    from app import db
    rows = await db.list_article_metas(kb_name)
    if not rows:
        return None
    out: list[dict] = []
    for r in rows:
        out.append({
            "slug": r.get("slug") or "",
            "title": r.get("title") or "",
            "summary": r.get("summary") or "",
            "tags": list(r.get("tags") or []),
            "status": r.get("status") or "draft",
            "confidence": r.get("confidence") or "medium",
            "updated": r.get("updated") or "",
            "source_type": r.get("source_type") or "manual",
            "source_files": list(r.get("source_files") or []),
            "word_count": int(r.get("word_count") or 0),
            "h2_count": int(r.get("h2_count") or 0),
            "link_count": int(r.get("link_count") or 0),
            "code_count": int(r.get("code_count") or 0),
            "wikilink_count": int(r.get("wikilink_count") or 0),
            "has_tech_tag": bool(r.get("has_tech_tag") or False),
            "kb": kb_name,
        })
    return out


def _run_sync(coro):
    """Run a coroutine from sync code, handling both 'no loop' and 'loop running' cases."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # We're inside an async context but called from a sync function.
    # Use a dedicated thread-backed loop via asyncio.run_coroutine_threadsafe.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(asyncio.run, coro)
        return fut.result()


def invalidate_articles_cache(kb_name: str | None = None) -> None:
    if kb_name is None:
        _ARTICLES_CACHE.clear()
    else:
        _ARTICLES_CACHE.pop(kb_name, None)
    # Quality scans derive from get_articles output, so bust them too.
    try:
        from app.quality import invalidate_quality_cache
        invalidate_quality_cache(kb_name)
    except ImportError:
        pass


def get_article(kb_name: str, slug: str) -> dict | None:
    """Get a single article with rendered HTML."""
    text = _safe_read(kb_name, f"wiki/{slug}.md")
    if text is None:
        return None
    meta, body = parse_frontmatter(text)

    # Convert wikilinks to internal links
    body = re.sub(
        r'\[\[([^\]]+)\]\]',
        lambda m: f'[{m.group(1)}](/wiki/{kb_name}/{_wikilink_to_slug(m.group(1))})',
        body,
    )

    try:
        html = md_lib.markdown(
            body,
            extensions=["fenced_code", "tables", "toc", "codehilite"],
            extension_configs={"codehilite": {"css_class": "highlight"}},
        )
    except Exception as exc:
        logger.warning("Markdown rendering failed for %s/%s: %s", kb_name, slug, exc)
        html = f"<pre>{body}</pre>"

    article = _normalize_article_meta(meta, slug, body, kb_name)
    article["html"] = html
    article["raw_markdown"] = body
    return article


# --- Redis caching ----------------------------------------------------------

_redis_pool = None


def set_redis_pool(pool):
    """Set the Redis pool for article caching (called from lifespan)."""
    global _redis_pool
    _redis_pool = pool


async def get_articles_cached(kb_name: str) -> list[dict]:
    """Get articles with Redis caching (TTL 60s). Falls back to disk read."""
    if _redis_pool is None:
        return get_articles(kb_name)

    cache_key = f"kb:articles:{kb_name}"
    try:
        cached = await _redis_pool.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    articles = get_articles(kb_name)
    try:
        await _redis_pool.set(cache_key, json.dumps(articles), ex=60)
    except Exception:
        pass
    return articles


async def invalidate_article_cache(kb_name: str):
    """Invalidate the Redis cache for a KB's articles."""
    if _redis_pool is None:
        return
    try:
        await _redis_pool.delete(f"kb:articles:{kb_name}")
    except Exception:
        pass


# --- Fuzzy matching ---------------------------------------------------------

_MATCH_STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "for", "in", "on", "to", "with",
    "vs", "how", "what", "why", "best", "is", "are", "was", "were",
}


def _significant_words(text: str) -> set[str]:
    """Extract significant words (lowercase, no stop words, len > 2)."""
    return {
        w for w in text.lower().split()
        if len(w) > 2 and w not in _MATCH_STOP_WORDS
    }


def find_related_article(kb_name: str, topic: str) -> dict | None:
    """Fuzzy match a topic against existing article titles.

    Uses both character similarity (SequenceMatcher >= 0.75) AND word overlap
    (>= 50% of significant words must match) to prevent false positives like
    "kubernetes operators" matching "kubernetes pod security standards".
    """
    articles = get_articles(kb_name)
    if not articles:
        return None

    topic_lower = topic.lower().strip()
    topic_words = _significant_words(topic)
    best_match = None
    best_ratio = 0.0

    for article in articles:
        title = article["title"].lower().strip()
        # Exact slug match
        slug = article["slug"].lower()
        if slug == topic_lower.replace(" ", "-"):
            return article

        ratio = SequenceMatcher(None, topic_lower, title).ratio()
        if ratio > best_ratio:
            # Also check word overlap to prevent false positives
            title_words = _significant_words(title)
            if topic_words and title_words:
                overlap = len(topic_words & title_words)
                min_words = min(len(topic_words), len(title_words))
                word_overlap = overlap / max(min_words, 1)
                if word_overlap < 0.5:
                    continue  # Not enough meaningful word overlap
            best_ratio = ratio
            best_match = article

    if best_match:
        # High word overlap (>= 80%) with reasonable ratio → match
        # This catches "Python Testing" → "Python Testing Best Practices"
        if best_ratio >= 0.60:
            title_words = _significant_words(best_match["title"])
            if topic_words and title_words:
                overlap = len(topic_words & title_words) / max(min(len(topic_words), len(title_words)), 1)
                if overlap >= 0.8 and best_ratio >= 0.60:
                    return best_match
        # Standard threshold for character similarity
        if best_ratio >= 0.75:
            return best_match
    return None


# --- Create / Update operations ---------------------------------------------

def extract_title(content: str, fallback: str = "") -> str:
    """Extract a clean title from synthesized content.

    Looks for the first # heading that isn't a meta line like "# Research:".
    Falls back to the provided fallback (usually the user's prompt).
    """
    for line in content.split("\n"):
        line = line.strip()
        if not line.startswith("# "):
            continue
        title = line[2:].strip()
        # Skip generic prefixes the LLM adds
        for prefix in ("Research:", "Research Document:", "Local Research:"):
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        # Strip trailing ": Research Document" / ": Comprehensive..." suffixes
        for suffix in (
            ": Research Document",
            ": Comprehensive Research Document",
            ": A Comprehensive Guide",
            ": A Research Document",
        ):
            if title.endswith(suffix):
                title = title[:-len(suffix)].strip()
        if title and len(title) > 3:
            return title
    return fallback


def _format_frontmatter_value(v) -> str:
    """Quote a frontmatter scalar safely. Keeps strings that contain
    colons / quotes / newlines valid YAML."""
    s = str(v)
    if any(ch in s for ch in (":", '"', "\n", "#")):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def create_article(
    kb_name: str, topic: str, content: str, source_type: str = "web",
    *, source_meta: Optional[dict] = None,
) -> str:
    """Create a new wiki article with frontmatter. Returns the slug.

    source_type: "web", "local", "media", or "manual"
    source_meta: optional dict of ``source_*`` fields (repo, branch,
        commit, homepage, docs, etc.) to stamp into frontmatter for
        deterministic backlinks. Callers build this via
        ``local_research.build_source_meta()``.
    """
    storage.init_kb(kb_name)

    title = extract_title(content, topic)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _topic_to_slug(title, today)

    storage.write_text(kb_name, f"raw/{slug}.md", f"# {title}\n\n{content}\n")

    stop_words = {
        "the", "a", "an", "and", "or", "of", "for", "in", "on",
        "to", "with", "vs", "how", "what", "why", "best",
    }
    tags = [
        w.lower()
        for w in title.split()
        if len(w) > 2 and w.lower() not in stop_words
    ][:5]

    first_line = content.split("\n")[0].strip().lstrip("#").strip()
    if len(first_line) > 200:
        first_line = first_line[:197] + "..."

    # Deterministic ordering of source_* keys so diffs stay tidy across
    # re-syntheses. Unknown source_* keys go at the end, alphabetical.
    source_lines = ""
    if source_meta:
        preferred_order = [
            "source_repo", "source_branch", "source_commit",
            "source_repo_subpath", "source_path",
            "source_homepage", "source_docs", "source_issues",
            "source_npm", "source_pypi", "source_crates", "source_go_module",
        ]
        seen = set()
        ordered_items = []
        for k in preferred_order:
            if k in source_meta and source_meta[k]:
                ordered_items.append((k, source_meta[k]))
                seen.add(k)
        for k in sorted(source_meta):
            if k not in seen and source_meta[k] and k.startswith("source_"):
                ordered_items.append((k, source_meta[k]))
        source_lines = "".join(
            f"{k}: {_format_frontmatter_value(v)}\n" for k, v in ordered_items
        )

    frontmatter = (
        f"---\n"
        f'title: "{title}"\n'
        f"aliases: []\n"
        f"created: {today}\n"
        f"updated: {today}\n"
        f"source_type: {source_type}\n"
        f"{source_lines}"
        f"source_files:\n"
        f"  - raw/{slug}.md\n"
        f"tags: [{', '.join(tags)}]\n"
        f"status: draft\n"
        f"confidence: medium\n"
        f'summary: "{first_line}"\n'
        f"---\n\n"
    )
    storage.write_text(kb_name, f"wiki/{slug}.md", frontmatter + content)
    invalidate_articles_cache(kb_name)
    _write_through_meta_index(kb_name, slug, frontmatter + content)
    logger.info("Created wiki article: %s/%s", kb_name, slug)
    return slug


def _write_through_meta_index(kb: str, slug: str, text: str) -> None:
    """Fire-and-forget dual-write to the DynamoDB article metadata index.

    Kept best-effort so a broken DynamoDB path never blocks the primary
    S3 write — we log and move on if the index lookup / write fails.
    The read path has an S3-scan fallback, so a stale / missing index
    just means a slower read, not a broken KB.
    """
    try:
        meta, body = parse_frontmatter(text)
        row = _normalize_article_meta(meta, slug, body, kb)
    except Exception as exc:
        logger.debug("meta-index: failed to parse %s/%s: %s", kb, slug, exc)
        return

    async def _run():
        try:
            from app import db
            await db.upsert_article_meta(kb, slug, row)
        except Exception as exc:
            logger.debug("meta-index: upsert failed for %s/%s: %s", kb, slug, exc)

    # Fire the task on the running loop if there is one; otherwise run
    # a short synchronous event loop. create_article is sync so we
    # might be called from either context.
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        try:
            asyncio.run(_run())
        except Exception as exc:
            logger.debug("meta-index: sync run failed for %s/%s: %s", kb, slug, exc)


async def update_article(
    kb_name: str, slug: str, new_content: str,
    *, source_meta: Optional[dict] = None,
) -> str:
    """Merge new research content into an existing wiki article.

    Finds the article by slug, appends new findings under a
    '## Recent Updates' section, and bumps the 'updated' date in frontmatter.
    Returns the slug of the updated article.

    Snapshots the pre-overwrite content to ``article_versions`` via
    ``db.save_article_version`` before any storage write happens.
    """
    text = _safe_read(kb_name, f"wiki/{slug}.md")
    if text is None:
        raise FileNotFoundError(f"Article not found: {kb_name}/{slug}")

    # Snapshot the current file content BEFORE overwriting. Logging-only
    # failure mode — never block an update on the snapshot write.
    try:
        from app import db
        await db.save_article_version(
            kb=kb_name,
            article_slug=slug,
            full_content=text,
            change_type="updated",
        )
    except Exception as exc:
        logger.warning(
            "Failed to save article version for %s/%s: %s", kb_name, slug, exc,
        )

    meta, body = parse_frontmatter(text)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Ensure essential fields are preserved (guard against corruption)
    if "title" not in meta:
        meta["title"] = slug.replace("-", " ").title()
        logger.warning("Article %s/%s missing title in frontmatter, using slug fallback", kb_name, slug)

    # Update the frontmatter date
    meta["updated"] = today

    # Stamp/refresh source_* fields if the caller supplied them. New
    # values overwrite old ones (e.g. commit SHA advances as the repo
    # moves) but keys not re-supplied are preserved.
    if source_meta:
        for k, v in source_meta.items():
            if k.startswith("source_") and v:
                meta[k] = v

    # Append under "## Recent Updates" section
    update_header = "## Recent Updates"
    date_header = f"### {today}"
    update_block = f"\n\n{date_header}\n\n{new_content}"

    if update_header in body:
        # Append after the existing Recent Updates header
        idx = body.index(update_header) + len(update_header)
        body = body[:idx] + update_block + body[idx:]
    else:
        # Add the section at the end
        body = body.rstrip() + f"\n\n{update_header}{update_block}"

    fm_str = _serialize_frontmatter(meta)
    new_text = fm_str + "\n\n" + body
    storage.write_text(kb_name, f"wiki/{slug}.md", new_text)
    invalidate_articles_cache(kb_name)
    _write_through_meta_index(kb_name, slug, new_text)
    logger.info("Updated wiki article: %s/%s", kb_name, slug)
    return slug


async def create_or_update_article(
    kb_name: str, topic: str, content: str, source_type: str = "web",
    *, source_meta: Optional[dict] = None,
) -> tuple[str, str]:
    """Smart create-or-merge: find related article, update if exists, create otherwise.

    Returns (slug, change_type) where change_type is 'created' or 'updated'.
    """
    existing = find_related_article(kb_name, topic)
    if existing:
        slug = await update_article(
            kb_name, existing["slug"], content, source_meta=source_meta,
        )
        return slug, "updated"
    else:
        slug = create_article(
            kb_name, topic, content, source_type=source_type, source_meta=source_meta,
        )
        return slug, "created"


# --- Delete operations ------------------------------------------------------

def delete_article(kb_name: str, slug: str) -> dict:
    """Delete a wiki article and its raw source file.

    Returns a summary dict with what was removed.
    """
    text = _safe_read(kb_name, f"wiki/{slug}.md")
    if text is None:
        raise FileNotFoundError(f"Article not found: {kb_name}/{slug}")

    meta, _ = parse_frontmatter(text)
    title = _to_str(meta.get("title")) or slug

    removed: list[str] = []
    if storage.delete(kb_name, f"wiki/{slug}.md"):
        removed.append(f"{kb_name}/wiki/{slug}.md")
    if storage.delete(kb_name, f"raw/{slug}.md"):
        removed.append(f"{kb_name}/raw/{slug}.md")

    invalidate_articles_cache(kb_name)

    # Best-effort drop from the DynamoDB metadata index — logged, never fatal.
    async def _drop():
        try:
            from app import db
            await db.delete_article_meta(kb_name, slug)
        except Exception as exc:
            logger.debug("meta-index: delete failed for %s/%s: %s", kb_name, slug, exc)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_drop())
    except RuntimeError:
        try:
            asyncio.run(_drop())
        except Exception:
            pass

    logger.info("Deleted article: %s/%s (%s)", kb_name, slug, title)
    return {"slug": slug, "kb": kb_name, "title": title, "files_removed": removed}


# --- Internal helpers -------------------------------------------------------

def _topic_to_slug(topic: str, fallback_date: str) -> str:
    """Convert a topic string to a filesystem-safe slug."""
    safe = "".join(c if c.isalnum() or c in " -" else "_" for c in topic)[:80]
    slug = safe.lower().replace(" ", "-").strip("-")
    return slug if slug else f"research-{fallback_date}"


def _serialize_frontmatter(meta: dict) -> str:
    """Serialize a metadata dict back to YAML frontmatter block."""
    lines = ["---"]
    for key, val in meta.items():
        if isinstance(val, list):
            if all(isinstance(v, str) for v in val) and len(val) <= 5:
                lines.append(f"{key}: [{', '.join(val)}]")
            else:
                lines.append(f"{key}:")
                for item in val:
                    lines.append(f"  - {item}")
        elif isinstance(val, str) and ("\n" in val or '"' in val or ":" in val):
            safe = val.replace('"', '\\"')
            lines.append(f'{key}: "{safe}"')
        else:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines)
