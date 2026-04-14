"""
Scaffolds: multi-file artifact storage for plug-and-play templates.

A scaffold is a small tree of source files produced by the scaffold
agent (HTML/CSS/JS for now; React/Vue projects later). Each scaffold
has:

  - a manifest describing what it is and which files it contains
  - one or more source files under a /files subdir
  - an optional paired wiki article for discoverability

Storage layout (per-KB, under the KB root):

    scaffolds/<slug>/manifest.json
    scaffolds/<slug>/files/<relative-path>
    scaffolds/_index.json               # denormalized listing for browse

The wiki article for a scaffold lives at ``wiki/scaffold-<slug>.md``
with ``source_type: scaffold`` in the frontmatter so it's
discoverable via existing search.

Manifest schema (version 1):

    {
      "slug": "minimalist-saas-landing",
      "title": "Minimalist SaaS Landing",
      "description": "...",
      "scaffold_type": "landing-page",
      "framework": "vanilla",
      "preview_type": "static",
      "entrypoint": "index.html",
      "build_cmd": null,
      "scaffold_version": 1,
      "topic": "minimalist saas landing page",
      "files": ["index.html", "styles.css", "scripts.js"],
      "created": "2026-04-14",
      "updated": "2026-04-14"
    }

The ``preview_type`` is the hook for Option B: today only ``static``
is implemented (served as sandboxed iframe); ``wc`` (WebContainer)
and ``container`` (backend sandbox) would be added without changing
the rest of the pipeline.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from app import storage

logger = logging.getLogger("kb-service.scaffolds")

SCAFFOLD_VERSION = 1

# Hand-declared top-level types. Extend as new verticals land.
# Kept as a flat enum rather than a nested tree so the UI facet
# stays simple; emergent sub-categorization lives in the scaffold
# article's tags.
SCAFFOLD_TYPES = {
    "landing-page",
    "dashboard",
    "component-kit",
    "email-template",
    "data-viz",
    "api-schema",
    "docker-stack",
    "other",
}

# Frameworks the MVP supports. Start with vanilla; others are hooks
# for Option B (full project scaffolds via WebContainer / backend
# sandbox).
FRAMEWORKS = {"vanilla", "react", "vue", "next", "svelte"}

# Preview rendering strategies. ``static`` is the only one wired
# today — it serves files directly from /sandbox/<kb>/<slug>/<path>
# with strict CSP. ``wc`` + ``container`` are placeholder values the
# manifest accepts so future work can flip a scaffold's preview mode
# without changing its storage layout.
PREVIEW_TYPES = {"static", "wc", "container"}

# File size guardrails — each file <= 256KB, whole scaffold <= 2MB.
# Agents occasionally emit giant base64 blobs or paste an entire
# framework; these caps stop that from blowing past sensible limits.
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_FILES = 50


def _now_iso_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _safe_rel_path(path: str) -> str:
    """Normalise a scaffold-relative path. Rejects escape attempts.

    The agent writes these paths; if it tries ``../../etc/passwd``,
    a backslash-escaped sibling, or an absolute path we refuse rather
    than writing outside the scaffold's own tree.
    """
    if not path:
        raise ValueError("empty file path")
    raw = path.replace("\\", "/")
    # Reject absolute paths BEFORE strip so "/etc/passwd" doesn't
    # silently become "etc/passwd". The earlier version normalised
    # away the leading slash, which let agents (or attackers) sneak
    # in a path that *looked* relative after stripping.
    if raw.startswith("/"):
        raise ValueError(f"unsafe path (absolute): {path!r}")
    if ":" in raw:
        # Catches Windows drive letters and URI-ish escapes.
        raise ValueError(f"unsafe path (drive/scheme): {path!r}")
    # Collapse leading `./` segments so the agent can write either
    # `./index.html` or `index.html` and we end up with the same key.
    segments = [s for s in raw.split("/") if s and s != "."]
    if any(s == ".." for s in segments):
        raise ValueError(f"path escape attempt: {path!r}")
    norm = "/".join(segments)
    if not norm:
        raise ValueError("empty file path")
    return norm


def _slugify(topic: str, fallback: str = "scaffold") -> str:
    safe = "".join(c if c.isalnum() or c in " -" else "-" for c in topic)[:80]
    slug = re.sub(r"-+", "-", safe.lower().replace(" ", "-")).strip("-")
    return slug or fallback


# ---------------------------------------------------------------------------
# Read / list
# ---------------------------------------------------------------------------

def get_manifest(kb: str, slug: str) -> Optional[dict]:
    """Load a scaffold's manifest, or None if it doesn't exist."""
    raw = storage.read_text(kb, f"scaffolds/{slug}/manifest.json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Malformed manifest for %s/%s: %s", kb, slug, exc)
        return None


def get_file(kb: str, slug: str, rel_path: str) -> Optional[str]:
    """Read a single scaffold file. Returns None if missing."""
    safe = _safe_rel_path(rel_path)
    return storage.read_text(kb, f"scaffolds/{slug}/files/{safe}")


def list_scaffolds(kb: str) -> list[dict]:
    """Return all scaffold manifests in a KB, newest first.

    Backed by ``scaffolds/_index.json`` which is maintained on each
    create/delete. A stale index is self-healing — we filter out
    entries whose manifest no longer loads.
    """
    raw = storage.read_text(kb, "scaffolds/_index.json") or "[]"
    try:
        idx = json.loads(raw)
    except (TypeError, ValueError):
        idx = []
    if not isinstance(idx, list):
        return []
    # Best-effort freshness check: drop entries whose manifest is
    # gone. Keeps list_scaffolds honest after manual S3 edits.
    alive: list[dict] = []
    for entry in idx:
        if not isinstance(entry, dict) or not entry.get("slug"):
            continue
        if storage.read_text(kb, f"scaffolds/{entry['slug']}/manifest.json"):
            alive.append(entry)
    alive.sort(key=lambda e: e.get("created", ""), reverse=True)
    return alive


# ---------------------------------------------------------------------------
# Write / delete
# ---------------------------------------------------------------------------

def _validate_manifest(manifest: dict) -> dict:
    """Fill defaults + check invariants. Returns the canonicalized dict.

    Raises ValueError on anything we can't auto-correct.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")

    slug = manifest.get("slug") or _slugify(manifest.get("title", ""))
    if not slug:
        raise ValueError("manifest missing slug/title")

    stype = manifest.get("scaffold_type", "other")
    if stype not in SCAFFOLD_TYPES:
        logger.info("scaffold_type %r not in enum; storing as 'other'", stype)
        stype = "other"

    framework = manifest.get("framework", "vanilla")
    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework: {framework!r}")

    preview_type = manifest.get("preview_type", "static")
    if preview_type not in PREVIEW_TYPES:
        raise ValueError(f"unknown preview_type: {preview_type!r}")

    entrypoint = manifest.get("entrypoint")
    if not entrypoint:
        raise ValueError("manifest.entrypoint is required")
    entrypoint = _safe_rel_path(entrypoint)

    today = _now_iso_date()
    return {
        "slug": slug,
        "title": manifest.get("title") or slug.replace("-", " ").title(),
        "description": manifest.get("description", "")[:2000],
        "topic": manifest.get("topic", ""),
        "scaffold_type": stype,
        "framework": framework,
        "preview_type": preview_type,
        "entrypoint": entrypoint,
        "build_cmd": manifest.get("build_cmd") or None,
        "scaffold_version": SCAFFOLD_VERSION,
        "files": list(manifest.get("files") or []),
        "created": manifest.get("created") or today,
        "updated": today,
    }


def create_scaffold(
    kb: str, manifest: dict, files: list[dict],
) -> str:
    """Write a scaffold to storage. Returns the slug.

    ``files`` is a list of ``{path, content}`` dicts. Sizes are
    validated server-side — agents occasionally emit huge blobs and
    we don't want a single scaffold to consume a KB's entire quota.
    """
    if not isinstance(files, list) or not files:
        raise ValueError("scaffold must have at least one file")
    if len(files) > MAX_FILES:
        raise ValueError(f"scaffold has {len(files)} files; max is {MAX_FILES}")

    canon = _validate_manifest(manifest)
    slug = canon["slug"]

    total = 0
    file_list: list[str] = []
    for f in files:
        if not isinstance(f, dict):
            raise ValueError("file entry must be a dict")
        path = _safe_rel_path(f.get("path", ""))
        content = f.get("content", "")
        if not isinstance(content, str):
            raise ValueError(f"file {path} content must be str")
        size = len(content.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise ValueError(
                f"file {path} is {size} bytes; max per file is {MAX_FILE_BYTES}"
            )
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError(f"scaffold exceeds {MAX_TOTAL_BYTES}-byte cap")
        storage.write_text(kb, f"scaffolds/{slug}/files/{path}", content)
        file_list.append(path)

    # Make sure the entrypoint is actually in the files we wrote —
    # otherwise the sandbox route will 404 on first render.
    if canon["entrypoint"] not in file_list:
        raise ValueError(
            f"entrypoint {canon['entrypoint']!r} not in written files "
            f"({file_list})"
        )
    canon["files"] = file_list

    storage.write_text(
        kb, f"scaffolds/{slug}/manifest.json",
        json.dumps(canon, indent=2, sort_keys=True),
    )
    _touch_index(kb, canon)
    logger.info("Created scaffold: %s/%s (%d files, %d bytes)", kb, slug, len(files), total)
    return slug


def delete_scaffold(kb: str, slug: str) -> None:
    """Delete all files for a scaffold + drop it from the index.

    Best-effort on each file deletion; the manifest + index update is
    authoritative so a partial deletion still removes the scaffold
    from the browse UI.
    """
    manifest = get_manifest(kb, slug)
    if manifest:
        for rel in manifest.get("files", []):
            try:
                storage.delete(kb, f"scaffolds/{slug}/files/{rel}")
            except Exception as exc:
                logger.warning("scaffold delete %s/%s: file %s: %s", kb, slug, rel, exc)
    storage.delete(kb, f"scaffolds/{slug}/manifest.json")
    _drop_from_index(kb, slug)
    logger.info("Deleted scaffold: %s/%s", kb, slug)


# ---------------------------------------------------------------------------
# Index maintenance
# ---------------------------------------------------------------------------

def _touch_index(kb: str, manifest: dict) -> None:
    """Upsert a summary row into scaffolds/_index.json."""
    raw = storage.read_text(kb, "scaffolds/_index.json") or "[]"
    try:
        idx = json.loads(raw)
        if not isinstance(idx, list):
            idx = []
    except (TypeError, ValueError):
        idx = []

    entry = {
        "slug": manifest["slug"],
        "title": manifest["title"],
        "scaffold_type": manifest["scaffold_type"],
        "framework": manifest["framework"],
        "preview_type": manifest["preview_type"],
        "created": manifest.get("created"),
        "updated": manifest.get("updated"),
    }
    filtered = [e for e in idx if isinstance(e, dict) and e.get("slug") != entry["slug"]]
    filtered.append(entry)
    storage.write_text(
        kb, "scaffolds/_index.json",
        json.dumps(filtered, indent=2, sort_keys=True),
    )


def _drop_from_index(kb: str, slug: str) -> None:
    raw = storage.read_text(kb, "scaffolds/_index.json") or "[]"
    try:
        idx = json.loads(raw)
        if not isinstance(idx, list):
            return
    except (TypeError, ValueError):
        return
    filtered = [e for e in idx if isinstance(e, dict) and e.get("slug") != slug]
    storage.write_text(
        kb, "scaffolds/_index.json",
        json.dumps(filtered, indent=2, sort_keys=True),
    )
