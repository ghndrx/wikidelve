"""
Documents: versioned rendered artifacts (PDF/PPTX/MD-export) produced by
the document drafting agent and iterated via agentic chat.

Unlike Scaffolds (source-code trees for developer consumption),
Documents are single rendered deliverables — a reader opens them and
reads them. Each Document has:

  - a canonical ``markdown`` source (edited each chat turn)
  - a rendered binary (PDF for MVP; PPTX later)
  - a history of versions so the chat agent can reason over drift
  - a decisions_log authored by the agent explaining each version

Storage layout (per-KB, under the KB root):

    documents/<slug>/manifest.json                 # metadata + version list
    documents/<slug>/v<N>.md                       # markdown at version N
    documents/<slug>/v<N>.<ext>                    # rendered binary
    documents/<slug>/history.jsonl                 # chat turns + decisions
    documents/_index.json                          # denormalized browse list

Manifest schema (version 1):

    {
      "slug": "q2-ingestion-brief",
      "title": "Q2 Ingestion Pipeline — One-Pager",
      "doc_type": "pdf",
      "autonomy_mode": "propose",
      "brief": "A sales one-pager on our ingestion pipeline",
      "seed_articles": [{"kb": "personal", "slug": "..."}, ...],
      "pinned_facts": ["pricing is $99/mo", "we launched in 2024"],
      "current_version": 3,
      "versions": [
        {"v": 1, "created": "...", "trigger": "initial draft", "size_bytes": 84301, "markdown_hash": "..."},
        ...
      ],
      "document_version": 1,
      "created": "2026-04-14",
      "updated": "2026-04-14"
    }

The ``autonomy_mode`` is per-document — default ``propose`` means the
chat UI shows a diff + ✓/✗ before committing the agent's proposed v+1.
``auto`` commits immediately, ``plan-first`` requires the agent to
announce a plan and wait for ✓ before executing. All three share the
same storage model.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from app import storage

logger = logging.getLogger("kb-service.documents")

DOCUMENT_VERSION = 1

# MVP supports PDF first (weasyprint). PPTX / DOCX / MD-export are
# hooks — the manifest accepts them, the renderer routes on them.
DOC_TYPES = {"pdf", "pptx", "docx", "md-export"}

# How much latitude the chat agent has per turn. See the PRD section
# "Agentic Document Chat" in-thread for the tradeoffs. Stored on the
# manifest so different documents can have different policies.
AUTONOMY_MODES = {"auto", "propose", "plan-first", "collab", "review-gate"}
DEFAULT_AUTONOMY = "propose"

# Guardrails — markdown source should not balloon unboundedly; the
# rendered binary is also capped so a runaway PDF doesn't eat storage.
MAX_MARKDOWN_BYTES = 512 * 1024       # 512KB of source markdown
MAX_RENDERED_BYTES = 10 * 1024 * 1024  # 10MB rendered artifact
MAX_VERSIONS = 50                      # trim history after this


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _slugify(title: str, fallback: str = "document") -> str:
    safe = "".join(c if c.isalnum() or c in " -" else "-" for c in title)[:80]
    slug = re.sub(r"-+", "-", safe.lower().replace(" ", "-")).strip("-")
    return slug or fallback


# ---------------------------------------------------------------------------
# Read / list
# ---------------------------------------------------------------------------

def get_manifest(kb: str, slug: str) -> Optional[dict]:
    raw = storage.read_text(kb, f"documents/{slug}/manifest.json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        logger.warning("Malformed manifest for %s/%s: %s", kb, slug, exc)
        return None


def get_markdown(kb: str, slug: str, version: Optional[int] = None) -> Optional[str]:
    """Read the markdown source for a given version (default: current)."""
    manifest = get_manifest(kb, slug)
    if not manifest:
        return None
    v = version if version is not None else manifest.get("current_version", 1)
    return storage.read_text(kb, f"documents/{slug}/v{v}.md")


def get_rendered(kb: str, slug: str, version: Optional[int] = None) -> Optional[bytes]:
    """Read the rendered binary for a version. Binary path — returns bytes.

    Storage.read_text decodes UTF-8 which would corrupt a PDF. We go
    via a path-reader interface so the binary survives round-trip.
    """
    manifest = get_manifest(kb, slug)
    if not manifest:
        return None
    v = version if version is not None else manifest.get("current_version", 1)
    doc_type = manifest.get("doc_type", "pdf")
    # Delegate to storage.read_binary if the backend supports it;
    # otherwise fall back to base64-encoded text stored with an .b64
    # suffix. For the current S3/filesystem backend, read_binary is
    # the right path — see app.storage for the wiring.
    try:
        return storage.read_binary(kb, f"documents/{slug}/v{v}.{doc_type}")
    except AttributeError:
        # Backend doesn't expose read_binary yet; fall back to the
        # base64 sidecar pattern so the feature doesn't hard-fail.
        raw_b64 = storage.read_text(kb, f"documents/{slug}/v{v}.{doc_type}.b64")
        if not raw_b64:
            return None
        import base64
        try:
            return base64.b64decode(raw_b64)
        except Exception:
            return None


def get_history(kb: str, slug: str, limit: int = 50) -> list[dict]:
    """Read the append-only turn/decision log for a document."""
    raw = storage.read_text(kb, f"documents/{slug}/history.jsonl") or ""
    events: list[dict] = []
    for line in raw.splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except (TypeError, ValueError):
            continue
    return events


def list_documents(kb: str) -> list[dict]:
    """List all documents in a KB, newest first. Self-heals stale index."""
    raw = storage.read_text(kb, "documents/_index.json") or "[]"
    try:
        idx = json.loads(raw)
    except (TypeError, ValueError):
        idx = []
    if not isinstance(idx, list):
        return []
    alive: list[dict] = []
    for entry in idx:
        if not isinstance(entry, dict) or not entry.get("slug"):
            continue
        if storage.read_text(kb, f"documents/{entry['slug']}/manifest.json"):
            alive.append(entry)
    alive.sort(key=lambda e: e.get("updated", e.get("created", "")), reverse=True)
    return alive


# ---------------------------------------------------------------------------
# Write / update
# ---------------------------------------------------------------------------

def _validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    title = manifest.get("title", "").strip()
    slug = manifest.get("slug") or _slugify(title)
    if not slug:
        raise ValueError("manifest needs a title or slug")

    doc_type = manifest.get("doc_type", "pdf")
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of {sorted(DOC_TYPES)}")

    autonomy = manifest.get("autonomy_mode", DEFAULT_AUTONOMY)
    if autonomy not in AUTONOMY_MODES:
        raise ValueError(f"autonomy_mode must be one of {sorted(AUTONOMY_MODES)}")

    seeds = manifest.get("seed_articles") or []
    if not isinstance(seeds, list):
        raise ValueError("seed_articles must be a list")
    for s in seeds:
        if not isinstance(s, dict) or not s.get("kb") or not s.get("slug"):
            raise ValueError(f"seed_article entries need kb+slug: {s!r}")

    pinned = manifest.get("pinned_facts") or []
    if not isinstance(pinned, list):
        raise ValueError("pinned_facts must be a list of strings")

    today = _now_date()
    return {
        "slug": slug,
        "title": title or slug.replace("-", " ").title(),
        "doc_type": doc_type,
        "autonomy_mode": autonomy,
        "brief": manifest.get("brief", "")[:4000],
        "seed_articles": seeds,
        "pinned_facts": [str(p)[:500] for p in pinned],
        "current_version": int(manifest.get("current_version", 0)),
        "versions": list(manifest.get("versions") or []),
        "document_version": DOCUMENT_VERSION,
        "created": manifest.get("created") or today,
        "updated": today,
    }


def create_document(
    kb: str, title: str, brief: str,
    *,
    doc_type: str = "pdf",
    autonomy_mode: str = DEFAULT_AUTONOMY,
    seed_articles: Optional[list[dict]] = None,
    pinned_facts: Optional[list[str]] = None,
) -> str:
    """Create an empty document shell — no version body yet.

    The drafting agent fills in v1 on the first chat turn. Returning
    the slug here lets the caller immediately open the chat UI.
    """
    manifest = _validate_manifest({
        "title": title, "brief": brief, "doc_type": doc_type,
        "autonomy_mode": autonomy_mode,
        "seed_articles": seed_articles or [],
        "pinned_facts": pinned_facts or [],
        "current_version": 0,
        "versions": [],
    })
    slug = manifest["slug"]
    storage.write_text(
        kb, f"documents/{slug}/manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    _append_history(kb, slug, {
        "type": "create", "title": manifest["title"],
        "brief": brief, "timestamp": _now_iso(),
    })
    _touch_index(kb, manifest)
    logger.info("Created document shell: %s/%s", kb, slug)
    return slug


def commit_version(
    kb: str, slug: str, markdown: str, rendered: Optional[bytes],
    *,
    trigger: str,
    agent_notes: Optional[str] = None,
) -> dict:
    """Write a new version of a document. Returns the new version entry.

    ``markdown`` is the canonical source for this version. ``rendered``
    is the binary (PDF/PPTX/etc.) — None is allowed when the doc_type
    is md-export or the renderer is deferred. ``trigger`` is a short
    description of why this version was made (user message or agent
    initiative).
    """
    manifest = get_manifest(kb, slug)
    if not manifest:
        raise ValueError(f"document not found: {kb}/{slug}")

    if not isinstance(markdown, str):
        raise ValueError("markdown must be str")
    md_bytes = markdown.encode("utf-8")
    if len(md_bytes) > MAX_MARKDOWN_BYTES:
        raise ValueError(
            f"markdown is {len(md_bytes)} bytes; max is {MAX_MARKDOWN_BYTES}"
        )
    if rendered is not None and len(rendered) > MAX_RENDERED_BYTES:
        raise ValueError(
            f"rendered artifact is {len(rendered)} bytes; "
            f"max is {MAX_RENDERED_BYTES}"
        )

    next_v = int(manifest.get("current_version", 0)) + 1
    doc_type = manifest.get("doc_type", "pdf")

    storage.write_text(kb, f"documents/{slug}/v{next_v}.md", markdown)
    if rendered is not None:
        _write_binary(kb, f"documents/{slug}/v{next_v}.{doc_type}", rendered)

    entry = {
        "v": next_v,
        "created": _now_iso(),
        "trigger": (trigger or "")[:500],
        "agent_notes": (agent_notes or "")[:2000] or None,
        "size_bytes": len(rendered) if rendered is not None else len(md_bytes),
        "markdown_hash": _hash(markdown),
    }
    versions = list(manifest.get("versions") or []) + [entry]
    # Trim history of old version BODIES once we exceed the cap.
    # We keep their manifest entries forever so diffing UI can show
    # "this changed in v7" even if v7's body is no longer stored.
    if len(versions) > MAX_VERSIONS:
        cutoff = len(versions) - MAX_VERSIONS
        for old in versions[:cutoff]:
            try:
                storage.delete(kb, f"documents/{slug}/v{old['v']}.md")
                storage.delete(kb, f"documents/{slug}/v{old['v']}.{doc_type}")
            except Exception as exc:
                logger.debug("trim v%s: %s", old.get("v"), exc)

    manifest["versions"] = versions
    manifest["current_version"] = next_v
    manifest["updated"] = _now_date()
    storage.write_text(
        kb, f"documents/{slug}/manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    _append_history(kb, slug, {
        "type": "version", "v": next_v,
        "trigger": trigger, "agent_notes": agent_notes,
        "timestamp": _now_iso(),
    })
    _touch_index(kb, manifest)
    logger.info("Committed %s/%s v%d (%d bytes md)", kb, slug, next_v, len(md_bytes))
    return entry


def append_chat_turn(
    kb: str, slug: str, role: str, content: str,
    *,
    metadata: Optional[dict] = None,
) -> None:
    """Append a user/agent turn to the document's history log.

    role: 'user' or 'agent'. Content is the message text. Metadata
    carries tool-call traces, planned steps, diff summaries, etc.
    """
    if role not in ("user", "agent", "system"):
        raise ValueError(f"invalid role: {role!r}")
    _append_history(kb, slug, {
        "type": "turn", "role": role, "content": content[:8000],
        "metadata": metadata or {}, "timestamp": _now_iso(),
    })


def delete_document(kb: str, slug: str) -> None:
    manifest = get_manifest(kb, slug)
    if manifest:
        doc_type = manifest.get("doc_type", "pdf")
        for entry in manifest.get("versions") or []:
            try:
                storage.delete(kb, f"documents/{slug}/v{entry['v']}.md")
                storage.delete(kb, f"documents/{slug}/v{entry['v']}.{doc_type}")
            except Exception as exc:
                logger.debug("delete v%s: %s", entry.get("v"), exc)
    storage.delete(kb, f"documents/{slug}/manifest.json")
    storage.delete(kb, f"documents/{slug}/history.jsonl")
    _drop_from_index(kb, slug)
    logger.info("Deleted document: %s/%s", kb, slug)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_binary(kb: str, rel_path: str, blob: bytes) -> None:
    """Persist a binary artifact. Falls back to base64 text if the
    storage backend doesn't support native binary writes yet.
    """
    try:
        storage.write_binary(kb, rel_path, blob)
    except AttributeError:
        import base64
        storage.write_text(kb, f"{rel_path}.b64", base64.b64encode(blob).decode("ascii"))


def _append_history(kb: str, slug: str, event: dict) -> None:
    """JSON-lines append. Best-effort — history is informational, not
    authoritative, so a failed write is logged and not raised."""
    try:
        existing = storage.read_text(kb, f"documents/{slug}/history.jsonl") or ""
        line = json.dumps(event, default=str)
        storage.write_text(kb, f"documents/{slug}/history.jsonl", existing + line + "\n")
    except Exception as exc:
        logger.debug("history append %s/%s: %s", kb, slug, exc)


def _touch_index(kb: str, manifest: dict) -> None:
    raw = storage.read_text(kb, "documents/_index.json") or "[]"
    try:
        idx = json.loads(raw)
        if not isinstance(idx, list):
            idx = []
    except (TypeError, ValueError):
        idx = []

    entry = {
        "slug": manifest["slug"],
        "title": manifest["title"],
        "doc_type": manifest["doc_type"],
        "autonomy_mode": manifest["autonomy_mode"],
        "current_version": manifest.get("current_version", 0),
        "created": manifest.get("created"),
        "updated": manifest.get("updated"),
    }
    filtered = [e for e in idx if isinstance(e, dict) and e.get("slug") != entry["slug"]]
    filtered.append(entry)
    storage.write_text(
        kb, "documents/_index.json",
        json.dumps(filtered, indent=2, sort_keys=True),
    )


def _drop_from_index(kb: str, slug: str) -> None:
    raw = storage.read_text(kb, "documents/_index.json") or "[]"
    try:
        idx = json.loads(raw)
        if not isinstance(idx, list):
            return
    except (TypeError, ValueError):
        return
    filtered = [e for e in idx if isinstance(e, dict) and e.get("slug") != slug]
    storage.write_text(
        kb, "documents/_index.json",
        json.dumps(filtered, indent=2, sort_keys=True),
    )
