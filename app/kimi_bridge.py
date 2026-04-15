"""HTTP client for the kimi-bridge sidecar.

The sidecar runs kimi-cli in a container and exposes a single
``POST /agent/run`` endpoint. We hand it a prompt + workdir; it
returns when the turn finishes (or we timeout), and the workdir is
populated with whatever files kimi wrote.

Workdir lifecycle lives on the Python side: we create an ephemeral
directory under ``KIMI_WORKDIR_ROOT`` (bind-mounted into the
sidecar at ``/workdirs``), call the sidecar with the container-path
view of it, then walk it to ingest results.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

KIMI_BRIDGE_URL = os.getenv("KIMI_BRIDGE_URL", "http://kimi-bridge:5555").rstrip("/")
KIMI_BRIDGE_SECRET = os.getenv("KIMI_BRIDGE_SECRET", "").strip()
# Host-side path where ephemeral workdirs live. Must be the same
# directory that docker-compose bind-mounts into the sidecar at
# /workdirs. In-container the worker sees it at KIMI_WORKDIR_HOST,
# but the sidecar sees it at KIMI_WORKDIR_CONTAINER.
KIMI_WORKDIR_HOST = Path(os.getenv("KIMI_WORKDIR_HOST", "/kb/kimi-workdirs"))
KIMI_WORKDIR_CONTAINER = os.getenv("KIMI_WORKDIR_CONTAINER", "/workdirs")
KIMI_ENABLED = os.getenv("KIMI_BRIDGE_ENABLED", "false").strip().lower() == "true"


class KimiBridgeError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    h = {"content-type": "application/json"}
    if KIMI_BRIDGE_SECRET:
        h["x-kimi-bridge-secret"] = KIMI_BRIDGE_SECRET
    return h


def make_workdir(prefix: str) -> tuple[Path, str]:
    """Create an ephemeral workdir on host; return (host_path, container_path)."""
    KIMI_WORKDIR_HOST.mkdir(parents=True, exist_ok=True)
    host = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=KIMI_WORKDIR_HOST))
    # Sidecar sees it at /workdirs/<name>.
    container = f"{KIMI_WORKDIR_CONTAINER}/{host.name}"
    return host, container


def cleanup_workdir(host_path: Path) -> None:
    try:
        shutil.rmtree(host_path, ignore_errors=True)
    except Exception:  # pragma: no cover
        logger.exception("Failed to clean up kimi workdir %s", host_path)


async def run_agent(
    prompt: str,
    workdir_container: str,
    *,
    model: str | None = None,
    thinking: bool = True,
    timeout_ms: int = 1_500_000,
) -> dict[str, Any]:
    """Call the sidecar. Raises KimiBridgeError on transport failure."""
    body = {
        "prompt": prompt,
        "workDir": workdir_container,
        "thinking": thinking,
        "timeoutMs": timeout_ms,
    }
    if model:
        body["model"] = model
    # httpx read timeout must exceed the sidecar's own turn timeout.
    read_timeout = max(60.0, (timeout_ms / 1000.0) + 60.0)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(read_timeout, connect=10.0)) as client:
            resp = await client.post(
                f"{KIMI_BRIDGE_URL}/agent/run",
                json=body,
                headers=_headers(),
            )
    except httpx.HTTPError as exc:
        raise KimiBridgeError(f"kimi-bridge transport error: {exc}") from exc
    if resp.status_code != 200:
        raise KimiBridgeError(
            f"kimi-bridge HTTP {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json()


def walk_workdir(host_path: Path, max_files: int = 50, max_bytes: int = 2_000_000) -> list[dict]:
    """Collect files kimi wrote. Mirrors scaffolds.create_scaffold shape."""
    results: list[dict] = []
    total = 0
    for p in sorted(host_path.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(host_path).as_posix()
        # Skip kimi's own metadata (session files, shadow git) + the
        # sidecar's screenshot cache (tools write there, not scaffold output).
        if (rel.startswith(".kimi/") or rel.startswith(".git/")
                or rel.startswith(".screenshots/")):
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        total += len(data)
        if len(results) >= max_files or total > max_bytes:
            logger.warning("Stopping workdir walk at %d files / %d bytes", len(results), total)
            break
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF8 file %s", rel)
            continue
        results.append({"path": rel, "content": text})
    return results
