"""
Optional S3 backing store for WikiDelve.

When S3_BUCKET is set, all wiki articles, raw sources, research files,
and the SQLite database are mirrored to S3. On startup, missing local
files are restored from S3 (pull-on-start).

Enable by setting:
    S3_BUCKET=my-wikidelve-backup
    S3_PREFIX=wikidelve/           # optional, default ""

Uses the same AWS credentials already configured for Bedrock.
"""

import asyncio
import logging
import os
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger("kb-service.s3")

S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_PREFIX = os.getenv("S3_PREFIX", "").strip().strip("/")

# Module-level flag — skip all S3 ops when bucket is not configured
_enabled = bool(S3_BUCKET)


def is_enabled() -> bool:
    return _enabled


@lru_cache(maxsize=1)
def _get_client():
    """Lazy-init boto3 S3 client, reusing Bedrock credentials."""
    import boto3
    from app.config import (
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_SESSION_TOKEN,
        BEDROCK_REGION,
    )

    kwargs: dict = {"region_name": BEDROCK_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = AWS_SESSION_TOKEN

    return boto3.client("s3", **kwargs)


def _s3_key(local_path: str | Path) -> str:
    """Convert a local absolute path to an S3 key.

    /kb/personal/wiki/my-article.md  →  {prefix}/personal/wiki/my-article.md
    /kb/research/output.md           →  {prefix}/research/output.md
    /kb/wikidelve.db                 →  {prefix}/wikidelve.db
    """
    path_str = str(local_path)

    # Strip the /kb/ root to get the relative portion
    if "/kb/" in path_str:
        relative = path_str.split("/kb/", 1)[1]
    else:
        relative = Path(path_str).name

    if S3_PREFIX:
        return f"{S3_PREFIX}/{relative}"
    return relative


# ---------------------------------------------------------------------------
# Upload / delete (fire-and-forget from sync context)
# ---------------------------------------------------------------------------

def upload_file(local_path: str | Path) -> None:
    """Upload a single file to S3. No-op if S3 is not configured."""
    if not _enabled:
        return
    local_path = Path(local_path)
    if not local_path.exists():
        return
    try:
        key = _s3_key(local_path)
        _get_client().upload_file(str(local_path), S3_BUCKET, key)
        logger.debug("S3 upload: %s → s3://%s/%s", local_path, S3_BUCKET, key)
    except Exception as exc:
        logger.warning("S3 upload failed for %s: %s", local_path, exc)


def delete_file(local_path: str | Path) -> None:
    """Delete a single file from S3. No-op if S3 is not configured."""
    if not _enabled:
        return
    try:
        key = _s3_key(local_path)
        _get_client().delete_object(Bucket=S3_BUCKET, Key=key)
        logger.debug("S3 delete: s3://%s/%s", S3_BUCKET, key)
    except Exception as exc:
        logger.warning("S3 delete failed for %s: %s", local_path, exc)


def upload_file_async(local_path: str | Path) -> None:
    """Schedule upload in a background thread (non-blocking for async code)."""
    if not _enabled:
        return
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, upload_file, local_path)


def delete_file_async(local_path: str | Path) -> None:
    """Schedule delete in a background thread (non-blocking for async code)."""
    if not _enabled:
        return
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, delete_file, local_path)


# ---------------------------------------------------------------------------
# Bulk sync: pull from S3 on startup, push local to S3
# ---------------------------------------------------------------------------

def pull_from_s3(local_root: str | Path = "/kb") -> int:
    """Download all files from S3 bucket/prefix to local filesystem.

    Only downloads files that are missing locally or older than S3 version.
    Returns count of files restored.
    """
    if not _enabled:
        return 0

    local_root = Path(local_root)
    client = _get_client()
    prefix = f"{S3_PREFIX}/" if S3_PREFIX else ""
    restored = 0

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Convert S3 key back to local path
                relative = key[len(prefix):] if prefix and key.startswith(prefix) else key
                if not relative:
                    continue

                local_path = local_root / relative
                # Skip if local file exists and is newer
                if local_path.exists():
                    local_mtime = local_path.stat().st_mtime
                    s3_mtime = obj["LastModified"].timestamp()
                    if local_mtime >= s3_mtime:
                        continue

                # Download
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(S3_BUCKET, key, str(local_path))
                restored += 1
                logger.debug("S3 restore: s3://%s/%s → %s", S3_BUCKET, key, local_path)

    except Exception as exc:
        logger.error("S3 pull failed: %s", exc)

    if restored:
        logger.info("S3 sync: restored %d files from s3://%s/%s", restored, S3_BUCKET, prefix)
    return restored


def push_to_s3(local_root: str | Path = "/kb") -> int:
    """Upload all local KB files to S3. Used for initial backup or manual sync.

    Returns count of files uploaded.
    """
    if not _enabled:
        return 0

    local_root = Path(local_root)
    uploaded = 0

    for path in local_root.rglob("*"):
        if path.is_dir():
            continue
        # Skip temp files and __pycache__
        if "__pycache__" in str(path) or path.suffix in (".pyc", ".tmp"):
            continue
        try:
            upload_file(path)
            uploaded += 1
        except Exception as exc:
            logger.warning("S3 push failed for %s: %s", path, exc)

    logger.info("S3 sync: pushed %d files to s3://%s/%s", uploaded, S3_BUCKET, S3_PREFIX)
    return uploaded


def sync_db() -> None:
    """Upload the SQLite database to S3."""
    if not _enabled:
        return
    from app.config import DB_PATH
    upload_file(DB_PATH)
