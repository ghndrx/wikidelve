"""
Article storage abstraction.

Two backends:
  - LocalStorage  → filesystem under data/{kb}/{rel}
  - S3Storage     → s3://{bucket}/{prefix}/{kb}/{rel}

Selected by env var STORAGE_BACKEND ("local" | "s3").

All paths are KB-relative (e.g. "wiki/tokio.md", "raw/tokio.md"). The
storage layer owns the KB-root prefix on both backends, so callers never
construct full filesystem or S3 keys themselves.

The API is sync. S3 calls block the calling thread; for hot async paths
that touch many objects (cold-start FTS rebuild, get_articles), use the
bulk helpers (`iter_articles`, `list_slugs`) which fan out via boto3
pagination + a thread pool.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Protocol

from app.config import KB_DIRS, KB_ROOT

logger = logging.getLogger("kb-service.storage")


class Storage(Protocol):
    """Backend protocol — every method takes a KB name + KB-relative path."""

    def read_text(self, kb: str, rel_path: str) -> str | None: ...
    def write_text(self, kb: str, rel_path: str, content: str) -> None: ...
    def delete(self, kb: str, rel_path: str) -> bool: ...
    def exists(self, kb: str, rel_path: str) -> bool: ...
    def list_slugs(self, kb: str, subdir: str = "wiki", suffix: str = ".md") -> list[str]: ...
    def iter_articles(self, kb: str, subdir: str = "wiki") -> Iterator[tuple[str, str]]: ...
    def init_kb(self, kb: str) -> None: ...
    def list_kbs(self) -> list[str]: ...


# ---------------------------------------------------------------------------
# LocalStorage — filesystem under KB_DIRS[kb]
# ---------------------------------------------------------------------------


class LocalStorage:
    def _root(self, kb: str) -> Path:
        root = KB_DIRS.get(kb)
        if root is None:
            root = KB_ROOT / kb
        return root

    def _path(self, kb: str, rel_path: str) -> Path:
        # Defence-in-depth against path traversal. Route handlers
        # already sanitise filenames, but a single missed caller
        # shouldn't let an attacker escape the KB root.
        if "\\" in rel_path or rel_path.startswith("/"):
            raise ValueError(f"Invalid rel_path: {rel_path!r}")
        for segment in rel_path.split("/"):
            if segment in ("..", ".") or segment.startswith(".."):
                raise ValueError(f"Invalid rel_path segment: {segment!r}")
        return self._root(kb) / rel_path

    def read_text(self, kb: str, rel_path: str) -> str | None:
        p = self._path(kb, rel_path)
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("LocalStorage read failed %s: %s", p, exc)
            return None

    def write_text(self, kb: str, rel_path: str, content: str) -> None:
        p = self._path(kb, rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def delete(self, kb: str, rel_path: str) -> bool:
        p = self._path(kb, rel_path)
        if not p.exists():
            return False
        try:
            p.unlink()
            return True
        except OSError as exc:
            logger.warning("LocalStorage delete failed %s: %s", p, exc)
            return False

    def exists(self, kb: str, rel_path: str) -> bool:
        return self._path(kb, rel_path).exists()

    def list_slugs(self, kb: str, subdir: str = "wiki", suffix: str = ".md") -> list[str]:
        d = self._root(kb) / subdir
        if not d.exists():
            return []
        return sorted(
            f.stem for f in d.glob(f"*{suffix}") if not f.name.startswith("_")
        )

    def iter_articles(self, kb: str, subdir: str = "wiki") -> Iterator[tuple[str, str]]:
        d = self._root(kb) / subdir
        if not d.exists():
            return iter(())
        out: list[tuple[str, str]] = []
        for f in sorted(d.glob("*.md")):
            if f.name.startswith("_"):
                continue
            try:
                out.append((f.stem, f.read_text(encoding="utf-8", errors="replace")))
            except OSError as exc:
                logger.warning("LocalStorage iter read failed %s: %s", f, exc)
        return iter(out)

    def init_kb(self, kb: str) -> None:
        root = self._root(kb)
        (root / "wiki").mkdir(parents=True, exist_ok=True)
        (root / "raw").mkdir(parents=True, exist_ok=True)

    def list_kbs(self) -> list[str]:
        return sorted(KB_DIRS.keys())


# ---------------------------------------------------------------------------
# S3Storage — articles live under s3://{bucket}/{prefix}{kb}/{rel}
# ---------------------------------------------------------------------------


class S3Storage:
    """S3-backed storage. No local filesystem at all.

    Layout:
        s3://{bucket}/{prefix}{kb}/wiki/{slug}.md
        s3://{bucket}/{prefix}{kb}/raw/{slug}.md

    `prefix` is read from S3_PREFIX (empty by default → keys begin with `{kb}/`).
    """

    def __init__(self, bucket: str, prefix: str = ""):
        if not bucket:
            raise ValueError("S3Storage requires a bucket name")
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    @property
    def _client(self):
        return _s3_client()

    def _key(self, kb: str, rel_path: str) -> str:
        # Parity with LocalStorage._path: reject backslash, absolute
        # paths, and any `..` / `.` segment. Keeping the rule set
        # identical means callers get the same ValueError regardless
        # of STORAGE_BACKEND, which is the whole point of the parity
        # suite under tests/parity/.
        if "\\" in rel_path or rel_path.startswith("/"):
            raise ValueError(f"Invalid rel_path: {rel_path!r}")
        for segment in rel_path.split("/"):
            if segment in ("..", ".") or segment.startswith(".."):
                raise ValueError(f"Invalid rel_path segment: {segment!r}")
        rel = rel_path.lstrip("/")
        if self.prefix:
            return f"{self.prefix}/{kb}/{rel}"
        return f"{kb}/{rel}"

    def _kb_prefix(self, kb: str, subdir: str | None = None) -> str:
        parts = [self.prefix, kb] if self.prefix else [kb]
        if subdir:
            parts.append(subdir)
        return "/".join(parts) + "/"

    def read_text(self, kb: str, rel_path: str) -> str | None:
        key = self._key(kb, rel_path)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode("utf-8", errors="replace")
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            # Most other errors (404 wrapped, AccessDenied) → return None and log.
            code = getattr(getattr(exc, "response", {}), "get", lambda *_: {})("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            logger.warning("S3 get_object failed %s: %s", key, exc)
            return None

    def write_text(self, kb: str, rel_path: str, content: str) -> None:
        key = self._key(kb, rel_path)
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )

    def delete(self, kb: str, rel_path: str) -> bool:
        # S3 DELETE is idempotent and returns 204 even for missing
        # keys, which breaks parity with LocalStorage (which returns
        # False when the path doesn't exist). HEAD first so callers
        # can rely on the return value to mean "something was
        # actually deleted" regardless of STORAGE_BACKEND.
        key = self._key(kb, rel_path)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return False
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            logger.warning("S3 delete_object failed %s: %s", key, exc)
            return False

    def exists(self, kb: str, rel_path: str) -> bool:
        key = self._key(kb, rel_path)
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_slugs(self, kb: str, subdir: str = "wiki", suffix: str = ".md") -> list[str]:
        prefix = self._kb_prefix(kb, subdir)
        slugs: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = key[len(prefix):]
                if "/" in name or name.startswith("_"):
                    continue
                if not name.endswith(suffix):
                    continue
                slugs.append(name[: -len(suffix)])
        slugs.sort()
        return slugs

    def iter_articles(self, kb: str, subdir: str = "wiki") -> Iterator[tuple[str, str]]:
        prefix = self._kb_prefix(kb, subdir)
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                name = key[len(prefix):]
                if "/" in name or name.startswith("_") or not name.endswith(".md"):
                    continue
                keys.append(key)
        if not keys:
            return iter(())

        def _fetch(key: str) -> tuple[str, str] | None:
            try:
                resp = self._client.get_object(Bucket=self.bucket, Key=key)
                slug = key[len(prefix):][:-len(".md")]
                return slug, resp["Body"].read().decode("utf-8", errors="replace")
            except Exception as exc:
                logger.warning("S3 iter fetch failed %s: %s", key, exc)
                return None

        results: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            for r in pool.map(_fetch, keys):
                if r is not None:
                    results.append(r)
        results.sort(key=lambda t: t[0])
        return iter(results)

    def init_kb(self, kb: str) -> None:
        # No-op: S3 has no directories. Marker keys are not needed because
        # list operations succeed against empty prefixes.
        return None

    def list_kbs(self) -> list[str]:
        """Discover KBs by listing top-level prefixes in the bucket."""
        prefix = (self.prefix + "/") if self.prefix else ""
        out: set[str] = set()
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []) or []:
                p = cp.get("Prefix", "")
                if not p:
                    continue
                # strip the configured prefix and trailing slash
                name = p[len(prefix):].rstrip("/")
                # Skip the tfstate prefix and any reserved KBs
                if not name or name.startswith("_") or name in ("tfstate",):
                    continue
                out.add(name)
        return sorted(out)


@lru_cache(maxsize=1)
def _s3_client():
    import boto3
    from botocore.config import Config
    region = os.getenv("AWS_REGION") or os.getenv("BEDROCK_REGION") or "us-west-2"
    kwargs: dict = {
        "region_name": region,
        "config": Config(max_pool_connections=64),
    }
    ak = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    sk = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if ak and sk:
        kwargs["aws_access_key_id"] = ak
        kwargs["aws_secret_access_key"] = sk
    return boto3.client("s3", **kwargs)


# ---------------------------------------------------------------------------
# Module-level singleton — pick backend at import time from env
# ---------------------------------------------------------------------------


_BACKEND = os.getenv("STORAGE_BACKEND", "local").strip().lower()
_S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
_S3_PREFIX = os.getenv("S3_PREFIX", "").strip()


def _build_default() -> Storage:
    if _BACKEND == "s3":
        if not _S3_BUCKET:
            raise RuntimeError("STORAGE_BACKEND=s3 but S3_BUCKET is not set")
        return S3Storage(_S3_BUCKET, _S3_PREFIX)
    return LocalStorage()


_default: Storage | None = None


def get_storage() -> Storage:
    global _default
    if _default is None:
        _default = _build_default()
    return _default


def set_storage(storage: Storage) -> None:
    """Override the storage backend (used in tests)."""
    global _default
    _default = storage


# Convenience module-level passthroughs ---------------------------------------


def read_text(kb: str, rel_path: str) -> str | None:
    return get_storage().read_text(kb, rel_path)


def write_text(kb: str, rel_path: str, content: str) -> None:
    get_storage().write_text(kb, rel_path, content)


def delete(kb: str, rel_path: str) -> bool:
    return get_storage().delete(kb, rel_path)


def exists(kb: str, rel_path: str) -> bool:
    return get_storage().exists(kb, rel_path)


def list_slugs(kb: str, subdir: str = "wiki", suffix: str = ".md") -> list[str]:
    return get_storage().list_slugs(kb, subdir, suffix)


def iter_articles(kb: str, subdir: str = "wiki") -> Iterator[tuple[str, str]]:
    return get_storage().iter_articles(kb, subdir)


def init_kb(kb: str) -> None:
    get_storage().init_kb(kb)


def list_kbs() -> list[str]:
    return get_storage().list_kbs()


def backend_name() -> str:
    s = get_storage()
    return "s3" if isinstance(s, S3Storage) else "local"
