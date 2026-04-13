"""
Vector store abstraction.

Two backends:
  - SQLiteVectorStore  → JSON-encoded vectors in `article_embeddings` (legacy)
  - S3VectorStore      → AWS S3 Vectors (per-KB index, server-side ANN search)

Selected by env var VECTOR_BACKEND ("sqlite" | "s3").

The store owns one collection per KB. With the S3 backend, collections map
1:1 to S3 Vectors indexes that are pre-provisioned by the wikidelve-infra
Terraform module (see `var.kb_names`).

API is async. SQLite uses aiosqlite. S3 Vectors uses boto3 in a thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Protocol

import aiosqlite

from app.config import DB_PATH

logger = logging.getLogger("kb-service.vector_store")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    async def upsert(self, kb: str, slug: str, vector: list[float]) -> None: ...
    async def delete(self, kb: str, slug: str) -> None: ...
    async def query(
        self, kb: str | None, vector: list[float], top_k: int = 10
    ) -> list[dict]:
        """Returns [{"slug": str, "kb": str, "score": float}, ...] sorted desc."""
        ...

    async def list_all(self, kb: str | None = None) -> list[dict]:
        """Returns [{"slug": str, "kb": str}, ...] for indexed-articles audit."""
        ...

    async def ensure_index(self, kb: str) -> None: ...


# ---------------------------------------------------------------------------
# SQLite backend (legacy — JSON in article_embeddings)
# ---------------------------------------------------------------------------


def _normalize(vec: list[float]) -> list[float]:
    mag = math.sqrt(sum(v * v for v in vec))
    if mag == 0:
        return vec
    return [v / mag for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SQLiteVectorStore:
    async def upsert(self, kb: str, slug: str, vector: list[float]) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(DB_PATH))
        try:
            await db.execute("PRAGMA journal_mode=WAL")
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                """INSERT INTO article_embeddings (slug, kb, embedding, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(slug, kb) DO UPDATE SET
                       embedding = excluded.embedding,
                       updated_at = excluded.updated_at""",
                (slug, kb, json.dumps(_normalize(vector)), now),
            )
            await db.commit()
        finally:
            await db.close()

    async def delete(self, kb: str, slug: str) -> None:
        db = await aiosqlite.connect(str(DB_PATH))
        try:
            await db.execute(
                "DELETE FROM article_embeddings WHERE kb = ? AND slug = ?",
                (kb, slug),
            )
            await db.commit()
        finally:
            await db.close()

    async def query(
        self, kb: str | None, vector: list[float], top_k: int = 10
    ) -> list[dict]:
        rows = await self._load(kb)
        if not rows:
            return []
        q = _normalize(vector)
        scored = []
        for r in rows:
            stored = json.loads(r["embedding"])
            scored.append({"slug": r["slug"], "kb": r["kb"], "score": _dot(q, stored)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def list_all(self, kb: str | None = None) -> list[dict]:
        rows = await self._load(kb)
        return [{"slug": r["slug"], "kb": r["kb"]} for r in rows]

    async def ensure_index(self, kb: str) -> None:
        return None  # nothing to provision

    async def _load(self, kb: str | None) -> list[dict]:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(DB_PATH))
        db.row_factory = aiosqlite.Row
        try:
            if kb:
                cur = await db.execute(
                    "SELECT slug, kb, embedding FROM article_embeddings WHERE kb = ?",
                    (kb,),
                )
            else:
                cur = await db.execute(
                    "SELECT slug, kb, embedding FROM article_embeddings",
                )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# S3 Vectors backend
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _s3vectors_client():
    import boto3
    region = os.getenv("AWS_REGION") or os.getenv("BEDROCK_REGION") or "us-west-2"
    kwargs: dict = {"region_name": region}
    ak = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    sk = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    if ak and sk:
        kwargs["aws_access_key_id"] = ak
        kwargs["aws_secret_access_key"] = sk
    return boto3.client("s3vectors", **kwargs)


class S3VectorStore:
    """One S3 Vectors index per KB.

    Index names match KB names (`personal`, `pluto`, ...). The Terraform
    module provisions them; this class only does data-plane ops.
    """

    def __init__(self, vector_bucket: str):
        if not vector_bucket:
            raise ValueError("S3VectorStore requires S3_VECTORS_BUCKET")
        self.bucket = vector_bucket

    @property
    def _client(self):
        return _s3vectors_client()

    def _index_name(self, kb: str) -> str:
        return kb

    async def ensure_index(self, kb: str) -> None:
        """Verify the index exists. We do not auto-create it — Terraform owns
        the control plane. If it's missing, raise so the operator notices and
        adds the KB to var.kb_names + reapplies."""
        def _check():
            try:
                self._client.get_index(
                    vectorBucketName=self.bucket,
                    indexName=self._index_name(kb),
                )
                return True
            except Exception:
                return False
        ok = await asyncio.to_thread(_check)
        if not ok:
            raise RuntimeError(
                f"S3 Vectors index '{kb}' missing in bucket '{self.bucket}'. "
                f"Add '{kb}' to var.kb_names in wikidelve-infra and apply."
            )

    async def upsert(self, kb: str, slug: str, vector: list[float]) -> None:
        normalized = _normalize(vector)
        index = self._index_name(kb)
        bucket = self.bucket

        def _put():
            self._client.put_vectors(
                vectorBucketName=bucket,
                indexName=index,
                vectors=[
                    {
                        "key": slug,
                        "data": {"float32": normalized},
                        "metadata": {"kb": kb, "slug": slug},
                    }
                ],
            )

        await asyncio.to_thread(_put)

    async def delete(self, kb: str, slug: str) -> None:
        index = self._index_name(kb)
        bucket = self.bucket

        def _del():
            self._client.delete_vectors(
                vectorBucketName=bucket,
                indexName=index,
                keys=[slug],
            )

        await asyncio.to_thread(_del)

    async def query(
        self, kb: str | None, vector: list[float], top_k: int = 10
    ) -> list[dict]:
        normalized = _normalize(vector)
        bucket = self.bucket

        if kb is None:
            # Cross-KB search → fan out across all known indexes.
            indexes = await self._list_indexes()
        else:
            indexes = [self._index_name(kb)]

        def _query_one(idx: str) -> list[dict]:
            try:
                resp = self._client.query_vectors(
                    vectorBucketName=bucket,
                    indexName=idx,
                    queryVector={"float32": normalized},
                    topK=top_k,
                    returnDistance=True,
                    returnMetadata=True,
                )
            except Exception as exc:
                logger.warning("S3Vectors query failed (index=%s): %s", idx, exc)
                return []
            out = []
            for v in resp.get("vectors", []):
                meta = v.get("metadata") or {}
                # cosine distance → similarity
                dist = v.get("distance")
                score = 1.0 - float(dist) if dist is not None else 0.0
                out.append({
                    "slug": meta.get("slug") or v.get("key"),
                    "kb": meta.get("kb") or idx,
                    "score": score,
                })
            return out

        async def _gather():
            results: list[dict] = []
            for idx in indexes:
                rows = await asyncio.to_thread(_query_one, idx)
                results.extend(rows)
            return results

        all_rows = await _gather()
        all_rows.sort(key=lambda r: r["score"], reverse=True)
        return all_rows[:top_k]

    async def list_all(self, kb: str | None = None) -> list[dict]:
        bucket = self.bucket
        indexes = [kb] if kb else await self._list_indexes()

        def _list_one(idx: str) -> list[dict]:
            out: list[dict] = []
            paginator_args = {
                "vectorBucketName": bucket,
                "indexName": idx,
                "returnMetadata": True,
            }
            try:
                token = None
                while True:
                    if token:
                        paginator_args["nextToken"] = token
                    resp = self._client.list_vectors(**paginator_args)
                    for v in resp.get("vectors", []):
                        meta = v.get("metadata") or {}
                        out.append({
                            "slug": meta.get("slug") or v.get("key"),
                            "kb": meta.get("kb") or idx,
                        })
                    token = resp.get("nextToken")
                    if not token:
                        break
            except Exception as exc:
                logger.warning("S3Vectors list failed (index=%s): %s", idx, exc)
            return out

        results: list[dict] = []
        for idx in indexes:
            results.extend(await asyncio.to_thread(_list_one, idx))
        return results

    async def _list_indexes(self) -> list[str]:
        def _list():
            try:
                resp = self._client.list_indexes(vectorBucketName=self.bucket)
                return [i["indexName"] for i in resp.get("indexes", [])]
            except Exception as exc:
                logger.warning("S3Vectors list_indexes failed: %s", exc)
                return []
        return await asyncio.to_thread(_list)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------


_BACKEND = os.getenv("VECTOR_BACKEND", "sqlite").strip().lower()
_VECTORS_BUCKET = os.getenv("S3_VECTORS_BUCKET", "").strip()


def _build_default() -> VectorStore:
    if _BACKEND == "s3":
        if not _VECTORS_BUCKET:
            raise RuntimeError("VECTOR_BACKEND=s3 but S3_VECTORS_BUCKET is not set")
        return S3VectorStore(_VECTORS_BUCKET)
    return SQLiteVectorStore()


_default: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _default
    if _default is None:
        _default = _build_default()
    return _default


def set_vector_store(store: VectorStore) -> None:
    """Override the vector store (used in tests)."""
    global _default
    _default = store
