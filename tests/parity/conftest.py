"""Backend-parity test fixtures.

Every test under `tests/parity/` is parameterized across the local
and AWS backends so one test body proves the two paths behave
identically. Three tiers:

    local        — LocalStorage on tmp_path, SQLite in tmp_path
    s3-moto      — moto in-memory replica of S3 + DynamoDB (always on)
    aws-real     — user's real AWS account, OPT-IN via INTEGRATION_AWS=1
                   with a dedicated test bucket + table

The aws-real tier is gated for safety: it refuses to run unless
``WIKIDELVE_AWS_TEST_BUCKET`` and ``WIKIDELVE_AWS_TEST_TABLE`` are set
AND they are DIFFERENT from the production ``S3_BUCKET`` /
``DYNAMODB_TABLE`` values. Every test uses a unique per-run prefix so
parallel runs don't collide and cleanup is guaranteed.

Fixtures exposed:
    parity_storage   — LocalStorage | S3Storage(moto|real), parameterized
    parity_db        — app.db module re-bound to sqlite | dynamodb(moto|real)
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Iterator

import pytest


# ---------------------------------------------------------------------------
# Real AWS gating
# ---------------------------------------------------------------------------

INTEGRATION_AWS = os.getenv("INTEGRATION_AWS", "").strip() == "1"

# Test-only env vars — must differ from prod S3_BUCKET / DYNAMODB_TABLE.
AWS_TEST_BUCKET = os.getenv("WIKIDELVE_AWS_TEST_BUCKET", "").strip()
AWS_TEST_TABLE = os.getenv("WIKIDELVE_AWS_TEST_TABLE", "").strip()

# Guard: refuse to run against production. These are the values that
# power the live deployment — blowing them up would destroy real data.
PROD_BUCKET = os.getenv("S3_BUCKET", "").strip()
PROD_TABLE = os.getenv("DYNAMODB_TABLE", "").strip()


def _aws_test_targets_are_safe() -> tuple[bool, str]:
    """Return (ok, reason). Never run real-AWS tests unless ok is True."""
    if not INTEGRATION_AWS:
        return False, "INTEGRATION_AWS not set — real-AWS parity skipped"
    if not AWS_TEST_BUCKET:
        return False, "WIKIDELVE_AWS_TEST_BUCKET not set"
    if not AWS_TEST_TABLE:
        return False, "WIKIDELVE_AWS_TEST_TABLE not set"
    if PROD_BUCKET and AWS_TEST_BUCKET == PROD_BUCKET:
        return False, (
            f"WIKIDELVE_AWS_TEST_BUCKET={AWS_TEST_BUCKET!r} matches "
            f"production S3_BUCKET — refusing to run"
        )
    if PROD_TABLE and AWS_TEST_TABLE == PROD_TABLE:
        return False, (
            f"WIKIDELVE_AWS_TEST_TABLE={AWS_TEST_TABLE!r} matches "
            f"production DYNAMODB_TABLE — refusing to run"
        )
    return True, ""


def _aws_parity_params() -> list[str]:
    """Return the backend list for parity tests.

    Always includes local + moto. Adds 'aws' only when the safety
    guard passes — otherwise the param list stays 2-wide and real-AWS
    tests are skipped entirely (not even collected, so failures don't
    surface in offline runs).
    """
    params = ["local", "s3-moto", "sqlite", "dynamodb-moto"]
    ok, _ = _aws_test_targets_are_safe()
    if ok:
        params += ["aws-real"]
    return params


# Per-run prefix so real-AWS artifacts are always isolated and
# cleanable. UUID4 is overkill but the collision probability is zero.
AWS_TEST_RUN_ID = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Storage fixture
# ---------------------------------------------------------------------------


def _storage_params() -> list:
    """Two or three tiers depending on INTEGRATION_AWS."""
    tiers = [
        pytest.param("local", id="local"),
        pytest.param("s3-moto", id="s3-moto"),
    ]
    ok, _ = _aws_test_targets_are_safe()
    if ok:
        tiers.append(pytest.param("s3-real", id="s3-real"))
    return tiers


@pytest.fixture(params=_storage_params())
def parity_storage(request, tmp_path: Path, monkeypatch) -> Iterator:
    """Yield a Storage instance for each backend tier.

    Tier selection:
      - local:    LocalStorage on tmp_path
      - s3-moto:  S3Storage against moto in-memory bucket
      - s3-real:  S3Storage against the user's real AWS_TEST_BUCKET,
                  scoped to a unique prefix per test run
    """
    tier = request.param

    # Always reset the singleton so the backend switch takes effect.
    from app import storage as storage_module
    monkeypatch.setattr(storage_module, "_default", None)

    if tier == "local":
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("KB_ROOT", str(tmp_path))

        # storage.py does `from app.config import KB_DIRS, KB_ROOT` so
        # those names are rebound in the storage module — patching
        # config.* alone is not enough. Patch BOTH modules, and mutate
        # the KB_DIRS dict IN PLACE so the reference storage.py holds
        # stays valid for the duration of the test.
        from app import config as config_module
        monkeypatch.setattr(config_module, "KB_ROOT", tmp_path)
        monkeypatch.setattr(storage_module, "KB_ROOT", tmp_path)

        # Snapshot and clear the dict, then install the test entry.
        original_kb_dirs = dict(storage_module.KB_DIRS)
        storage_module.KB_DIRS.clear()
        storage_module.KB_DIRS["test-kb"] = tmp_path / "test-kb"
        # config.KB_DIRS is the same dict object (module-level import
        # chain) so we don't need a second mutation, but assert it to
        # catch future refactors that might break the link.
        assert config_module.KB_DIRS is storage_module.KB_DIRS

        inst = storage_module.LocalStorage()
        storage_module.set_storage(inst)
        inst.init_kb("test-kb")
        try:
            yield inst
        finally:
            storage_module.KB_DIRS.clear()
            storage_module.KB_DIRS.update(original_kb_dirs)

    elif tier == "s3-moto":
        from moto import mock_aws
        import boto3

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="wikidelve-moto-test")

            monkeypatch.setenv("STORAGE_BACKEND", "s3")
            monkeypatch.setenv("S3_BUCKET", "wikidelve-moto-test")
            monkeypatch.setenv("S3_PREFIX", "")
            monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto-test")
            monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto-test")
            monkeypatch.setenv("AWS_REGION", "us-east-1")

            from app.storage import _s3_client
            _s3_client.cache_clear()

            inst = storage_module.S3Storage(
                bucket="wikidelve-moto-test", prefix="",
            )
            storage_module.set_storage(inst)
            yield inst

    elif tier == "s3-real":
        # Safety check — refuse if the env is mis-configured.
        ok, reason = _aws_test_targets_are_safe()
        if not ok:
            pytest.skip(reason)

        # Every test gets its own prefix so parallel runs and leaked
        # cleanups don't step on each other.
        prefix = f"parity-test/{AWS_TEST_RUN_ID}/{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv("STORAGE_BACKEND", "s3")
        monkeypatch.setenv("S3_BUCKET", AWS_TEST_BUCKET)
        monkeypatch.setenv("S3_PREFIX", prefix)

        from app.storage import _s3_client
        _s3_client.cache_clear()

        inst = storage_module.S3Storage(bucket=AWS_TEST_BUCKET, prefix=prefix)
        storage_module.set_storage(inst)
        try:
            yield inst
        finally:
            # Clean up every key under our test prefix. This must run
            # even if the test fails.
            import boto3
            client = boto3.client("s3")
            paginator = client.get_paginator("list_objects_v2")
            to_delete: list[dict] = []
            for page in paginator.paginate(
                Bucket=AWS_TEST_BUCKET, Prefix=prefix + "/",
            ):
                for obj in page.get("Contents", []) or []:
                    to_delete.append({"Key": obj["Key"]})
                    if len(to_delete) == 1000:
                        client.delete_objects(
                            Bucket=AWS_TEST_BUCKET,
                            Delete={"Objects": to_delete},
                        )
                        to_delete = []
            if to_delete:
                client.delete_objects(
                    Bucket=AWS_TEST_BUCKET,
                    Delete={"Objects": to_delete},
                )
    else:
        raise ValueError(f"Unknown storage tier: {tier}")

    # Final cleanup: drop the singleton so the next test starts fresh.
    monkeypatch.setattr(storage_module, "_default", None)


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------


def _db_params() -> list:
    tiers = [
        pytest.param("sqlite", id="sqlite"),
        pytest.param("dynamodb-moto", id="dynamodb-moto"),
    ]
    ok, _ = _aws_test_targets_are_safe()
    if ok:
        tiers.append(pytest.param("dynamodb-real", id="dynamodb-real"))
    return tiers


def _create_test_ddb_table(ddb_client, table_name: str) -> None:
    """Create the single-table schema that prod uses. Waits for ACTIVE."""
    ddb_client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
            {"AttributeName": "GSI1PK", "AttributeType": "S"},
            {"AttributeName": "GSI1SK", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "GSI1",
                "KeySchema": [
                    {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                    {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5,
                },
            },
        ],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5,
        },
    )
    ddb_client.get_waiter("table_exists").wait(TableName=table_name)


def _scan_and_clear_table(ddb_resource, table_name: str, pk_prefix: str) -> None:
    """Delete every item whose PK starts with the given per-run prefix."""
    table = ddb_resource.Table(table_name)
    # Scan with a filter — the test data volume is small so scan is fine.
    last_key = None
    while True:
        kwargs = {
            "FilterExpression": "begins_with(PK, :p)",
            "ExpressionAttributeValues": {":p": pk_prefix},
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = table.scan(**kwargs)
        for item in resp.get("Items", []):
            table.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break


@pytest.fixture(params=_db_params())
def parity_db(request, tmp_path: Path, monkeypatch):
    """Yield the `app.db` module bound to each backend.

    sqlite         — fresh temp database
    dynamodb-moto  — moto in-memory DynamoDB + single-table schema
    dynamodb-real  — user's real AWS_TEST_TABLE, items scoped to a
                     per-run PK prefix so cleanup is bounded
    """
    tier = request.param

    if tier == "sqlite":
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("DB_PATH", str(db_path))

        import importlib
        from app import config as config_module
        monkeypatch.setattr(config_module, "DB_PATH", db_path)

        # Drop the cached db + db_dynamo imports so the trailer runs
        # in sqlite mode (prior test may have left dynamodb bound).
        for mod_name in ("app.db", "app.db_dynamo"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        db_module = importlib.import_module("app.db")
        # Also patch the module-level DB_PATH binding that db.py
        # imported by name — same pattern as storage.KB_DIRS.
        monkeypatch.setattr(db_module, "DB_PATH", db_path, raising=False)

        import asyncio
        # asyncio.get_event_loop() is deprecated in 3.12; use
        # new_event_loop().run_until_complete so pytest-asyncio's
        # auto-mode doesn't clobber us.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(db_module.init_db())
        finally:
            loop.close()
        yield db_module

    elif tier == "dynamodb-moto":
        from moto import mock_aws
        import boto3

        with mock_aws():
            # Env vars — must be set BEFORE any re-import of db_dynamo
            # so its module-level `TABLE_NAME = os.getenv(...)` reads
            # the right value.
            monkeypatch.setenv("DB_BACKEND", "dynamodb")
            monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto-test")
            monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto-test")
            monkeypatch.setenv("AWS_REGION", "us-east-1")
            monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
            monkeypatch.setenv("DB_DYNAMO_TABLE", "wikidelve-moto-test")

            # db_dynamo imports `BEDROCK_REGION` from app.config by
            # name — env vars don't flow through unless we also patch
            # the already-cached app.config module. Same pattern as
            # the storage KB_DIRS fix.
            from app import config as config_module
            monkeypatch.setattr(config_module, "BEDROCK_REGION", "us-east-1")
            monkeypatch.setattr(config_module, "AWS_ACCESS_KEY_ID", "moto-test")
            monkeypatch.setattr(config_module, "AWS_SECRET_ACCESS_KEY", "moto-test")
            monkeypatch.setattr(config_module, "AWS_SESSION_TOKEN", "")

            ddb = boto3.client("dynamodb", region_name="us-east-1")
            _create_test_ddb_table(ddb, "wikidelve-moto-test")

            # Flush cached app.db + app.db_dynamo so the trailer re-runs
            # with DB_BACKEND=dynamodb AND so db_dynamo's module-level
            # `TABLE_NAME = os.getenv("DB_DYNAMO_TABLE", ...)` re-reads
            # the env var we just set. Do NOT clear app.config — storage
            # imports KB_DIRS/KB_ROOT from it by name, and wiping config
            # cascades into broken storage parity tests.
            import importlib
            for mod_name in ("app.db", "app.db_dynamo"):
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            # Fresh import triggers TABLE_NAME = "wikidelve-moto-test".
            dyn = importlib.import_module("app.db_dynamo")
            assert dyn.TABLE_NAME == "wikidelve-moto-test", (
                f"DB_DYNAMO_TABLE env var not picked up; got {dyn.TABLE_NAME!r}"
            )
            dyn._table = None  # force lazy client rebuild under mock_aws
            db_module = importlib.import_module("app.db")
            yield db_module

    elif tier == "dynamodb-real":
        ok, reason = _aws_test_targets_are_safe()
        if not ok:
            pytest.skip(reason)

        # We write into the user's real table. To keep cleanup tractable,
        # every test prefixes its PKs with a per-run sentinel so
        # `_scan_and_clear_table` can delete exactly what it wrote.
        pk_prefix = f"TEST-{AWS_TEST_RUN_ID}-{uuid.uuid4().hex[:8]}#"
        monkeypatch.setenv("DB_BACKEND", "dynamodb")
        monkeypatch.setenv("DYNAMODB_TABLE", AWS_TEST_TABLE)
        monkeypatch.setenv("WIKIDELVE_PK_PREFIX", pk_prefix)

        for mod_name in ("app.db", "app.db_dynamo"):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        import app.db as db_module

        import boto3
        ddb_resource = boto3.resource("dynamodb")

        try:
            yield db_module
        finally:
            _scan_and_clear_table(ddb_resource, AWS_TEST_TABLE, pk_prefix)

    else:
        raise ValueError(f"Unknown db tier: {tier}")

    # Drop cached imports so the next test starts fresh.
    for mod_name in ("app.db", "app.db_dynamo"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
