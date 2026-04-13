"""Benchmark fixtures — parameterized storage + db backends for latency
comparison. Reuses the same moto plumbing as parity tests but tuned
for pytest-benchmark: fewer assertions, more timing.
"""

from __future__ import annotations

import importlib
import os
import sys
import uuid
from pathlib import Path

import pytest


# Mark every test in this package as 'bench' so `make bench` can
# select them and the default `make test` excludes them.
def pytest_collection_modifyitems(items):
    for item in items:
        if "bench" in str(item.fspath):
            item.add_marker(pytest.mark.bench)


@pytest.fixture(params=["local", "s3-moto"], ids=["local", "s3-moto"])
def bench_storage(request, tmp_path, monkeypatch):
    """Same logic as parity_storage but without real-AWS tier
    (benchmarking moto vs local is the goal)."""
    from app import storage as storage_module
    monkeypatch.setattr(storage_module, "_default", None)

    if request.param == "local":
        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        from app import config as cfg
        monkeypatch.setattr(cfg, "KB_ROOT", tmp_path)
        orig = dict(storage_module.KB_DIRS)
        storage_module.KB_DIRS.clear()
        storage_module.KB_DIRS["bench-kb"] = tmp_path / "bench-kb"
        inst = storage_module.LocalStorage()
        storage_module.set_storage(inst)
        inst.init_kb("bench-kb")
        try:
            yield inst, "local"
        finally:
            storage_module.KB_DIRS.clear()
            storage_module.KB_DIRS.update(orig)
    else:
        from moto import mock_aws
        import boto3
        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(
                Bucket="bench-moto",
            )
            monkeypatch.setenv("STORAGE_BACKEND", "s3")
            monkeypatch.setenv("S3_BUCKET", "bench-moto")
            monkeypatch.setenv("S3_PREFIX", "")
            monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto")
            monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto")
            monkeypatch.setenv("AWS_REGION", "us-east-1")
            from app.storage import _s3_client
            _s3_client.cache_clear()
            inst = storage_module.S3Storage(bucket="bench-moto", prefix="")
            storage_module.set_storage(inst)
            yield inst, "s3-moto"

    monkeypatch.setattr(storage_module, "_default", None)


@pytest.fixture(params=["sqlite", "dynamodb-moto"], ids=["sqlite", "ddb-moto"])
def bench_db(request, tmp_path, monkeypatch):
    """Same logic as parity_db but yields (db_module, backend_name)."""
    if request.param == "sqlite":
        monkeypatch.setenv("DB_BACKEND", "sqlite")
        db_path = tmp_path / "bench.db"
        monkeypatch.setenv("DB_PATH", str(db_path))
        from app import config as cfg
        monkeypatch.setattr(cfg, "DB_PATH", db_path)
        for mod in ("app.db", "app.db_dynamo"):
            if mod in sys.modules:
                del sys.modules[mod]
        db_mod = importlib.import_module("app.db")
        monkeypatch.setattr(db_mod, "DB_PATH", db_path, raising=False)
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(db_mod.init_db())
        finally:
            loop.close()
        yield db_mod, "sqlite"
    else:
        from moto import mock_aws
        import boto3
        with mock_aws():
            monkeypatch.setenv("DB_BACKEND", "dynamodb")
            monkeypatch.setenv("AWS_ACCESS_KEY_ID", "moto")
            monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "moto")
            monkeypatch.setenv("AWS_REGION", "us-east-1")
            monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
            monkeypatch.setenv("DB_DYNAMO_TABLE", "bench-moto")
            from app import config as cfg
            monkeypatch.setattr(cfg, "BEDROCK_REGION", "us-east-1")
            monkeypatch.setattr(cfg, "AWS_ACCESS_KEY_ID", "moto")
            monkeypatch.setattr(cfg, "AWS_SECRET_ACCESS_KEY", "moto")
            monkeypatch.setattr(cfg, "AWS_SESSION_TOKEN", "")

            ddb = boto3.client("dynamodb", region_name="us-east-1")
            ddb.create_table(
                TableName="bench-moto",
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
                GlobalSecondaryIndexes=[{
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
                }],
                BillingMode="PROVISIONED",
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )
            ddb.get_waiter("table_exists").wait(TableName="bench-moto")

            for mod in ("app.db", "app.db_dynamo"):
                if mod in sys.modules:
                    del sys.modules[mod]
            dyn = importlib.import_module("app.db_dynamo")
            dyn._table = None
            dyn.TABLE_NAME = "bench-moto"
            db_mod = importlib.import_module("app.db")
            yield db_mod, "dynamodb-moto"

    for mod in ("app.db", "app.db_dynamo"):
        if mod in sys.modules:
            del sys.modules[mod]
