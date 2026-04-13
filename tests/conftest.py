"""Shared test fixtures for WikiDelve tests."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Force a local, isolated environment before any app.* modules import.
# The .env file ships production settings (DynamoDB, S3) which unit tests
# must not touch.
os.environ["DB_BACKEND"] = "sqlite"
os.environ["STORAGE_BACKEND"] = "local"
os.environ.pop("S3_BUCKET", None)
os.environ.pop("DYNAMODB_TABLE", None)

# Point KB_ROOT at a writeable tmp dir BEFORE app.config imports. The
# default is /kb which test processes usually can't create. Tests that
# need a specific KB layout override via monkeypatch later; this just
# stops import-time `KB_ROOT.iterdir()` calls from exploding.
_TEST_KB_ROOT = tempfile.mkdtemp(prefix="wikidelve-tests-kb-")
os.environ["KB_ROOT"] = _TEST_KB_ROOT
os.environ["PERSONAL_KB_PATH"] = str(Path(_TEST_KB_ROOT) / "personal")

# Extend the host allowlist so tailnet-style host tests pass without
# hardcoding a real personal tailnet name in the repo.
os.environ["ALLOWED_HOSTS"] = (
    "localhost,127.0.0.1,kb-service,wikidelve,testserver,"
    "wikidelve.example.ts.net,*.example.ts.net"
)
Path(_TEST_KB_ROOT, "personal", "wiki").mkdir(parents=True, exist_ok=True)

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOTE: the custom `event_loop` fixture we used to ship here is
# deprecated in pytest-asyncio 0.23+ and, under auto mode, it causes
# async tests to dispatch to a different loop than the one their
# fixtures are attached to — breaking ~200 tests at once. Remove the
# override and let pytest-asyncio manage the loop itself. If a test
# needs a session-scoped loop, use `@pytest.mark.asyncio(scope="session")`
# directly on the test.


@pytest.fixture
def tmp_kb(tmp_path):
    """Create a temporary KB directory structure with sample articles."""
    kb_dir = tmp_path / "test-kb"
    wiki_dir = kb_dir / "wiki"
    raw_dir = kb_dir / "raw"
    wiki_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    # Sample article with frontmatter
    (wiki_dir / "kubernetes-basics.md").write_text(
        '---\ntitle: "Kubernetes Basics"\ntags: [kubernetes, containers, orchestration]\n'
        'status: published\nconfidence: high\nupdated: 2026-04-01\n'
        'summary: "An introduction to Kubernetes container orchestration"\n'
        'source_type: web\nsource_files:\n  - raw/kubernetes-basics.md\n---\n\n'
        "## Executive Summary\n\n"
        "Kubernetes is an open-source container orchestration platform.\n\n"
        "## Key Concepts\n\n"
        "### Pods\n\nA pod is the smallest deployable unit in Kubernetes.\n\n"
        "### Services\n\nServices expose pods to the network.\n\n"
        "## Best Practices\n\n- Use namespaces\n- Set resource limits\n- Use liveness probes\n"
    )

    (wiki_dir / "docker-networking.md").write_text(
        '---\ntitle: "Docker Networking"\ntags: [docker, networking]\n'
        'status: draft\nconfidence: medium\nupdated: 2026-03-15\n'
        'summary: "Docker networking modes and configuration"\n---\n\n'
        "Docker provides several networking drivers.\n\n"
        "## Bridge\n\nThe default networking mode.\n"
    )

    (wiki_dir / "python-testing.md").write_text(
        '---\ntitle: "Python Testing Best Practices"\ntags: [python, testing, pytest]\n'
        'status: published\nconfidence: high\nupdated: 2026-04-05\n'
        'summary: "How to write effective Python tests"\n---\n\n'
        "## Introduction\n\nTesting is essential for software quality.\n\n"
        "## Using pytest\n\n```python\ndef test_example():\n    assert 1 + 1 == 2\n```\n\n"
        "## Fixtures\n\nFixtures provide test data and setup.\n"
    )

    (wiki_dir / "_index.md").write_text("# Index\nThis should be skipped\n")

    return kb_dir


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def mock_kb_dirs(tmp_kb, monkeypatch):
    """Point storage + KB_DIRS at the tmp KB so file-level tests are isolated."""
    dirs = {"test": tmp_kb}
    import app.config as _config
    import app.storage as _storage
    import app.wiki as _wiki

    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(dirs)

    local = _storage.LocalStorage()
    prev_backend = _storage._default
    _storage._default = local

    _wiki.invalidate_articles_cache()

    yield dirs

    _storage._default = prev_backend
    _wiki.invalidate_articles_cache()


@pytest.fixture
def mock_db_path(tmp_db, monkeypatch):
    """Patch DB_PATH to use a temporary database."""
    monkeypatch.setattr("app.config.DB_PATH", tmp_db)
    monkeypatch.setattr("app.search.DB_PATH", tmp_db)
    return tmp_db
