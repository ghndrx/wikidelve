"""Shared fixtures for unit tests."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


ARTICLE_1 = """\
---
title: Python Basics
summary: Introduction to Python programming
tags: [python, tutorial, beginner]
---
Python is a versatile programming language used for web development,
data science, and automation. It has a simple syntax that makes it
great for beginners.
"""

ARTICLE_2 = """\
---
title: Rust Memory Safety
summary: How Rust ensures memory safety without garbage collection
tags: [rust, memory, systems]
---
Rust uses ownership and borrowing to guarantee memory safety at compile time.
The borrow checker prevents data races and dangling pointers without runtime
overhead, making Rust suitable for systems programming.
"""

ARTICLE_3 = """\
---
title: Docker Containers
summary: Container orchestration with Docker
tags: [docker, containers, devops]
---
Docker containers package applications with their dependencies for consistent
deployment across environments. Docker Compose simplifies multi-container setups
and networking between services.
"""

UNDERSCORE_ARTICLE = """\
---
title: Hidden Template
summary: This should be skipped
tags: [template]
---
This file starts with underscore and should be excluded from indexing.
"""


@pytest.fixture
def tmp_kb(tmp_path):
    """Create a temporary KB directory with 3 articles and 1 underscore file."""
    kb_dir = tmp_path / "test_kb"
    wiki_dir = kb_dir / "wiki"
    raw_dir = kb_dir / "raw"
    wiki_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    (wiki_dir / "python-basics.md").write_text(ARTICLE_1)
    (wiki_dir / "rust-memory-safety.md").write_text(ARTICLE_2)
    (wiki_dir / "docker-containers.md").write_text(ARTICLE_3)
    (wiki_dir / "_template.md").write_text(UNDERSCORE_ARTICLE)

    return kb_dir


@pytest.fixture
def mock_kb_dirs(tmp_kb, monkeypatch):
    """Point storage + KB_DIRS at the tmp KB and isolate the storage singleton."""
    import app.config as _config
    import app.storage as _storage
    import app.wiki as _wiki

    dirs = {"test": tmp_kb}

    prev_kb_dirs = dict(_config.KB_DIRS)
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(dirs)

    # Redirect the fallback root onto the tmp dir so code paths that auto-
    # create a KB on demand don't hit the real /kb mount.
    fallback_root = tmp_kb.parent / "_fallback"
    fallback_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_storage, "KB_ROOT", fallback_root)

    prev_backend = _storage._default
    _storage._default = _storage.LocalStorage()
    _wiki.invalidate_articles_cache()

    yield dirs

    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(prev_kb_dirs)
    _storage._default = prev_backend
    _wiki.invalidate_articles_cache()


@pytest.fixture
def mock_db_path(tmp_path):
    """Return a temp DB path for tests that need one."""
    return tmp_path / "test_search.db"
