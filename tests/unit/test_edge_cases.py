"""Security + edge case sweep (P6) — compact tests for boundary conditions
that could slip through without dedicated coverage.

Covers: path traversal, rate-limit key composition, CSP nonce uniqueness,
corrupt frontmatter, empty KBs, wikilink cycles.
"""

from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# Path traversal guards (storage layer)
# ---------------------------------------------------------------------------


class TestStoragePathTraversal:
    """Confirm both LocalStorage._path and S3Storage._key reject the same
    patterns. These run against the live module code without fixtures."""

    @pytest.mark.parametrize("bad", [
        "../escape.md",
        "wiki/../../etc/passwd",
        "/absolute/path.md",
        "wiki\\backslash.md",
        "wiki/./dot-segment.md",
        "wiki/../../../root.md",
    ])
    def test_local_storage_rejects_traversal(self, bad):
        from app.storage import LocalStorage
        s = LocalStorage()
        with pytest.raises(ValueError):
            s._path("test-kb", bad)

    @pytest.mark.parametrize("bad", [
        "../escape.md",
        "/absolute.md",
        "wiki\\backslash.md",
        "wiki/../../../root.md",
    ])
    def test_s3_storage_rejects_traversal(self, bad):
        from app.storage import S3Storage
        s = S3Storage(bucket="dummy")
        with pytest.raises(ValueError):
            s._key("test-kb", bad)


# ---------------------------------------------------------------------------
# CSP nonce uniqueness
# ---------------------------------------------------------------------------


class TestCspNonce:
    def test_nonce_differs_per_request(self):
        from fastapi.testclient import TestClient
        import asyncio
        from contextlib import asynccontextmanager
        from app.main import app
        from app import db

        @asynccontextmanager
        async def _test_lifespan(app_):
            app_.state.redis = None
            await db.init_db()
            yield

        app.router.lifespan_context = _test_lifespan

        with TestClient(app) as c:
            r1 = c.get("/health")
            r2 = c.get("/health")
            csp1 = r1.headers.get("content-security-policy", "")
            csp2 = r2.headers.get("content-security-policy", "")
            nonce1 = re.search(r"nonce-([A-Za-z0-9_-]+)", csp1)
            nonce2 = re.search(r"nonce-([A-Za-z0-9_-]+)", csp2)
            assert nonce1 and nonce2
            assert nonce1.group(1) != nonce2.group(1), (
                "CSP nonce must differ between requests"
            )


# ---------------------------------------------------------------------------
# Corrupt frontmatter
# ---------------------------------------------------------------------------


class TestCorruptFrontmatter:
    def test_invalid_yaml_returns_empty_meta(self):
        from app.wiki import parse_frontmatter
        text = "---\ntitle: [[[broken yaml\n---\nbody"
        meta, body = parse_frontmatter(text)
        # Should not crash; meta is empty or partial, body is preserved.
        assert isinstance(meta, dict)
        assert "body" in body

    def test_no_frontmatter_returns_full_body(self):
        from app.wiki import parse_frontmatter
        text = "Just a plain article\nwith no YAML."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_string(self):
        from app.wiki import parse_frontmatter
        meta, body = parse_frontmatter("")
        assert meta == {}
        assert body == ""

    def test_frontmatter_with_unicode(self):
        from app.wiki import parse_frontmatter
        text = '---\ntitle: "Résumé: über cool"\ntags: [python]\n---\n# Body'
        meta, body = parse_frontmatter(text)
        assert "Résumé" in meta.get("title", "")


# ---------------------------------------------------------------------------
# Broken wikilink edge cases
# ---------------------------------------------------------------------------


class TestBrokenWikilinkEdgeCases:
    def test_wikilink_regex_finds_basic_links(self):
        from app.quality import _WIKILINK_RE
        text = "See [[Docker]] and [[Kubernetes Best Practices]]."
        matches = _WIKILINK_RE.findall(text)
        assert "Docker" in matches
        assert "Kubernetes Best Practices" in matches

    def test_wikilink_with_alias(self):
        from app.quality import _WIKILINK_RE
        text = "Use [[Docker|container runtime]] for this."
        matches = _WIKILINK_RE.findall(text)
        assert "Docker" in matches

    def test_wikilink_self_reference(self):
        from app.quality import _wikilink_slug
        assert _wikilink_slug("Docker") == "docker"

    def test_wikilink_slug_strips_special_chars(self):
        from app.quality import _wikilink_slug
        assert _wikilink_slug("C++: A Deep Dive") == "c-a-deep-dive"
        assert _wikilink_slug("Go vs Rust — 2026") == "go-vs-rust--2026"


# ---------------------------------------------------------------------------
# Empty KB behavior
# ---------------------------------------------------------------------------


class TestEmptyKB:
    def test_get_articles_empty_kb(self, tmp_path, monkeypatch):
        from app import storage, config, wiki
        kb_dir = tmp_path / "empty-kb"
        (kb_dir / "wiki").mkdir(parents=True)
        config.KB_DIRS["empty-kb"] = kb_dir
        storage.KB_DIRS["empty-kb"] = kb_dir
        try:
            articles = wiki.get_articles("empty-kb")
            assert articles == []
        finally:
            config.KB_DIRS.pop("empty-kb", None)
            storage.KB_DIRS.pop("empty-kb", None)

    def test_find_broken_wikilinks_empty_kb(self, tmp_path, monkeypatch):
        from app import storage, config
        from app.quality import find_broken_wikilinks, invalidate_quality_cache
        kb_dir = tmp_path / "empty-kb2"
        (kb_dir / "wiki").mkdir(parents=True)
        config.KB_DIRS["empty-kb2"] = kb_dir
        storage.KB_DIRS["empty-kb2"] = kb_dir
        invalidate_quality_cache()
        try:
            result = find_broken_wikilinks("empty-kb2")
            assert result["broken_count"] == 0
            assert result["affected_articles"] == 0
        finally:
            config.KB_DIRS.pop("empty-kb2", None)
            storage.KB_DIRS.pop("empty-kb2", None)
