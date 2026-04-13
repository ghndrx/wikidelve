"""Storage backend parity — every `Storage` Protocol method is
exercised across LocalStorage and S3Storage (moto + real-AWS tiers).

One test body per method, parameterized by backend via the
`parity_storage` fixture in `tests/parity/conftest.py`. A failure on
one backend but not the other points at a bug in the specific
implementation — usually the interesting kind of bug.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Write → read round-trips
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_write_then_read_roundtrips(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/hello.md", "# Hello")
        assert parity_storage.read_text("test-kb", "wiki/hello.md") == "# Hello"

    def test_read_missing_returns_none(self, parity_storage):
        assert parity_storage.read_text("test-kb", "wiki/nope.md") is None

    def test_overwrite_replaces_content(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/overwrite.md", "v1")
        parity_storage.write_text("test-kb", "wiki/overwrite.md", "v2")
        assert parity_storage.read_text("test-kb", "wiki/overwrite.md") == "v2"

    def test_unicode_content_roundtrips(self, parity_storage):
        body = "# 你好\n\nこれはテスト — емодзі 🚀\n"
        parity_storage.write_text("test-kb", "wiki/unicode.md", body)
        assert parity_storage.read_text("test-kb", "wiki/unicode.md") == body

    def test_empty_string_roundtrips(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/empty.md", "")
        assert parity_storage.read_text("test-kb", "wiki/empty.md") == ""

    def test_large_body_roundtrips(self, parity_storage):
        body = "x" * 100_000  # 100 KB
        parity_storage.write_text("test-kb", "wiki/big.md", body)
        assert parity_storage.read_text("test-kb", "wiki/big.md") == body


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    def test_exists_true_after_write(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/a.md", "a")
        assert parity_storage.exists("test-kb", "wiki/a.md") is True

    def test_exists_false_for_missing(self, parity_storage):
        assert parity_storage.exists("test-kb", "wiki/ghost.md") is False

    def test_exists_false_after_delete(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/b.md", "b")
        parity_storage.delete("test-kb", "wiki/b.md")
        assert parity_storage.exists("test-kb", "wiki/b.md") is False


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing_returns_true(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/del.md", "doomed")
        assert parity_storage.delete("test-kb", "wiki/del.md") is True

    def test_delete_missing_returns_false(self, parity_storage):
        # Local returns False (file.exists() check), S3 is a tri-state
        # but our S3Storage wrapper also returns False here.
        assert parity_storage.delete("test-kb", "wiki/never.md") is False

    def test_double_delete_returns_false(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/twice.md", "x")
        parity_storage.delete("test-kb", "wiki/twice.md")
        assert parity_storage.delete("test-kb", "wiki/twice.md") is False


# ---------------------------------------------------------------------------
# list_slugs
# ---------------------------------------------------------------------------


class TestListSlugs:
    def test_empty_kb_returns_empty(self, parity_storage):
        assert parity_storage.list_slugs("test-kb") == []

    def test_lists_written_slugs_sorted(self, parity_storage):
        for slug in ("zebra", "apple", "mango"):
            parity_storage.write_text("test-kb", f"wiki/{slug}.md", slug)
        assert parity_storage.list_slugs("test-kb") == ["apple", "mango", "zebra"]

    def test_excludes_underscore_prefixed(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/real.md", "x")
        parity_storage.write_text("test-kb", "wiki/_index.md", "x")
        slugs = parity_storage.list_slugs("test-kb")
        assert "real" in slugs
        assert "_index" not in slugs

    def test_subdir_parameter_honored(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/in-wiki.md", "w")
        parity_storage.write_text("test-kb", "raw/in-raw.md", "r")
        assert parity_storage.list_slugs("test-kb", subdir="wiki") == ["in-wiki"]
        assert parity_storage.list_slugs("test-kb", subdir="raw") == ["in-raw"]


# ---------------------------------------------------------------------------
# iter_articles — single most likely place for backend divergence
# ---------------------------------------------------------------------------


class TestIterArticles:
    def test_empty_kb_yields_nothing(self, parity_storage):
        assert list(parity_storage.iter_articles("test-kb")) == []

    def test_iter_returns_slug_body_tuples(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/one.md", "# One")
        parity_storage.write_text("test-kb", "wiki/two.md", "# Two")
        results = dict(parity_storage.iter_articles("test-kb"))
        assert results == {"one": "# One", "two": "# Two"}

    def test_iter_excludes_underscore_prefixed(self, parity_storage):
        parity_storage.write_text("test-kb", "wiki/public.md", "p")
        parity_storage.write_text("test-kb", "wiki/_hidden.md", "h")
        slugs = [slug for slug, _ in parity_storage.iter_articles("test-kb")]
        assert "public" in slugs
        assert "_hidden" not in slugs

    def test_iter_many_articles_matches_list_slugs(self, parity_storage):
        """Parallel S3 fan-out must produce the same set as list_slugs."""
        for i in range(50):
            parity_storage.write_text("test-kb", f"wiki/art-{i:02d}.md", f"body {i}")
        iter_slugs = sorted(s for s, _ in parity_storage.iter_articles("test-kb"))
        list_slugs = parity_storage.list_slugs("test-kb")
        assert iter_slugs == list_slugs


# ---------------------------------------------------------------------------
# Path traversal guards
# ---------------------------------------------------------------------------


class TestPathTraversal:
    @pytest.mark.parametrize(
        "bad_path",
        [
            "../escape.md",
            "wiki/../../etc/passwd",
            "/absolute/path.md",
            "wiki\\windows.md",
            "wiki/./sneaky.md",
        ],
    )
    def test_traversal_rejected_on_read(self, parity_storage, bad_path):
        with pytest.raises((ValueError, Exception)):
            parity_storage.read_text("test-kb", bad_path)

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../escape.md",
            "/absolute.md",
            "wiki/../../../root.md",
        ],
    )
    def test_traversal_rejected_on_write(self, parity_storage, bad_path):
        with pytest.raises((ValueError, Exception)):
            parity_storage.write_text("test-kb", bad_path, "payload")
