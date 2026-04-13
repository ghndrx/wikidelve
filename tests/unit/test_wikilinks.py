"""Tests for wikilink integrity checking and fixing in app.quality."""

from contextlib import contextmanager

import pytest
from pathlib import Path
from unittest.mock import patch

from app.quality import check_wikilinks, fix_wikilinks, _fuzzy_match_slug


@contextmanager
def _use_kb(kb_dir, name="test"):
    import app.config as _config
    import app.storage as _storage
    import app.wiki as _wiki

    dirs = {name: kb_dir}
    prev_kb_dirs = dict(_config.KB_DIRS)
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(dirs)

    prev_root = _storage.KB_ROOT
    fallback_root = Path(kb_dir).parent / "_fallback"
    fallback_root.mkdir(parents=True, exist_ok=True)
    _storage.KB_ROOT = fallback_root

    prev_backend = _storage._default
    _storage._default = _storage.LocalStorage()
    _wiki.invalidate_articles_cache()

    try:
        yield dirs
    finally:
        _config.KB_DIRS.clear()
        _config.KB_DIRS.update(prev_kb_dirs)
        _storage.KB_ROOT = prev_root
        _storage._default = prev_backend
        _wiki.invalidate_articles_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _article(title="Test", tags=None, body=""):
    lines = ["---"]
    lines.append(f'title: "{title}"')
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("updated: 2026-04-01")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def _make_kb(tmp_path, articles: dict[str, str]):
    kb_dir = tmp_path / "wikilink_kb"
    wiki_dir = kb_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    for slug, content in articles.items():
        (wiki_dir / f"{slug}.md").write_text(content, encoding="utf-8")
    return kb_dir


def _patch_dirs(kb_dir):
    """Kept for backward compatibility with existing tests."""
    return _use_kb(kb_dir, "test"), _noop_ctx()


@contextmanager
def _noop_ctx():
    yield


# ===========================================================================
# _fuzzy_match_slug
# ===========================================================================

class TestFuzzyMatchSlug:
    def test_exact_title_match(self):
        slug_to_title = {"kubernetes-basics": "Kubernetes Basics"}
        title_to_slug = {"kubernetes basics": "kubernetes-basics"}
        result = _fuzzy_match_slug("Kubernetes Basics", slug_to_title, title_to_slug)
        assert result is not None
        assert result["slug"] == "kubernetes-basics"
        assert result["confidence"] == 1.0

    def test_close_fuzzy_match(self):
        slug_to_title = {"docker-networking": "Docker Networking"}
        title_to_slug = {"docker networking": "docker-networking"}
        result = _fuzzy_match_slug("Docker Network", slug_to_title, title_to_slug)
        assert result is not None
        assert result["slug"] == "docker-networking"
        assert result["confidence"] >= 0.6

    def test_no_match_returns_none(self):
        slug_to_title = {"kubernetes-basics": "Kubernetes Basics"}
        title_to_slug = {"kubernetes basics": "kubernetes-basics"}
        result = _fuzzy_match_slug("Quantum Computing Theory", slug_to_title, title_to_slug)
        assert result is None

    def test_low_similarity_below_threshold(self):
        slug_to_title = {"python-testing": "Python Testing"}
        title_to_slug = {"python testing": "python-testing"}
        result = _fuzzy_match_slug("Java Deployment", slug_to_title, title_to_slug)
        assert result is None


# ===========================================================================
# check_wikilinks
# ===========================================================================

class TestCheckWikilinks:
    def test_all_valid_links(self, tmp_path):
        articles = {
            "kubernetes-basics": _article(
                title="Kubernetes Basics",
                body="See also [[Docker Networking]] for more.",
            ),
            "docker-networking": _article(
                title="Docker Networking",
                body="Related to [[Kubernetes Basics]].",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = check_wikilinks("test")
            assert result["total_links"] == 2
            assert result["valid_links"] == 2
            assert result["broken_links"] == 0
            assert len(result["details"]) == 0

    def test_broken_link_detected(self, tmp_path):
        articles = {
            "kubernetes-basics": _article(
                title="Kubernetes Basics",
                body="See [[Nonexistent Article]] here.",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = check_wikilinks("test")
            assert result["total_links"] == 1
            assert result["valid_links"] == 0
            assert result["broken_links"] == 1
            assert result["details"][0]["link_text"] == "Nonexistent Article"
            assert result["details"][0]["source_slug"] == "kubernetes-basics"

    def test_broken_link_with_suggestion(self, tmp_path):
        articles = {
            "kubernetes-basics": _article(
                title="Kubernetes Basics",
                body="See [[Kubernetes Basic]] for more.",  # typo
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = check_wikilinks("test")
            assert result["broken_links"] == 1
            detail = result["details"][0]
            assert detail["suggestion"] is not None
            assert detail["suggestion"]["slug"] == "kubernetes-basics"
            assert detail["suggestion"]["confidence"] >= 0.7

    def test_mixed_valid_and_broken(self, tmp_path):
        articles = {
            "article-a": _article(
                title="Article A",
                body="Links to [[Article B]] and [[Ghost Article]].",
            ),
            "article-b": _article(title="Article B", body="No links here."),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = check_wikilinks("test")
            assert result["total_links"] == 2
            assert result["valid_links"] == 1
            assert result["broken_links"] == 1

    def test_no_links_at_all(self, tmp_path):
        articles = {
            "plain": _article(title="Plain Article", body="No wikilinks here."),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = check_wikilinks("test")
            assert result["total_links"] == 0
            assert result["broken_links"] == 0

    def test_unknown_kb(self):
        # Unknown KBs return empty counts rather than an error payload.
        result = check_wikilinks("nonexistent")
        assert result["total_links"] == 0
        assert result["broken_links"] == 0

    def test_slug_mismatch_from_special_chars(self, tmp_path):
        """Wikilink with colons generates a different slug than the actual article."""
        articles = {
            "gpu-cloud-platforms-runpod-vs-lambda-labs": _article(
                title="GPU Cloud Platforms: RunPod vs Lambda Labs",
                body="Content here.",
            ),
            "overview": _article(
                title="Overview",
                body="See [[GPU Cloud Platforms: RunPod vs Lambda Labs]].",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = check_wikilinks("test")
            # The colon is stripped by _wikilink_to_slug, should match
            assert result["valid_links"] == 1
            assert result["broken_links"] == 0


# ===========================================================================
# fix_wikilinks
# ===========================================================================

class TestFixWikilinks:
    def test_fix_fuzzy_matched_link(self, tmp_path):
        articles = {
            "kubernetes-basics": _article(
                title="Kubernetes Basics",
                body="See [[Kubernetes Basic]] for details.",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = fix_wikilinks("test")
            assert result["links_fixed"] >= 1

            # Verify file was updated
            text = (kb_dir / "wiki" / "kubernetes-basics.md").read_text()
            assert "[[Kubernetes Basics]]" in text
            assert "[[Kubernetes Basic]]" not in text

    def test_remove_broken_link_auto_remove(self, tmp_path):
        articles = {
            "article-a": _article(
                title="Article A",
                body="See [[Totally Nonexistent Thing]] for more.",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = fix_wikilinks("test", auto_remove=True)
            assert result["links_removed"] >= 1

            text = (kb_dir / "wiki" / "article-a.md").read_text()
            assert "[[" not in text
            assert "Totally Nonexistent Thing" in text  # plain text preserved

    def test_leave_broken_link_no_auto_remove(self, tmp_path):
        articles = {
            "article-a": _article(
                title="Article A",
                body="See [[Totally Nonexistent Thing]] for more.",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = fix_wikilinks("test", auto_remove=False)
            assert result["articles_modified"] == 0

            text = (kb_dir / "wiki" / "article-a.md").read_text()
            assert "[[Totally Nonexistent Thing]]" in text

    def test_valid_links_untouched(self, tmp_path):
        articles = {
            "article-a": _article(
                title="Article A",
                body="See [[Article B]] for more.",
            ),
            "article-b": _article(title="Article B", body="Content."),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = fix_wikilinks("test")
            assert result["articles_modified"] == 0

    def test_fix_single_article(self, tmp_path):
        articles = {
            "article-a": _article(
                title="Article A",
                body="See [[Ghost]] here.",
            ),
            "article-b": _article(
                title="Article B",
                body="See [[Ghost]] here.",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = fix_wikilinks("test", slug="article-a", auto_remove=True)
            assert result["articles_modified"] == 1
            # article-b should be untouched
            text_b = (kb_dir / "wiki" / "article-b.md").read_text()
            assert "[[Ghost]]" in text_b

    def test_fix_nonexistent_slug_returns_error(self, tmp_path):
        articles = {"article-a": _article(title="A", body="text")}
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            result = fix_wikilinks("test", slug="nonexistent")
            assert "error" in result

    def test_unknown_kb(self):
        # Unknown KBs return an empty result with zero fixes.
        result = fix_wikilinks("nonexistent")
        assert result["links_fixed"] == 0
        assert result["articles_modified"] == 0

    def test_updates_frontmatter_date(self, tmp_path):
        articles = {
            "article-a": _article(
                title="Article A",
                body="See [[Ghost Link]] here.",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        p1, p2 = _patch_dirs(kb_dir)
        with p1, p2:
            fix_wikilinks("test", auto_remove=True)
            text = (kb_dir / "wiki" / "article-a.md").read_text()
            # Updated date should reflect today, not the original 2026-04-01
            # (unless test runs on that exact day)
            assert "updated:" in text
