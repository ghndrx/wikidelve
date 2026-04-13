"""Tests for app.quality — article scoring, duplicates, shallow detection."""

from contextlib import contextmanager

import pytest
from pathlib import Path
from unittest.mock import patch

from app.quality import (
    score_article_quality,
    score_all_articles,
    find_duplicates,
    find_shallow_articles,
)


@contextmanager
def _use_kb(kb_dir, name="t"):
    """Route the storage backend at a tmp KB dir for the duration of the block."""
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
# Helpers to build markdown articles with frontmatter
# ---------------------------------------------------------------------------

def _article(
    title="Test Article",
    summary="",
    tags=None,
    source_files=None,
    updated="2026-04-01",
    body="",
):
    """Build a markdown string with YAML frontmatter + body."""
    lines = ["---"]
    lines.append(f'title: "{title}"')
    if summary:
        lines.append(f'summary: "{summary}"')
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    if source_files:
        lines.append("source_files:")
        for sf in source_files:
            lines.append(f"  - {sf}")
    if updated:
        lines.append(f"updated: {updated}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


def _words(n):
    """Generate a body with exactly *n* words."""
    return " ".join(["word"] * n)


def _make_kb(tmp_path, articles: dict[str, str]):
    """Create a KB directory with the given slug->content mapping.

    Returns (kb_dir, kb_name, patched_dirs).
    """
    kb_dir = tmp_path / "score_kb"
    wiki_dir = kb_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    for slug, content in articles.items():
        (wiki_dir / f"{slug}.md").write_text(content, encoding="utf-8")
    return kb_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kb(tmp_path):
    """Minimal KB with one well-structured article."""
    body = (
        "## Executive Summary\n\n"
        + _words(500)
        + "\n\n## Key Concepts\n\n"
        + "Some text with [[Wikilink One]] and [[Wikilink Two]] and "
        "[ext](http://example.com).\n\n"
        "```python\nprint('hello')\n```\n\n"
        "```bash\necho hi\n```\n\n"
        "| Col A | Col B |\n|---|---|\n| 1 | 2 |\n\n"
        "- item one\n- item two\n- item three\n"
    )
    content = _article(
        title="Good Article",
        summary="A solid article",
        tags=["python", "testing"],
        source_files=["raw/good.md"],
        updated="2026-04-01",
        body=body,
    )
    kb_dir = _make_kb(tmp_path, {"good-article": content})
    dirs = {"test": kb_dir}
    with _use_kb(kb_dir, next(iter(dirs))):
        yield dirs


@pytest.fixture
def empty_kb(tmp_path):
    """KB directory exists but contains no articles."""
    kb_dir = tmp_path / "empty_kb"
    (kb_dir / "wiki").mkdir(parents=True)
    dirs = {"empty": kb_dir}
    with _use_kb(kb_dir, next(iter(dirs))):
        yield dirs


# ===========================================================================
# 1. score_article_quality
# ===========================================================================

class TestScoreArticleQuality:
    """Tests for score_article_quality."""

    def test_well_structured_article_scores_high(self, kb):
        result = score_article_quality("test", "good-article")
        assert result["score"] >= 60
        assert "breakdown" in result

    def test_empty_body_low_score(self, tmp_path):
        """Empty body should have 0 for words, structure, links, code."""
        kb_dir = _make_kb(tmp_path, {"empty": _article(body="")})
        with _use_kb(kb_dir, "t"):
            result = score_article_quality("t", "empty")
            assert result["breakdown"]["words"] == 0
            assert result["breakdown"]["structure"] == 0
            assert result["breakdown"]["links"] == 0
            assert result["breakdown"]["code"] == 0

    def test_missing_kb_returns_zero(self):
        result = score_article_quality("nonexistent", "slug")
        assert result["score"] == 0

    def test_missing_slug_returns_zero(self, kb):
        result = score_article_quality("test", "no-such-article")
        assert result["score"] == 0

    # --- word count tiers ---

    def test_word_tier_below_200(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"short": _article(body=_words(50))})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "short")["breakdown"]
            assert bd["words"] == 50 // 25  # max(0, 50//25) = 2

    def test_word_tier_200(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"med": _article(body=_words(200))})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "med")["breakdown"]
            assert bd["words"] == 8

    def test_word_tier_400(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"med2": _article(body=_words(400))})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "med2")["breakdown"]
            assert bd["words"] == 15

    def test_word_tier_800(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"long": _article(body=_words(800))})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "long")["breakdown"]
            assert bd["words"] == 20

    def test_word_tier_1500(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"xl": _article(body=_words(1500))})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "xl")["breakdown"]
            assert bd["words"] == 25

    # --- structure (H2 count) ---

    def test_h2_count_scoring(self, tmp_path):
        body = "## One\ntext\n\n## Two\ntext\n\n## Three\ntext"
        kb_dir = _make_kb(tmp_path, {"h2": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "h2")["breakdown"]
            assert bd["structure"] == 9  # 3 * 3

    def test_h2_capped_at_15(self, tmp_path):
        body = "\n\n".join([f"## Section {i}\ntext" for i in range(10)])
        kb_dir = _make_kb(tmp_path, {"many": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "many")["breakdown"]
            assert bd["structure"] == 15

    # --- links ---

    def test_wikilinks_score_double(self, tmp_path):
        body = "See [[Article A]] and [[Article B]]."
        kb_dir = _make_kb(tmp_path, {"lnk": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "lnk")["breakdown"]
            assert bd["links"] == 4  # 2 wikilinks * 2

    def test_external_links(self, tmp_path):
        body = "[a](http://a.com) and [b](http://b.com)"
        kb_dir = _make_kb(tmp_path, {"ext": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "ext")["breakdown"]
            assert bd["links"] == 2

    def test_links_capped_at_15(self, tmp_path):
        body = " ".join([f"[[Link{i}]]" for i in range(20)])
        kb_dir = _make_kb(tmp_path, {"cap": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "cap")["breakdown"]
            assert bd["links"] == 15

    # --- code blocks ---

    def test_code_blocks_scoring(self, tmp_path):
        body = "```python\nprint(1)\n```\n\n```bash\necho hi\n```"
        kb_dir = _make_kb(tmp_path, {"code": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "code")["breakdown"]
            assert bd["code"] == 6  # 2 blocks * 3

    def test_code_capped_at_10(self, tmp_path):
        blocks = "\n\n".join(["```\ncode\n```"] * 6)
        kb_dir = _make_kb(tmp_path, {"mcode": _article(body=blocks)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "mcode")["breakdown"]
            assert bd["code"] == 10

    # --- frontmatter completeness ---

    def test_full_frontmatter(self, kb):
        bd = score_article_quality("test", "good-article")["breakdown"]
        assert bd["frontmatter"] == 15  # title+summary+tags+source_files+updated

    def test_minimal_frontmatter(self, tmp_path):
        content = "---\ntitle: Only Title\n---\n\nSome body text."
        kb_dir = _make_kb(tmp_path, {"min": content})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "min")["breakdown"]
            assert bd["frontmatter"] == 3  # title only

    # --- freshness ---

    def test_freshness_2026(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"f26": _article(updated="2026-01-01", body="word")})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "f26")["breakdown"]
            assert bd["freshness"] == 10

    def test_freshness_2025(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"f25": _article(updated="2025-06-15", body="word")})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "f25")["breakdown"]
            assert bd["freshness"] == 5

    def test_freshness_old(self, tmp_path):
        kb_dir = _make_kb(tmp_path, {"old": _article(updated="2023-01-01", body="word")})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "old")["breakdown"]
            assert bd["freshness"] == 0

    # --- tables & lists ---

    def test_tables_scoring(self, tmp_path):
        body = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        kb_dir = _make_kb(tmp_path, {"tbl": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "tbl")["breakdown"]
            assert bd["tables"] >= 1

    def test_lists_scoring(self, tmp_path):
        body = "- one\n- two\n- three\n- four"
        kb_dir = _make_kb(tmp_path, {"lst": _article(body=body)})
        with _use_kb(kb_dir, "t"):
            bd = score_article_quality("t", "lst")["breakdown"]
            assert bd["lists"] == 4

    def test_score_capped_at_100(self, tmp_path):
        """Even with maxed-out everything the total never exceeds 100."""
        body = (
            "\n\n".join([f"## S{i}\n" + _words(200) for i in range(10)])
            + "\n\n"
            + " ".join([f"[[L{i}]]" for i in range(20)])
            + "\n\n"
            + "\n\n".join(["```\ncode\n```"] * 8)
            + "\n\n"
            + "| A | B |\n" * 10
            + "\n"
            + "- item\n" * 10
        )
        content = _article(
            title="Max", summary="max", tags=["a"], source_files=["r"],
            updated="2026-01-01", body=body,
        )
        kb_dir = _make_kb(tmp_path, {"max": content})
        with _use_kb(kb_dir, "t"):
            result = score_article_quality("t", "max")
            assert result["score"] <= 100


# ===========================================================================
# 2. score_all_articles
# ===========================================================================

class TestScoreAllArticles:
    """Tests for score_all_articles."""

    def test_sorted_ascending(self, tmp_path):
        """Returned list is sorted by score ascending (worst first)."""
        articles = {
            "poor": _article(body=_words(10)),
            "decent": _article(body="## Intro\n\n" + _words(400) + "\n\n## More\ntext"),
            "good": _article(
                title="Good", summary="s", tags=["t"], source_files=["r"],
                updated="2026-01-01",
                body="## A\n\n" + _words(900) + "\n\n## B\ntext",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            scores = score_all_articles("t")
            values = [s["score"] for s in scores]
            assert values == sorted(values)

    def test_handles_empty_kb(self, empty_kb):
        result = score_all_articles("empty")
        assert result == []

    def test_returns_all_articles(self, tmp_path):
        articles = {f"art{i}": _article(body=_words(50)) for i in range(5)}
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            assert len(score_all_articles("t")) == 5


# ===========================================================================
# 3. find_duplicates
# ===========================================================================

class TestFindDuplicates:
    """Tests for find_duplicates."""

    def test_similar_titles_detected(self, tmp_path):
        articles = {
            "kubernetes-basics": _article(title="Kubernetes Basics", tags=["k8s"]),
            "kubernetes-fundamentals": _article(title="Kubernetes Fundamentals", tags=["k8s"]),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            dupes = find_duplicates("t", threshold=0.4)
            assert len(dupes) >= 1
            assert dupes[0]["article_a"] in ("kubernetes-basics", "kubernetes-fundamentals")

    def test_dissimilar_not_matched(self, tmp_path):
        articles = {
            "python-basics": _article(title="Python Basics", tags=["python"]),
            "docker-networking": _article(title="Docker Networking", tags=["docker"]),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            dupes = find_duplicates("t", threshold=0.5)
            assert len(dupes) == 0

    def test_tag_overlap_contributes(self, tmp_path):
        articles = {
            "go-concurrency": _article(
                title="Go Concurrency", tags=["go", "concurrency", "goroutines"]
            ),
            "golang-parallelism": _article(
                title="Golang Parallelism", tags=["go", "concurrency", "goroutines"]
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            dupes = find_duplicates("t", threshold=0.3)
            assert len(dupes) >= 1
            assert dupes[0]["tag_overlap"] > 0

    def test_high_threshold_filters_more(self, tmp_path):
        articles = {
            "react-hooks": _article(title="React Hooks", tags=["react"]),
            "react-hooks-guide": _article(title="React Hooks Guide", tags=["react"]),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            low = find_duplicates("t", threshold=0.3)
            high = find_duplicates("t", threshold=0.9)
            assert len(low) >= len(high)

    def test_empty_kb_no_duplicates(self, empty_kb):
        assert find_duplicates("empty") == []

    def test_duplicate_score_fields(self, tmp_path):
        articles = {
            "api-auth": _article(title="API Authentication", tags=["auth", "api"]),
            "api-authentication": _article(title="API Authentication Guide", tags=["auth", "api"]),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            dupes = find_duplicates("t", threshold=0.3)
            assert len(dupes) >= 1
            d = dupes[0]
            assert "score" in d
            assert "title_similarity" in d
            assert "tag_overlap" in d
            assert 0 <= d["score"] <= 1


# ===========================================================================
# 4. find_shallow_articles
# ===========================================================================

class TestFindShallowArticles:
    """Tests for find_shallow_articles."""

    def test_short_articles_flagged(self, tmp_path):
        articles = {
            "tiny": _article(title="Tiny", body="Just a few words here."),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            shallow = find_shallow_articles("t", min_words=300)
            slugs = [s["slug"] for s in shallow]
            assert "tiny" in slugs
            issues = next(s for s in shallow if s["slug"] == "tiny")["issues"]
            assert any("words" in i for i in issues)

    def test_missing_structure_flagged(self, tmp_path):
        # >200 words but no H2 headers → "only 0 sections"
        articles = {
            "flat": _article(title="Flat", body=_words(250)),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            shallow = find_shallow_articles("t", min_words=100)
            flat_entry = next((s for s in shallow if s["slug"] == "flat"), None)
            assert flat_entry is not None
            assert any("sections" in i for i in flat_entry["issues"])

    def test_few_links_flagged(self, tmp_path):
        articles = {
            "nolinks": _article(title="No Links", body="## Intro\n\n" + _words(400) + "\n\n## More\ntext"),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            shallow = find_shallow_articles("t", min_words=100)
            entry = next((s for s in shallow if s["slug"] == "nolinks"), None)
            assert entry is not None
            assert any("links" in i for i in entry["issues"])

    def test_technical_article_without_code_flagged(self, tmp_path):
        articles = {
            "nocode": _article(
                title="Docker Setup",
                tags=["docker"],
                body="## Intro\n\n" + _words(400) + "\n\n## Setup\n[ref](http://x.com) and [ref2](http://y.com)\n",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            shallow = find_shallow_articles("t", min_words=100)
            entry = next((s for s in shallow if s["slug"] == "nocode"), None)
            assert entry is not None
            assert any("code" in i for i in entry["issues"])

    def test_well_structured_not_flagged(self, tmp_path):
        body = (
            "## Intro\n\n" + _words(500) + "\n\n"
            "## Details\n[a](http://a.com) [b](http://b.com)\n\n"
            "```python\nprint(1)\n```\n"
        )
        articles = {"good": _article(title="Good", body=body)}
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            shallow = find_shallow_articles("t", min_words=100)
            slugs = [s["slug"] for s in shallow]
            assert "good" not in slugs

    def test_empty_kb_returns_empty(self, empty_kb):
        assert find_shallow_articles("empty") == []

    def test_sorted_by_issue_count_descending(self, tmp_path):
        articles = {
            "bad": _article(title="Bad", tags=["python"], body="tiny"),
            "ok": _article(
                title="Ok",
                body="## Intro\n\n" + _words(500) + "\n\n## More\n[l](http://x.com) [l2](http://y.com)\n",
            ),
        }
        kb_dir = _make_kb(tmp_path, articles)
        with _use_kb(kb_dir, "t"):
            shallow = find_shallow_articles("t", min_words=300)
            if len(shallow) >= 2:
                assert len(shallow[0]["issues"]) >= len(shallow[1]["issues"])
