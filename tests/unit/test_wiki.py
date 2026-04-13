"""Comprehensive tests for app.wiki module."""

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from app.wiki import (
    _safe_read,
    _serialize_frontmatter,
    _to_str,
    _to_str_list,
    _topic_to_slug,
    create_article,
    create_or_update_article,
    delete_article,
    find_related_article,
    get_article,
    get_articles,
    parse_frontmatter,
    update_article,
)


# ---------------------------------------------------------------------------
# 1. parse_frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = '---\ntitle: "Hello"\ntags: [a, b]\n---\n\nBody here'
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Hello"
        assert meta["tags"] == ["a", "b"]
        assert body == "Body here"

    def test_empty_frontmatter(self):
        # Empty YAML between --- markers parses as None, so returns ({}, original_text)
        text = "---\n---\n\nBody only"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        # parse_frontmatter returns original text when YAML is not a dict (None)
        assert "Body only" in body

    def test_no_frontmatter(self):
        text = "Just a plain markdown body"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_malformed_yaml(self):
        text = "---\n: [invalid yaml\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_non_dict_yaml(self):
        """YAML that parses to a list instead of dict should be rejected."""
        text = "---\n- item1\n- item2\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_frontmatter_with_only_one_delimiter(self):
        text = "---\ntitle: hello\nbody without closing delimiter"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_frontmatter_preserves_types(self):
        text = "---\ncount: 42\nenabled: true\nweight: 3.14\n---\n\nBody"
        meta, body = parse_frontmatter(text)
        assert meta["count"] == 42
        assert meta["enabled"] is True
        assert meta["weight"] == pytest.approx(3.14)

    def test_empty_string_input(self):
        meta, body = parse_frontmatter("")
        assert meta == {}
        assert body == ""


# ---------------------------------------------------------------------------
# 2. _to_str
# ---------------------------------------------------------------------------

class TestToStr:
    def test_none_returns_default(self):
        assert _to_str(None) == ""
        assert _to_str(None, "fallback") == "fallback"

    def test_dict_returns_default(self):
        assert _to_str({"key": "val"}) == ""

    def test_list_returns_default(self):
        assert _to_str([1, 2]) == ""

    def test_int_coerced(self):
        assert _to_str(42) == "42"

    def test_float_coerced(self):
        assert _to_str(3.14) == "3.14"

    def test_bool_coerced(self):
        assert _to_str(True) == "True"

    def test_str_passthrough(self):
        assert _to_str("hello") == "hello"

    def test_empty_string(self):
        assert _to_str("") == ""


# ---------------------------------------------------------------------------
# 3. _to_str_list
# ---------------------------------------------------------------------------

class TestToStrList:
    def test_none_returns_empty(self):
        assert _to_str_list(None) == []

    def test_string_csv(self):
        assert _to_str_list("a, b, c") == ["a", "b", "c"]

    def test_single_string(self):
        assert _to_str_list("only") == ["only"]

    def test_empty_string(self):
        assert _to_str_list("") == []

    def test_list_of_strings(self):
        assert _to_str_list(["x", "y"]) == ["x", "y"]

    def test_int_becomes_single_list(self):
        assert _to_str_list(42) == ["42"]

    def test_float_becomes_single_list(self):
        assert _to_str_list(3.14) == ["3.14"]

    def test_bool_becomes_single_list(self):
        assert _to_str_list(True) == ["True"]

    def test_nested_lists_skipped(self):
        assert _to_str_list(["a", ["nested"], "b"]) == ["a", "b"]

    def test_dicts_in_list_skipped(self):
        assert _to_str_list(["a", {"k": "v"}, "b"]) == ["a", "b"]

    def test_none_items_in_list_skipped(self):
        assert _to_str_list(["a", None, "b"]) == ["a", "b"]

    def test_ints_in_list(self):
        assert _to_str_list([1, 2, 3]) == ["1", "2", "3"]

    def test_whitespace_csv(self):
        assert _to_str_list("  a ,  , b  ") == ["a", "b"]


# ---------------------------------------------------------------------------
# 4. _safe_read
# ---------------------------------------------------------------------------

class TestSafeRead:
    def test_existing_file(self, mock_kb_dirs, tmp_kb):
        (tmp_kb / "wiki" / "hello.md").write_text("content", encoding="utf-8")
        assert _safe_read("test", "wiki/hello.md") == "content"

    def test_missing_file(self, mock_kb_dirs):
        assert _safe_read("test", "wiki/nope.md") is None

    def test_unknown_kb_returns_none(self, mock_kb_dirs):
        assert _safe_read("nonexistent", "wiki/anything.md") is None

    def test_binary_file_read_with_replace(self, mock_kb_dirs, tmp_kb):
        (tmp_kb / "wiki" / "binary.md").write_bytes(b"hello \xff world")
        result = _safe_read("test", "wiki/binary.md")
        assert result is not None
        assert "hello" in result


# ---------------------------------------------------------------------------
# 5. get_articles
# ---------------------------------------------------------------------------

class TestGetArticles:
    def test_returns_all_articles(self, mock_kb_dirs):
        articles = get_articles("test")
        assert len(articles) == 3
        slugs = {a["slug"] for a in articles}
        assert slugs == {"python-basics", "docker-containers", "rust-memory-safety"}

    def test_skips_underscore_files(self, mock_kb_dirs, tmp_kb):
        """Files starting with _ (like _index.md) should be excluded."""
        articles = get_articles("test")
        slugs = {a["slug"] for a in articles}
        assert "_index" not in slugs

    def test_missing_kb(self, mock_kb_dirs):
        assert get_articles("nonexistent") == []

    def test_empty_kb(self, mock_kb_dirs, tmp_kb):
        """A KB whose wiki dir has no .md files returns empty list."""
        wiki_dir = tmp_kb / "wiki"
        for f in wiki_dir.glob("*.md"):
            f.unlink()
        assert get_articles("test") == []

    def test_article_metadata_fields(self, mock_kb_dirs):
        articles = get_articles("test")
        art = next(a for a in articles if a["slug"] == "python-basics")
        assert art["title"] == "Python Basics"
        assert art["status"] == "draft"  # fixture has no status field → defaults to "draft"
        assert "python" in art["tags"]
        assert art["kb"] == "test"
        assert art["word_count"] > 0


# ---------------------------------------------------------------------------
# 6. get_article
# ---------------------------------------------------------------------------

class TestGetArticle:
    def test_existing_article(self, mock_kb_dirs):
        art = get_article("test", "python-basics")
        assert art is not None
        assert art["slug"] == "python-basics"
        assert art["title"] == "Python Basics"
        assert "html" in art
        assert "raw_markdown" in art

    def test_missing_article(self, mock_kb_dirs):
        assert get_article("test", "nonexistent") is None

    def test_missing_kb(self, mock_kb_dirs):
        assert get_article("nope", "anything") is None

    def test_html_rendering(self, mock_kb_dirs):
        art = get_article("test", "python-basics")
        # Fixture body has no headers, but should still have <p> tags from markdown
        assert "<p>" in art["html"]

    def test_wikilink_conversion(self, mock_kb_dirs, tmp_kb):
        """[[Topic Name]] should be converted to markdown links."""
        wiki_dir = tmp_kb / "wiki"
        (wiki_dir / "wikilink-test.md").write_text(
            '---\ntitle: "Wikilink Test"\n---\n\nSee [[Docker Containers]] for details.',
            encoding="utf-8",
        )
        art = get_article("test", "wikilink-test")
        assert art is not None
        # The raw_markdown should have the converted link
        assert "[Docker Containers](/wiki/test/docker-containers)" in art["raw_markdown"]
        # The HTML should contain an anchor tag
        assert "docker-containers" in art["html"]

    def test_code_block_rendering(self, mock_kb_dirs, tmp_kb):
        """Article with code fences should render <code> tags in HTML."""
        (tmp_kb / "wiki" / "code-test.md").write_text(
            '---\ntitle: "Code Test"\n---\n\n```python\nprint("hello")\n```\n',
            encoding="utf-8",
        )
        art = get_article("test", "code-test")
        assert art is not None
        assert "<code" in art["html"] or "highlight" in art["html"]


# ---------------------------------------------------------------------------
# 7. find_related_article
# ---------------------------------------------------------------------------

class TestFindRelatedArticle:
    def test_exact_slug_match(self, mock_kb_dirs):
        result = find_related_article("test", "python-basics")
        assert result is not None
        assert result["slug"] == "python-basics"

    def test_fuzzy_match(self, mock_kb_dirs):
        """'Python Basic' is close enough to 'Python Basics' (above 0.55 threshold)."""
        result = find_related_article("test", "Python Basic")
        assert result is not None
        assert result["slug"] == "python-basics"

    def test_below_threshold(self, mock_kb_dirs):
        result = find_related_article("test", "completely unrelated xyz")
        assert result is None

    def test_empty_kb(self, mock_kb_dirs, tmp_kb):
        for f in (tmp_kb / "wiki").glob("*.md"):
            f.unlink()
        assert find_related_article("test", "anything") is None

    def test_missing_kb(self, mock_kb_dirs):
        assert find_related_article("nonexistent", "anything") is None

    def test_exact_slug_with_spaces(self, mock_kb_dirs):
        """'docker containers' should match slug 'docker-containers' via exact slug match."""
        result = find_related_article("test", "docker containers")
        assert result is not None
        assert result["slug"] == "docker-containers"


# ---------------------------------------------------------------------------
# 8. create_article
# ---------------------------------------------------------------------------

class TestCreateArticle:
    def test_creates_files(self, mock_kb_dirs, tmp_kb):
        slug = create_article("test", "New Topic", "Some content", "web")
        wiki_file = tmp_kb / "wiki" / f"{slug}.md"
        raw_file = tmp_kb / "raw" / f"{slug}.md"
        assert wiki_file.exists()
        assert raw_file.exists()

    def test_generated_slug(self, mock_kb_dirs):
        slug = create_article("test", "My Cool Topic", "content")
        assert slug == "my-cool-topic"

    def test_frontmatter_correct(self, mock_kb_dirs, tmp_kb):
        slug = create_article("test", "Integration Testing", "The first line.", "local")
        wiki_file = tmp_kb / "wiki" / f"{slug}.md"
        text = wiki_file.read_text()
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Integration Testing"
        assert meta["source_type"] == "local"
        assert meta["status"] == "draft"
        assert meta["confidence"] == "medium"
        assert "integration" in meta["tags"]
        assert "testing" in meta["tags"]

    def test_unknown_kb_auto_initializes(self, mock_kb_dirs):
        # create_article delegates to storage.init_kb which creates the KB
        # on demand under the configured root.
        slug = create_article("nonexistent", "Topic", "content")
        assert slug

    def test_summary_truncation(self, mock_kb_dirs, tmp_kb):
        long_line = "A" * 300
        slug = create_article("test", "Long Summary", long_line)
        wiki_file = tmp_kb / "wiki" / f"{slug}.md"
        text = wiki_file.read_text()
        meta, _ = parse_frontmatter(text)
        assert len(meta["summary"]) <= 200

    def test_tags_stop_words_excluded(self, mock_kb_dirs):
        slug = create_article("test", "How to Use the Best Approach for Testing", "c")
        wiki_file = Path(mock_kb_dirs["test"]) / "wiki" / f"{slug}.md"
        text = wiki_file.read_text()
        meta, _ = parse_frontmatter(text)
        for tag in meta["tags"]:
            assert tag not in {"how", "the", "for", "best"}


# ---------------------------------------------------------------------------
# 9. update_article
# ---------------------------------------------------------------------------

class TestUpdateArticle:
    @pytest.mark.asyncio
    async def test_appends_recent_updates(self, mock_kb_dirs, tmp_kb):
        slug = await update_article("test", "python-basics", "New finding here")
        wiki_file = tmp_kb / "wiki" / f"{slug}.md"
        text = wiki_file.read_text()
        assert "## Recent Updates" in text
        assert "New finding here" in text

    @pytest.mark.asyncio
    async def test_bumps_updated_date(self, mock_kb_dirs, tmp_kb):
        await update_article("test", "python-basics", "something new")
        wiki_file = tmp_kb / "wiki" / "python-basics.md"
        text = wiki_file.read_text()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        meta, _ = parse_frontmatter(text)
        assert str(meta["updated"]) == today

    @pytest.mark.asyncio
    async def test_missing_article_raises(self, mock_kb_dirs):
        with pytest.raises(FileNotFoundError):
            await update_article("test", "nonexistent-slug", "content")

    @pytest.mark.asyncio
    async def test_unknown_kb_raises(self, mock_kb_dirs):
        with pytest.raises(FileNotFoundError):
            await update_article("nonexistent", "slug", "content")

    @pytest.mark.asyncio
    async def test_second_update_appends(self, mock_kb_dirs, tmp_kb):
        await update_article("test", "docker-containers", "First update")
        await update_article("test", "docker-containers", "Second update")
        text = (tmp_kb / "wiki" / "docker-containers.md").read_text()
        assert "First update" in text
        assert "Second update" in text


# ---------------------------------------------------------------------------
# 10. create_or_update_article
# ---------------------------------------------------------------------------

class TestCreateOrUpdateArticle:
    @pytest.mark.asyncio
    async def test_creates_new_article(self, mock_kb_dirs):
        slug, change = await create_or_update_article("test", "Brand New Topic", "content")
        assert change == "created"
        assert slug == "brand-new-topic"

    @pytest.mark.asyncio
    async def test_updates_existing_article(self, mock_kb_dirs):
        slug, change = await create_or_update_article(
            "test", "Python Basics", "additional info"
        )
        assert change == "updated"
        assert slug == "python-basics"

    @pytest.mark.asyncio
    async def test_fuzzy_update(self, mock_kb_dirs):
        """A fuzzy-close topic should trigger update, not create."""
        slug, change = await create_or_update_article(
            "test", "Docker Containers Guide", "more info"
        )
        assert change == "updated"
        assert slug == "docker-containers"


# ---------------------------------------------------------------------------
# 11. delete_article
# ---------------------------------------------------------------------------

class TestDeleteArticle:
    def test_deletes_wiki_file(self, mock_kb_dirs, tmp_kb):
        wiki_file = tmp_kb / "wiki" / "python-basics.md"
        assert wiki_file.exists()
        result = delete_article("test", "python-basics")
        assert not wiki_file.exists()
        assert result["slug"] == "python-basics"
        assert result["kb"] == "test"
        assert len(result["files_removed"]) >= 1

    def test_deletes_raw_file_too(self, mock_kb_dirs, tmp_kb):
        raw_file = tmp_kb / "raw" / "docker-containers.md"
        raw_file.write_text("raw content")
        delete_article("test", "docker-containers")
        assert not raw_file.exists()

    def test_missing_article_raises(self, mock_kb_dirs):
        with pytest.raises(FileNotFoundError):
            delete_article("test", "does-not-exist")

    def test_unknown_kb_raises(self, mock_kb_dirs):
        with pytest.raises(FileNotFoundError):
            delete_article("nonexistent", "slug")

    def test_returns_title(self, mock_kb_dirs):
        result = delete_article("test", "python-basics")
        assert result["title"] == "Python Basics"


# ---------------------------------------------------------------------------
# 12. _topic_to_slug
# ---------------------------------------------------------------------------

class TestTopicToSlug:
    def test_normal_topic(self):
        assert _topic_to_slug("Hello World", "2026-01-01") == "hello-world"

    def test_special_chars_replaced(self):
        slug = _topic_to_slug("What's C++ & C#?", "2026-01-01")
        # Special chars become underscores, spaces become dashes
        assert " " not in slug
        assert slug == slug.lower()
        assert all(c.isalnum() or c in "-_" for c in slug)

    def test_empty_topic_uses_fallback(self):
        assert _topic_to_slug("", "2026-04-08") == "research-2026-04-08"

    def test_only_special_chars(self):
        # "!@#$%" → each char becomes "_", result is non-empty so no fallback
        result = _topic_to_slug("!@#$%", "2026-04-08")
        assert all(c == "_" for c in result)

    def test_truncation_at_50(self):
        long_topic = "a" * 100
        slug = _topic_to_slug(long_topic, "2026-01-01")
        assert len(slug) <= 80

    def test_leading_trailing_hyphens_stripped(self):
        slug = _topic_to_slug(" -hello- ", "2026-01-01")
        assert not slug.startswith("-")
        assert not slug.endswith("-")


# ---------------------------------------------------------------------------
# 13. _serialize_frontmatter
# ---------------------------------------------------------------------------

class TestSerializeFrontmatter:
    def test_round_trip_basic(self):
        meta = {"title": "Test", "status": "draft", "tags": ["a", "b"]}
        text = _serialize_frontmatter(meta)
        assert text.startswith("---")
        assert text.endswith("---")
        # Parse back
        parsed = yaml.safe_load(text.strip("---").strip("-"))
        assert parsed["title"] == "Test"
        assert parsed["tags"] == ["a", "b"]

    def test_list_inline_short(self):
        meta = {"tags": ["x", "y", "z"]}
        text = _serialize_frontmatter(meta)
        assert "tags: [x, y, z]" in text

    def test_list_block_long(self):
        meta = {"items": list(range(10))}
        text = _serialize_frontmatter(meta)
        assert "  - " in text

    def test_string_with_newline(self):
        meta = {"note": "line1\nline2"}
        text = _serialize_frontmatter(meta)
        assert '"line1\nline2"' in text

    def test_integer_value(self):
        meta = {"count": 42}
        text = _serialize_frontmatter(meta)
        assert "count: 42" in text

    def test_none_value(self):
        meta = {"empty": None}
        text = _serialize_frontmatter(meta)
        assert "empty: None" in text

    def test_preserves_key_order(self):
        meta = {"alpha": 1, "beta": 2, "gamma": 3}
        text = _serialize_frontmatter(meta)
        alpha_idx = text.index("alpha")
        beta_idx = text.index("beta")
        gamma_idx = text.index("gamma")
        assert alpha_idx < beta_idx < gamma_idx
