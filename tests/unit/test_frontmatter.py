"""Tests for frontmatter parsing, serialization, and edge cases."""

import pytest
from app.wiki import parse_frontmatter, _parse_yaml_lenient, _serialize_frontmatter


# ---------------------------------------------------------------------------
# _parse_yaml_lenient
# ---------------------------------------------------------------------------

class TestParseYamlLenient:
    """Tests for lenient YAML parsing of broken frontmatter."""

    def test_unquoted_colon_in_title(self):
        raw = "title: Foo: Bar\nsummary: A summary"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["title"] == "Foo: Bar"

    def test_unquoted_colon_in_summary(self):
        raw = "title: Hello\nsummary: Key takeaway: it works"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["summary"] == "Key takeaway: it works"

    def test_multiple_colons_in_value(self):
        raw = "title: GPU Cloud: RunPod vs Lambda: A Comparison"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert "RunPod vs Lambda" in result["title"]

    def test_already_quoted_values_untouched(self):
        raw = 'title: "Foo: Bar"\nsummary: "Key: Value"'
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["title"] == "Foo: Bar"

    def test_list_values_untouched(self):
        raw = "title: Hello\ntags: [python, docker]"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["tags"] == ["python", "docker"]

    def test_simple_valid_yaml(self):
        raw = "title: Hello World\nstatus: draft"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["title"] == "Hello World"
        assert result["status"] == "draft"

    def test_completely_broken_yaml_returns_none(self):
        raw = "::::\n[[[[invalid"
        result = _parse_yaml_lenient(raw)
        assert result is None

    def test_empty_string(self):
        result = _parse_yaml_lenient("")
        # empty YAML parses to None
        assert result is None

    def test_single_colon_no_value(self):
        raw = "title:"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["title"] is None

    def test_value_starting_with_single_quote_not_wrapped(self):
        raw = "title: 'Already quoted: value'"
        result = _parse_yaml_lenient(raw)
        assert result is not None
        assert result["title"] == "Already quoted: value"


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    """Tests for the main frontmatter parser."""

    def test_simple_frontmatter(self):
        text = "---\ntitle: Hello\n---\nBody text here."
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Hello"
        assert body == "Body text here."

    def test_no_frontmatter(self):
        text = "Just some text without frontmatter."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self):
        text = "---\n---\nBody only."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text  # safe_load of empty returns None, so falls through

    def test_unquoted_colon_in_title_field(self):
        """The production bug: unquoted colons caused parse failures."""
        text = "---\ntitle: Kubernetes: Deep Dive\nsummary: A guide\n---\nContent."
        meta, body = parse_frontmatter(text)
        assert meta.get("title") is not None
        assert "Kubernetes" in str(meta["title"])
        assert body == "Content."

    def test_unquoted_colon_in_summary_field(self):
        text = "---\ntitle: Test\nsummary: Key point: always test\n---\nContent."
        meta, body = parse_frontmatter(text)
        assert "Key point" in str(meta.get("summary", ""))

    def test_multiple_frontmatter_blocks_merged(self):
        """The production bug: article updates created duplicate frontmatter."""
        text = (
            "---\ntitle: Original Title\nstatus: draft\n---\n"
            "---\ntitle: Updated Title\nconfidence: high\n---\n"
            "Final body content."
        )
        meta, body = parse_frontmatter(text)
        # First block takes precedence for existing keys
        assert meta["title"] == "Original Title"
        assert meta["status"] == "draft"
        # Second block fills in missing keys
        assert meta["confidence"] == "high"
        assert body == "Final body content."

    def test_three_frontmatter_blocks(self):
        text = (
            "---\ntitle: First\n---\n"
            "---\nstatus: review\n---\n"
            "---\nconfidence: low\n---\n"
            "Body."
        )
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "First"
        assert meta["status"] == "review"
        assert meta["confidence"] == "low"
        assert body == "Body."

    def test_frontmatter_with_em_dash(self):
        text = "---\ntitle: Python \u2014 A Modern Language\n---\nContent."
        meta, body = parse_frontmatter(text)
        assert "\u2014" in meta["title"]

    def test_frontmatter_with_brackets_in_value(self):
        text = '---\ntitle: "Arrays [and] Lists"\n---\nContent.'
        meta, body = parse_frontmatter(text)
        assert "[and]" in meta["title"]

    def test_frontmatter_with_curly_quotes(self):
        text = "---\ntitle: The \u201cBest\u201d Practices\n---\nContent."
        meta, body = parse_frontmatter(text)
        assert "\u201cBest\u201d" in meta["title"]

    def test_frontmatter_preserves_tags_list(self):
        text = "---\ntitle: Test\ntags: [a, b, c]\n---\nContent."
        meta, body = parse_frontmatter(text)
        assert meta["tags"] == ["a", "b", "c"]

    def test_incomplete_frontmatter_no_closing(self):
        text = "---\ntitle: Hello\nNo closing delimiter"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_frontmatter_with_date_values(self):
        text = "---\ntitle: Test\ncreated: 2024-01-15\nupdated: 2024-06-01\n---\nBody."
        meta, body = parse_frontmatter(text)
        # YAML may parse dates as date objects
        assert meta["title"] == "Test"
        assert body == "Body."

    def test_empty_body_after_frontmatter(self):
        text = "---\ntitle: Empty Body\n---\n"
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Empty Body"
        assert body == ""

    def test_body_with_triple_dashes_not_at_start(self):
        """Triple dashes in body content should not confuse parser."""
        text = "---\ntitle: Test\n---\nSome text\n---\nMore text after hr"
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Test"
        # The body parsing may attempt to merge, but the inner block
        # "Some text" is not valid YAML, so it should stop
        # The exact behavior depends on whether "Some text" parses as YAML


# ---------------------------------------------------------------------------
# _serialize_frontmatter
# ---------------------------------------------------------------------------

class TestSerializeFrontmatter:
    """Tests for frontmatter serialization."""

    def test_simple_dict(self):
        meta = {"title": "Hello", "status": "draft"}
        result = _serialize_frontmatter(meta)
        assert result.startswith("---")
        assert result.endswith("---")
        assert "title: Hello" in result
        assert "status: draft" in result

    def test_value_with_colon_is_quoted(self):
        """The production fix: values containing colons must be quoted."""
        meta = {"title": "Foo: Bar", "summary": "Key: Value"}
        result = _serialize_frontmatter(meta)
        assert 'title: "Foo: Bar"' in result
        assert 'summary: "Key: Value"' in result

    def test_value_with_double_quotes_escaped(self):
        meta = {"title": 'He said "hello"'}
        result = _serialize_frontmatter(meta)
        assert '\\"hello\\"' in result

    def test_list_short_inline(self):
        meta = {"tags": ["python", "rust"]}
        result = _serialize_frontmatter(meta)
        assert "tags: [python, rust]" in result

    def test_list_long_multiline(self):
        meta = {"items": ["a", "b", "c", "d", "e", "f"]}
        result = _serialize_frontmatter(meta)
        assert "  - a" in result
        assert "  - f" in result

    def test_none_value(self):
        meta = {"title": "Test", "summary": None}
        result = _serialize_frontmatter(meta)
        assert "summary: None" in result

    def test_integer_value(self):
        meta = {"title": "Test", "version": 3}
        result = _serialize_frontmatter(meta)
        assert "version: 3" in result

    def test_empty_dict(self):
        result = _serialize_frontmatter({})
        assert result == "---\n---"

    def test_multiline_string_quoted(self):
        meta = {"description": "Line one\nLine two"}
        result = _serialize_frontmatter(meta)
        assert 'description: "Line one' in result

    def test_empty_list(self):
        meta = {"tags": []}
        result = _serialize_frontmatter(meta)
        assert "tags: []" in result


# ---------------------------------------------------------------------------
# Round-trip: serialize -> parse
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Verify that serialize -> parse produces equivalent data."""

    def test_round_trip_simple(self):
        original = {"title": "Hello World", "status": "draft", "tags": ["a", "b"]}
        serialized = _serialize_frontmatter(original)
        text = serialized + "\n\nBody content."
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Hello World"
        assert meta["status"] == "draft"
        assert meta["tags"] == ["a", "b"]
        assert body == "Body content."

    def test_round_trip_colon_in_value(self):
        original = {"title": "Foo: Bar", "summary": "Key: takeaway here"}
        serialized = _serialize_frontmatter(original)
        text = serialized + "\n\nBody."
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Foo: Bar"
        assert meta["summary"] == "Key: takeaway here"

    def test_round_trip_empty_tags(self):
        original = {"title": "Test", "tags": []}
        serialized = _serialize_frontmatter(original)
        text = serialized + "\n\nBody."
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "Test"
        assert meta["tags"] == []
