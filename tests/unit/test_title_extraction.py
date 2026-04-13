"""Unit tests for title extraction and refinement."""

from app.wiki import extract_title


class TestExtractTitle:
    def test_basic_heading(self):
        content = "# My Article Title\n\nSome content here."
        assert extract_title(content) == "My Article Title"

    def test_strips_research_prefix(self):
        content = "# Research: Kubernetes Networking\n\nContent"
        assert extract_title(content) == "Kubernetes Networking"

    def test_strips_document_suffix(self):
        content = "# Kubernetes Networking: Research Document\n\nContent"
        assert extract_title(content) == "Kubernetes Networking"

    def test_strips_comprehensive_suffix(self):
        content = "# PostgreSQL Tuning: Comprehensive Research Document\n\nContent"
        assert extract_title(content) == "PostgreSQL Tuning"

    def test_skips_blank_lines(self):
        content = "\n\n# Actual Title\n\nContent"
        assert extract_title(content) == "Actual Title"

    def test_fallback_to_default(self):
        content = "No heading here, just a paragraph."
        assert extract_title(content, "fallback topic") == "fallback topic"

    def test_empty_content(self):
        assert extract_title("", "default") == "default"
        assert extract_title("", "") == ""

    def test_skips_short_titles(self):
        content = "# Hi\n\n# Real Title Here\n\nContent"
        # "Hi" is only 2 chars, should skip to "Real Title Here"
        title = extract_title(content)
        assert title == "Real Title Here"

    def test_real_world_example(self):
        content = "# Slow Roasted Pork for Dinner: Comprehensive Recipe and Meal Planning Guide\n\n## Executive Summary\n\nContent"
        title = extract_title(content)
        assert "Slow Roasted Pork" in title
        # "Comprehensive" in middle is fine — only exact trailing suffixes get stripped
        assert title == "Slow Roasted Pork for Dinner: Comprehensive Recipe and Meal Planning Guide"

    def test_local_research_prefix(self):
        content = "# Local Research: Auth Flow\n\nContent"
        assert extract_title(content) == "Auth Flow"

    def test_preserves_colons_in_middle(self):
        content = "# Rust vs Go: A Comparison\n\nContent"
        assert extract_title(content) == "Rust vs Go: A Comparison"
