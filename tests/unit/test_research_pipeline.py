"""Unit tests for the research pipeline pure helpers.

The heavy-lifting async functions (_search_serper, _chat, _synthesize,
_critique_draft) hit real LLMs/HTTP and are covered by the respx tests
in test_research_llm_helpers.py. This file sticks to the deterministic
helpers — they're the highest-value targets because they run on every
research job and are easy to break without noticing.

Covered here:
    _score_source_tier   — URL → tier integer
    _tier_label          — tier → human label
    _normalize_url_for_match — www/protocol/slash stripping
    _lint_citations      — hallucinated link scrubbing
    _truncate_for_tier   — per-tier content capping
    _deduplicate_results — URL + near-dup content dedup
    _format_sources_for_prompt — prompt builder
"""

from __future__ import annotations

import pytest

from app.research import (
    _deduplicate_results,
    _format_sources_for_prompt,
    _lint_citations,
    _normalize_url_for_match,
    _score_source_tier,
    _tier_label,
    _truncate_for_tier,
)


# ---------------------------------------------------------------------------
# _score_source_tier
# ---------------------------------------------------------------------------


class TestScoreSourceTier:
    @pytest.mark.parametrize(
        "url,expected_tier",
        [
            # Tier 1 — .gov/.edu/.mil TLDs
            ("https://www.nist.gov/cybersecurity", 1),
            ("https://mit.edu/papers", 1),
            # Tier 1 — substring whitelist (official docs + engineering blogs)
            ("https://docs.python.org/3/library/asyncio.html", 1),
            ("https://kubernetes.io/docs/concepts/", 1),
            ("https://github.com/user/repo", 1),
            ("https://en.wikipedia.org/wiki/Kubernetes", 1),
            ("https://arxiv.org/abs/2301.00001", 1),
            # Tier 2 — community and news
            ("https://stackoverflow.com/questions/12345", 2),
            ("https://medium.com/@author/article", 2),
            ("https://www.theverge.com/post", 2),
            # Tier 3 — generic
            ("https://some-random-blog.com/post", 3),
            ("", 3),
        ],
    )
    def test_tier_mapping(self, url, expected_tier):
        assert _score_source_tier(url) == expected_tier

    def test_none_and_empty_fall_back_to_tier_3(self):
        assert _score_source_tier("") == 3
        # A malformed URL shouldn't crash the scorer
        assert _score_source_tier("not-a-url") == 3


# ---------------------------------------------------------------------------
# _tier_label
# ---------------------------------------------------------------------------


class TestTierLabel:
    def test_known_tiers(self):
        assert "Tier 1" in _tier_label(1)
        assert "Tier 2" in _tier_label(2)
        assert "Tier 3" in _tier_label(3)
        assert "Authoritative" in _tier_label(1)

    def test_unknown_tier_raises(self):
        with pytest.raises(KeyError):
            _tier_label(99)


# ---------------------------------------------------------------------------
# _normalize_url_for_match
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://example.com/path", "example.com/path"),
            ("http://example.com/path", "example.com/path"),
            ("https://www.example.com/path", "example.com/path"),
            ("https://example.com/path/", "example.com/path"),
            ("HTTPS://EXAMPLE.COM/Path", "example.com/path"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalization(self, raw, expected):
        assert _normalize_url_for_match(raw) == expected

    def test_two_equivalent_urls_match(self):
        assert _normalize_url_for_match("https://www.nist.gov/about/") == \
               _normalize_url_for_match("http://nist.gov/about")


# ---------------------------------------------------------------------------
# _lint_citations
# ---------------------------------------------------------------------------


class TestLintCitations:
    def test_keeps_real_citations(self):
        markdown = "See [the docs](https://docs.python.org/3/) for details."
        cleaned, stripped = _lint_citations(
            markdown, ["https://docs.python.org/3/"],
        )
        assert cleaned == markdown
        assert stripped == []

    def test_strips_hallucinated_link_preserves_label(self):
        markdown = "As noted in [fake source](https://made-up-source.example/doc)."
        cleaned, stripped = _lint_citations(markdown, [])
        # Link gone, label preserved
        assert "made-up-source.example" not in cleaned
        assert "fake source" in cleaned
        assert stripped == ["https://made-up-source.example/doc"]

    def test_mixed_real_and_fake(self):
        markdown = (
            "See [docs](https://docs.python.org/3/) and [fake](https://nope.example)."
        )
        cleaned, stripped = _lint_citations(
            markdown, ["https://docs.python.org/3/"],
        )
        assert "[docs](https://docs.python.org/3/)" in cleaned
        assert "https://nope.example" not in cleaned
        assert stripped == ["https://nope.example"]

    def test_fuzzy_url_match_with_www_variant(self):
        markdown = "See [ref](https://www.nist.gov/about/)."
        cleaned, stripped = _lint_citations(
            markdown, ["https://nist.gov/about"],
        )
        assert stripped == []
        assert "[ref](https://www.nist.gov/about/)" in cleaned

    def test_empty_markdown(self):
        cleaned, stripped = _lint_citations("", ["https://x.com"])
        assert cleaned == ""
        assert stripped == []

    def test_multiple_hallucinations(self):
        markdown = (
            "[A](https://a.fake) and [B](https://b.fake) and [C](https://c.fake)."
        )
        cleaned, stripped = _lint_citations(markdown, [])
        assert len(stripped) == 3
        assert "fake" not in cleaned  # all three links stripped


# ---------------------------------------------------------------------------
# _truncate_for_tier
# ---------------------------------------------------------------------------


class TestTruncateForTier:
    def test_tier_1_gets_8k_cap(self):
        content = "x" * 9000
        result = _truncate_for_tier(content, 1)
        # 8000 + " ..." suffix
        assert len(result) <= 8010
        assert result.endswith(" ...")

    def test_tier_2_gets_3k_cap(self):
        content = "y" * 4000
        result = _truncate_for_tier(content, 2)
        assert len(result) <= 3010

    def test_tier_3_gets_1k_cap(self):
        content = "z" * 2000
        result = _truncate_for_tier(content, 3)
        assert len(result) <= 1010

    def test_unknown_tier_gets_1k_cap(self):
        content = "w" * 2000
        result = _truncate_for_tier(content, 99)
        assert len(result) <= 1010

    def test_short_content_unchanged(self):
        assert _truncate_for_tier("hello", 1) == "hello"

    def test_empty_content(self):
        assert _truncate_for_tier("", 1) == ""
        assert _truncate_for_tier(None, 1) == ""


# ---------------------------------------------------------------------------
# _deduplicate_results
# ---------------------------------------------------------------------------


class TestDeduplicateResults:
    def test_empty_returns_empty(self):
        assert _deduplicate_results([]) == []

    def test_unique_urls_all_kept(self):
        results = [
            {"url": "https://a.com", "content": "a" * 50},
            {"url": "https://b.com", "content": "b" * 50},
            {"url": "https://c.com", "content": "c" * 50},
        ]
        assert len(_deduplicate_results(results)) == 3

    def test_duplicate_urls_dedup(self):
        results = [
            {"url": "https://example.com/path", "content": "abc def ghi jkl mno"},
            {"url": "https://example.com/path/", "content": "zzz zzz zzz zzz zzz"},
        ]
        # Trailing-slash variants count as duplicates
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1

    def test_near_duplicate_content_dedup(self):
        body = "The quick brown fox jumps over the lazy dog"
        results = [
            {"url": "https://a.com", "content": body},
            {"url": "https://b.com", "content": body},  # same content, different URL
        ]
        deduped = _deduplicate_results(results)
        # Content near-dup pass should remove one
        assert len(deduped) == 1

    def test_short_content_not_deduplicated(self):
        # Content under 20 chars is too short to reliably dedup
        results = [
            {"url": "https://a.com", "content": "short"},
            {"url": "https://b.com", "content": "brief"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# _format_sources_for_prompt
# ---------------------------------------------------------------------------


class TestFormatSourcesForPrompt:
    def test_empty_list(self):
        result = _format_sources_for_prompt([])
        assert isinstance(result, str)

    def test_single_source_renders_title_and_url(self):
        results = [
            {
                "url": "https://example.com/doc",
                "title": "Example Doc",
                "content": "This is the body",
                "tier": 2,
            },
        ]
        out = _format_sources_for_prompt(results)
        assert "Example Doc" in out
        assert "https://example.com/doc" in out
        assert "This is the body" in out

    def test_respects_tier_content_cap(self):
        # A 10KB Tier 3 source should appear truncated in the output
        long_body = "x" * 10000
        results = [
            {
                "url": "https://example.com",
                "title": "Big Doc",
                "content": long_body,
                "tier": 3,
            },
        ]
        out = _format_sources_for_prompt(results)
        # Tier 3 cap is 1000, so 'x' count in output should be far less
        # than the original 10k.
        assert out.count("x") < 2000
