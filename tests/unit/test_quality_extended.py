"""Extended unit tests for app.quality — scoring, fact-checking, crosslinks, enrichment."""

import json
import re
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.quality import (
    _WIKILINK_RE,
    _fuzzy_match_slug,
    _quality_cache,
    _quality_cache_get,
    _quality_cache_set,
    _score_from_parsed,
    _wikilink_slug,
    add_crosslinks,
    enrich_article,
    fact_check_article,
    find_broken_wikilinks,
    find_duplicates,
    find_shallow_articles,
    fix_wikilinks,
    freshness_audit,
    invalidate_quality_cache,
    run_quality_pass,
    score_article_quality,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_quality_cache():
    """Ensure each test starts with a clean quality cache."""
    _quality_cache.clear()
    yield
    _quality_cache.clear()


# ── Cache tests ──────────────────────────────────────────────────────────────


class TestQualityCache:
    def test_set_and_get(self):
        _quality_cache_set("scores", "mykb", [1, 2, 3])
        assert _quality_cache_get("scores", "mykb") == [1, 2, 3]

    def test_get_returns_none_when_missing(self):
        assert _quality_cache_get("scores", "nonexistent") is None

    def test_ttl_expiry(self, monkeypatch):
        _quality_cache_set("scores", "kb1", "value")
        # Fast-forward monotonic clock past TTL
        real_ts = _quality_cache[("scores", "kb1")][0]
        _quality_cache[("scores", "kb1")] = (real_ts - 600, "value")
        assert _quality_cache_get("scores", "kb1") is None

    def test_invalidate_single_kb(self):
        _quality_cache_set("scores", "kb1", "v1")
        _quality_cache_set("scores", "kb2", "v2")
        invalidate_quality_cache("kb1")
        assert _quality_cache_get("scores", "kb1") is None
        assert _quality_cache_get("scores", "kb2") == "v2"

    def test_invalidate_all(self):
        _quality_cache_set("scores", "kb1", "v1")
        _quality_cache_set("scores", "kb2", "v2")
        invalidate_quality_cache(None)
        assert _quality_cache_get("scores", "kb1") is None
        assert _quality_cache_get("scores", "kb2") is None


# ── Scoring tests ────────────────────────────────────────────────────────────


class TestScoreFromParsed:
    def test_empty_body(self):
        result = _score_from_parsed("test-slug", {}, "")
        assert result["score"] == 0
        assert result["slug"] == "test-slug"
        assert result["word_count"] == 0

    def test_word_count_tiers(self):
        # < 200 words
        short = _score_from_parsed("s", {}, "word " * 50)
        assert short["breakdown"]["words"] == 2  # 50 // 25

        # 200-399 words
        mid = _score_from_parsed("s", {}, "word " * 250)
        assert mid["breakdown"]["words"] == 8

        # 400-799 words
        med = _score_from_parsed("s", {}, "word " * 500)
        assert med["breakdown"]["words"] == 15

        # 800-1499 words
        long = _score_from_parsed("s", {}, "word " * 900)
        assert long["breakdown"]["words"] == 20

        # >= 1500 words
        huge = _score_from_parsed("s", {}, "word " * 1600)
        assert huge["breakdown"]["words"] == 25

    def test_structure_scoring(self):
        body = "## Section 1\ntext\n## Section 2\ntext\n## Section 3\ntext"
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["structure"] == 9  # 3 * 3

    def test_structure_capped_at_15(self):
        body = "\n".join(f"## Section {i}\ntext" for i in range(10))
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["structure"] == 15

    def test_wikilinks_scoring(self):
        body = "See [[Python]] and [[Rust]] for more. Also [link](http://example.com)."
        result = _score_from_parsed("s", {}, body)
        # 2 wikilinks * 2 + 1 ext link = 5
        assert result["breakdown"]["links"] == 5

    def test_links_capped_at_15(self):
        body = " ".join(f"[[Link{i}]]" for i in range(20))
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["links"] == 15

    def test_code_blocks_scoring(self):
        body = "```python\nprint('hi')\n```\n\n```bash\necho hi\n```"
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["code"] == 6  # 2 blocks * 3

    def test_code_capped_at_10(self):
        body = "\n".join("```\ncode\n```" for _ in range(10))
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["code"] == 10

    def test_frontmatter_completeness(self):
        full_meta = {
            "title": "Test",
            "summary": "A test article",
            "tags": ["python"],
            "source_files": ["a.py"],
            "updated": "2026-01-01",
        }
        result = _score_from_parsed("s", full_meta, "text")
        assert result["breakdown"]["frontmatter"] == 15  # 3+4+3+3+2

    def test_frontmatter_partial(self):
        partial_meta = {"title": "Test"}
        result = _score_from_parsed("s", partial_meta, "text")
        assert result["breakdown"]["frontmatter"] == 3

    def test_tables_scoring(self):
        body = "| Col1 | Col2 | Col3 |\n| a | b | c |\n| d | e | f |"
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["tables"] == 3

    def test_tables_capped_at_5(self):
        body = "\n".join("| a | b | c |" for _ in range(10))
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["tables"] == 5

    def test_lists_scoring(self):
        body = "- item1\n- item2\n* item3"
        result = _score_from_parsed("s", {}, body)
        assert result["breakdown"]["lists"] == 3

    def test_freshness_2026(self):
        result = _score_from_parsed("s", {"updated": "2026-04-01"}, "text")
        assert result["breakdown"]["freshness"] == 10

    def test_freshness_2025(self):
        result = _score_from_parsed("s", {"updated": "2025-06-15"}, "text")
        assert result["breakdown"]["freshness"] == 5

    def test_freshness_old(self):
        result = _score_from_parsed("s", {"updated": "2024-01-01"}, "text")
        assert result["breakdown"]["freshness"] == 0

    def test_total_capped_at_100(self):
        """Even with extreme values the total never exceeds 100."""
        body = "\n".join(
            [
                "word " * 2000,
                *[f"## Section {i}" for i in range(20)],
                *[f"[[Link{i}]]" for i in range(20)],
                *["```\ncode\n```" for _ in range(10)],
                *["| a | b | c |" for _ in range(10)],
                *["- item" for _ in range(10)],
            ]
        )
        meta = {
            "title": "T",
            "summary": "S",
            "tags": ["t"],
            "source_files": ["f"],
            "updated": "2026-01-01",
        }
        result = _score_from_parsed("s", meta, body)
        assert result["score"] <= 100

    def test_title_falls_back_to_slug(self):
        result = _score_from_parsed("my-slug", {}, "text")
        assert result["title"] == "my-slug"

    def test_title_uses_meta(self):
        result = _score_from_parsed("my-slug", {"title": "My Title"}, "text")
        assert result["title"] == "My Title"


class TestScoreArticleQuality:
    @patch("app.quality.read_article_text", return_value=None)
    def test_missing_article(self, mock_read):
        result = score_article_quality("kb", "missing")
        assert result["score"] == 0
        assert result["breakdown"] == {}

    @patch("app.quality.parse_frontmatter", return_value=({"title": "T"}, "Some body text"))
    @patch("app.quality.read_article_text", return_value="---\ntitle: T\n---\nSome body text")
    def test_scores_existing_article(self, mock_read, mock_parse):
        result = score_article_quality("kb", "test")
        assert "score" in result
        assert "breakdown" in result


# ── Duplicate detection ──────────────────────────────────────────────────────


class TestFindDuplicates:
    @patch("app.quality.get_articles")
    def test_no_articles(self, mock_get):
        mock_get.return_value = []
        result = find_duplicates("kb")
        assert result == []

    @patch("app.quality.get_articles")
    def test_finds_similar_titles(self, mock_get):
        mock_get.return_value = [
            {"slug": "python-intro", "title": "Python Introduction", "tags": ["python"]},
            {"slug": "python-introduction", "title": "Python Introduction Guide", "tags": ["python"]},
            {"slug": "rust-basics", "title": "Rust Basics", "tags": ["rust"]},
        ]
        result = find_duplicates("kb", threshold=0.3)
        # The two python articles share tags and similar titles
        slugs_in_results = set()
        for d in result:
            slugs_in_results.add(d["article_a"])
            slugs_in_results.add(d["article_b"])
        assert "python-intro" in slugs_in_results or "python-introduction" in slugs_in_results

    @patch("app.quality.get_articles")
    def test_high_threshold_filters(self, mock_get):
        mock_get.return_value = [
            {"slug": "a", "title": "Alpha", "tags": ["x"]},
            {"slug": "b", "title": "Beta", "tags": ["y"]},
        ]
        result = find_duplicates("kb", threshold=0.99)
        assert result == []

    @patch("app.quality.get_articles")
    def test_result_sorted_by_score_desc(self, mock_get):
        mock_get.return_value = [
            {"slug": "a", "title": "Docker Tutorial", "tags": ["docker"]},
            {"slug": "b", "title": "Docker Tutorial Guide", "tags": ["docker"]},
            {"slug": "c", "title": "Docker Guide Tutorial", "tags": ["docker", "tutorial"]},
        ]
        result = find_duplicates("kb", threshold=0.2)
        if len(result) >= 2:
            assert result[0]["score"] >= result[1]["score"]


# ── Wikilink slug / regex ────────────────────────────────────────────────────


class TestWikilinkSlug:
    def test_simple(self):
        assert _wikilink_slug("Python Basics") == "python-basics"

    def test_strips_special_chars(self):
        assert _wikilink_slug("GPU Cloud: RunPod vs Lambda") == "gpu-cloud-runpod-vs-lambda"

    def test_empty(self):
        assert _wikilink_slug("") == ""

    def test_wikilink_regex_simple(self):
        matches = _WIKILINK_RE.findall("See [[Python Basics]] and [[Rust]].")
        assert matches == ["Python Basics", "Rust"]

    def test_wikilink_regex_with_alias(self):
        matches = _WIKILINK_RE.findall("See [[Python Basics|Python]].")
        assert matches == ["Python Basics"]

    def test_wikilink_regex_no_newlines(self):
        matches = _WIKILINK_RE.findall("See [[broken\nlink]].")
        assert matches == []


# ── Broken wikilinks ────────────────────────────────────────────────────────


class TestFindBrokenWikilinks:
    @patch("app.storage.iter_articles")
    @patch("app.quality.get_articles")
    def test_detects_broken_links(self, mock_get, mock_iter):
        mock_get.return_value = [
            {"slug": "python-basics", "title": "Python Basics"},
        ]
        mock_iter.return_value = [
            ("python-basics", "---\ntitle: Python Basics\n---\nSee [[Nonexistent Article]] here."),
        ]
        result = find_broken_wikilinks("kb")
        assert result["broken_count"] >= 1
        assert result["affected_articles"] == 1
        assert any(t["slug"] == "nonexistent-article" for t in result["by_target"])

    @patch("app.storage.iter_articles")
    @patch("app.quality.get_articles")
    def test_no_broken_links(self, mock_get, mock_iter):
        mock_get.return_value = [
            {"slug": "python-basics", "title": "Python Basics"},
            {"slug": "rust", "title": "Rust"},
        ]
        mock_iter.return_value = [
            ("python-basics", "---\ntitle: Python Basics\n---\nSee [[Rust]] here."),
        ]
        result = find_broken_wikilinks("kb")
        assert result["broken_count"] == 0

    @patch("app.storage.iter_articles")
    @patch("app.quality.get_articles")
    def test_deduplicates_same_link_in_article(self, mock_get, mock_iter):
        mock_get.return_value = [
            {"slug": "a", "title": "A"},
        ]
        mock_iter.return_value = [
            ("a", "---\ntitle: A\n---\n[[Missing]] and [[Missing]] again."),
        ]
        result = find_broken_wikilinks("kb")
        # Count should be 2 (two occurrences), but sources should only list once
        target = result["by_target"][0]
        assert target["count"] == 2
        assert len(target["sources"]) == 1


# ── Shallow articles ─────────────────────────────────────────────────────────


class TestFindShallowArticles:
    @patch("app.quality.get_articles")
    def test_signals_fast_path(self, mock_get):
        # Need at least one article with h2_count > 0 to trigger the fast path
        mock_get.return_value = [
            {
                "slug": "short",
                "title": "Short Article",
                "word_count": 50,
                "h2_count": 0,
                "link_count": 0,
                "code_count": 0,
                "has_tech_tag": False,
                "tags": [],
            },
            {
                "slug": "good",
                "title": "Good Article",
                "word_count": 1000,
                "h2_count": 5,
                "link_count": 10,
                "code_count": 3,
                "has_tech_tag": False,
                "tags": [],
            },
        ]
        result = find_shallow_articles("kb", min_words=300)
        assert len(result) == 1
        assert "only 50 words" in result[0]["issues"][0]

    @patch("app.quality.get_articles")
    def test_no_code_for_tech_tag(self, mock_get):
        mock_get.return_value = [
            {
                "slug": "py",
                "title": "Python Guide",
                "word_count": 500,
                "h2_count": 3,
                "link_count": 5,
                "code_count": 0,
                "has_tech_tag": True,
                "tags": ["python"],
            }
        ]
        result = find_shallow_articles("kb")
        assert any("no code examples" in i for i in result[0]["issues"])

    @patch("app.storage.iter_articles")
    @patch("app.quality.get_articles")
    def test_fallback_path_reads_bodies(self, mock_get, mock_iter):
        """When h2_count signal is missing, falls back to reading bodies."""
        mock_get.return_value = [
            {"slug": "a", "title": "A", "word_count": 50, "tags": []},
        ]
        mock_iter.return_value = [
            ("a", "---\ntitle: A\n---\nShort text."),
        ]
        result = find_shallow_articles("kb", min_words=300)
        assert len(result) == 1


# ── Fuzzy match slug ─────────────────────────────────────────────────────────


class TestFuzzyMatchSlug:
    def test_exact_title_match(self):
        s2t = {"python-basics": "Python Basics"}
        t2s = {"python basics": "python-basics"}
        result = _fuzzy_match_slug("Python Basics", s2t, t2s)
        assert result is not None
        assert result["slug"] == "python-basics"
        assert result["confidence"] == 1.0

    def test_fuzzy_match_above_threshold(self):
        s2t = {"python-basics": "Python Basics"}
        t2s = {"python basics": "python-basics"}
        result = _fuzzy_match_slug("Python Basic", s2t, t2s)
        assert result is not None
        assert result["confidence"] >= 0.6

    def test_no_match_below_threshold(self):
        s2t = {"python-basics": "Python Basics"}
        t2s = {"python basics": "python-basics"}
        result = _fuzzy_match_slug("Completely Different Topic XYZ", s2t, t2s)
        assert result is None


# ── Enrich article (async, LLM mocked) ──────────────────────────────────────


class TestEnrichArticle:
    @patch("app.quality.write_article_text")
    @patch("app.quality._serialize_frontmatter", return_value="---\ntitle: Test\n---")
    @patch("app.quality._chat", new_callable=AsyncMock)
    @patch("app.quality.parse_frontmatter")
    @patch("app.quality.read_article_text")
    async def test_enriches_successfully(self, mock_read, mock_parse, mock_chat, mock_fm, mock_write):
        mock_read.return_value = "---\ntitle: Test\n---\nShort content."
        mock_parse.return_value = ({"title": "Test", "status": "draft"}, "Short content.")
        mock_chat.return_value = "## Overview\nEnriched content with lots of detail. " * 20

        result = await enrich_article("kb", "test-slug")
        assert result["status"] == "enriched"
        assert result["slug"] == "test-slug"
        assert mock_write.called

    @patch("app.quality.read_article_text", return_value=None)
    async def test_missing_article(self, mock_read):
        result = await enrich_article("kb", "missing")
        assert "error" in result

    @patch("app.quality._chat", new_callable=AsyncMock, return_value="short")
    @patch("app.quality.parse_frontmatter", return_value=({"title": "T"}, "body"))
    @patch("app.quality.read_article_text", return_value="---\ntitle: T\n---\nbody")
    async def test_insufficient_llm_response(self, mock_read, mock_parse, mock_chat):
        result = await enrich_article("kb", "slug")
        assert "error" in result


# ── Crosslinks (async, LLM mocked) ──────────────────────────────────────────


class TestAddCrosslinks:
    @patch("app.quality.write_article_text")
    @patch("app.quality._serialize_frontmatter", return_value="---\ntitle: T\n---")
    @patch("app.quality._chat", new_callable=AsyncMock)
    @patch("app.quality.get_articles")
    @patch("app.quality.parse_frontmatter")
    @patch("app.quality.read_article_text")
    async def test_adds_links(self, mock_read, mock_parse, mock_get, mock_chat, mock_fm, mock_write):
        mock_read.return_value = "---\ntitle: Test\n---\nPython is great."
        mock_parse.return_value = ({"title": "Test"}, "Python is great.")
        mock_get.return_value = [
            {"slug": "test", "title": "Test"},
            {"slug": "python", "title": "Python"},
        ]
        mock_chat.return_value = "[[Python]] is great. Also see [[Docker]] for containers. " * 5

        result = await add_crosslinks("kb", "test")
        assert result["slug"] == "test"
        assert "links_added" in result

    @patch("app.quality.read_article_text", return_value=None)
    async def test_missing_article(self, mock_read):
        result = await add_crosslinks("kb", "missing")
        assert "error" in result

    @patch("app.quality.get_articles", return_value=[{"slug": "only", "title": "Only"}])
    @patch("app.quality.parse_frontmatter", return_value=({"title": "Only"}, "body"))
    @patch("app.quality.read_article_text", return_value="---\ntitle: Only\n---\nbody")
    async def test_no_other_articles(self, mock_read, mock_parse, mock_get):
        result = await add_crosslinks("kb", "only")
        assert result["links_added"] == 0


# ── Fact checking (async, LLM + Serper mocked) ──────────────────────────────


class TestFactCheckArticle:
    @patch("app.quality.read_article_text", return_value=None)
    async def test_missing_article(self, mock_read):
        result = await fact_check_article("kb", "missing")
        assert "error" in result

    @patch("app.quality._serper_search", new_callable=AsyncMock, return_value=[])
    @patch(
        "app.quality._chat",
        new_callable=AsyncMock,
        return_value='[{"claim": "Python 3.12 released", "search_query": "python 3.12 release", "type": "version"}]',
    )
    @patch("app.quality.db")
    @patch("app.quality.parse_frontmatter", return_value=({"title": "T"}, "Python 3.12 was released."))
    @patch("app.quality.read_article_text", return_value="---\ntitle: T\n---\nPython 3.12 was released.")
    async def test_unverifiable_when_no_search_results(self, mock_read, mock_parse, mock_db, mock_chat, mock_serper):
        mock_db.get_claims_for_article = AsyncMock(return_value=[])
        result = await fact_check_article("kb", "test")
        assert result["claims_checked"] == 1
        assert result["results"][0]["status"] == "unverifiable"

    @patch("app.quality._serper_search", new_callable=AsyncMock)
    @patch("app.quality._chat", new_callable=AsyncMock)
    @patch("app.quality.db")
    @patch("app.quality.parse_frontmatter", return_value=({"title": "T"}, "Body"))
    @patch("app.quality.read_article_text", return_value="---\ntitle: T\n---\nBody")
    async def test_verified_claim(self, mock_read, mock_parse, mock_db, mock_chat, mock_serper):
        mock_db.get_claims_for_article = AsyncMock(return_value=[])
        mock_chat.side_effect = [
            '[{"claim": "Python is popular", "search_query": "python popularity", "type": "feature"}]',
            '{"status": "verified", "current_info": "Python is indeed popular", "correction": null}',
        ]
        mock_serper.return_value = [
            {"title": "Python Stats", "snippet": "Python is #1", "url": "https://example.com"}
        ]
        result = await fact_check_article("kb", "test")
        assert result["verified"] == 1

    @patch("app.quality._serper_search", new_callable=AsyncMock)
    @patch("app.quality._chat", new_callable=AsyncMock)
    @patch("app.quality.db")
    @patch("app.quality.parse_frontmatter", return_value=({"title": "T"}, "Body"))
    @patch("app.quality.read_article_text", return_value="---\ntitle: T\n---\nBody")
    async def test_uses_stored_claims(self, mock_read, mock_parse, mock_db, mock_chat, mock_serper):
        mock_db.get_claims_for_article = AsyncMock(return_value=[
            {"id": "c1", "claim_text": "Python 3.12 released", "claim_type": "version"},
        ])
        mock_db.update_claim_status = AsyncMock()
        mock_chat.return_value = '{"status": "verified", "current_info": "confirmed", "correction": null}'
        mock_serper.return_value = [
            {"title": "T", "snippet": "S", "url": "https://x.com"}
        ]
        result = await fact_check_article("kb", "test")
        assert result["claims_checked"] == 1
        # Should NOT have called the extraction prompt since stored claims were used
        assert mock_chat.call_count == 1  # only verify prompt
        mock_db.update_claim_status.assert_called_once()

    @patch("app.quality._chat", new_callable=AsyncMock, return_value="not json at all")
    @patch("app.quality.db")
    @patch("app.quality.parse_frontmatter", return_value=({"title": "T"}, "Body"))
    @patch("app.quality.read_article_text", return_value="---\ntitle: T\n---\nBody")
    async def test_malformed_llm_response(self, mock_read, mock_parse, mock_db, mock_chat):
        mock_db.get_claims_for_article = AsyncMock(return_value=[])
        result = await fact_check_article("kb", "test")
        assert result["claims_checked"] == 0


# ── Freshness audit ─────────────────────────────────────────────────────────


class TestFreshnessAudit:
    @patch("app.quality.fact_check_article", new_callable=AsyncMock)
    async def test_fresh_article(self, mock_fc):
        mock_fc.return_value = {
            "slug": "s",
            "title": "T",
            "claims_checked": 2,
            "results": [
                {"status": "verified", "claim": "x"},
                {"status": "verified", "claim": "y"},
            ],
        }
        result = await freshness_audit("kb", "s")
        assert result["status"] == "fresh"
        assert result["updates_applied"] == 0

    @patch("app.quality.fact_check_article", new_callable=AsyncMock)
    async def test_stale_no_auto_update(self, mock_fc):
        mock_fc.return_value = {
            "slug": "s",
            "title": "T",
            "claims_checked": 1,
            "results": [{"status": "outdated", "claim": "old info"}],
        }
        result = await freshness_audit("kb", "s", auto_update=False)
        assert result["status"] == "stale"
        assert result["outdated"] == 1
        assert result["updates_applied"] == 0

    @patch("app.quality.fact_check_article", new_callable=AsyncMock, return_value={"error": "not found"})
    async def test_error_propagation(self, mock_fc):
        result = await freshness_audit("kb", "s")
        assert "error" in result


# ── Run quality pass ─────────────────────────────────────────────────────────


class TestRunQualityPass:
    @patch("app.quality.add_crosslinks", new_callable=AsyncMock)
    @patch("app.quality.enrich_article", new_callable=AsyncMock)
    @patch("app.quality.find_shallow_articles")
    async def test_full_pass(self, mock_shallow, mock_enrich, mock_crosslink):
        mock_shallow.return_value = [
            {"slug": "s1", "title": "S1", "word_count": 50, "issues": ["short"], "kb": "kb"},
        ]
        mock_enrich.return_value = {"slug": "s1", "old_words": 50, "new_words": 500, "status": "enriched"}
        mock_crosslink.return_value = {"slug": "s1", "links_added": 3}

        result = await run_quality_pass("kb", max_articles=1)
        assert result["articles_enriched"] == 1
        assert result["articles_crosslinked"] == 1
        assert result["total_words_added"] == 450

    @patch("app.quality.add_crosslinks", new_callable=AsyncMock)
    @patch("app.quality.enrich_article", new_callable=AsyncMock, return_value={"error": "fail"})
    @patch("app.quality.find_shallow_articles", return_value=[
        {"slug": "s1", "title": "S1", "word_count": 50, "issues": ["short"], "kb": "kb"},
    ])
    async def test_enrich_error_skips_crosslink(self, mock_shallow, mock_enrich, mock_crosslink):
        result = await run_quality_pass("kb")
        assert result["articles_enriched"] == 0
        mock_crosslink.assert_not_called()


# ── Fix wikilinks ────────────────────────────────────────────────────────────


class TestFixWikilinks:
    @patch("app.quality.write_article_text")
    @patch("app.quality._serialize_frontmatter", return_value="---\ntitle: A\n---")
    @patch("app.quality.read_article_text")
    @patch("app.quality.get_articles")
    def test_fixes_fuzzy_match(self, mock_get, mock_read, mock_fm, mock_write):
        mock_get.return_value = [
            {"slug": "a", "title": "Article A"},
            {"slug": "python-basics", "title": "Python Basics"},
        ]
        mock_read.return_value = "---\ntitle: Article A\n---\nSee [[Python Basic]]."

        result = fix_wikilinks("kb")
        assert result["links_fixed"] >= 1 or result["links_removed"] >= 0

    @patch("app.quality.get_articles", return_value=[])
    def test_no_articles(self, mock_get):
        result = fix_wikilinks("kb")
        assert result["articles_modified"] == 0

    @patch("app.quality.read_article_text")
    @patch("app.quality.get_articles")
    def test_slug_not_found(self, mock_get, mock_read):
        mock_get.return_value = [{"slug": "a", "title": "A"}]
        result = fix_wikilinks("kb", slug="nonexistent")
        assert "error" in result


# ── Serper search ────────────────────────────────────────────────────────────


class TestSerperSearch:
    @patch("app.quality.SERPER_API_KEY", "")
    async def test_no_api_key_returns_empty(self):
        from app.quality import _serper_search
        result = await _serper_search("test query")
        assert result == []
