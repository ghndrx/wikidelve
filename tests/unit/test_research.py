"""Tests for the research pipeline (app.research)."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import httpx

from app.research import (
    _score_source_tier,
    _tier_label,
    _deduplicate_results,
    _format_sources_for_prompt,
    _generate_sub_questions,
    _generate_verification_queries,
    _synthesize,
    _normalize_url_for_match,
    _lint_citations,
    _truncate_for_tier,
    _render_confidence_notes,
    _critique_draft,
    run_research,
    refine_query,
)


# ===========================================================================
# 1. Source scoring / tier logic
# ===========================================================================

class TestScoreSourceTier:
    """Tests for _score_source_tier URL classification."""

    def test_empty_url_returns_tier3(self):
        assert _score_source_tier("") == 3

    def test_none_url_returns_tier3(self):
        assert _score_source_tier(None) == 3

    def test_gov_domain_is_tier1(self):
        assert _score_source_tier("https://www.whitehouse.gov/report") == 1

    def test_edu_domain_is_tier1(self):
        assert _score_source_tier("https://cs.stanford.edu/papers") == 1

    def test_mil_domain_is_tier1(self):
        assert _score_source_tier("https://www.defense.mil/info") == 1

    def test_tier1_substring_python_docs(self):
        assert _score_source_tier("https://docs.python.org/3/library/asyncio.html") == 1

    def test_tier1_substring_mdn(self):
        assert _score_source_tier("https://developer.mozilla.org/en-US/docs/Web") == 1

    def test_tier1_substring_aws_docs(self):
        assert _score_source_tier("https://docs.aws.amazon.com/lambda/latest/dg/") == 1

    def test_tier2_substring_stackoverflow(self):
        assert _score_source_tier("https://stackoverflow.com/questions/12345") == 2

    def test_tier2_substring_medium(self):
        assert _score_source_tier("https://medium.com/@user/article") == 2

    def test_tier2_substring_devto(self):
        assert _score_source_tier("https://dev.to/user/post") == 2

    def test_random_blog_is_tier3(self):
        assert _score_source_tier("https://someguy.blog/random-post") == 3

    def test_case_insensitive(self):
        assert _score_source_tier("https://DOCS.PYTHON.ORG/3/lib") == 1

    def test_malformed_url_returns_tier3(self):
        assert _score_source_tier("not-a-url") == 3


class TestTierLabel:
    """Tests for _tier_label display string."""

    def test_tier1_label(self):
        assert _tier_label(1) == "Tier 1 - Authoritative"

    def test_tier2_label(self):
        assert _tier_label(2) == "Tier 2 - Reputable"

    def test_tier3_label(self):
        assert _tier_label(3) == "Tier 3 - General"

    def test_invalid_tier_raises(self):
        with pytest.raises(KeyError):
            _tier_label(99)


# ===========================================================================
# 2. URL normalization and citation linting
# ===========================================================================

class TestNormalizeUrl:

    def test_strips_protocol_and_www(self):
        assert _normalize_url_for_match("https://www.example.com/page/") == "example.com/page"

    def test_empty_string(self):
        assert _normalize_url_for_match("") == ""

    def test_http_protocol(self):
        assert _normalize_url_for_match("http://example.com") == "example.com"


class TestLintCitations:

    def test_keeps_valid_citations(self):
        md = "[Python](https://docs.python.org/3/) is great"
        cleaned, stripped = _lint_citations(md, ["https://docs.python.org/3/"])
        assert cleaned == md
        assert stripped == []

    def test_strips_hallucinated_citation(self):
        md = "See [fake](https://hallucinated.example.com/page) for details"
        cleaned, stripped = _lint_citations(md, ["https://real.example.com"])
        assert "fake" in cleaned
        assert "https://hallucinated.example.com" not in cleaned
        assert len(stripped) == 1

    def test_empty_markdown(self):
        cleaned, stripped = _lint_citations("", [])
        assert cleaned == ""
        assert stripped == []

    def test_mixed_valid_and_invalid(self):
        md = "[A](https://a.com) and [B](https://b.com)"
        cleaned, stripped = _lint_citations(md, ["https://a.com"])
        assert "[A](https://a.com)" in cleaned
        assert "https://b.com" not in cleaned
        assert len(stripped) == 1


# ===========================================================================
# 3. Content truncation
# ===========================================================================

class TestTruncateForTier:

    def test_tier1_gets_8000_chars(self):
        content = "x" * 10000
        result = _truncate_for_tier(content, 1)
        assert len(result) == 8000 + len(" ...")

    def test_tier2_gets_3000_chars(self):
        content = "x" * 5000
        result = _truncate_for_tier(content, 2)
        assert len(result) == 3000 + len(" ...")

    def test_tier3_gets_1000_chars(self):
        content = "x" * 2000
        result = _truncate_for_tier(content, 3)
        assert len(result) == 1000 + len(" ...")

    def test_short_content_unchanged(self):
        assert _truncate_for_tier("short", 1) == "short"

    def test_empty_content(self):
        assert _truncate_for_tier("", 1) == ""

    def test_unknown_tier_defaults_1000(self):
        content = "x" * 2000
        result = _truncate_for_tier(content, 99)
        assert len(result) == 1000 + len(" ...")


# ===========================================================================
# 4. Deduplication logic
# ===========================================================================

class TestDeduplicateResults:

    def test_removes_exact_url_duplicates(self):
        results = [
            {"url": "https://example.com/a", "title": "A", "content": "aaa"},
            {"url": "https://example.com/a", "title": "A dup", "content": "bbb"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1
        assert deduped[0]["title"] == "A"

    def test_url_dedup_case_insensitive(self):
        results = [
            {"url": "https://Example.COM/page", "title": "A", "content": "aaa"},
            {"url": "https://example.com/page", "title": "B", "content": "bbb"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1

    def test_url_dedup_ignores_protocol(self):
        results = [
            {"url": "https://example.com/page", "title": "A", "content": "aaa"},
            {"url": "http://example.com/page", "title": "B", "content": "bbb"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1

    def test_url_dedup_strips_trailing_slash(self):
        results = [
            {"url": "https://example.com/page/", "title": "A", "content": "aaa"},
            {"url": "https://example.com/page", "title": "B", "content": "bbb"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1

    def test_keeps_results_without_url(self):
        results = [
            {"title": "no url 1", "content": "unique content one here"},
            {"title": "no url 2", "content": "unique content two here"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2

    def test_removes_near_duplicate_content(self):
        long_text = "the quick brown fox jumped over the lazy dog " * 10
        results = [
            {"url": "https://a.com", "title": "A", "content": long_text},
            {"url": "https://b.com", "title": "B", "content": long_text},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1

    def test_keeps_short_content_without_dedup(self):
        results = [
            {"url": "https://a.com", "title": "A", "content": "short"},
            {"url": "https://b.com", "title": "B", "content": "short"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2

    def test_empty_list(self):
        assert _deduplicate_results([]) == []

    def test_distinct_results_preserved(self):
        results = [
            {"url": "https://a.com", "title": "A", "content": "alpha bravo charlie delta echo foxtrot golf hotel"},
            {"url": "https://b.com", "title": "B", "content": "india juliet kilo lima mike november oscar papa"},
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2


# ===========================================================================
# 5. Source formatting
# ===========================================================================

class TestFormatSourcesForPrompt:

    def test_formats_with_tier_labels(self):
        results = [
            {"title": "Python Docs", "url": "https://docs.python.org", "content": "Python is great", "tier": 1},
        ]
        text = _format_sources_for_prompt(results)
        assert "[Tier 1 - Authoritative]" in text
        assert "**Python Docs**" in text
        assert "https://docs.python.org" in text
        assert "Python is great" in text

    def test_skips_empty_content(self):
        results = [
            {"title": "Empty", "url": "https://empty.com", "content": "", "tier": 2},
            {"title": "Full", "url": "https://full.com", "content": "Has content", "tier": 3},
        ]
        text = _format_sources_for_prompt(results)
        assert "Empty" not in text
        assert "Full" in text

    def test_extracted_marker(self):
        results = [
            {"title": "Extracted", "url": "https://e.com", "content": "- fact 1", "tier": 1, "extracted": True},
        ]
        text = _format_sources_for_prompt(results)
        assert "[EXTRACTED FACTS]" in text

    def test_no_extracted_marker_when_false(self):
        results = [
            {"title": "Normal", "url": "https://n.com", "content": "content", "tier": 2},
        ]
        text = _format_sources_for_prompt(results)
        assert "[EXTRACTED FACTS]" not in text

    def test_empty_results(self):
        assert _format_sources_for_prompt([]) == ""

    def test_default_tier_is_3(self):
        results = [{"title": "T", "url": "https://t.com", "content": "c"}]
        text = _format_sources_for_prompt(results)
        assert "[Tier 3 - General]" in text


# ===========================================================================
# 6. Confidence notes rendering
# ===========================================================================

class TestRenderConfidenceNotes:

    def test_empty_claims(self):
        assert _render_confidence_notes([]) == ""

    def test_supported_claims(self):
        claims = [{"text": "Python is fast", "status": "supported", "note": "confirmed"}]
        result = _render_confidence_notes(claims)
        assert "## Confidence Notes" in result
        assert "**1 supported**" in result
        assert "**0 unsupported**" in result

    def test_unsupported_claims_listed(self):
        claims = [{"text": "Wrong claim", "status": "unsupported", "note": "no evidence"}]
        result = _render_confidence_notes(claims)
        assert "Wrong claim" in result
        assert "no evidence" in result
        assert "Claims the sources don't fully support" in result

    def test_missing_claims_listed(self):
        claims = [{"text": "Missing topic", "status": "missing", "note": "important"}]
        result = _render_confidence_notes(claims)
        assert "Missing topic" in result
        assert "Topics the sources cover that the draft omitted" in result


# ===========================================================================
# 7. Sub-question generation (mocked LLM)
# ===========================================================================

class TestGenerateSubQuestions:

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_parses_multiline_response(self, mock_chat):
        mock_chat.return_value = (
            "What are the performance benchmarks for Python 3.12?\n"
            "How does Python compare to Rust for web backends?\n"
            "What new features are in Python 3.13?"
        )
        client = MagicMock()
        questions = await _generate_sub_questions(client, "Python performance", "some summaries")
        assert len(questions) == 3
        assert "Python 3.12" in questions[0]

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_strips_numbering(self, mock_chat):
        mock_chat.return_value = (
            "1. What is the GIL in Python?\n"
            "2. How does asyncio work under the hood?\n"
            "3- What are coroutines?"
        )
        client = MagicMock()
        questions = await _generate_sub_questions(client, "Python internals", "summaries")
        for q in questions:
            assert not q.startswith("1")
            assert not q.startswith("2")
            assert not q.startswith("3")

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_filters_short_lines(self, mock_chat):
        mock_chat.return_value = "A real question about Python internals?\nShort\n\n"
        client = MagicMock()
        questions = await _generate_sub_questions(client, "Python", "summaries")
        assert len(questions) == 1

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_limits_to_4_questions(self, mock_chat):
        mock_chat.return_value = "\n".join(
            [f"Question number {i} about the topic at hand?" for i in range(10)]
        )
        client = MagicMock()
        questions = await _generate_sub_questions(client, "topic", "summaries")
        assert len(questions) <= 4


# ===========================================================================
# 8. Verification query generation (mocked LLM)
# ===========================================================================

class TestGenerateVerificationQueries:

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_parses_response(self, mock_chat):
        mock_chat.return_value = (
            "Python 3.12 release date confirmation\n"
            "GIL removal PEP 703 status check\n"
            "CPython performance improvement percentage"
        )
        client = MagicMock()
        queries = await _generate_verification_queries(client, "Python 3.12", "sources text")
        assert len(queries) == 3

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_limits_to_3_queries(self, mock_chat):
        mock_chat.return_value = "\n".join(
            [f"Verification query number {i} about the topic" for i in range(10)]
        )
        client = MagicMock()
        queries = await _generate_verification_queries(client, "topic", "sources")
        assert len(queries) <= 3

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_strips_numbering_and_bullets(self, mock_chat):
        mock_chat.return_value = (
            "1) First verification query about claims\n"
            "- Second verification query about data"
        )
        client = MagicMock()
        queries = await _generate_verification_queries(client, "topic", "sources")
        for q in queries:
            assert not q.startswith("1")
            assert not q.startswith("-")


# ===========================================================================
# 9. Critique draft (mocked LLM)
# ===========================================================================

class TestCritiqueDraft:

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_parses_valid_json(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "claims": [
                {"text": "Python is fast", "status": "supported", "note": "confirmed"},
                {"text": "Python 4 exists", "status": "unsupported", "note": "not real"},
            ]
        })
        client = MagicMock()
        claims = await _critique_draft(client, "Python", "draft text", "sources text")
        assert len(claims) == 2
        assert claims[0]["status"] == "supported"
        assert claims[1]["status"] == "unsupported"

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_handles_markdown_fenced_json(self, mock_chat):
        mock_chat.return_value = '```json\n{"claims": [{"text": "A", "status": "supported", "note": "ok"}]}\n```'
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert len(claims) == 1

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_empty_on_invalid_json(self, mock_chat):
        mock_chat.return_value = "This is not JSON at all"
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert claims == []

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_empty_on_exception(self, mock_chat):
        mock_chat.side_effect = Exception("LLM down")
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert claims == []

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_normalizes_invalid_status(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "claims": [{"text": "Claim", "status": "unknown_status", "note": ""}]
        })
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert claims[0]["status"] == "unsupported"

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_skips_claims_without_text(self, mock_chat):
        mock_chat.return_value = json.dumps({
            "claims": [
                {"text": "", "status": "supported", "note": ""},
                {"text": "Valid", "status": "supported", "note": "ok"},
            ]
        })
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert len(claims) == 1


# ===========================================================================
# 10. Synthesis (mocked LLM + DB)
# ===========================================================================

class TestSynthesize:

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    async def test_returns_article_and_claims(self, mock_db, mock_chat, mock_extract, mock_critique):
        mock_chat.return_value = "## Executive Summary\nGreat findings."
        mock_extract.return_value = 0
        mock_critique.return_value = [
            {"text": "Claim A", "status": "supported", "note": "ok"},
        ]
        mock_db.get_kb_settings = AsyncMock(return_value=None)

        client = MagicMock()
        results = [
            {"title": "Source", "url": "https://example.com", "content": "content here", "tier": 1},
        ]
        article, claims = await _synthesize(client, "Test topic", results)
        assert "Executive Summary" in article
        assert "Confidence Notes" in article
        assert len(claims) == 1

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    async def test_strips_hallucinated_links(self, mock_db, mock_chat, mock_extract, mock_critique):
        mock_chat.return_value = "See [fake](https://hallucinated.example.com) for info."
        mock_extract.return_value = 0
        mock_critique.return_value = []
        mock_db.get_kb_settings = AsyncMock(return_value=None)

        client = MagicMock()
        results = [
            {"title": "Real", "url": "https://real.example.com", "content": "data", "tier": 2},
        ]
        article, _ = await _synthesize(client, "Topic", results)
        assert "https://hallucinated.example.com" not in article
        assert "fake" in article  # label kept

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    async def test_uses_kb_persona(self, mock_db, mock_chat, mock_extract, mock_critique):
        mock_chat.return_value = "## Executive Summary\nDone."
        mock_extract.return_value = 0
        mock_critique.return_value = []
        mock_db.get_kb_settings = AsyncMock(return_value={
            "synthesis_provider": "openai",
            "synthesis_model": "gpt-4o",
            "persona": "You are a security engineer.",
        })

        from app.research import _current_kb
        token = _current_kb.set("security-kb")
        try:
            client = MagicMock()
            results = [{"title": "S", "url": "https://s.com", "content": "c", "tier": 1}]
            await _synthesize(client, "Topic", results)

            # Check that _chat was called with persona-prefixed system prompt
            call_kwargs = mock_chat.call_args
            system_msg = call_kwargs[1].get("system_msg") or call_kwargs[0][1]
            assert "security engineer" in system_msg
        finally:
            _current_kb.reset(token)

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    @patch("app.config.RESEARCH_MAX_RETRIES", 1)
    @patch("app.config.RESEARCH_RETRY_DELAY", 0)
    async def test_retries_on_timeout(self, mock_db, mock_chat, mock_extract, mock_critique):
        mock_chat.side_effect = [httpx.TimeoutException("timeout"), "## Summary\nRetried OK."]
        mock_extract.return_value = 0
        mock_critique.return_value = []
        mock_db.get_kb_settings = AsyncMock(return_value=None)

        client = MagicMock()
        results = [{"title": "S", "url": "https://s.com", "content": "c", "tier": 1}]
        article, _ = await _synthesize(client, "Topic", results)
        assert "Retried OK" in article
        assert mock_chat.call_count == 2


# ===========================================================================
# 11. Query refinement
# ===========================================================================

class TestRefineQuery:

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    async def test_returns_refined_query(self, mock_llm):
        mock_llm.return_value = "machine learning"
        result = await refine_query("machin lerning")
        assert result == "machine learning"

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    async def test_returns_original_on_error(self, mock_llm):
        mock_llm.side_effect = Exception("LLM down")
        result = await refine_query("some query")
        assert result == "some query"

    @patch("app.research.MINIMAX_API_KEY", "")
    @patch("app.research.LLM_PROVIDER", "minimax")
    async def test_returns_original_when_no_llm(self):
        result = await refine_query("some query")
        assert result == "some query"


# ===========================================================================
# 12. Search functions (mocked providers)
# ===========================================================================

class TestSearchSerper:

    @patch("app.sources.SerperProvider", autospec=True)
    async def test_delegates_to_provider(self, MockProvider):
        from app.research import _search_serper

        mock_instance = MockProvider.return_value
        mock_instance.search = AsyncMock(return_value=[
            {"title": "Result", "url": "https://example.com", "content": "snippet"},
        ])

        client = MagicMock()
        results = await _search_serper(client, "test query", num=5)
        assert len(results) == 1
        assert results[0]["title"] == "Result"
        mock_instance.search.assert_awaited_once_with("test query", num=5)


class TestRound1SearchAllProviders:

    @patch("app.sources.get_provider_classes")
    async def test_merges_results_from_multiple_providers(self, mock_get_providers):
        from app.research import _round1_search_all_providers

        class FakeProviderA:
            name = "fake_a"
            def __init__(self, client): pass
            async def search(self, query, num=5):
                return [{"title": "A1", "url": "https://a.com", "content": "a content"}]

        class FakeProviderB:
            name = "fake_b"
            def __init__(self, client): pass
            async def search(self, query, num=5):
                return [{"title": "B1", "url": "https://b.com", "content": "b content"}]

        mock_get_providers.return_value = [FakeProviderA, FakeProviderB]

        client = MagicMock()
        results = await _round1_search_all_providers(client, "test query")
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"A1", "B1"}

    @patch("app.sources.get_provider_classes")
    async def test_handles_provider_failure_gracefully(self, mock_get_providers):
        from app.research import _round1_search_all_providers

        class GoodProvider:
            name = "good"
            def __init__(self, client): pass
            async def search(self, query, num=5):
                return [{"title": "Good", "url": "https://g.com", "content": "content"}]

        class BadProvider:
            name = "bad"
            def __init__(self, client): pass
            async def search(self, query, num=5):
                raise RuntimeError("Provider crashed")

        mock_get_providers.return_value = [GoodProvider, BadProvider]

        client = MagicMock()
        results = await _round1_search_all_providers(client, "test")
        assert len(results) == 1
        assert results[0]["title"] == "Good"

    @patch("app.sources.get_provider_classes")
    async def test_returns_empty_when_no_providers(self, mock_get_providers):
        from app.research import _round1_search_all_providers

        mock_get_providers.return_value = []
        client = MagicMock()
        results = await _round1_search_all_providers(client, "test")
        assert results == []


# ===========================================================================
# 13. Full pipeline (run_research) with mocked dependencies
# ===========================================================================

class TestRunResearch:

    @patch("app.storage.write_text")
    @patch("app.research._synthesize", new_callable=AsyncMock)
    @patch("app.browser.read_pages_smart", new_callable=AsyncMock, create=True)
    @patch("app.browser.find_document_urls", return_value=[], create=True)
    @patch("app.research._generate_verification_queries", new_callable=AsyncMock)
    @patch("app.research._search_serper", new_callable=AsyncMock)
    @patch("app.research._generate_sub_questions", new_callable=AsyncMock)
    @patch("app.research._round1_search_all_providers", new_callable=AsyncMock)
    @patch("app.research.refine_query", new_callable=AsyncMock)
    @patch("app.research.db")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    async def test_successful_pipeline(
        self, mock_db, mock_refine, mock_round1, mock_subq,
        mock_serper, mock_verif, mock_find_docs, mock_read_pages,
        mock_synthesize, mock_write_text,
    ):
        mock_refine.return_value = "refined topic"
        mock_round1.return_value = [
            {"title": "R1", "url": "https://r1.com", "content": "round 1 content"},
        ]
        mock_subq.return_value = ["Sub question one about the topic?"]
        mock_serper.return_value = [
            {"title": "R2", "url": "https://r2.com", "content": "round 2 content"},
        ]
        mock_verif.return_value = ["Verify this specific claim about topic"]
        mock_synthesize.return_value = ("## Article\nGreat research.", [])

        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        await run_research("test topic", job_id=1, kb_name="test")

        # Check final status is complete
        final_call = mock_db.update_job.call_args_list[-1]
        assert final_call[1].get("status") == "complete"

    @patch("app.research.db")
    @patch("app.research.SERPER_API_KEY", "")
    async def test_errors_without_serper_key(self, mock_db):
        mock_db.update_job = AsyncMock()
        await run_research("topic", job_id=1)
        # Should set error status
        calls = mock_db.update_job.call_args_list
        error_call = calls[-1]
        assert error_call[1].get("status") == "error"
        assert "SERPER_API_KEY" in error_call[1].get("error", "")

    @patch("app.research.db")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "")
    @patch("app.research.LLM_PROVIDER", "minimax")
    async def test_errors_without_minimax_key(self, mock_db):
        mock_db.update_job = AsyncMock()
        await run_research("topic", job_id=1)
        calls = mock_db.update_job.call_args_list
        error_call = calls[-1]
        assert error_call[1].get("status") == "error"
        assert "MINIMAX_API_KEY" in error_call[1].get("error", "")

    @patch("app.research.refine_query", new_callable=AsyncMock)
    @patch("app.research._round1_search_all_providers", new_callable=AsyncMock)
    @patch("app.research.db")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    async def test_handles_no_results(self, mock_db, mock_round1, mock_refine):
        mock_db.update_job = AsyncMock()
        mock_refine.return_value = "topic"
        mock_round1.return_value = []

        await run_research("topic", job_id=1)
        calls = mock_db.update_job.call_args_list
        last_call = calls[-1]
        assert last_call[1].get("status") == "no_results"

    @patch("app.research.refine_query", new_callable=AsyncMock)
    @patch("app.research._round1_search_all_providers", new_callable=AsyncMock)
    @patch("app.research.db")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "bedrock")
    async def test_handles_round1_timeout(self, mock_db, mock_round1, mock_refine):
        mock_db.update_job = AsyncMock()
        mock_refine.return_value = "topic"
        mock_round1.side_effect = httpx.TimeoutException("timeout")

        await run_research("topic", job_id=1)
        calls = mock_db.update_job.call_args_list
        last_call = calls[-1]
        assert last_call[1].get("status") == "error"
        assert "timed out" in last_call[1].get("error", "")

    @patch("app.research.refine_query", new_callable=AsyncMock)
    @patch("app.research._round1_search_all_providers", new_callable=AsyncMock)
    @patch("app.research.db")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "bedrock")
    async def test_handles_round1_http_error(self, mock_db, mock_round1, mock_refine):
        mock_db.update_job = AsyncMock()
        mock_refine.return_value = "topic"
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.example.com"))
        mock_round1.side_effect = httpx.HTTPStatusError("rate limited", request=response.request, response=response)

        await run_research("topic", job_id=1)
        calls = mock_db.update_job.call_args_list
        last_call = calls[-1]
        assert last_call[1].get("status") == "error"
        assert "429" in last_call[1].get("error", "")


# ===========================================================================
# 14. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_score_source_tier_with_special_chars_in_url(self):
        url = "https://docs.python.org/3/library/re.html#module-re"
        assert _score_source_tier(url) == 1

    def test_dedup_preserves_order(self):
        results = [
            {"url": "https://first.com", "title": "First", "content": "aaa bbb ccc ddd eee fff"},
            {"url": "https://second.com", "title": "Second", "content": "ggg hhh iii jjj kkk lll"},
            {"url": "https://third.com", "title": "Third", "content": "mmm nnn ooo ppp qqq rrr"},
        ]
        deduped = _deduplicate_results(results)
        assert [r["title"] for r in deduped] == ["First", "Second", "Third"]

    def test_format_sources_handles_missing_tier(self):
        results = [
            {"title": "No tier", "url": "https://x.com", "content": "stuff"},
        ]
        text = _format_sources_for_prompt(results)
        assert "[Tier 3 - General]" in text

    def test_lint_citations_preserves_non_link_text(self):
        md = "Some text without any links at all."
        cleaned, stripped = _lint_citations(md, [])
        assert cleaned == md
        assert stripped == []

    def test_truncate_for_tier_none_content(self):
        assert _truncate_for_tier(None, 1) == ""

    def test_normalize_url_none_safe(self):
        # _normalize_url_for_match expects a string; test empty
        assert _normalize_url_for_match("") == ""
