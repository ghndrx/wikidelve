"""Extended tests for app.research — covering uncovered lines."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx

from app.research import (
    _critique_draft,
    _extract_facts_from_source,
    _extract_facts_for_tier1_sources,
    _render_confidence_notes,
    _synthesize,
    refine_query,
    run_research,
    run_research_collect,
    run_research_synthesize,
    _current_kb,
    _current_job_id,
)


# ===========================================================================
# 1. _critique_draft — lines 390-392 (JSONDecodeError path)
# ===========================================================================

class TestCritiqueDraftExtended:

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_empty_on_json_decode_error(self, mock_chat):
        """Lines 390-392: JSON that looks like an object but is invalid."""
        mock_chat.return_value = '{"claims": [{"text": bad json}]}'
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert claims == []

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_empty_when_claims_not_list(self, mock_chat):
        """Line 395-396: claims key present but not a list."""
        mock_chat.return_value = json.dumps({"claims": "not a list"})
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert claims == []

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_skips_non_dict_claims(self, mock_chat):
        """Line 400-401: non-dict items in claims list are skipped."""
        mock_chat.return_value = json.dumps({
            "claims": [
                "just a string",
                42,
                {"text": "Valid", "status": "supported", "note": "ok"},
            ]
        })
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert len(claims) == 1
        assert claims[0]["text"] == "Valid"

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_handles_markdown_fence_without_language(self, mock_chat):
        """Line 376-381: markdown fence without json tag."""
        mock_chat.return_value = '```\n{"claims": [{"text": "A", "status": "supported", "note": "ok"}]}\n```'
        client = MagicMock()
        claims = await _critique_draft(client, "topic", "draft", "sources")
        assert len(claims) == 1


# ===========================================================================
# 2. _extract_facts_from_source — lines 496-498
# ===========================================================================

class TestExtractFactsFromSource:

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_none_for_short_content(self, mock_chat):
        """Line 474: content shorter than _EXTRACTION_MIN_CHARS."""
        source = {"content": "short", "title": "T", "url": "https://x.com"}
        result = await _extract_facts_from_source(MagicMock(), "topic", source)
        assert result is None
        mock_chat.assert_not_awaited()

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_extracted_facts(self, mock_chat):
        mock_chat.return_value = "- Fact 1\n- Fact 2"
        source = {"content": "x" * 2000, "title": "Source", "url": "https://s.com"}
        result = await _extract_facts_from_source(MagicMock(), "topic", source)
        assert result == "- Fact 1\n- Fact 2"

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_returns_none_on_exception(self, mock_chat):
        """Lines 496-498: exception during extraction."""
        mock_chat.side_effect = Exception("LLM failed")
        source = {"content": "x" * 2000, "title": "T", "url": "https://x.com"}
        result = await _extract_facts_from_source(MagicMock(), "topic", source)
        assert result is None

    @patch("app.research._chat", new_callable=AsyncMock)
    async def test_handles_missing_title_and_url(self, mock_chat):
        mock_chat.return_value = "- Fact"
        source = {"content": "x" * 2000}
        result = await _extract_facts_from_source(MagicMock(), "topic", source)
        assert result == "- Fact"


# ===========================================================================
# 3. _extract_facts_for_tier1_sources
# ===========================================================================

class TestExtractFactsForTier1Sources:

    @patch("app.research._extract_facts_from_source", new_callable=AsyncMock)
    async def test_extracts_from_tier1_only(self, mock_extract):
        mock_extract.return_value = "- extracted fact"
        results = [
            {"title": "T1", "url": "https://t1.com", "content": "x" * 2000, "tier": 1},
            {"title": "T2", "url": "https://t2.com", "content": "x" * 2000, "tier": 2},
        ]
        count = await _extract_facts_for_tier1_sources(MagicMock(), "topic", results)
        assert count == 1
        assert results[0]["extracted"] is True
        assert results[0]["content"] == "- extracted fact"
        # Tier 2 source should be untouched
        assert results[1].get("extracted") is None

    @patch("app.research._extract_facts_from_source", new_callable=AsyncMock)
    async def test_returns_zero_when_no_targets(self, mock_extract):
        results = [
            {"title": "T", "url": "https://t.com", "content": "short", "tier": 1},
        ]
        count = await _extract_facts_for_tier1_sources(MagicMock(), "topic", results)
        assert count == 0
        mock_extract.assert_not_awaited()

    @patch("app.research._extract_facts_from_source", new_callable=AsyncMock)
    async def test_handles_extraction_exception(self, mock_extract):
        mock_extract.return_value = None  # Simulate failure (returns None)
        results = [
            {"title": "T1", "url": "https://t1.com", "content": "x" * 2000, "tier": 1},
        ]
        count = await _extract_facts_for_tier1_sources(MagicMock(), "topic", results)
        assert count == 0

    @patch("app.research._extract_facts_from_source", new_callable=AsyncMock)
    async def test_skips_empty_extraction_results(self, mock_extract):
        mock_extract.return_value = "   "  # Empty after strip
        results = [
            {"title": "T1", "url": "https://t1.com", "content": "x" * 2000, "tier": 1},
        ]
        count = await _extract_facts_for_tier1_sources(MagicMock(), "topic", results)
        assert count == 0

    @patch("app.research._extract_facts_from_source", new_callable=AsyncMock)
    async def test_caps_at_max_sources(self, mock_extract):
        """Only extracts from up to _EXTRACTION_MAX_SOURCES (6) sources."""
        mock_extract.return_value = "- fact"
        results = [
            {"title": f"T{i}", "url": f"https://t{i}.com", "content": "x" * 2000, "tier": 1}
            for i in range(10)
        ]
        count = await _extract_facts_for_tier1_sources(MagicMock(), "topic", results)
        assert count == 6  # capped at _EXTRACTION_MAX_SOURCES


# ===========================================================================
# 4. run_research — lines 816-818, 847-863, 930-951, 971-992, 998-1021
# ===========================================================================

class TestRunResearchExtended:

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test topic")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_handles_generic_round1_exception(self, mock_db, mock_refine):
        """Lines 816-818: generic exception in round 1."""
        mock_db.update_job = AsyncMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, side_effect=RuntimeError("unexpected")):
            await run_research("test topic", 1, "personal")

        # Should set error status
        error_call = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" or
               (len(c.args) > 1 and "error" in str(c))
        ]
        assert len(error_call) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test topic")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_handles_round2_sub_question_failure(self, mock_db, mock_refine):
        """Lines 847-848: sub-question generation exception."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "content", "tier": 1},
        ]

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, side_effect=Exception("LLM down")), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._synthesize",
                   new_callable=AsyncMock, return_value=("# Article", [])), \
             patch("app.storage") as mock_storage, \
             patch("app.config.RESEARCH_KB", "research"):
            mock_storage.write_text = MagicMock()
            await run_research("test topic", 2, "personal")

        # Should still complete despite sub-question failure
        complete_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "complete"
        ]
        assert len(complete_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_handles_round2_search_phase_failure(self, mock_db, mock_refine):
        """Lines 862-863: round 2 search phase exception."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "content", "tier": 1},
        ]

        async def fail_search(*args, **kwargs):
            raise Exception("Search phase crashed")

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, return_value=["sub question here?"]), \
             patch("app.research._search_serper",
                   new_callable=AsyncMock, side_effect=Exception("search failed")), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._synthesize",
                   new_callable=AsyncMock, return_value=("# Done", [])), \
             patch("app.storage") as mock_storage, \
             patch("app.config.RESEARCH_KB", "research"):
            mock_storage.write_text = MagicMock()
            await run_research("test", 3, "personal")

        # Should still reach completion
        complete_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "complete"
        ]
        assert len(complete_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_no_results_sets_no_results_status(self, mock_db, mock_refine):
        """Line 820-822: empty round 1 results."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=[]):
            await run_research("test", 4, "personal")

        no_result_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "no_results"
        ]
        assert len(no_result_calls) == 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_synthesis_timeout_sets_error(self, mock_db, mock_refine):
        """Lines 1013-1018: synthesis timeout."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "c", "tier": 1},
        ]

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._synthesize",
                   new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")):
            await run_research("test", 5, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_synthesis_http_error_sets_error(self, mock_db, mock_refine):
        """Lines 1007-1012: synthesis HTTP error."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "c", "tier": 1},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_request = MagicMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._synthesize",
                   new_callable=AsyncMock,
                   side_effect=httpx.HTTPStatusError("error", request=mock_request, response=mock_response)):
            await run_research("test", 6, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_synthesis_generic_error(self, mock_db, mock_refine):
        """Lines 1019-1021: generic synthesis exception."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "c", "tier": 1},
        ]

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._synthesize",
                   new_callable=AsyncMock, side_effect=RuntimeError("unexpected")):
            await run_research("test", 7, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_write_failure_sets_error(self, mock_db, mock_refine):
        """Lines 1035-1037: storage write failure."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "c", "tier": 1},
        ]

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._synthesize",
                   new_callable=AsyncMock, return_value=("# Article", [])), \
             patch("app.storage") as mock_storage, \
             patch("app.config.RESEARCH_KB", "research"):
            mock_storage.write_text = MagicMock(side_effect=OSError("disk full"))
            await run_research("test", 8, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" and "write" in str(c.kwargs.get("error", "")).lower()
        ]
        assert len(error_calls) >= 1

    @patch("app.research.SERPER_API_KEY", "")
    @patch("app.research.db")
    async def test_missing_serper_key_sets_error(self, mock_db):
        """Lines 775-780: no SERPER_API_KEY."""
        mock_db.update_job = AsyncMock()
        await run_research("test", 9, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" and "SERPER" in str(c.kwargs.get("error", ""))
        ]
        assert len(error_calls) >= 1

    @patch("app.research.SERPER_API_KEY", "test-key")
    @patch("app.research.MINIMAX_API_KEY", "")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_missing_minimax_key_sets_error(self, mock_db):
        """Lines 782-783: minimax provider but no key."""
        mock_db.update_job = AsyncMock()
        await run_research("test", 10, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" and "MINIMAX" in str(c.kwargs.get("error", ""))
        ]
        assert len(error_calls) >= 1


# ===========================================================================
# 5. run_research_collect — lines 1063-1069, 1073-1248
# ===========================================================================

class TestRunResearchCollect:

    @patch("app.research.SERPER_API_KEY", "")
    @patch("app.research.db")
    async def test_missing_serper_key(self, mock_db):
        """Lines 1075-1080."""
        mock_db.update_job = AsyncMock()
        await run_research_collect("test", 1, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_missing_minimax_key(self, mock_db):
        """Lines 1082-1083."""
        mock_db.update_job = AsyncMock()
        await run_research_collect("test", 2, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_no_results_sets_no_results(self, mock_db, mock_refine):
        """Lines 1114-1115."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=[]):
            await run_research_collect("test", 3, "personal")

        no_result_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "no_results"
        ]
        assert len(no_result_calls) == 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_full_collect_reaches_awaiting_review(self, mock_db, mock_refine):
        """Lines 1242-1248: full path to awaiting_review."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        round1_results = [
            {"title": "R1", "url": "https://r1.com", "content": "content here", "tier": 1},
        ]

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=round1_results), \
             patch("app.research._generate_sub_questions",
                   new_callable=AsyncMock, return_value=[]), \
             patch("app.research._generate_verification_queries",
                   new_callable=AsyncMock, return_value=[]):
            await run_research_collect("test topic", 4, "personal")

        awaiting_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "awaiting_review"
        ]
        assert len(awaiting_calls) == 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_round1_http_error(self, mock_db, mock_refine):
        """Lines 1101-1106: HTTPStatusError in round 1."""
        mock_db.update_job = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"
        mock_request = MagicMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock,
                   side_effect=httpx.HTTPStatusError("err", request=mock_request, response=mock_response)):
            await run_research_collect("test", 5, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_round1_timeout(self, mock_db, mock_refine):
        """Lines 1107-1108."""
        mock_db.update_job = AsyncMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")):
            await run_research_collect("test", 6, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_round1_generic_exception(self, mock_db, mock_refine):
        """Lines 1110-1112."""
        mock_db.update_job = AsyncMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            await run_research_collect("test", 7, "personal")

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.refine_query", new_callable=AsyncMock, return_value="test")
    @patch("app.research.SERPER_API_KEY", "key")
    @patch("app.research.MINIMAX_API_KEY", "key")
    @patch("app.research.LLM_PROVIDER", "minimax")
    @patch("app.research.db")
    async def test_collect_sets_context_vars(self, mock_db, mock_refine):
        """Lines 1063-1069: context vars set and reset."""
        mock_db.update_job = AsyncMock()
        mock_db.save_sources = AsyncMock()

        with patch("app.research._round1_search_all_providers",
                   new_callable=AsyncMock, return_value=[]):
            await run_research_collect("test", 99, "my-kb")

        # Verify context vars were reset (default is None)
        assert _current_job_id.get() is None
        assert _current_kb.get() is None


# ===========================================================================
# 6. run_research_synthesize — lines 1259-1318
# ===========================================================================

class TestRunResearchSynthesize:

    @patch("app.research.db")
    async def test_no_sources_sets_error(self, mock_db):
        """Lines 1260-1261: no selected sources."""
        mock_db.get_selected_sources = AsyncMock(return_value=[])
        mock_db.update_job = AsyncMock()

        await run_research_synthesize("test topic", 1)

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" and "No sources" in str(c.kwargs.get("error", ""))
        ]
        assert len(error_calls) == 1

    @patch("app.research.db")
    async def test_successful_synthesis(self, mock_db):
        """Lines 1265-1318: full successful path."""
        mock_db.get_selected_sources = AsyncMock(return_value=[
            {"title": "Source A", "content": "Data", "url": "https://a.com", "tier": 1},
        ])
        mock_db.update_job = AsyncMock()

        with patch("app.research._synthesize",
                   new_callable=AsyncMock, return_value=("# Research Output", [{"text": "c", "status": "supported", "note": "ok"}])), \
             patch("app.storage") as mock_storage, \
             patch("app.config.RESEARCH_KB", "research"):
            mock_storage.write_text = MagicMock()
            await run_research_synthesize("test topic", 2)

        complete_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "complete"
        ]
        assert len(complete_calls) == 1
        # Check claims_json is populated
        complete_kwargs = complete_calls[0].kwargs
        assert complete_kwargs.get("claims_json") is not None

    @patch("app.research.db")
    async def test_synthesis_http_error(self, mock_db):
        """Lines 1280-1285."""
        mock_db.get_selected_sources = AsyncMock(return_value=[
            {"title": "S", "content": "c", "url": "https://s.com", "tier": 1},
        ])
        mock_db.update_job = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.text = "Bad Gateway"
        mock_request = MagicMock()

        with patch("app.research._synthesize",
                   new_callable=AsyncMock,
                   side_effect=httpx.HTTPStatusError("err", request=mock_request, response=mock_response)):
            await run_research_synthesize("topic", 3)

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.db")
    async def test_synthesis_timeout(self, mock_db):
        """Lines 1286-1288."""
        mock_db.get_selected_sources = AsyncMock(return_value=[
            {"title": "S", "content": "c", "url": "https://s.com", "tier": 1},
        ])
        mock_db.update_job = AsyncMock()

        with patch("app.research._synthesize",
                   new_callable=AsyncMock, side_effect=httpx.TimeoutException("timeout")):
            await run_research_synthesize("topic", 4)

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" and "timed out" in str(c.kwargs.get("error", ""))
        ]
        assert len(error_calls) >= 1

    @patch("app.research.db")
    async def test_synthesis_generic_error(self, mock_db):
        """Lines 1289-1291."""
        mock_db.get_selected_sources = AsyncMock(return_value=[
            {"title": "S", "content": "c", "url": "https://s.com", "tier": 1},
        ])
        mock_db.update_job = AsyncMock()

        with patch("app.research._synthesize",
                   new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            await run_research_synthesize("topic", 5)

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error"
        ]
        assert len(error_calls) >= 1

    @patch("app.research.db")
    async def test_write_failure(self, mock_db):
        """Lines 1305-1307: storage write failure."""
        mock_db.get_selected_sources = AsyncMock(return_value=[
            {"title": "S", "content": "c", "url": "https://s.com", "tier": 1},
        ])
        mock_db.update_job = AsyncMock()

        with patch("app.research._synthesize",
                   new_callable=AsyncMock, return_value=("# Output", [])), \
             patch("app.storage") as mock_storage, \
             patch("app.config.RESEARCH_KB", "research"):
            mock_storage.write_text = MagicMock(side_effect=OSError("disk full"))
            await run_research_synthesize("topic", 6)

        error_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "error" and "write" in str(c.kwargs.get("error", "")).lower()
        ]
        assert len(error_calls) >= 1

    @patch("app.research.db")
    async def test_no_critique_claims_sets_null_json(self, mock_db):
        """Lines 1316: claims_json is None when no critique."""
        mock_db.get_selected_sources = AsyncMock(return_value=[
            {"title": "S", "content": "c", "url": "https://s.com", "tier": 1},
        ])
        mock_db.update_job = AsyncMock()

        with patch("app.research._synthesize",
                   new_callable=AsyncMock, return_value=("# Output", [])), \
             patch("app.storage") as mock_storage, \
             patch("app.config.RESEARCH_KB", "research"):
            mock_storage.write_text = MagicMock()
            await run_research_synthesize("topic", 7)

        complete_calls = [
            c for c in mock_db.update_job.call_args_list
            if c.kwargs.get("status") == "complete"
        ]
        assert len(complete_calls) == 1
        assert complete_calls[0].kwargs.get("claims_json") is None


# ===========================================================================
# 7. refine_query — additional edge cases
# ===========================================================================

class TestRefineQueryExtended:

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    async def test_rejects_too_long_refinement(self, mock_llm):
        """Line 742: refined query too long (>3x original) is rejected."""
        mock_llm.return_value = "a " * 500  # Way too long
        result = await refine_query("short")
        assert result == "short"

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    async def test_rejects_too_short_refinement(self, mock_llm):
        """Line 742: refined query too short (<3 chars)."""
        mock_llm.return_value = "ab"
        result = await refine_query("some query")
        assert result == "some query"

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    async def test_rejects_empty_refinement(self, mock_llm):
        """Line 741: empty refined result."""
        mock_llm.return_value = "  "
        result = await refine_query("some query")
        assert result == "some query"

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.LLM_PROVIDER", "bedrock")
    async def test_bedrock_provider_uses_llm(self, mock_llm):
        """Line 726: bedrock provider triggers refinement."""
        mock_llm.return_value = "refined query"
        result = await refine_query("refned qury")
        assert result == "refined query"

    @patch("app.research.llm_chat", new_callable=AsyncMock)
    @patch("app.research.MINIMAX_API_KEY", "test-key")
    async def test_strips_quotes_from_refinement(self, mock_llm):
        """Line 741: quotes stripped from refined result."""
        mock_llm.return_value = '"machine learning"'
        result = await refine_query("machin lerning")
        assert result == "machine learning"


# ===========================================================================
# 8. _synthesize — extended paths
# ===========================================================================

class TestSynthesizeExtended:

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    async def test_extraction_failure_doesnt_block_synthesis(self, mock_db, mock_chat, mock_extract, mock_critique):
        """Lines 596-598: extraction pass exception is caught."""
        mock_chat.return_value = "## Executive Summary\nDone."
        mock_extract.side_effect = RuntimeError("extraction crashed")
        mock_critique.return_value = []
        mock_db.get_kb_settings = AsyncMock(return_value=None)

        client = MagicMock()
        results = [{"title": "S", "url": "https://s.com", "content": "c", "tier": 1}]
        article, claims = await _synthesize(client, "topic", results)
        assert "Executive Summary" in article

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    async def test_critique_crash_doesnt_block(self, mock_db, mock_chat, mock_extract, mock_critique):
        """Lines 708-709: critique crash is caught."""
        mock_chat.return_value = "## Executive Summary\nDone."
        mock_extract.return_value = 0
        mock_critique.side_effect = RuntimeError("critique crashed")
        mock_db.get_kb_settings = AsyncMock(return_value=None)

        client = MagicMock()
        results = [{"title": "S", "url": "https://s.com", "content": "c", "tier": 1}]
        article, claims = await _synthesize(client, "topic", results)
        assert "Executive Summary" in article
        assert claims == []

    @patch("app.research._critique_draft", new_callable=AsyncMock)
    @patch("app.research._extract_facts_for_tier1_sources", new_callable=AsyncMock)
    @patch("app.research._chat", new_callable=AsyncMock)
    @patch("app.research.db")
    async def test_kb_settings_failure_uses_defaults(self, mock_db, mock_chat, mock_extract, mock_critique):
        """Lines 615-616: kb_settings load failure."""
        mock_chat.return_value = "## Executive Summary\nDone."
        mock_extract.return_value = 0
        mock_critique.return_value = []
        mock_db.get_kb_settings = AsyncMock(side_effect=RuntimeError("DB error"))

        token = _current_kb.set("broken-kb")
        try:
            client = MagicMock()
            results = [{"title": "S", "url": "https://s.com", "content": "c", "tier": 1}]
            article, _ = await _synthesize(client, "topic", results)
            assert "Executive Summary" in article
        finally:
            _current_kb.reset(token)
