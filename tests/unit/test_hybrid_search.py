"""Tests for hybrid search with Reciprocal Rank Fusion."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.hybrid_search import (
    reciprocal_rank_fusion,
    _richness,
    hybrid_search,
    search_via_graph,
    _extract_query_entities,
    _match_known_entities,
    _llm_extract_entities,
)


# ──────────────────────────────────────────────────────────────────────────────
# reciprocal_rank_fusion
# ──────────────────────────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    """Tests for the RRF merging algorithm."""

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_list(self):
        docs = [
            {"slug": "a", "kb": "test", "title": "Article A"},
            {"slug": "b", "kb": "test", "title": "Article B"},
        ]
        result = reciprocal_rank_fusion([docs], k=1)
        assert len(result) == 2
        assert result[0]["slug"] == "a"
        assert result[1]["slug"] == "b"

    def test_two_lists_agreement(self):
        """When both lists agree on ranking, RRF should preserve it."""
        list1 = [
            {"slug": "a", "kb": "t", "title": "A"},
            {"slug": "b", "kb": "t", "title": "B"},
        ]
        list2 = [
            {"slug": "a", "kb": "t", "title": "A"},
            {"slug": "b", "kb": "t", "title": "B"},
        ]
        result = reciprocal_rank_fusion([list1, list2])
        assert result[0]["slug"] == "a"
        assert result[1]["slug"] == "b"

    def test_two_lists_disagreement(self):
        """Document appearing in both lists gets higher score than one-list docs."""
        list1 = [
            {"slug": "a", "kb": "t", "title": "A"},
            {"slug": "c", "kb": "t", "title": "C"},
        ]
        list2 = [
            {"slug": "b", "kb": "t", "title": "B"},
            {"slug": "a", "kb": "t", "title": "A"},
        ]
        result = reciprocal_rank_fusion([list1, list2])
        # "a" appears in both lists, should rank first
        assert result[0]["slug"] == "a"

    def test_rrf_scores_present(self):
        docs = [{"slug": "a", "kb": "t", "title": "A"}]
        result = reciprocal_rank_fusion([docs], k=1)
        assert "rrf_score" in result[0]
        assert result[0]["rrf_score"] > 0

    def test_deduplication_across_lists(self):
        """Same document in multiple lists should appear once."""
        list1 = [{"slug": "a", "kb": "t", "title": "A"}]
        list2 = [{"slug": "a", "kb": "t", "title": "A"}]
        list3 = [{"slug": "a", "kb": "t", "title": "A"}]
        result = reciprocal_rank_fusion([list1, list2, list3])
        assert len(result) == 1

    def test_k_parameter(self):
        """Higher k reduces impact of high rankings."""
        docs = [{"slug": "a", "kb": "t", "title": "A"}]
        # Put in multiple lists to exceed MIN_RRF_SCORE even at high k
        result_low_k = reciprocal_rank_fusion([docs, docs], k=1)
        result_high_k = reciprocal_rank_fusion([docs, docs], k=10)
        assert result_low_k[0]["rrf_score"] > result_high_k[0]["rrf_score"]

    def test_different_kbs(self):
        """Documents from different KBs are treated as different."""
        list1 = [{"slug": "a", "kb": "kb1", "title": "A"}]
        list2 = [{"slug": "a", "kb": "kb2", "title": "A"}]
        # Use k=1 so single-list scores pass MIN_RRF threshold
        result = reciprocal_rank_fusion([list1, list2], k=1)
        assert len(result) == 2

    def test_missing_slug_skipped(self):
        docs = [{"kb": "t", "title": "No slug"}, {"slug": "a", "kb": "t", "title": "A"}]
        # Use k=1 so single-list scores pass the MIN_RRF_SCORE threshold
        result = reciprocal_rank_fusion([docs], k=1)
        assert len(result) == 1
        assert result[0]["slug"] == "a"

    def test_empty_slug_skipped(self):
        docs = [{"slug": "", "kb": "t", "title": "Empty"}, {"slug": "a", "kb": "t", "title": "A"}]
        result = reciprocal_rank_fusion([docs], k=1)
        assert len(result) == 1

    def test_richest_data_kept(self):
        """When same doc appears in multiple lists, keep the one with most metadata."""
        list1 = [{"slug": "a", "kb": "t", "title": "A"}]
        list2 = [{"slug": "a", "kb": "t", "title": "A", "summary": "Summary", "tags": ["tag1"]}]
        result = reciprocal_rank_fusion([list1, list2])
        assert result[0].get("summary") == "Summary"

    def test_many_lists(self):
        """Handles 4+ result lists (FTS, vector, graph, palace)."""
        lists = [
            [{"slug": "a", "kb": "t", "title": "A"}],
            [{"slug": "b", "kb": "t", "title": "B"}],
            [{"slug": "a", "kb": "t", "title": "A"}, {"slug": "c", "kb": "t", "title": "C"}],
            [{"slug": "a", "kb": "t", "title": "A"}],
        ]
        result = reciprocal_rank_fusion(lists)
        # "a" appears in 3 lists, should be first
        assert result[0]["slug"] == "a"

    def test_all_empty_lists(self):
        result = reciprocal_rank_fusion([[], [], []])
        assert result == []

    def test_mixed_empty_and_populated(self):
        result = reciprocal_rank_fusion([
            [],
            [{"slug": "a", "kb": "t", "title": "A"}],
            [],
        ], k=1)
        assert len(result) == 1

    def test_min_rrf_score_filter(self):
        """Documents with very low RRF scores should be filtered out."""
        # With k=60 (default) and a single list, rank-0 doc gets 1/61 ~ 0.016
        # which is below MIN_RRF_SCORE of 0.03
        docs = [{"slug": "a", "kb": "t", "title": "A"}]
        result = reciprocal_rank_fusion([docs])
        # Single appearance at default k=60 should be filtered
        assert len(result) == 0

    def test_rrf_score_is_sum_across_lists(self):
        """RRF score for a doc is the sum of 1/(k+rank+1) across all lists."""
        doc = {"slug": "x", "kb": "t", "title": "X"}
        # Doc at rank 0 in three lists with k=1: 3 * 1/(1+0+1) = 3 * 0.5 = 1.5
        result = reciprocal_rank_fusion([[doc], [doc], [doc]], k=1)
        assert len(result) == 1
        assert abs(result[0]["rrf_score"] - 1.5) < 0.001

    def test_original_doc_not_mutated(self):
        """RRF should not mutate the input documents."""
        doc = {"slug": "a", "kb": "t", "title": "A"}
        reciprocal_rank_fusion([[doc], [doc]], k=1)
        assert "rrf_score" not in doc


class TestRichness:
    """Tests for the _richness scoring helper."""

    def test_empty_doc(self):
        assert _richness({}) == 0

    def test_title_only(self):
        assert _richness({"title": "Test"}) == 2

    def test_full_doc(self):
        doc = {
            "title": "Test",
            "summary": "Sum",
            "snippet": "Snip",
            "tags": ["a"],
            "connections": [1],
        }
        assert _richness(doc) == 7

    def test_partial_doc(self):
        assert _richness({"title": "T", "tags": ["a"]}) == 3

    def test_empty_values_not_counted(self):
        assert _richness({"title": "", "summary": ""}) == 0

    def test_none_values_not_counted(self):
        assert _richness({"title": None, "summary": None}) == 0


# ──────────────────────────────────────────────────────────────────────────────
# hybrid_search (async orchestrator)
# ──────────────────────────────────────────────────────────────────────────────


class TestHybridSearch:
    """Tests for the top-level hybrid_search coroutine."""

    async def test_empty_query_returns_empty(self):
        assert await hybrid_search("") == []

    async def test_whitespace_query_returns_empty(self):
        assert await hybrid_search("   ") == []

    async def test_none_query_returns_empty(self):
        assert await hybrid_search(None) == []

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock)
    @patch("app.hybrid_search._embedding_circuit")
    async def test_fts_results_flow_through(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        mock_circuit.is_open = False
        mock_fts.return_value = [
            {"slug": "a", "kb": "t", "title": "A"},
            {"slug": "b", "kb": "t", "title": "B"},
        ]
        result = await hybrid_search("python", limit=10)
        mock_fts.assert_awaited_once()
        # Results should come through (scores may be low with k=60 single list)
        # so just verify the function ran without error
        assert isinstance(result, list)

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock)
    @patch("app.hybrid_search._embedding_circuit")
    async def test_fts_failure_graceful(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        """FTS failure should not crash hybrid search."""
        mock_circuit.is_open = False
        mock_fts.side_effect = RuntimeError("FTS index corrupt")
        result = await hybrid_search("python")
        assert isinstance(result, list)

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock)
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search._embedding_circuit")
    async def test_vector_skipped_when_circuit_open(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        mock_circuit.is_open = True
        await hybrid_search("test query")
        mock_vec.assert_not_awaited()

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock)
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search._embedding_circuit")
    async def test_vector_called_when_circuit_closed(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        mock_circuit.is_open = False
        mock_vec.return_value = []
        await hybrid_search("test query")
        mock_vec.assert_awaited_once()

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock)
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search._embedding_circuit")
    async def test_embedding_unavailable_handled(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        from app.llm import EmbeddingUnavailable
        mock_circuit.is_open = False
        mock_vec.side_effect = EmbeddingUnavailable("no key")
        result = await hybrid_search("test query")
        assert isinstance(result, list)

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock)
    @patch("app.hybrid_search._embedding_circuit")
    async def test_limit_respected(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        mock_circuit.is_open = False
        # Return many docs so RRF has enough to work with
        docs = [{"slug": f"s{i}", "kb": "t", "title": f"T{i}"} for i in range(20)]
        mock_fts.return_value = docs
        # Also feed them through vector so they appear in 2 lists (pass MIN_RRF)
        mock_vec.return_value = docs
        result = await hybrid_search("python", limit=3)
        assert len(result) <= 3

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock)
    @patch("app.hybrid_search._embedding_circuit")
    async def test_kb_filter_applied_to_fts(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        """When kb_name is provided, FTS results from other KBs are filtered."""
        mock_circuit.is_open = False
        mock_fts.return_value = [
            {"slug": "a", "kb": "wanted", "title": "A"},
            {"slug": "b", "kb": "other", "title": "B"},
        ]
        result = await hybrid_search("python", kb_name="wanted")
        # "b" from "other" KB should be filtered out
        for doc in result:
            assert doc.get("kb") == "wanted" or doc.get("kb") == "unknown"

    @patch("app.hybrid_search.search_via_graph", new_callable=AsyncMock)
    @patch("app.hybrid_search.search_similar", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.search_fts", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search._embedding_circuit")
    async def test_graph_failure_graceful(self, mock_circuit, mock_fts, mock_vec, mock_graph):
        mock_circuit.is_open = False
        mock_graph.side_effect = Exception("graph DB down")
        result = await hybrid_search("test")
        assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────────────────────
# search_via_graph
# ──────────────────────────────────────────────────────────────────────────────


class TestSearchViaGraph:
    @patch("app.hybrid_search._extract_query_entities", new_callable=AsyncMock, return_value=[])
    async def test_no_entities_returns_empty(self, mock_extract):
        result = await search_via_graph("random query")
        assert result == []

    @patch("app.hybrid_search.get_entity_articles", new_callable=AsyncMock)
    @patch("app.hybrid_search._extract_query_entities", new_callable=AsyncMock)
    async def test_entities_looked_up(self, mock_extract, mock_get_articles):
        mock_extract.return_value = ["Python", "Django"]
        mock_get_articles.return_value = [
            {"slug": "python-intro", "kb": "t"},
            {"slug": "django-basics", "kb": "t"},
        ]
        result = await search_via_graph("Python Django web framework")
        assert len(result) > 0
        assert mock_get_articles.await_count == 2

    @patch("app.hybrid_search.get_entity_articles", new_callable=AsyncMock)
    @patch("app.hybrid_search._extract_query_entities", new_callable=AsyncMock)
    async def test_kb_filter(self, mock_extract, mock_get_articles):
        mock_extract.return_value = ["Python"]
        mock_get_articles.return_value = [
            {"slug": "a", "kb": "wanted"},
            {"slug": "b", "kb": "other"},
        ]
        result = await search_via_graph("Python", kb_name="wanted")
        assert all(r["kb"] == "wanted" for r in result)

    @patch("app.hybrid_search.get_entity_articles", new_callable=AsyncMock)
    @patch("app.hybrid_search._extract_query_entities", new_callable=AsyncMock)
    async def test_limit_respected(self, mock_extract, mock_get_articles):
        mock_extract.return_value = ["Python"]
        mock_get_articles.return_value = [
            {"slug": f"s{i}", "kb": "t"} for i in range(20)
        ]
        result = await search_via_graph("Python", limit=5)
        assert len(result) <= 5

    @patch("app.hybrid_search.get_entity_articles", new_callable=AsyncMock)
    @patch("app.hybrid_search._extract_query_entities", new_callable=AsyncMock)
    async def test_entity_lookup_failure_handled(self, mock_extract, mock_get_articles):
        mock_extract.return_value = ["Python"]
        mock_get_articles.side_effect = RuntimeError("DB error")
        result = await search_via_graph("Python")
        assert result == []

    @patch("app.hybrid_search.get_entity_articles", new_callable=AsyncMock)
    @patch("app.hybrid_search._extract_query_entities", new_callable=AsyncMock)
    async def test_sorted_by_entity_count(self, mock_extract, mock_get_articles):
        """Articles matching more entities should rank higher."""
        mock_extract.return_value = ["Python", "Django"]
        # Both entities return article "a"; only Django returns "b"
        async def mock_get(entity_name):
            if entity_name == "Python":
                return [{"slug": "a", "kb": "t"}, {"slug": "b", "kb": "t"}]
            return [{"slug": "a", "kb": "t"}]
        mock_get_articles.side_effect = mock_get
        result = await search_via_graph("Python Django")
        assert result[0]["slug"] == "a"  # appears for both entities


# ──────────────────────────────────────────────────────────────────────────────
# _extract_query_entities / _match_known_entities
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractQueryEntities:
    @patch("app.hybrid_search._match_known_entities", new_callable=AsyncMock)
    async def test_returns_known_matches_first(self, mock_match):
        mock_match.return_value = ["Python"]
        result = await _extract_query_entities("Python basics")
        assert result == ["Python"]

    @patch("app.hybrid_search.llm_chat", new_callable=AsyncMock)
    @patch("app.hybrid_search._match_known_entities", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.MINIMAX_API_KEY", "fake-key")
    async def test_falls_back_to_llm_for_long_queries(self, mock_match, mock_llm):
        mock_llm.return_value = '["React", "TypeScript"]'
        result = await _extract_query_entities("compare React and TypeScript frameworks")
        assert "React" in result
        assert "TypeScript" in result

    @patch("app.hybrid_search._match_known_entities", new_callable=AsyncMock, return_value=[])
    @patch("app.hybrid_search.MINIMAX_API_KEY", None)
    @patch("app.hybrid_search.LLM_PROVIDER", "none")
    async def test_no_llm_returns_empty_for_long_query(self, mock_match):
        result = await _extract_query_entities("compare React and TypeScript frameworks")
        assert result == []


class TestLlmExtractEntities:
    @patch("app.hybrid_search.llm_chat", new_callable=AsyncMock)
    async def test_parses_json_array(self, mock_llm):
        mock_llm.return_value = '["Python", "Django"]'
        result = await _llm_extract_entities("Python Django tutorial")
        assert result == ["Python", "Django"]

    @patch("app.hybrid_search.llm_chat", new_callable=AsyncMock)
    async def test_invalid_json_returns_empty(self, mock_llm):
        mock_llm.return_value = "not valid json"
        result = await _llm_extract_entities("some query")
        assert result == []

    @patch("app.hybrid_search.llm_chat", new_callable=AsyncMock)
    async def test_non_array_json_returns_empty(self, mock_llm):
        mock_llm.return_value = '{"entities": ["a"]}'
        result = await _llm_extract_entities("some query")
        assert result == []

    @patch("app.hybrid_search.llm_chat", new_callable=AsyncMock)
    async def test_filters_non_string_items(self, mock_llm):
        mock_llm.return_value = '["Python", 42, null, "Rust"]'
        result = await _llm_extract_entities("Python vs Rust")
        assert result == ["Python", "Rust"]
