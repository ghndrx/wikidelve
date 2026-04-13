"""Unit tests for app.chat._decompose_query + cross-encoder rerank blend."""

from unittest.mock import AsyncMock, patch

import pytest

from app.chat import _decompose_query, _rerank_with_llm


class TestDecomposeQuery:
    @pytest.mark.asyncio
    async def test_short_question_returns_empty(self):
        # Too few words — decomposition is a waste.
        result = await _decompose_query("what is python")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_question(self):
        assert await _decompose_query("") == []
        assert await _decompose_query("   ") == []

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        with patch(
            "app.llm.llm_chat",
            new=AsyncMock(side_effect=RuntimeError("minimax down")),
        ):
            result = await _decompose_query(
                "compare kubernetes and nomad for running stateful workloads at scale"
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_parses_multiline_response(self):
        fake = (
            "How does Kubernetes handle stateful workloads?\n"
            "How does Nomad handle stateful workloads?\n"
            "What are the operational tradeoffs at scale?"
        )
        with patch("app.llm.llm_chat", new=AsyncMock(return_value=fake)):
            result = await _decompose_query(
                "compare kubernetes and nomad for running stateful workloads at scale"
            )
        assert len(result) == 3
        assert all("?" in s for s in result)

    @pytest.mark.asyncio
    async def test_deduplicates_original(self):
        # LLM returns the original question verbatim — should be dropped.
        q = "how do I deploy a python fastapi app to production"
        fake = f"{q}\nWhat process manager should I use?"
        with patch("app.llm.llm_chat", new=AsyncMock(return_value=fake)):
            result = await _decompose_query(q)
        assert q.lower() not in [r.lower() for r in result]


class TestRerankWithLlm:
    @pytest.mark.asyncio
    async def test_empty_candidates_noop(self):
        result = await _rerank_with_llm("question", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_preserves_input(self):
        candidates = [
            {"title": "a", "chunk_text": "alpha", "score": 0.9},
            {"title": "b", "chunk_text": "beta", "score": 0.5},
        ]
        with patch("app.llm.llm_chat", new=AsyncMock(return_value="not json")):
            result = await _rerank_with_llm("q", candidates)
        assert result == candidates

    @pytest.mark.asyncio
    async def test_blended_scores_preserve_order_on_agreement(self):
        candidates = [
            {"title": "a", "chunk_text": "alpha", "score": 0.9},
            {"title": "b", "chunk_text": "beta", "score": 0.5},
        ]
        fake = '[{"id": 0, "score": 9}, {"id": 1, "score": 5}]'
        with patch("app.llm.llm_chat", new=AsyncMock(return_value=fake)):
            result = await _rerank_with_llm("q", candidates)
        assert result[0]["score"] > result[1]["score"]

    @pytest.mark.asyncio
    async def test_blended_scores_reorder_on_disagreement(self):
        candidates = [
            {"title": "bi-encoder favourite", "chunk_text": "alpha", "score": 0.9},
            {"title": "actually relevant", "chunk_text": "beta", "score": 0.1},
        ]
        # LLM strongly prefers passage 1 — blended score should bump it up.
        fake = '[{"id": 0, "score": 1}, {"id": 1, "score": 10}]'
        with patch("app.llm.llm_chat", new=AsyncMock(return_value=fake)):
            result = await _rerank_with_llm("q", candidates)
        # rerank_score is in [0, 1]; higher new score should boost.
        assert result[1]["rerank_score"] == 1.0
        assert result[0]["rerank_score"] == 0.1

    @pytest.mark.asyncio
    async def test_handles_code_fences(self):
        candidates = [{"title": "a", "chunk_text": "alpha", "score": 0.9}]
        fake = '```json\n[{"id": 0, "score": 8}]\n```'
        with patch("app.llm.llm_chat", new=AsyncMock(return_value=fake)):
            result = await _rerank_with_llm("q", candidates)
        assert result[0].get("rerank_score") == 0.8
