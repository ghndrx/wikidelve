"""Unit tests for the embedding circuit breaker in app.llm."""

from unittest.mock import AsyncMock, patch

import pytest

from app import llm
from app.llm import EmbeddingUnavailable, _EmbeddingCircuit


@pytest.fixture(autouse=True)
def reset_circuit():
    llm._embedding_circuit.reset()
    yield
    llm._embedding_circuit.reset()


class TestCircuitStateMachine:
    def test_starts_closed(self):
        c = _EmbeddingCircuit()
        assert c.allow() is True
        assert c.is_open is False

    def test_opens_after_threshold(self):
        c = _EmbeddingCircuit(failure_threshold=2, cooldown_seconds=30)
        c.record_failure()
        assert c.allow() is True
        c.record_failure()
        assert c.is_open is True
        assert c.allow() is False

    def test_cooldown_half_open_probe(self):
        c = _EmbeddingCircuit(failure_threshold=1, cooldown_seconds=0.01)
        c.record_failure()
        assert c.allow() is False
        import time
        time.sleep(0.02)
        # First probe after cooldown should be allowed.
        assert c.allow() is True
        assert c.is_open is False

    def test_success_resets_failure_count(self):
        c = _EmbeddingCircuit(failure_threshold=3)
        c.record_failure()
        c.record_failure()
        c.record_success()
        # Would need 3 more failures now, not 1.
        c.record_failure()
        assert c.is_open is False


class TestLlmEmbedIntegration:
    @pytest.mark.asyncio
    async def test_success_path(self):
        fake = [[0.1, 0.2], [0.3, 0.4]]
        with patch("app.llm._minimax_embed", new=AsyncMock(return_value=fake)):
            vectors = await llm.llm_embed(["a", "b"])
        assert vectors == fake
        assert llm._embedding_circuit.is_open is False

    @pytest.mark.asyncio
    async def test_exception_feeds_circuit(self):
        with patch(
            "app.llm._minimax_embed",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            for _ in range(llm._embedding_circuit.failure_threshold):
                with pytest.raises(RuntimeError):
                    await llm.llm_embed(["a"])

        # Next call should short-circuit with EmbeddingUnavailable, no API hit.
        with patch(
            "app.llm._minimax_embed", new=AsyncMock(side_effect=AssertionError("should not call"))
        ):
            with pytest.raises(EmbeddingUnavailable):
                await llm.llm_embed(["a"])

    @pytest.mark.asyncio
    async def test_empty_response_counts_as_failure(self):
        with patch("app.llm._minimax_embed", new=AsyncMock(return_value=[])):
            with pytest.raises(EmbeddingUnavailable):
                await llm.llm_embed(["a"])

    @pytest.mark.asyncio
    async def test_mismatched_vector_count_fails(self):
        with patch("app.llm._minimax_embed", new=AsyncMock(return_value=[[0.1]])):
            with pytest.raises(EmbeddingUnavailable):
                await llm.llm_embed(["a", "b"])
