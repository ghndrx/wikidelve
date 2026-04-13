"""LLM provider tests: minimax + bedrock call paths mocked at the
internal-function level so the public API (llm_chat, llm_embed) parsing,
usage tracking, and circuit breaker all run against real-shaped data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app import llm


@pytest.fixture(autouse=True)
def _reset_llm_state():
    llm._embedding_circuit.reset()
    yield
    llm._embedding_circuit.reset()


# ---------------------------------------------------------------------------
# Minimax chat via mocked internal
# ---------------------------------------------------------------------------


class TestMinimaxChat:
    @pytest.mark.asyncio
    async def test_happy_path_returns_content(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_PROVIDER", "minimax")

        async def fake_chat(sys, usr, max_tokens, temp, *, client=None, model=None):
            return "Hello from minimax!"

        monkeypatch.setattr(llm, "_minimax_chat", fake_chat)
        result = await llm.llm_chat("system msg", "user msg")
        assert result == "Hello from minimax!"

    def test_think_tag_regex_strips_think_blocks(self):
        """The _THINK_RE regex is applied inside _minimax_chat and
        _bedrock_chat — verify the pattern itself works correctly."""
        raw = "<think>internal reasoning goes here</think>The answer is 42."
        cleaned = llm._THINK_RE.sub("", raw).strip()
        assert cleaned == "The answer is 42."
        assert "<think>" not in cleaned

    def test_think_tag_multiline(self):
        raw = "<think>\n  step 1\n  step 2\n</think>\nFinal answer."
        cleaned = llm._THINK_RE.sub("", raw).strip()
        assert cleaned == "Final answer."

    @pytest.mark.asyncio
    async def test_bedrock_provider_routes_correctly(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_PROVIDER", "bedrock")

        async def fake_bedrock(sys, usr, max_tokens, temp, *, model=None):
            return "Hello from bedrock!"

        monkeypatch.setattr(llm, "_bedrock_chat", fake_bedrock)
        result = await llm.llm_chat("sys", "usr", provider="bedrock")
        assert result == "Hello from bedrock!"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            await llm.llm_chat("sys", "usr", provider="openai")


# ---------------------------------------------------------------------------
# Minimax embeddings
# ---------------------------------------------------------------------------


class TestMinimaxEmbed:
    @pytest.mark.asyncio
    async def test_happy_path_returns_vectors(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_PROVIDER", "minimax")
        fake_vec = [0.1] * 1024

        async def fake_embed(texts, embed_type):
            return [fake_vec] * len(texts)

        monkeypatch.setattr(llm, "_minimax_embed", fake_embed)
        result = await llm.llm_embed(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 1024

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_PROVIDER", "minimax")

        async def bomb(texts, embed_type):
            raise RuntimeError("minimax 500")

        monkeypatch.setattr(llm, "_minimax_embed", bomb)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await llm.llm_embed(["test"])

        with pytest.raises(llm.EmbeddingUnavailable):
            await llm.llm_embed(["test"])

    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_PROVIDER", "minimax")
        fake_vec = [0.1] * 1024

        async def good(texts, embed_type):
            return [fake_vec] * len(texts)

        monkeypatch.setattr(llm, "_minimax_embed", good)
        llm._embedding_circuit._failures = 2
        result = await llm.llm_embed(["test"])
        assert len(result) == 1
        assert llm._embedding_circuit._failures == 0


# ---------------------------------------------------------------------------
# Non-Latin character stripping
# ---------------------------------------------------------------------------


class TestStripNonLatin:
    def test_ascii_unchanged(self):
        assert llm._strip_non_latin("Hello world") == "Hello world"

    def test_removes_cjk(self):
        result = llm._strip_non_latin("Hello 你好 world")
        assert "你好" not in result
        assert "Hello" in result
        assert "world" in result

    def test_preserves_accented_latin(self):
        assert "é" in llm._strip_non_latin("café")

    def test_preserves_common_symbols(self):
        result = llm._strip_non_latin("price: $100 (50% off)")
        assert "$100" in result
        assert "50%" in result
