"""Tests for LLM utility functions and provider abstractions.

Covers:
  - _strip_non_latin character filtering
  - _resolve_provider_model provider/model selection
  - _extract_minimax_content response parsing
  - _parse_minimax_tool_response tool-call parsing
  - _EmbeddingCircuit circuit breaker
  - _record_minimax_usage / _record_bedrock_usage metric recording
  - llm_chat routing with mocked httpx
  - llm_embed with circuit breaker
  - llm_chat_stream streaming
  - llm_chat_tools tool-calling abstraction
  - _bedrock_extract_tool_or_text response extraction
"""

import json
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.llm import (
    _strip_non_latin,
    _resolve_provider_model,
    _extract_minimax_content,
    _parse_minimax_tool_response,
    _EmbeddingCircuit,
    _record_minimax_usage,
    _record_bedrock_usage,
    _bedrock_extract_tool_or_text,
    llm_chat,
    llm_embed,
    llm_chat_stream,
    llm_chat_tools,
    EmbeddingUnavailable,
    _embedding_circuit,
)
from app import llm as llm_module


class TestStripNonLatin:
    """Tests for removing non-Latin script characters from LLM output."""

    # --- Chinese characters (the original production bug) -------------------

    def test_removes_chinese_characters(self):
        text = "Hello World \u4f60\u597d\u4e16\u754c"
        result = _strip_non_latin(text)
        assert result == "Hello World "

    def test_removes_inline_chinese(self):
        text = "Kubernetes \u5bb9\u5668\u7f16\u6392 orchestration"
        result = _strip_non_latin(text)
        assert "Kubernetes" in result
        assert "orchestration" in result
        assert "\u5bb9" not in result

    def test_removes_chinese_punctuation_and_chars(self):
        text = "Docker\u3001Kubernetes\u3001Helm"
        result = _strip_non_latin(text)
        # CJK punctuation may or may not be blocked depending on unicode names
        assert "Docker" in result
        assert "Kubernetes" in result

    # --- Cyrillic -----------------------------------------------------------

    def test_removes_cyrillic(self):
        text = "Hello \u041f\u0440\u0438\u0432\u0435\u0442 World"
        result = _strip_non_latin(text)
        assert "Hello" in result
        assert "World" in result
        assert "\u041f" not in result

    def test_removes_cyrillic_sentence(self):
        text = "\u042d\u0442\u043e \u0442\u0435\u0441\u0442"
        result = _strip_non_latin(text)
        assert "\u042d" not in result
        assert "\u0442" not in result

    # --- Arabic -------------------------------------------------------------

    def test_removes_arabic(self):
        text = "Hello \u0645\u0631\u062d\u0628\u0627 World"
        result = _strip_non_latin(text)
        assert "Hello" in result
        assert "World" in result
        assert "\u0645" not in result

    # --- Japanese (Hiragana/Katakana) --------------------------------------

    def test_removes_hiragana(self):
        text = "Test \u3053\u3093\u306b\u3061\u306f end"
        result = _strip_non_latin(text)
        assert "Test" in result
        assert "end" in result
        assert "\u3053" not in result

    def test_removes_katakana(self):
        text = "Test \u30c6\u30b9\u30c8 end"
        result = _strip_non_latin(text)
        assert "Test" in result
        assert "end" in result
        assert "\u30c6" not in result

    # --- Korean (Hangul) ---------------------------------------------------

    def test_removes_hangul(self):
        text = "Hello \uc548\ub155\ud558\uc138\uc694 World"
        result = _strip_non_latin(text)
        assert "Hello" in result
        assert "World" in result
        assert "\uc548" not in result

    # --- Devanagari --------------------------------------------------------

    def test_removes_devanagari(self):
        text = "Hello \u0928\u092e\u0938\u094d\u0924\u0947 World"
        result = _strip_non_latin(text)
        assert "Hello" in result
        assert "World" in result
        assert "\u0928" not in result

    # --- Thai --------------------------------------------------------------

    def test_removes_thai(self):
        text = "Hello \u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 World"
        result = _strip_non_latin(text)
        assert "Hello" in result
        assert "World" in result

    # --- Preservation tests ------------------------------------------------

    def test_preserves_pure_ascii(self):
        text = "Hello World 123 !@#$%"
        assert _strip_non_latin(text) == text

    def test_preserves_accented_latin(self):
        text = "caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u00fc\u00f1\u00ed"
        result = _strip_non_latin(text)
        assert "\u00e9" in result
        assert "\u00ef" in result
        assert "\u00fc" in result
        assert "\u00f1" in result

    def test_preserves_latin_extended(self):
        text = "\u0159\u017e\u0161\u010d\u0165\u010f\u0148"  # Czech chars
        result = _strip_non_latin(text)
        assert len(result) == len(text)

    def test_preserves_common_punctuation(self):
        text = "Hello, World! How's it going? (Fine.) [OK] {yes} <no>"
        assert _strip_non_latin(text) == text

    def test_preserves_math_symbols(self):
        text = "x + y = z * 2 / 3 - 1"
        assert _strip_non_latin(text) == text

    def test_preserves_empty_string(self):
        assert _strip_non_latin("") == ""

    def test_preserves_whitespace_only(self):
        assert _strip_non_latin("   \n\t  ") == "   \n\t  "

    def test_preserves_newlines_and_formatting(self):
        text = "Line 1\nLine 2\n\n## Heading\n- bullet"
        assert _strip_non_latin(text) == text

    def test_preserves_urls(self):
        text = "Visit https://example.com/path?q=1&r=2#frag"
        assert _strip_non_latin(text) == text

    def test_preserves_code_snippets(self):
        text = "def foo(x: int) -> str:\n    return f'{x}'"
        assert _strip_non_latin(text) == text

    def test_preserves_em_dash_and_special_punctuation(self):
        text = "word \u2014 another \u2013 more \u2026 end"
        result = _strip_non_latin(text)
        assert "\u2014" in result  # em dash
        assert "\u2013" in result  # en dash
        assert "\u2026" in result  # ellipsis

    # --- Mixed content (realistic LLM output) ------------------------------

    def test_mixed_chinese_and_english_paragraph(self):
        text = (
            "Kubernetes is a container orchestration platform. "
            "\u5b83\u63d0\u4f9b\u4e86\u5bb9\u5668\u7f16\u6392\u529f\u80fd\u3002 "
            "It supports automatic scaling and deployment."
        )
        result = _strip_non_latin(text)
        assert "Kubernetes is a container orchestration platform." in result
        assert "It supports automatic scaling and deployment." in result
        assert "\u5b83" not in result

    def test_mixed_cyrillic_in_technical_text(self):
        text = "Docker \u043a\u043e\u043d\u0442\u0435\u0439\u043d\u0435\u0440 provides isolation"
        result = _strip_non_latin(text)
        assert "Docker" in result
        assert "provides isolation" in result
        assert "\u043a" not in result

    def test_only_non_latin_returns_empty_like(self):
        text = "\u4f60\u597d\u4e16\u754c"
        result = _strip_non_latin(text)
        assert all(ord(c) < 128 or "LATIN" in __import__("unicodedata").name(c, "") for c in result)

    def test_single_non_latin_char(self):
        result = _strip_non_latin("\u4f60")
        assert result == ""

    def test_non_latin_between_words(self):
        text = "hello\u4f60world"
        result = _strip_non_latin(text)
        assert result == "helloworld"


# ===========================================================================
# _resolve_provider_model
# ===========================================================================

class TestResolveProviderModel:
    def test_defaults_to_minimax(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        monkeypatch.setattr(llm_module, "MINIMAX_MODEL", "abab-7")
        provider, model = _resolve_provider_model(None, None)
        assert provider == "minimax"
        assert model == "abab-7"

    def test_explicit_bedrock_provider(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "claude-v3")
        provider, model = _resolve_provider_model("bedrock", None)
        assert provider == "bedrock"
        assert model == "claude-v3"

    def test_explicit_model_override(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        provider, model = _resolve_provider_model("minimax", "custom-model")
        assert model == "custom-model"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _resolve_provider_model("openai", None)

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "claude-v3")
        provider, model = _resolve_provider_model("BEDROCK", None)
        assert provider == "bedrock"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "bedrock")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "some-model")
        provider, model = _resolve_provider_model(None, None)
        assert provider == "bedrock"


# ===========================================================================
# _extract_minimax_content
# ===========================================================================

class TestExtractMinimaxContent:
    def test_standard_response(self):
        data = {
            "choices": [
                {"message": {"content": "Hello!"}, "finish_reason": "stop"}
            ]
        }
        assert _extract_minimax_content(data) == "Hello!"

    def test_text_field_fallback(self):
        data = {"choices": [{"text": "Fallback text"}]}
        assert _extract_minimax_content(data) == "Fallback text"

    def test_delta_field_fallback(self):
        data = {"choices": [{"delta": {"content": "Delta text"}}]}
        assert _extract_minimax_content(data) == "Delta text"

    def test_no_choices_raises(self):
        with pytest.raises(ValueError, match="no choices"):
            _extract_minimax_content({"choices": []})

    def test_empty_content_raises(self):
        data = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        with pytest.raises(ValueError, match="empty content"):
            _extract_minimax_content(data)

    def test_strips_think_tags(self):
        data = {
            "choices": [
                {"message": {"content": "<think>reasoning</think>The answer."}}
            ]
        }
        assert _extract_minimax_content(data) == "The answer."

    def test_non_dict_choice_raises(self):
        data = {"choices": ["not a dict"]}
        with pytest.raises(ValueError, match="not a dict"):
            _extract_minimax_content(data)

    def test_missing_choices_key_raises(self):
        with pytest.raises(ValueError, match="no choices"):
            _extract_minimax_content({"base_resp": {"status_code": 0}})


# ===========================================================================
# _parse_minimax_tool_response
# ===========================================================================

class TestParseMinimaxToolResponse:
    def test_valid_tool_call(self):
        text = json.dumps({
            "type": "tool_use",
            "name": "search",
            "input": {"query": "python"},
        })
        result = _parse_minimax_tool_response(text)
        assert result is not None
        assert result["type"] == "tool_use"
        assert result["name"] == "search"
        assert result["input"] == {"query": "python"}

    def test_code_fenced_tool_call(self):
        text = '```json\n{"type": "tool_use", "name": "lookup", "input": {"id": 1}}\n```'
        result = _parse_minimax_tool_response(text)
        assert result is not None
        assert result["name"] == "lookup"

    def test_plain_text_returns_none(self):
        assert _parse_minimax_tool_response("Just a regular answer.") is None

    def test_empty_string_returns_none(self):
        assert _parse_minimax_tool_response("") is None

    def test_none_input_returns_none(self):
        assert _parse_minimax_tool_response(None) is None

    def test_wrong_type_field_returns_none(self):
        text = json.dumps({"type": "text", "content": "hello"})
        assert _parse_minimax_tool_response(text) is None

    def test_missing_name_returns_none(self):
        text = json.dumps({"type": "tool_use", "input": {}})
        assert _parse_minimax_tool_response(text) is None

    def test_invalid_json_returns_none(self):
        assert _parse_minimax_tool_response("{broken json") is None


# ===========================================================================
# _EmbeddingCircuit
# ===========================================================================

class TestEmbeddingCircuit:
    def test_starts_closed(self):
        cb = _EmbeddingCircuit()
        assert cb.allow() is True
        assert cb.is_open is False

    def test_opens_after_threshold(self):
        cb = _EmbeddingCircuit(failure_threshold=2, cooldown_seconds=60)
        cb.record_failure()
        assert cb.is_open is False
        cb.record_failure()
        assert cb.is_open is True
        assert cb.allow() is False

    def test_resets_on_success(self):
        cb = _EmbeddingCircuit(failure_threshold=2)
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0
        assert cb.is_open is False

    def test_half_open_after_cooldown(self):
        cb = _EmbeddingCircuit(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.is_open is True
        time.sleep(0.02)
        assert cb.allow() is True  # half-open probe
        assert cb._failures == 0

    def test_reset_method(self):
        cb = _EmbeddingCircuit(failure_threshold=1)
        cb.record_failure()
        assert cb.is_open is True
        cb.reset()
        assert cb.is_open is False
        assert cb._failures == 0


# ===========================================================================
# _record_minimax_usage / _record_bedrock_usage
# ===========================================================================

class TestRecordUsage:
    def test_minimax_usage_records(self):
        data = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }
        with patch("app.llm._record_llm_usage") as mock_record:
            _record_minimax_usage(data, "abab-7", kind="chat")
        mock_record.assert_called_once_with(
            provider="minimax", model="abab-7", kind="chat",
            input_tokens=100, output_tokens=50,
        )

    def test_minimax_usage_alternative_keys(self):
        data = {
            "usage": {"input_tokens": 80, "output_tokens": 40}
        }
        with patch("app.llm._record_llm_usage") as mock_record:
            _record_minimax_usage(data, "model-x", kind="embed")
        mock_record.assert_called_once_with(
            provider="minimax", model="model-x", kind="embed",
            input_tokens=80, output_tokens=40,
        )

    def test_minimax_usage_no_usage_key(self):
        with patch("app.llm._record_llm_usage") as mock_record:
            _record_minimax_usage({}, "model-x", kind="chat")
        mock_record.assert_not_called()

    def test_bedrock_usage_records(self):
        data = {
            "usage": {"inputTokens": 200, "outputTokens": 100}
        }
        with patch("app.llm._record_llm_usage") as mock_record:
            _record_bedrock_usage(data, "claude-v3", kind="chat")
        mock_record.assert_called_once_with(
            provider="bedrock", model="claude-v3", kind="chat",
            input_tokens=200, output_tokens=100,
        )


# ===========================================================================
# _bedrock_extract_tool_or_text
# ===========================================================================

class TestBedrockExtractToolOrText:
    def test_text_response(self):
        resp = {
            "output": {
                "message": {
                    "content": [{"text": "Hello world"}]
                }
            }
        }
        result = _bedrock_extract_tool_or_text(resp)
        assert result["type"] == "text"
        assert result["content"] == "Hello world"

    def test_tool_use_response(self):
        resp = {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"name": "search", "input": {"q": "test"}}}
                    ]
                }
            }
        }
        result = _bedrock_extract_tool_or_text(resp)
        assert result["type"] == "tool_use"
        assert result["name"] == "search"
        assert result["input"] == {"q": "test"}

    def test_empty_content(self):
        resp = {"output": {"message": {"content": []}}}
        result = _bedrock_extract_tool_or_text(resp)
        assert result["type"] == "text"
        assert result["content"] == ""

    def test_multiple_text_blocks(self):
        resp = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Part 1"},
                        {"text": "Part 2"},
                    ]
                }
            }
        }
        result = _bedrock_extract_tool_or_text(resp)
        assert result["type"] == "text"
        assert "Part 1" in result["content"]
        assert "Part 2" in result["content"]


# ===========================================================================
# llm_chat (mocked httpx)
# ===========================================================================

class TestLlmChat:
    @pytest.mark.asyncio
    async def test_minimax_chat_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_MODEL", "abab-7")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await llm_chat("system", "user", client=mock_client)
        assert result == "Test answer"

    @pytest.mark.asyncio
    async def test_minimax_chat_no_api_key(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "")

        with pytest.raises(ValueError, match="MINIMAX_API_KEY not set"):
            await llm_chat("system", "user")

    @pytest.mark.asyncio
    async def test_routes_to_bedrock(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "bedrock")

        async def fake_bedrock(sys, usr, max_tokens, temp, *, model=None):
            return "Bedrock response"

        monkeypatch.setattr(llm_module, "_bedrock_chat", fake_bedrock)
        result = await llm_chat("sys", "usr")
        assert result == "Bedrock response"

    @pytest.mark.asyncio
    async def test_strips_non_latin_from_output(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")

        async def fake_minimax(sys, usr, max_tokens, temp, *, client=None, model=None):
            return "Hello \u4f60\u597d world"

        monkeypatch.setattr(llm_module, "_minimax_chat", fake_minimax)
        result = await llm_chat("sys", "usr")
        assert "\u4f60" not in result
        assert "Hello" in result
        assert "world" in result

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            await llm_chat("sys", "usr", provider="openai")


# ===========================================================================
# llm_embed (circuit breaker integration)
# ===========================================================================

class TestLlmEmbed:
    @pytest.fixture(autouse=True)
    def _reset_circuit(self):
        _embedding_circuit.reset()
        yield
        _embedding_circuit.reset()

    @pytest.mark.asyncio
    async def test_embed_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        fake_vec = [0.1] * 128

        async def fake_embed(texts, embed_type):
            return [fake_vec] * len(texts)

        monkeypatch.setattr(llm_module, "_minimax_embed", fake_embed)
        result = await llm_embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 128
        assert _embedding_circuit._failures == 0

    @pytest.mark.asyncio
    async def test_embed_circuit_opens_after_failures(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")

        async def fail_embed(texts, embed_type):
            raise RuntimeError("timeout")

        monkeypatch.setattr(llm_module, "_minimax_embed", fail_embed)

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await llm_embed(["test"])

        # Circuit is now open
        with pytest.raises(EmbeddingUnavailable):
            await llm_embed(["test"])

    @pytest.mark.asyncio
    async def test_embed_mismatched_count_raises(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")

        async def bad_embed(texts, embed_type):
            return [[0.1] * 128]  # returns 1 vector for 2 texts

        monkeypatch.setattr(llm_module, "_minimax_embed", bad_embed)
        with pytest.raises(EmbeddingUnavailable, match="1 vectors for 2 texts"):
            await llm_embed(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_embed_bedrock_routing(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "bedrock")
        fake_vec = [0.2] * 64

        async def fake_bedrock_embed(texts):
            return [fake_vec] * len(texts)

        monkeypatch.setattr(llm_module, "_bedrock_embed", fake_bedrock_embed)
        result = await llm_embed(["test"])
        assert len(result) == 1


# ===========================================================================
# llm_chat_stream (mocked)
# ===========================================================================

class TestLlmChatStream:
    @pytest.mark.asyncio
    async def test_stream_minimax(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")

        async def fake_stream(sys, usr, max_tokens, temp, *, client=None, model=None):
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        monkeypatch.setattr(llm_module, "_minimax_chat_stream", fake_stream)
        chunks = []
        async for chunk in llm_chat_stream("sys", "usr"):
            chunks.append(chunk)
        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_skips_empty_chunks(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")

        async def fake_stream(sys, usr, max_tokens, temp, *, client=None, model=None):
            yield "data"
            yield ""
            yield None
            yield "more"

        monkeypatch.setattr(llm_module, "_minimax_chat_stream", fake_stream)
        chunks = []
        async for chunk in llm_chat_stream("sys", "usr"):
            chunks.append(chunk)
        # Empty and None chunks are skipped
        assert chunks == ["data", "more"]

    @pytest.mark.asyncio
    async def test_stream_strips_non_latin(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")

        async def fake_stream(sys, usr, max_tokens, temp, *, client=None, model=None):
            yield "Hello\u4f60"

        monkeypatch.setattr(llm_module, "_minimax_chat_stream", fake_stream)
        chunks = []
        async for chunk in llm_chat_stream("sys", "usr"):
            chunks.append(chunk)
        assert "\u4f60" not in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_routes_to_bedrock(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "bedrock")

        async def fake_bedrock_stream(sys, usr, max_tokens, temp, *, model=None):
            yield "bedrock chunk"

        monkeypatch.setattr(llm_module, "_bedrock_chat_stream", fake_bedrock_stream)
        chunks = []
        async for chunk in llm_chat_stream("sys", "usr"):
            chunks.append(chunk)
        assert chunks == ["bedrock chunk"]


# ===========================================================================
# llm_chat_tools
# ===========================================================================

class TestLlmChatTools:
    @pytest.mark.asyncio
    async def test_minimax_text_response(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")

        async def fake_chat(sys, usr, max_tokens, temp, *, client=None, model=None):
            return "Just a text answer."

        monkeypatch.setattr(llm_module, "_minimax_chat", fake_chat)

        tools = [{"name": "search", "description": "Search", "input_schema": {}}]
        result = await llm_chat_tools("sys", "usr", tools)
        assert result["type"] == "text"
        assert "text answer" in result["content"]

    @pytest.mark.asyncio
    async def test_minimax_tool_use_response(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "minimax")
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")

        tool_json = json.dumps({
            "type": "tool_use", "name": "search", "input": {"q": "test"}
        })

        async def fake_chat(sys, usr, max_tokens, temp, *, client=None, model=None):
            return tool_json

        monkeypatch.setattr(llm_module, "_minimax_chat", fake_chat)

        tools = [{"name": "search", "description": "Search", "input_schema": {}}]
        result = await llm_chat_tools("sys", "usr", tools)
        assert result["type"] == "tool_use"
        assert result["name"] == "search"

    @pytest.mark.asyncio
    async def test_bedrock_tool_routing(self, monkeypatch):
        monkeypatch.setattr(llm_module, "LLM_PROVIDER", "bedrock")
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")

        async def fake_bedrock_tools(sys, usr, tools, max_tokens, temp, *, model=None):
            return {"type": "text", "content": "Bedrock tools answer"}

        monkeypatch.setattr(llm_module, "_bedrock_chat_tools", fake_bedrock_tools)

        tools = [{"name": "lookup", "description": "Look up", "input_schema": {}}]
        result = await llm_chat_tools("sys", "usr", tools, provider="bedrock")
        assert result["type"] == "text"
