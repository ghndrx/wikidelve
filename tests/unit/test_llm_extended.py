"""Extended tests for app/llm.py — covers uncovered lines for 80%+ coverage.

Targets:
  - _minimax_chat shared-pool path (243-244)
  - _record_llm_usage failure path (293-294)
  - _record_bedrock_usage edge (312)
  - _minimax_embed full path (322-362)
  - _get_bedrock_client (377-386)
  - _bedrock_base_url (391)
  - _bedrock_chat boto3 path (408-444)
  - _bedrock_chat_bearer (456-492)
  - _bedrock_embed boto3 path (500-537)
  - _bedrock_embed_bearer (542-577)
  - _minimax_chat_stream internals (629, 650-684)
  - _bedrock_chat_stream boto3 path (700-753)
  - _bedrock_chat_stream_bearer (769-813)
  - _record_llm_usage happy + error (856-860)
  - _bedrock_chat_tools boto3 path (935-972)
  - _bedrock_chat_tools_bearer (984-1023)
"""

import asyncio
import io
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock

from app import llm as llm_module
from app.llm import (
    _strip_non_latin,
    _resolve_provider_model,
    _extract_minimax_content,
    _parse_minimax_tool_response,
    _EmbeddingCircuit,
    _record_minimax_usage,
    _record_bedrock_usage,
    _record_llm_usage,
    _bedrock_extract_tool_or_text,
    _bedrock_base_url,
    _get_bedrock_client,
    llm_chat,
    llm_embed,
    llm_chat_stream,
    llm_chat_tools,
    EmbeddingUnavailable,
    _embedding_circuit,
    _minimax_chat,
    _minimax_embed,
    _minimax_chat_stream,
    _bedrock_chat,
    _bedrock_chat_bearer,
    _bedrock_chat_stream,
    _bedrock_chat_stream_bearer,
    _bedrock_embed,
    _bedrock_embed_bearer,
    _bedrock_chat_tools,
    _bedrock_chat_tools_bearer,
)


# ===========================================================================
# _minimax_chat — shared-pool path (lines 243-244)
# ===========================================================================

class TestMinimaxChatSharedPool:
    """When no client is passed, _minimax_chat uses get_http_client()."""

    @pytest.mark.asyncio
    async def test_shared_pool_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_MODEL", "abab-7")
        monkeypatch.setattr(llm_module, "MINIMAX_BASE", "https://api.example.com")
        monkeypatch.setattr(llm_module, "MINIMAX_TIMEOUT", 60)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "pooled response"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.http_client.get_http_client", return_value=mock_client) as mock_get:
            # Call WITHOUT passing client= so it hits the shared pool path
            result = await _minimax_chat("sys", "usr", 100, 0.5)

        mock_get.assert_called_once()
        assert result == "pooled response"


# ===========================================================================
# _record_llm_usage (lines 278-294, 856-860)
# ===========================================================================

class TestRecordLlmUsage:
    def test_happy_path(self):
        with patch("app.metrics.record_llm_call") as mock_call:
            _record_llm_usage(provider="minimax", model="m1", kind="chat",
                              input_tokens=10, output_tokens=5)
            mock_call.assert_called_once_with(
                provider="minimax", model="m1", kind="chat",
                input_tokens=10, output_tokens=5,
            )

    def test_swallows_exception(self):
        """Metrics failure must not propagate (line 293-294)."""
        with patch("app.metrics.record_llm_call", side_effect=RuntimeError("boom")):
            # Should NOT raise
            _record_llm_usage(provider="bedrock", model="m2", kind="embed",
                              input_tokens=0, output_tokens=0)

    def test_import_error_swallowed(self):
        """If app.metrics can't be imported the call still succeeds."""
        with patch.dict("sys.modules", {"app.metrics": None}):
            # Force reimport path to fail
            _record_llm_usage(provider="x", model="y", kind="z",
                              input_tokens=0, output_tokens=0)


# ===========================================================================
# _record_bedrock_usage edge — no usage key (line 312)
# ===========================================================================

class TestRecordBedrockUsageEdge:
    def test_no_usage_key_noop(self):
        with patch("app.llm._record_llm_usage") as mock:
            _record_bedrock_usage({}, "model", kind="chat")
        mock.assert_not_called()

    def test_non_dict_data(self):
        with patch("app.llm._record_llm_usage") as mock:
            _record_bedrock_usage("not-a-dict", "model", kind="chat")
        mock.assert_not_called()

    def test_alternative_keys(self):
        data = {"usage": {"input_tokens": 30, "output_tokens": 15}}
        with patch("app.llm._record_llm_usage") as mock:
            _record_bedrock_usage(data, "model", kind="chat")
        mock.assert_called_once_with(
            provider="bedrock", model="model", kind="chat",
            input_tokens=30, output_tokens=15,
        )


# ===========================================================================
# _minimax_embed (lines 322-362)
# ===========================================================================

class TestMinimaxEmbed:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_BASE", "https://api.example.com")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "vectors": [[0.1, 0.2], [0.3, 0.4]],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.http_client.get_http_client", return_value=mock_client), \
             patch("app.llm._record_llm_usage") as mock_usage:
            result = await _minimax_embed(["hello", "world"], "db")

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_usage.assert_called_once()
        call_kw = mock_usage.call_args[1]
        assert call_kw["provider"] == "minimax"
        assert call_kw["model"] == "embo-01"
        assert call_kw["kind"] == "embed"

    @pytest.mark.asyncio
    async def test_empty_texts_returns_empty(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        result = await _minimax_embed([], "db")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "")
        with pytest.raises(ValueError, match="MINIMAX_API_KEY not set"):
            await _minimax_embed(["text"], "db")

    @pytest.mark.asyncio
    async def test_truncates_long_text(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_BASE", "https://api.example.com")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"vectors": [[0.5]]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        long_text = "x" * 10000
        with patch("app.http_client.get_http_client", return_value=mock_client):
            await _minimax_embed([long_text], "db")

        # Verify the text was truncated to 8000 chars
        call_json = mock_client.post.call_args[1]["json"]
        assert len(call_json["texts"][0]) == 8000


# ===========================================================================
# _get_bedrock_client (lines 377-386)
# ===========================================================================

class TestGetBedrockClient:
    def test_with_explicit_credentials(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")
        monkeypatch.setattr(llm_module, "AWS_ACCESS_KEY_ID", "AKID")
        monkeypatch.setattr(llm_module, "AWS_SECRET_ACCESS_KEY", "SECRET")
        monkeypatch.setattr(llm_module, "AWS_SESSION_TOKEN", "TOKEN")

        with patch("boto3.client") as mock_boto:
            _get_bedrock_client()
        mock_boto.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            aws_session_token="TOKEN",
        )

    def test_without_explicit_credentials(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "eu-west-1")
        monkeypatch.setattr(llm_module, "AWS_ACCESS_KEY_ID", "")
        monkeypatch.setattr(llm_module, "AWS_SECRET_ACCESS_KEY", "")
        monkeypatch.setattr(llm_module, "AWS_SESSION_TOKEN", "")

        with patch("boto3.client") as mock_boto:
            _get_bedrock_client()
        mock_boto.assert_called_once_with(
            "bedrock-runtime", region_name="eu-west-1",
        )

    def test_without_session_token(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-west-2")
        monkeypatch.setattr(llm_module, "AWS_ACCESS_KEY_ID", "AKID")
        monkeypatch.setattr(llm_module, "AWS_SECRET_ACCESS_KEY", "SECRET")
        monkeypatch.setattr(llm_module, "AWS_SESSION_TOKEN", "")

        with patch("boto3.client") as mock_boto:
            _get_bedrock_client()
        call_kwargs = mock_boto.call_args[1]
        assert "aws_session_token" not in call_kwargs


# ===========================================================================
# _bedrock_base_url (line 391)
# ===========================================================================

class TestBedrockBaseUrl:
    def test_url_format(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "ap-southeast-1")
        url = _bedrock_base_url()
        assert url == "https://bedrock-runtime.ap-southeast-1.amazonaws.com"


# ===========================================================================
# _bedrock_chat boto3 path (lines 408-444)
# ===========================================================================

class TestBedrockChatBoto3:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "anthropic.claude-v3")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Hello from Bedrock"}]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage") as mock_usage:
            result = await _bedrock_chat("system prompt", "user msg", 1000, 0.3)

        assert result == "Hello from Bedrock"
        mock_client.converse.assert_called_once()
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "anthropic.claude-v3"
        assert call_kwargs["system"] == [{"text": "system prompt"}]

    @pytest.mark.asyncio
    async def test_empty_system_msg(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "response"}]}},
            "usage": {},
        }

        with patch("app.llm._get_bedrock_client", return_value=mock_client):
            result = await _bedrock_chat("", "user msg", 100, 0.2)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["system"] == []

    @pytest.mark.asyncio
    async def test_empty_content_raises(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": []}},
            "usage": {},
        }

        with patch("app.llm._get_bedrock_client", return_value=mock_client):
            with pytest.raises(ValueError, match="empty content"):
                await _bedrock_chat("sys", "usr", 100, 0.2)

    @pytest.mark.asyncio
    async def test_strips_think_tags(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "<think>reasoning</think>The answer."}]}},
            "usage": {},
        }

        with patch("app.llm._get_bedrock_client", return_value=mock_client):
            result = await _bedrock_chat("sys", "usr", 100, 0.2)

        assert result == "The answer."

    @pytest.mark.asyncio
    async def test_routes_to_bearer_when_api_key_set(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key-123")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        async def fake_bearer(*args, **kwargs):
            return "bearer response"

        with patch("app.llm._bedrock_chat_bearer", side_effect=fake_bearer) as mock_bearer:
            result = await _bedrock_chat("sys", "usr", 100, 0.2)

        assert result == "bearer response"
        mock_bearer.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_content_blocks(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"text": "Part A"},
                {"text": "Part B"},
            ]}},
            "usage": {},
        }

        with patch("app.llm._get_bedrock_client", return_value=mock_client):
            result = await _bedrock_chat("sys", "usr", 100, 0.2)

        assert "Part A" in result
        assert "Part B" in result


# ===========================================================================
# _bedrock_chat_bearer (lines 456-492)
# ===========================================================================

class TestBedrockChatBearer:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "anthropic.claude-v3")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [{"text": "bearer chat"}]}},
            "usage": {"inputTokens": 5, "outputTokens": 3},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.http_client.get_http_client", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage") as mock_usage:
            result = await _bedrock_chat_bearer("system", "user", 500, 0.3)

        assert result == "bearer chat"
        mock_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_content_raises(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": []}},
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.http_client.get_http_client", return_value=mock_client):
            with pytest.raises(ValueError, match="empty content"):
                await _bedrock_chat_bearer("sys", "usr", 100, 0.2)

    @pytest.mark.asyncio
    async def test_no_system_msg(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("app.http_client.get_http_client", return_value=mock_client):
            result = await _bedrock_chat_bearer("", "user msg", 100, 0.2)

        # Verify system was not included in payload
        call_json = mock_client.post.call_args[1]["json"]
        assert "system" not in call_json
        assert result == "ok"


# ===========================================================================
# _bedrock_embed boto3 path (lines 500-537)
# ===========================================================================

class TestBedrockEmbedBoto3:
    @pytest.mark.asyncio
    async def test_titan_model(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_EMBED_MODEL", "amazon.titan-embed-v1")

        body_bytes = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = body_bytes

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_llm_usage"):
            result = await _bedrock_embed(["hello world"])

        assert result == [[0.1, 0.2, 0.3]]
        call_kwargs = mock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert "inputText" in body

    @pytest.mark.asyncio
    async def test_cohere_model(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_EMBED_MODEL", "cohere.embed-english-v3")

        body_bytes = json.dumps({"embeddings": [[0.4, 0.5]]}).encode()
        mock_body = MagicMock()
        mock_body.read.return_value = body_bytes

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {"body": mock_body}

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_llm_usage"):
            result = await _bedrock_embed(["test text"])

        assert result == [[0.4, 0.5]]
        call_kwargs = mock_client.invoke_model.call_args[1]
        body = json.loads(call_kwargs["body"])
        assert "texts" in body

    @pytest.mark.asyncio
    async def test_multiple_texts(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_EMBED_MODEL", "amazon.titan-embed-v1")

        call_count = 0

        def mock_invoke(**kwargs):
            nonlocal call_count
            call_count += 1
            body_bytes = json.dumps({"embedding": [float(call_count)]}).encode()
            mock_body = MagicMock()
            mock_body.read.return_value = body_bytes
            return {"body": mock_body}

        mock_client = MagicMock()
        mock_client.invoke_model = mock_invoke

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_llm_usage") as mock_usage:
            result = await _bedrock_embed(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [1.0]
        assert result[1] == [2.0]
        mock_usage.assert_called_once()


# ===========================================================================
# _bedrock_embed_bearer (lines 542-577)
# ===========================================================================

class TestBedrockEmbedBearer:
    @pytest.mark.asyncio
    async def test_titan_model(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_EMBED_MODEL", "amazon.titan-embed-v1")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("app.llm._record_llm_usage"):
            result = await _bedrock_embed_bearer(["hello"])

        assert result == [[0.1, 0.2]]

    @pytest.mark.asyncio
    async def test_cohere_model(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_EMBED_MODEL", "cohere.embed-v3")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.3, 0.4]]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("app.llm._record_llm_usage"):
            result = await _bedrock_embed_bearer(["test"])

        assert result == [[0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_routes_from_bedrock_embed(self, monkeypatch):
        """_bedrock_embed delegates to _bedrock_embed_bearer when API key is set."""
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_EMBED_MODEL", "amazon.titan-embed-v1")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        async def fake_bearer(texts):
            return [[0.9]] * len(texts)

        with patch("app.llm._bedrock_embed_bearer", side_effect=fake_bearer):
            result = await _bedrock_embed(["hi"])

        assert result == [[0.9]]


# ===========================================================================
# _minimax_chat_stream internals (lines 629, 650-684)
# ===========================================================================

class TestMinimaxChatStreamInternals:
    @pytest.mark.asyncio
    async def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "")

        with pytest.raises(ValueError, match="MINIMAX_API_KEY not set"):
            async for _ in _minimax_chat_stream("sys", "usr", 100, 0.2):
                pass

    @pytest.mark.asyncio
    async def test_stream_with_client(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_MODEL", "model")
        monkeypatch.setattr(llm_module, "MINIMAX_BASE", "https://api.example.com")

        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
            "",
            "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}],
                                    "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)

        chunks = []
        async for chunk in _minimax_chat_stream("sys", "usr", 100, 0.2, client=mock_client):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_shared_pool(self, monkeypatch):
        """Without client= it uses get_http_client (lines 682-684)."""
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_MODEL", "model")
        monkeypatch.setattr(llm_module, "MINIMAX_BASE", "https://api.example.com")

        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "pooled"}}]}),
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)

        with patch("app.http_client.get_http_client", return_value=mock_client) as mock_get:
            chunks = []
            async for chunk in _minimax_chat_stream("sys", "usr", 100, 0.2):
                chunks.append(chunk)

        mock_get.assert_called_once()
        assert chunks == ["pooled"]

    @pytest.mark.asyncio
    async def test_stream_skips_empty_choices(self, monkeypatch):
        monkeypatch.setattr(llm_module, "MINIMAX_API_KEY", "test-key")
        monkeypatch.setattr(llm_module, "MINIMAX_MODEL", "model")
        monkeypatch.setattr(llm_module, "MINIMAX_BASE", "https://api.example.com")

        lines = [
            "data: " + json.dumps({"choices": []}),  # no choices
            "data: " + json.dumps({"choices": [{"delta": {}}]}),  # no content
            "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
            "data: invalid json",  # JSONDecodeError
            "no-data-prefix",  # skipped
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)

        chunks = []
        async for chunk in _minimax_chat_stream("sys", "usr", 100, 0.2, client=mock_client):
            chunks.append(chunk)

        assert chunks == ["ok"]


# ===========================================================================
# _bedrock_chat_stream boto3 path (lines 700-753)
# ===========================================================================

class TestBedrockChatStreamBoto3:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "claude-v3")

        events = [
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"delta": {"text": " world"}}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]

        mock_client = MagicMock()
        mock_client.converse_stream.return_value = {"stream": events}

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage"):
            chunks = []
            async for chunk in _bedrock_chat_stream("sys", "usr", 100, 0.2):
                chunks.append(chunk)

        assert "".join(chunks) == "Hello world"

    @pytest.mark.asyncio
    async def test_routes_to_bearer_when_api_key_set(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        async def fake_bearer_stream(*args, **kwargs):
            yield "bearer-chunk"

        with patch("app.llm._bedrock_chat_stream_bearer", side_effect=fake_bearer_stream):
            chunks = []
            async for chunk in _bedrock_chat_stream("sys", "usr", 100, 0.2):
                chunks.append(chunk)

        assert chunks == ["bearer-chunk"]

    @pytest.mark.asyncio
    async def test_producer_exception(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        mock_client = MagicMock()
        mock_client.converse_stream.side_effect = RuntimeError("boto3 error")

        with patch("app.llm._get_bedrock_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="boto3 error"):
                async for _ in _bedrock_chat_stream("sys", "usr", 100, 0.2):
                    pass


# ===========================================================================
# _bedrock_chat_stream_bearer (lines 769-813)
# ===========================================================================

class TestBedrockChatStreamBearer:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        lines = [
            "data: " + json.dumps({"contentBlockDelta": {"delta": {"text": "Hi"}}}),
            "data: " + json.dumps({"contentBlockDelta": {"delta": {"text": " there"}}}),
            "data: " + json.dumps({"metadata": {"usage": {"inputTokens": 5, "outputTokens": 3}}}),
            "data: " + json.dumps({"messageStop": {"stopReason": "end_turn"}}),
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage"):
            chunks = []
            async for chunk in _bedrock_chat_stream_bearer("sys", "usr", 100, 0.2):
                chunks.append(chunk)

        assert "".join(chunks) == "Hi there"

    @pytest.mark.asyncio
    async def test_skips_non_data_and_done(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        lines = [
            "",  # empty
            "non-data line",  # no data: prefix
            "data: [DONE]",  # done marker
            "data: invalid json",
            "data: " + json.dumps({"contentBlockDelta": {"delta": {"text": "ok"}}}),
            "data: " + json.dumps({"messageStop": {}}),
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in _bedrock_chat_stream_bearer("sys", "usr", 100, 0.2):
                chunks.append(chunk)

        assert chunks == ["ok"]

    @pytest.mark.asyncio
    async def test_no_system_msg(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        lines = [
            "data: " + json.dumps({"contentBlockDelta": {"delta": {"text": "x"}}}),
            "data: " + json.dumps({"messageStop": {}}),
        ]

        async def mock_aiter_lines():
            for line in lines:
                yield line

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in _bedrock_chat_stream_bearer("", "usr", 100, 0.2):
                chunks.append(chunk)

        # Verify system not in payload
        call_args = mock_client.stream.call_args
        # The json kwarg contains the payload
        payload = call_args[1].get("json") or call_args.kwargs.get("json")
        if payload:
            assert "system" not in payload


# ===========================================================================
# _bedrock_chat_tools boto3 path (lines 935-972)
# ===========================================================================

class TestBedrockChatToolsBoto3:
    @pytest.mark.asyncio
    async def test_happy_path_text(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "claude-v3")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "No tool needed"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage"):
            result = await _bedrock_chat_tools("sys", "usr", tools, 100, 0.2)

        assert result["type"] == "text"
        assert "No tool needed" in result["content"]

    @pytest.mark.asyncio
    async def test_happy_path_tool_use(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "claude-v3")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"name": "search", "input": {"q": "test"}}}
            ]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

        with patch("app.llm._get_bedrock_client", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage"):
            result = await _bedrock_chat_tools("sys", "usr", tools, 100, 0.2)

        assert result["type"] == "tool_use"
        assert result["name"] == "search"

    @pytest.mark.asyncio
    async def test_routes_to_bearer_when_key_set(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        async def fake_bearer(*args, **kwargs):
            return {"type": "text", "content": "bearer tools"}

        with patch("app.llm._bedrock_chat_tools_bearer", side_effect=fake_bearer):
            tools = [{"name": "t", "description": "d", "input_schema": {}}]
            result = await _bedrock_chat_tools("sys", "usr", tools, 100, 0.2)

        assert result["content"] == "bearer tools"

    @pytest.mark.asyncio
    async def test_tool_config_shape(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")

        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }

        tools = [
            {"name": "lookup", "description": "Look up info", "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}}},
        ]

        with patch("app.llm._get_bedrock_client", return_value=mock_client):
            await _bedrock_chat_tools("sys", "usr", tools, 100, 0.2)

        call_kwargs = mock_client.converse.call_args[1]
        tc = call_kwargs["toolConfig"]
        assert len(tc["tools"]) == 1
        spec = tc["tools"][0]["toolSpec"]
        assert spec["name"] == "lookup"
        assert spec["description"] == "Look up info"
        assert "json" in spec["inputSchema"]


# ===========================================================================
# _bedrock_chat_tools_bearer (lines 984-1023)
# ===========================================================================

class TestBedrockChatToolsBearer:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [{"text": "bearer tool text"}]}},
            "usage": {"inputTokens": 5, "outputTokens": 3},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tools = [{"name": "search", "description": "Search", "input_schema": {"type": "object"}}]

        with patch("app.http_client.get_http_client", return_value=mock_client), \
             patch("app.llm._record_bedrock_usage"):
            result = await _bedrock_chat_tools_bearer("sys", "usr", tools, 100, 0.2)

        assert result["type"] == "text"
        assert "bearer tool text" in result["content"]

    @pytest.mark.asyncio
    async def test_tool_use_response(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [
                {"toolUse": {"name": "calc", "input": {"expr": "2+2"}}}
            ]}},
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tools = [{"name": "calc", "description": "Calculate", "input_schema": {}}]

        with patch("app.http_client.get_http_client", return_value=mock_client):
            result = await _bedrock_chat_tools_bearer("sys", "usr", tools, 100, 0.2)

        assert result["type"] == "tool_use"
        assert result["name"] == "calc"

    @pytest.mark.asyncio
    async def test_no_system_msg(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tools = [{"name": "t", "description": "d", "input_schema": {}}]

        with patch("app.http_client.get_http_client", return_value=mock_client):
            await _bedrock_chat_tools_bearer("", "usr", tools, 100, 0.2)

        call_json = mock_client.post.call_args[1]["json"]
        assert "system" not in call_json

    @pytest.mark.asyncio
    async def test_payload_includes_tool_config(self, monkeypatch):
        monkeypatch.setattr(llm_module, "BEDROCK_API_KEY", "br-key")
        monkeypatch.setattr(llm_module, "BEDROCK_MODEL", "model")
        monkeypatch.setattr(llm_module, "BEDROCK_REGION", "us-east-1")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]

        with patch("app.http_client.get_http_client", return_value=mock_client):
            await _bedrock_chat_tools_bearer("sys", "usr", tools, 100, 0.2)

        call_json = mock_client.post.call_args[1]["json"]
        assert "toolConfig" in call_json
        spec = call_json["toolConfig"]["tools"][0]["toolSpec"]
        assert spec["name"] == "t"
