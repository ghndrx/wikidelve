"""Unit tests for the cross-cutting foundation primitives.

Covers:
  - LLM layer: per-call model+provider override, streaming, tool calling
  - SSE helper
  - app/sources/ provider package + registry
  - article_versions write hook
  - article_claims dedup + status updates
  - kb_settings upsert + clear semantics
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# --- Isolated DB path ------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "phase0_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            yield


# ===========================================================================
# 0.2 — Per-call model + provider override
# ===========================================================================


class TestLLMModelOverride:
    @pytest.mark.asyncio
    async def test_provider_validation_rejects_unknown(self):
        from app.llm import _resolve_provider_model
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            _resolve_provider_model("openai", None)

    def test_default_resolution(self):
        from app.llm import _resolve_provider_model
        provider, model = _resolve_provider_model(None, None)
        # Default LLM_PROVIDER is "minimax" unless env says otherwise
        assert provider in ("minimax", "bedrock")
        assert isinstance(model, str)
        assert model

    def test_explicit_model_override(self):
        from app.llm import _resolve_provider_model
        provider, model = _resolve_provider_model("minimax", "custom-model-id")
        assert provider == "minimax"
        assert model == "custom-model-id"

    @pytest.mark.asyncio
    async def test_llm_chat_passes_model_to_minimax(self):
        from app import llm
        captured = {}

        async def fake_minimax(system_msg, user_msg, max_tokens, temperature, *, client=None, model=None):
            captured["model"] = model
            return "fake response"

        with patch.object(llm, "_minimax_chat", side_effect=fake_minimax):
            result = await llm.llm_chat(
                "sys", "user", model="custom-x", provider="minimax",
            )

        assert captured["model"] == "custom-x"
        assert result == "fake response"


# ===========================================================================
# 0.1 — LLM streaming
# ===========================================================================


class TestLLMStreaming:
    @pytest.mark.asyncio
    async def test_minimax_stream_parses_sse_chunks(self):
        from app import llm

        # Three SSE chunks + a [DONE] terminator.
        fake_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            'data: {"choices":[{"delta":{"content":"!"}}]}',
            'data: [DONE]',
        ]

        class FakeStreamCtx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                for line in fake_lines:
                    yield line

        class FakeClient:
            def stream(self, method, url, headers=None, json=None):
                return FakeStreamCtx()

        with patch("app.llm.MINIMAX_API_KEY", "fake-key"):
            chunks = []
            async for chunk in llm._minimax_chat_stream(
                "sys", "user", 100, 0.2, client=FakeClient(), model="MiniMax-test",
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]


# ===========================================================================
# 0.3 — Tool-calling abstraction
# ===========================================================================


class TestLLMToolCalling:
    def test_parse_minimax_tool_response_valid(self):
        from app.llm import _parse_minimax_tool_response
        out = _parse_minimax_tool_response(
            '{"type": "tool_use", "name": "search", "input": {"q": "rust"}}'
        )
        assert out == {"type": "tool_use", "name": "search", "input": {"q": "rust"}}

    def test_parse_minimax_tool_response_fenced(self):
        from app.llm import _parse_minimax_tool_response
        text = '```json\n{"type": "tool_use", "name": "x", "input": {"a": 1}}\n```'
        out = _parse_minimax_tool_response(text)
        assert out["type"] == "tool_use"
        assert out["name"] == "x"
        assert out["input"] == {"a": 1}

    def test_parse_minimax_tool_response_text_returns_none(self):
        from app.llm import _parse_minimax_tool_response
        out = _parse_minimax_tool_response("Just plain text response.")
        assert out is None

    def test_parse_minimax_tool_response_malformed_returns_none(self):
        from app.llm import _parse_minimax_tool_response
        # Missing 'input'
        out = _parse_minimax_tool_response('{"type": "tool_use", "name": "x"}')
        assert out is None

    @pytest.mark.asyncio
    async def test_minimax_tool_call_returns_tool_use(self):
        from app import llm

        async def fake_minimax(system_msg, user_msg, max_tokens, temperature, *, client=None, model=None):
            return '{"type": "tool_use", "name": "search_kb", "input": {"query": "rust"}}'

        with patch.object(llm, "_minimax_chat", side_effect=fake_minimax):
            result = await llm._minimax_chat_tools(
                "sys", "user",
                tools=[{"name": "search_kb", "description": "x", "input_schema": {}}],
                max_tokens=100, temperature=0.0,
            )

        assert result == {
            "type": "tool_use",
            "name": "search_kb",
            "input": {"query": "rust"},
        }

    @pytest.mark.asyncio
    async def test_minimax_tool_call_falls_back_to_text(self):
        from app import llm

        async def fake_minimax(*args, **kwargs):
            return "I can answer this directly: Rust is fast."

        with patch.object(llm, "_minimax_chat", side_effect=fake_minimax):
            result = await llm._minimax_chat_tools(
                "sys", "user",
                tools=[{"name": "search_kb", "description": "x", "input_schema": {}}],
                max_tokens=100, temperature=0.0,
            )

        assert result["type"] == "text"
        assert "Rust is fast" in result["content"]


# ===========================================================================
# 0.4 — SSE helper
# ===========================================================================


class TestSSEHelper:
    @pytest.mark.asyncio
    async def test_sse_response_format(self):
        from app.sse import sse_response

        async def gen():
            yield {"event": "tick", "data": {"i": 0}}
            yield {"event": "tick", "data": {"i": 1}}
            yield {"event": "done", "data": "complete"}

        resp = sse_response(gen())
        assert resp.media_type == "text/event-stream"
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"

        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        body = "".join(chunks)

        assert "event: tick" in body
        assert 'data: {"i": 0}' in body
        assert 'data: {"i": 1}' in body
        assert "event: done" in body
        assert 'data: "complete"' in body
        # Each event ends with a blank line
        assert "\n\n" in body

    @pytest.mark.asyncio
    async def test_sse_response_handles_generator_error(self):
        from app.sse import sse_response

        async def bad_gen():
            yield {"event": "ok", "data": "first"}
            raise RuntimeError("boom")

        resp = sse_response(bad_gen())
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
        body = "".join(chunks)
        assert "first" in body
        assert "boom" in body
        assert "error" in body


# ===========================================================================
# 0.5 — app/sources/ provider package
# ===========================================================================


class TestSourcesPackage:
    def test_serper_and_tavily_registered(self):
        from app.sources import get_provider_classes
        names = [p.name for p in get_provider_classes()]
        assert "serper" in names
        assert "tavily" in names

    def test_serper_attributes(self):
        from app.sources import SerperProvider
        assert SerperProvider.name == "serper"
        assert SerperProvider.tier_default == 1
        assert SerperProvider.budget_attribution is True

    def test_tavily_attributes(self):
        from app.sources import TavilyProvider
        assert TavilyProvider.name == "tavily"
        assert TavilyProvider.tier_default == 2
        assert TavilyProvider.budget_attribution is False

    def test_provider_lookup(self):
        from app.sources import get_provider_class
        cls = get_provider_class("serper")
        assert cls is not None
        assert cls.name == "serper"
        assert get_provider_class("nonexistent") is None

    def test_register_rejects_empty_name(self):
        from app.sources.base import register

        class BadProvider:
            name = ""
            tier_default = 3
            budget_attribution = False

        with pytest.raises(ValueError, match="non-empty"):
            register(BadProvider)

    @pytest.mark.asyncio
    async def test_serper_provider_logs_to_budget(self):
        """SerperProvider must call db.log_serper_call against the current KB."""
        from app import db
        from app.sources import SerperProvider
        from app.research import _current_kb, _current_job_id

        await db.init_db()

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "organic": [
                        {"title": "T", "snippet": "S", "link": "https://example.com"},
                    ]
                }

        class FakeClient:
            async def post(self, *args, **kwargs):
                return FakeResp()

        kb_token = _current_kb.set("personal")
        job_token = _current_job_id.set(42)
        try:
            with patch("app.sources.serper.SERPER_API_KEY", "fake-key"):
                provider = SerperProvider(FakeClient())
                results = await provider.search("test query", num=8)
        finally:
            _current_kb.reset(kb_token)
            _current_job_id.reset(job_token)

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        # Budget should have been logged against 'personal'
        calls_today = await db.serper_calls_today("personal")
        assert calls_today == 1


# ===========================================================================
# 0.6 — article_versions table + write hook
# ===========================================================================


class TestArticleVersions:
    @pytest.mark.asyncio
    async def test_save_and_get_versions(self):
        from app import db
        await db.init_db()

        await db.save_article_version(
            kb="personal", article_slug="tokio",
            full_content="version 1 content",
            change_type="created",
        )
        await db.save_article_version(
            kb="personal", article_slug="tokio",
            full_content="version 2 content",
            change_type="updated",
        )

        versions = await db.get_article_versions("personal", "tokio")
        assert len(versions) == 2
        # Newest first
        assert versions[0]["full_content"] == "version 2 content"
        assert versions[1]["full_content"] == "version 1 content"
        # Hash is set
        assert versions[0]["content_hash"]
        assert len(versions[0]["content_hash"]) == 64

    @pytest.mark.asyncio
    async def test_update_article_writes_snapshot(self, tmp_path):
        """update_article must snapshot the pre-overwrite content."""
        from app import db
        from app.wiki import update_article

        await db.init_db()

        # Set up a temp KB with one article
        kb_dir = tmp_path / "personal"
        wiki = kb_dir / "wiki"
        wiki.mkdir(parents=True)
        article_path = wiki / "tokio.md"
        article_path.write_text(
            '---\ntitle: "Tokio"\ntags: [rust]\nupdated: 2026-01-01\n---\n\nOriginal body.',
        )

        import app.config as _config, app.storage as _storage, app.wiki as _wiki
        dirs = {"personal": kb_dir}
        _config.KB_DIRS.clear()
        _config.KB_DIRS.update(dirs)
        _storage._default = _storage.LocalStorage()
        _wiki.invalidate_articles_cache()
        await update_article("personal", "tokio", "New finding here")

        versions = await db.get_article_versions("personal", "tokio")
        assert len(versions) == 1
        assert "Original body." in versions[0]["full_content"]
        assert "New finding here" not in versions[0]["full_content"]


# ===========================================================================
# 0.7 — article_claims sidecar table
# ===========================================================================


class TestArticleClaims:
    @pytest.mark.asyncio
    async def test_save_and_get_claims(self):
        from app import db
        await db.init_db()

        cid = await db.save_claim(
            "personal", "tokio", "Tokio is async",
            confidence=0.9, status="verified",
        )
        assert cid > 0

        claims = await db.get_claims_for_article("personal", "tokio")
        assert len(claims) == 1
        assert claims[0]["confidence"] == 0.9
        assert claims[0]["status"] == "verified"

    @pytest.mark.asyncio
    async def test_claim_dedup_updates_existing(self):
        from app import db
        await db.init_db()

        await db.save_claim("personal", "tokio", "Tokio is async", confidence=0.9, status="verified")
        await db.save_claim("personal", "tokio", "Tokio is async", confidence=0.5, status="unverified")

        claims = await db.get_claims_for_article("personal", "tokio")
        assert len(claims) == 1
        # Updated to latest values
        assert claims[0]["confidence"] == 0.5
        assert claims[0]["status"] == "unverified"

    @pytest.mark.asyncio
    async def test_update_claim_status(self):
        from app import db
        await db.init_db()

        cid = await db.save_claim("personal", "x", "Some claim", confidence=0.5, status="unverified")
        await db.update_claim_status(cid, "outdated", 0.2)

        claims = await db.get_claims_for_article("personal", "x")
        assert claims[0]["status"] == "outdated"
        assert claims[0]["confidence"] == 0.2
        assert claims[0]["last_checked_at"] is not None


# ===========================================================================
# 0.8 — kb_settings table
# ===========================================================================


class TestKBSettings:
    @pytest.mark.asyncio
    async def test_upsert_roundtrip(self):
        from app import db
        await db.init_db()

        cfg = await db.upsert_kb_settings(
            "personal",
            synthesis_provider="minimax",
            synthesis_model="MiniMax-M2.7",
            persona="You are a Rust expert",
        )
        assert cfg["synthesis_provider"] == "minimax"
        assert cfg["synthesis_model"] == "MiniMax-M2.7"
        assert cfg["persona"] == "You are a Rust expert"

    @pytest.mark.asyncio
    async def test_partial_update_preserves_other_fields(self):
        from app import db
        await db.init_db()

        await db.upsert_kb_settings(
            "personal",
            synthesis_provider="minimax",
            persona="Original persona",
        )
        # Now update just the persona
        cfg = await db.upsert_kb_settings("personal", persona="New persona")
        assert cfg["synthesis_provider"] == "minimax"  # preserved
        assert cfg["persona"] == "New persona"  # updated

    @pytest.mark.asyncio
    async def test_empty_string_clears_field(self):
        from app import db
        await db.init_db()

        await db.upsert_kb_settings("personal", persona="Some persona")
        cfg = await db.upsert_kb_settings("personal", persona="")
        assert cfg["persona"] is None  # Cleared

    @pytest.mark.asyncio
    async def test_unknown_field_ignored(self):
        from app import db
        await db.init_db()

        cfg = await db.upsert_kb_settings(
            "personal", synthesis_provider="bedrock", nonsense_field="x",
        )
        assert "nonsense_field" not in cfg
        assert cfg["synthesis_provider"] == "bedrock"

    @pytest.mark.asyncio
    async def test_get_missing_kb_returns_none(self):
        from app import db
        await db.init_db()
        assert await db.get_kb_settings("nonexistent") is None


# ===========================================================================
# DynamoDB stubs (import-time check only)
# ===========================================================================


class TestDynamoStubs:
    def test_dynamo_module_exports_versioning_helpers(self):
        from app import db_dynamo
        # The article_versions / article_claims / kb_settings helpers must
        # exist as importable names so the DynamoDB re-export block in
        # app/db.py doesn't ImportError.
        assert hasattr(db_dynamo, "save_article_version")
        assert hasattr(db_dynamo, "get_article_versions")
        assert hasattr(db_dynamo, "get_article_at_timestamp")
        assert hasattr(db_dynamo, "save_claim")
        assert hasattr(db_dynamo, "get_claims_for_article")
        assert hasattr(db_dynamo, "get_stale_claims")
        assert hasattr(db_dynamo, "update_claim_status")
        assert hasattr(db_dynamo, "get_kb_settings")
        assert hasattr(db_dynamo, "upsert_kb_settings")
