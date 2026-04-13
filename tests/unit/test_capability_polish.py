"""Unit tests for the user-facing capability layer.

Covers:
  - Per-KB persona prepended to ``SYNTHESIS_SYSTEM_PROMPT``
  - ``article_versions`` diff helpers (``get_article_version_by_id``)
  - Confidence summary computation (``article_claims`` aggregation)
  - Suggestions endpoint composition (mocked hybrid_search + graph)
  - ``refine_research`` tool dispatch
  - Extended MCP tools surface (``chat_ask``, ``get_graph_neighbors``,
    ``enqueue_auto_discovery``, ``list_topic_candidates``,
    ``get_article_versions``)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "phase5_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            yield


# ===========================================================================
# 5.1 — Per-KB personas
# ===========================================================================


class TestPersonaInjection:
    @pytest.mark.asyncio
    async def test_persona_prepended_to_system_prompt(self):
        """Verify _synthesize uses kb_settings.persona to prefix the system msg."""
        from app import db
        from app import research

        await db.init_db()
        await db.upsert_kb_settings(
            "personal", persona="You are a senior security engineer.",
        )

        captured_system: list[str] = []

        async def fake_chat(client, system_msg, user_msg, **kwargs):
            captured_system.append(system_msg)
            return "Fake synthesized draft."

        async def fake_critique(*args, **kwargs):
            return []

        # Set the ContextVar so _synthesize loads the right kb_settings
        from app.research import _current_kb
        token = _current_kb.set("personal")
        try:
            with patch.object(research, "_chat", side_effect=fake_chat), \
                 patch.object(research, "_critique_draft", side_effect=fake_critique), \
                 patch.object(research, "_extract_facts_for_tier1_sources", AsyncMock(return_value=0)):
                fake_results = [{"title": "T", "content": "c", "url": "https://x", "tier": 1}]
                cleaned, claims = await research._synthesize(None, "Topic", fake_results)
        finally:
            _current_kb.reset(token)

        assert captured_system, "synthesize did not call _chat"
        # The persona text should appear at the START of the system message
        assert captured_system[0].startswith("You are a senior security engineer.")
        # And the original synthesis prompt body should still be there
        assert "knowledge base" in captured_system[0].lower() or "research" in captured_system[0].lower()

    @pytest.mark.asyncio
    async def test_no_persona_uses_default_prompt(self):
        from app import db
        from app import research

        await db.init_db()
        # No kb_settings → default prompt only
        captured_system: list[str] = []

        async def fake_chat(client, system_msg, user_msg, **kwargs):
            captured_system.append(system_msg)
            return "draft"

        async def fake_critique(*args, **kwargs):
            return []

        from app.research import _current_kb
        token = _current_kb.set("personal")
        try:
            with patch.object(research, "_chat", side_effect=fake_chat), \
                 patch.object(research, "_critique_draft", side_effect=fake_critique), \
                 patch.object(research, "_extract_facts_for_tier1_sources", AsyncMock(return_value=0)):
                await research._synthesize(None, "Topic", [{"title": "T", "content": "c", "url": "https://x", "tier": 1}])
        finally:
            _current_kb.reset(token)

        from app.config import SYNTHESIS_SYSTEM_PROMPT
        # No persona configured → system_msg should equal the default exactly
        assert captured_system[0] == SYNTHESIS_SYSTEM_PROMPT


# ===========================================================================
# 5.2 — Article version helpers
# ===========================================================================


class TestArticleVersionLookup:
    @pytest.mark.asyncio
    async def test_get_version_by_id_roundtrip(self):
        from app import db
        await db.init_db()

        vid = await db.save_article_version(
            "personal", "tokio", "snapshot content", change_type="updated",
        )
        v = await db.get_article_version_by_id(vid)
        assert v is not None
        assert v["full_content"] == "snapshot content"
        assert v["article_slug"] == "tokio"

    @pytest.mark.asyncio
    async def test_missing_version_returns_none(self):
        from app import db
        await db.init_db()
        assert await db.get_article_version_by_id(99999) is None


# ===========================================================================
# 5.3 — Confidence summary computation
# ===========================================================================


class TestConfidenceSummary:
    """The summary logic lives inline in main.view_article. We exercise it
    indirectly by computing it the same way the route does."""

    @pytest.mark.asyncio
    async def test_high_confidence_when_all_verified(self):
        from app import db
        await db.init_db()

        for i in range(3):
            await db.save_claim(
                "personal", "x", f"claim {i}",
                confidence=0.9, status="verified",
            )

        claims = await db.get_claims_for_article("personal", "x")
        avg = sum(float(c["confidence"]) for c in claims) / len(claims)
        assert avg >= 0.75
        verified = sum(1 for c in claims if c["status"] == "verified")
        assert verified == 3

    @pytest.mark.asyncio
    async def test_low_confidence_when_unverified(self):
        from app import db
        await db.init_db()

        for i in range(3):
            await db.save_claim(
                "personal", "y", f"claim {i}",
                confidence=0.3, status="unverified",
            )

        claims = await db.get_claims_for_article("personal", "y")
        avg = sum(float(c["confidence"]) for c in claims) / len(claims)
        assert avg < 0.5

    @pytest.mark.asyncio
    async def test_no_claims_yields_no_summary(self):
        from app import db
        await db.init_db()
        claims = await db.get_claims_for_article("personal", "missing")
        assert claims == []


# ===========================================================================
# 5.5 — refine_research tool dispatch
# ===========================================================================


class TestRefineResearchTool:
    @pytest.mark.asyncio
    async def test_refine_research_queues_scoped_topic(self):
        from app import chat, db
        await db.init_db()

        fake_redis = MagicMock()
        fake_redis.enqueue_job = AsyncMock(return_value=None)

        fake_article = {
            "slug": "tokio",
            "title": "Tokio Runtime",
            "raw_markdown": "body",
        }

        with patch.object(chat, "get_article", return_value=fake_article), \
             patch.object(chat.db, "check_cooldown", AsyncMock(return_value=None)):
            result = await chat.run_tool(
                "refine_research",
                {"kb": "personal", "slug": "tokio", "section": "Limitations"},
                redis=fake_redis,
            )

        assert result["status"] == "queued"
        assert "Limitations" in result["topic"]
        assert "Tokio Runtime" in result["topic"]
        assert result["parent_slug"] == "tokio"
        # Verify _queue_name is correctly passed
        args, kwargs = fake_redis.enqueue_job.call_args
        assert args[0] == "research_task"
        assert kwargs.get("_queue_name") == "wikidelve"

    @pytest.mark.asyncio
    async def test_refine_research_respects_cooldown(self):
        from app import chat, db
        await db.init_db()

        fake_redis = MagicMock()
        fake_redis.enqueue_job = AsyncMock()

        fake_article = {
            "slug": "tokio",
            "title": "Tokio Runtime",
            "raw_markdown": "body",
        }

        with patch.object(chat, "get_article", return_value=fake_article), \
             patch.object(
                 chat.db, "check_cooldown",
                 AsyncMock(return_value={"id": 42}),
             ):
            result = await chat.run_tool(
                "refine_research",
                {"kb": "personal", "slug": "tokio", "section": "Limitations"},
                redis=fake_redis,
            )

        assert result["error"] == "cooldown"
        fake_redis.enqueue_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_refine_research_missing_article(self):
        from app import chat
        fake_redis = MagicMock()
        with patch.object(chat, "get_article", return_value=None):
            result = await chat.run_tool(
                "refine_research",
                {"kb": "personal", "slug": "ghost", "section": "x"},
                redis=fake_redis,
            )
        assert "error" in result
        assert "not found" in result["error"].lower()


# ===========================================================================
# 5.7 — New MCP tools
# ===========================================================================


class TestMCPToolsSurface:
    """Skipped wholesale if optional `fastmcp` isn't installed.

    app.mcp_server imports fastmcp at module load. In CI/dev shells
    without the MCP SDK, every test in this class would fail with a
    ModuleNotFoundError — which is noise, not signal, because the MCP
    surface is an optional feature.
    """

    @classmethod
    def setup_class(cls):
        pytest.importorskip("fastmcp")

    def test_mcp_module_imports_cleanly(self):
        from app import mcp_server
        assert hasattr(mcp_server, "mcp")

    def test_extended_tools_defined(self):
        """The chat / graph / auto-discovery / versioning tools should exist."""
        import app.mcp_server as mcp_server
        for name in (
            "chat_ask",
            "get_graph_neighbors",
            "enqueue_auto_discovery",
            "list_topic_candidates",
            "get_article_versions",
        ):
            assert hasattr(mcp_server, name), f"Missing MCP tool: {name}"

    @pytest.mark.asyncio
    async def test_list_topic_candidates_returns_pending(self):
        """Verify the MCP list_topic_candidates wrapper returns the same data
        the underlying db helper does."""
        from app import db
        from app import mcp_server

        await db.init_db()
        await db.insert_topic_candidate("personal", "Test Topic A", "kg_entity", None, 5.0)
        await db.insert_topic_candidate("personal", "Test Topic B", "rss", "https://x", 1.0)

        result = await mcp_server.list_topic_candidates("personal", limit=10)
        assert len(result) == 2
        topics = {r["topic"] for r in result}
        assert topics == {"Test Topic A", "Test Topic B"}
        sources = {r["source"] for r in result}
        assert sources == {"kg_entity", "rss"}

    @pytest.mark.asyncio
    async def test_get_article_versions_mcp_wraps_db(self):
        from app import db
        from app import mcp_server

        await db.init_db()
        await db.save_article_version(
            "personal", "tokio", "version 1 body", change_type="created",
        )
        await db.save_article_version(
            "personal", "tokio", "version 2 longer body content here",
            change_type="updated",
        )

        result = await mcp_server.get_article_versions("personal", "tokio")
        assert len(result) == 2
        # Newest first
        assert result[0]["change_type"] == "updated"
        # word_count is computed
        assert result[0]["word_count"] > 0


# ===========================================================================
# Source-review UI route
# ===========================================================================


class TestSourceReviewRoute:
    """The view_source_review handler should resolve a job to a template
    context with sources grouped for rendering."""

    @pytest.mark.asyncio
    async def test_handler_returns_404_for_missing_job(self):
        from fastapi import HTTPException
        from app import main as main_mod
        from app import db

        await db.init_db()
        with pytest.raises(HTTPException) as exc_info:
            await main_mod.view_source_review(request=None, job_id=99999)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_handler_collects_sources(self, tmp_path, monkeypatch):
        from app import main as main_mod
        from app import db

        # Use a temp KB so render() doesn't blow up on missing dirs
        import app.config as _config, app.storage as _storage
        kb_dir = tmp_path / "personal"
        (kb_dir / "wiki").mkdir(parents=True)
        _config.KB_DIRS.clear()
        _config.KB_DIRS["personal"] = kb_dir
        _storage._default = _storage.LocalStorage()

        await db.init_db()
        job_id = await db.create_job("Test review topic")
        await db.update_job(job_id, status="awaiting_review")
        await db.save_sources(
            job_id,
            [
                {"url": "https://a.example/1", "title": "A", "content": "snippet a", "tier": 1},
                {"url": "https://b.example/2", "title": "B", "content": "snippet b", "tier": 2},
            ],
            round_num=1,
        )

        # Call the handler with a fake request — render() needs the
        # template, so we patch it out and just check the context.
        from unittest.mock import patch
        with patch.object(main_mod, "render") as mock_render:
            mock_render.return_value = "ok"
            result = await main_mod.view_source_review(request=None, job_id=job_id)

        mock_render.assert_called_once()
        args, kwargs = mock_render.call_args
        assert args[0] == "source_review.html"
        assert kwargs["job"]["id"] == job_id
        assert len(kwargs["sources"]) == 2
        assert "personal" in kwargs["kb_names"]


# ===========================================================================
# Audio transcription worker task
# ===========================================================================


class TestMediaAudioTask:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self):
        from app.worker import media_audio_task
        result = await media_audio_task({}, "", title="x")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_failed_transcription_short_circuits(self):
        from app import worker
        with patch.object(
            worker, "media_audio_task", wraps=worker.media_audio_task,
        ):
            with patch("app.transcribe.transcribe_audio_url", AsyncMock(return_value="")):
                result = await worker.media_audio_task(
                    {}, "https://example.com/audio.mp3", title="Test",
                )
        assert "error" in result
        assert "transcription" in result["error"].lower()
