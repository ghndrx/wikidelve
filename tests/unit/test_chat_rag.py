"""Unit tests for ``app/chat.py`` — RAG retrieval, prompt assembly, tool dispatch.

The sibling ``tests/unit/test_chat.py`` covers the chat sessions/events
DB layer; this file covers the chat orchestration module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "chat_rag_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            yield


# ===========================================================================
# 1.4 — _estimate_tokens + chunking
# ===========================================================================


class TestEstimateTokens:
    def test_empty(self):
        from app.chat import _estimate_tokens
        assert _estimate_tokens("") == 0
        assert _estimate_tokens(None) == 0

    def test_basic_math(self):
        from app.chat import _estimate_tokens
        assert _estimate_tokens("a") == 0
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("a" * 400) == 100


class TestChunkForRetrieval:
    def test_short_body_returns_single_chunk(self):
        from app.chat import _chunk_for_retrieval
        body = "Short paragraph here."
        assert _chunk_for_retrieval(body, max_chars=2000) == [body]

    def test_paragraph_split(self):
        from app.chat import _chunk_for_retrieval
        # Two paragraphs, each ~1500 chars → should split into 2 chunks
        para = "x" * 1500
        body = para + "\n\n" + para
        chunks = _chunk_for_retrieval(body, max_chars=2000)
        assert len(chunks) == 2

    def test_giant_paragraph_hard_splits(self):
        from app.chat import _chunk_for_retrieval
        # Single paragraph 5000 chars → hard-splits into 3 content chunks.
        # overlap_chars=0 keeps the strict max_chars cap.
        body = "y" * 5000
        chunks = _chunk_for_retrieval(body, max_chars=2000, overlap_chars=0)
        assert len(chunks) == 3
        assert all(len(c) <= 2000 for c in chunks)

    def test_empty_body(self):
        from app.chat import _chunk_for_retrieval
        assert _chunk_for_retrieval("") == []
        assert _chunk_for_retrieval(None) == []


# ===========================================================================
# 1.6 — [GAP] sentinel parser
# ===========================================================================


class TestParseGapTopics:
    def test_no_gaps(self):
        from app.chat import parse_gap_topics
        assert parse_gap_topics("Just a regular answer.") == []

    def test_single_gap(self):
        from app.chat import parse_gap_topics
        text = "Here is the answer.\n[GAP] Tokio runtime internals\n"
        assert parse_gap_topics(text) == ["Tokio runtime internals"]

    def test_multiple_gaps(self):
        from app.chat import parse_gap_topics
        text = (
            "Some answer.\n"
            "[GAP] Async cancellation patterns\n"
            "More text.\n"
            "[GAP] Tokio scheduler tuning\n"
        )
        assert parse_gap_topics(text) == [
            "Async cancellation patterns",
            "Tokio scheduler tuning",
        ]

    def test_dedupes(self):
        from app.chat import parse_gap_topics
        text = "[GAP] same topic\n[GAP] same topic\n"
        assert parse_gap_topics(text) == ["same topic"]

    def test_strips_whitespace(self):
        from app.chat import parse_gap_topics
        text = "[GAP]    spaced topic    \n"
        assert parse_gap_topics(text) == ["spaced topic"]

    def test_rejects_too_short(self):
        from app.chat import parse_gap_topics
        # < 5 chars → ignored as noise
        text = "[GAP] x\n"
        assert parse_gap_topics(text) == []


# ===========================================================================
# 1.1 — retrieve_context
# ===========================================================================


class TestRetrieveContext:
    @pytest.mark.asyncio
    async def test_empty_question_returns_empty(self):
        from app.chat import retrieve_context
        assert await retrieve_context("") == []
        assert await retrieve_context("   ") == []

    @pytest.mark.asyncio
    async def test_no_hybrid_results_returns_empty(self):
        from app import chat
        with patch.object(chat, "hybrid_search", AsyncMock(return_value=[])):
            assert await chat.retrieve_context("anything") == []

    @pytest.mark.asyncio
    async def test_orders_chunks_by_score(self):
        """Top-k chunks should be sorted by cosine similarity descending."""
        from app import chat

        fake_hits = [
            {"slug": "tokio", "kb": "personal", "rrf_score": 0.9},
            {"slug": "asyncio", "kb": "personal", "rrf_score": 0.7},
        ]

        def fake_get_article(kb, slug):
            return {
                "title": slug.title(),
                "raw_markdown": f"This is the body of {slug}. It has content.",
            }

        # Question vec aligns more with the second chunk
        question_vec = [1.0, 0.0]
        chunk_vecs = [
            [0.0, 1.0],  # tokio body — orthogonal → low score
            [1.0, 0.0],  # asyncio body — parallel → high score
        ]

        with patch.object(chat, "hybrid_search", AsyncMock(return_value=fake_hits)), \
             patch.object(chat, "get_article", side_effect=fake_get_article), \
             patch.object(chat, "embed_text", AsyncMock(return_value=question_vec)), \
             patch.object(chat, "embed_texts", AsyncMock(return_value=chunk_vecs)):
            results = await chat.retrieve_context("question", k=2)

        assert len(results) == 2
        # asyncio should come first (higher score)
        assert results[0]["slug"] == "asyncio"
        assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.asyncio
    async def test_skips_articles_that_404(self):
        from app import chat
        fake_hits = [
            {"slug": "missing", "kb": "personal", "rrf_score": 0.9},
            {"slug": "exists", "kb": "personal", "rrf_score": 0.8},
        ]

        def fake_get_article(kb, slug):
            if slug == "missing":
                return None
            return {"title": "Exists", "raw_markdown": "body content"}

        with patch.object(chat, "hybrid_search", AsyncMock(return_value=fake_hits)), \
             patch.object(chat, "get_article", side_effect=fake_get_article), \
             patch.object(chat, "embed_text", AsyncMock(return_value=[1.0])), \
             patch.object(chat, "embed_texts", AsyncMock(return_value=[[1.0]])):
            results = await chat.retrieve_context("q")

        assert len(results) == 1
        assert results[0]["slug"] == "exists"


# ===========================================================================
# 1.4 — get_recent_history
# ===========================================================================


class TestGetRecentHistory:
    @pytest.mark.asyncio
    async def test_empty_session_returns_empty(self):
        from app import chat
        with patch.object(chat.db, "get_chat_messages", AsyncMock(return_value=[])):
            assert await chat.get_recent_history("session-1") == []

    @pytest.mark.asyncio
    async def test_respects_token_budget(self):
        from app import chat
        # 4 messages, each ~2000 chars (~500 tokens). Budget=1000 → only fit 2
        msgs = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "y" * 2000},
            {"role": "user", "content": "z" * 2000},
            {"role": "assistant", "content": "w" * 2000},
        ]
        with patch.object(chat.db, "get_chat_messages", AsyncMock(return_value=msgs)):
            kept = await chat.get_recent_history("session-1", max_tokens=1000)
        # Should keep at most 2 messages from the newest end
        assert 1 <= len(kept) <= 2
        # Should be the newest ones
        assert kept[-1]["content"] == "w" * 2000

    @pytest.mark.asyncio
    async def test_returns_chronological_order(self):
        from app import chat
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        with patch.object(chat.db, "get_chat_messages", AsyncMock(return_value=msgs)):
            kept = await chat.get_recent_history("session-1", max_tokens=10000)
        assert [m["content"] for m in kept] == ["first", "second", "third"]


# ===========================================================================
# 1.1 — build_chat_prompt
# ===========================================================================


class TestBuildChatPrompt:
    def test_includes_question_passages_history(self):
        from app.chat import build_chat_prompt

        passages = [
            {
                "slug": "tokio",
                "kb": "personal",
                "title": "Tokio",
                "chunk_text": "Tokio is an async runtime.",
                "url": "/wiki/personal/tokio",
            }
        ]
        history = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ]
        question = "What is Tokio?"

        system, user = build_chat_prompt(question, passages, history)

        assert "research assistant" in system.lower()
        assert "[GAP]" in system
        assert "Earlier question" in user
        assert "Earlier answer" in user
        assert "Tokio is an async runtime" in user
        assert "/wiki/personal/tokio" in user
        assert "What is Tokio?" in user

    def test_handles_no_passages(self):
        from app.chat import build_chat_prompt
        system, user = build_chat_prompt("Question?", [], [])
        assert "no passages matched" in user
        assert "Question?" in user


# ===========================================================================
# 1.3 — Tool dispatcher
# ===========================================================================


class TestRunTool:
    @pytest.mark.asyncio
    async def test_search_kb_dispatches_to_hybrid_search(self):
        from app import chat
        with patch.object(chat, "hybrid_search", AsyncMock(return_value=[{"slug": "x"}])) as mock:
            result = await chat.run_tool("search_kb", {"query": "rust", "limit": 5})
        mock.assert_called_once_with("rust", kb_name=None, limit=5)
        assert result["results"] == [{"slug": "x"}]

    @pytest.mark.asyncio
    async def test_get_article_returns_full_markdown(self):
        from app import chat

        def fake_get(kb, slug):
            return {
                "title": "T",
                "summary": "S",
                "tags": ["a"],
                "raw_markdown": "body",
                "word_count": 2,
            }

        with patch.object(chat, "get_article", side_effect=fake_get):
            result = await chat.run_tool(
                "get_article", {"kb": "personal", "slug": "t"},
            )
        assert result["title"] == "T"
        assert result["raw_markdown"] == "body"
        assert result["tags"] == ["a"]

    @pytest.mark.asyncio
    async def test_get_article_handles_missing(self):
        from app import chat
        with patch.object(chat, "get_article", return_value=None):
            result = await chat.run_tool(
                "get_article", {"kb": "personal", "slug": "missing"},
            )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_find_related_dispatches(self):
        from app import chat
        with patch.object(chat, "find_related_article", return_value={"slug": "match"}) as mock:
            result = await chat.run_tool(
                "find_related", {"kb": "personal", "topic": "Tokio"},
            )
        mock.assert_called_once_with("personal", "Tokio")
        assert result["match"]["slug"] == "match"

    @pytest.mark.asyncio
    async def test_get_graph_neighbors_dispatches(self):
        from app import chat
        with patch.object(
            chat, "get_related_by_graph", AsyncMock(return_value=[{"slug": "n1"}])
        ) as mock:
            result = await chat.run_tool(
                "get_graph_neighbors",
                {"kb": "personal", "slug": "tokio", "depth": 3},
            )
        mock.assert_called_once_with("tokio", kb_name="personal", depth=3)
        assert result["neighbors"][0]["slug"] == "n1"

    @pytest.mark.asyncio
    async def test_enqueue_research_creates_job_and_enqueues(self):
        """Verify _queue_name is correctly passed — critical for arq routing."""
        from app import chat

        await chat.db.init_db()

        fake_redis = MagicMock()
        fake_redis.enqueue_job = AsyncMock(return_value=None)

        with patch.object(chat.db, "check_cooldown", AsyncMock(return_value=None)):
            result = await chat.run_tool(
                "enqueue_research",
                {"topic": "A new topic for research", "kb": "personal"},
                redis=fake_redis,
            )

        assert result["status"] == "queued"
        assert "job_id" in result
        # Verify arq enqueue call shape
        fake_redis.enqueue_job.assert_called_once()
        args, kwargs = fake_redis.enqueue_job.call_args
        assert args[0] == "research_task"
        assert args[1] == "A new topic for research"
        assert args[3] == "personal"
        assert kwargs.get("_queue_name") == "wikidelve"

    @pytest.mark.asyncio
    async def test_enqueue_research_respects_cooldown(self):
        from app import chat

        await chat.db.init_db()

        fake_redis = MagicMock()
        fake_redis.enqueue_job = AsyncMock()

        with patch.object(
            chat.db, "check_cooldown",
            AsyncMock(return_value={"id": 99}),
        ):
            result = await chat.run_tool(
                "enqueue_research",
                {"topic": "Already researched topic", "kb": "personal"},
                redis=fake_redis,
            )

        assert result["error"] == "cooldown"
        assert result["existing_job_id"] == 99
        fake_redis.enqueue_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_enqueue_research_requires_redis(self):
        from app import chat
        result = await chat.run_tool(
            "enqueue_research", {"topic": "x", "kb": "personal"}, redis=None,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        from app import chat
        result = await chat.run_tool("nonsense", {})
        assert "error" in result


# ===========================================================================
# CHAT_TOOLS shape sanity check
# ===========================================================================


class TestChatToolsShape:
    def test_all_tools_have_required_fields(self):
        from app.chat import CHAT_TOOLS
        for tool in CHAT_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert isinstance(tool["input_schema"], dict)
            assert tool["input_schema"].get("type") == "object"

    def test_all_tools_present(self):
        from app.chat import CHAT_TOOLS
        names = {t["name"] for t in CHAT_TOOLS}
        # 5 base tools + refine_research.
        assert names == {
            "search_kb",
            "get_article",
            "find_related",
            "get_graph_neighbors",
            "enqueue_research",
            "refine_research",
        }
