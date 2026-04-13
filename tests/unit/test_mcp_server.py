"""Tests for app.mcp_server — MCP tool handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ──────────────────────────────────────────────────────────────────────────────
# search tool
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpSearch:
    @patch("app.mcp_server.search_fts", new_callable=AsyncMock)
    async def test_basic_search(self, mock_fts):
        from app.mcp_server import search

        mock_fts.return_value = [
            {"title": "Python Basics", "slug": "python-basics", "kb": "personal",
             "snippet": "Intro to Python", "tags": ["python"]},
        ]
        results = await search("python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Basics"
        assert results[0]["slug"] == "python-basics"
        mock_fts.assert_awaited_once_with("python", limit=10)

    @patch("app.mcp_server.search_fts", new_callable=AsyncMock)
    async def test_search_limit_capped_at_30(self, mock_fts):
        from app.mcp_server import search

        mock_fts.return_value = []
        await search("query", limit=100)
        mock_fts.assert_awaited_once_with("query", limit=30)

    @patch("app.mcp_server.search_fts", new_callable=AsyncMock)
    async def test_search_empty_results(self, mock_fts):
        from app.mcp_server import search

        mock_fts.return_value = []
        results = await search("nonexistent")
        assert results == []

    @patch("app.mcp_server.search_fts", new_callable=AsyncMock)
    async def test_search_uses_fallback_fields(self, mock_fts):
        """When title/snippet not present, falls back to slug/summary."""
        from app.mcp_server import search

        mock_fts.return_value = [
            {"slug": "my-article", "summary": "A summary"},
        ]
        results = await search("test")
        assert results[0]["title"] == "my-article"
        assert results[0]["snippet"] == "A summary"


# ──────────────────────────────────────────────────────────────────────────────
# get_article_content tool
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpGetArticleContent:
    @patch("app.mcp_server.get_article")
    async def test_article_found(self, mock_get):
        from app.mcp_server import get_article_content

        mock_get.return_value = {
            "title": "Docker Guide",
            "raw_markdown": "# Docker\nContent here",
            "summary": "Docker overview",
            "tags": ["docker"],
            "status": "published",
            "updated": "2026-01-01",
            "word_count": 150,
        }
        result = await get_article_content("personal", "docker-guide")
        assert result["title"] == "Docker Guide"
        assert result["markdown"] == "# Docker\nContent here"
        assert result["word_count"] == 150

    @patch("app.mcp_server.get_article")
    async def test_article_not_found(self, mock_get):
        from app.mcp_server import get_article_content

        mock_get.return_value = None
        result = await get_article_content("personal", "nonexistent")
        assert "error" in result
        assert "not found" in result["error"].lower()

    @patch("app.mcp_server.get_article")
    async def test_article_with_missing_fields(self, mock_get):
        """Fields not present in article dict get defaults."""
        from app.mcp_server import get_article_content

        mock_get.return_value = {"title": "Minimal"}
        result = await get_article_content("personal", "minimal")
        assert result["title"] == "Minimal"
        assert result["markdown"] == ""
        assert result["tags"] == []
        assert result["word_count"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# list_knowledge_bases tool
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpListKBs:
    @patch("app.mcp_server.get_articles")
    @patch("app.mcp_server.storage.list_kbs")
    async def test_list_kbs(self, mock_list, mock_articles):
        from app.mcp_server import list_knowledge_bases

        mock_list.return_value = ["personal", "work"]
        mock_articles.side_effect = [
            [{"word_count": 100}, {"word_count": 200}],
            [{"word_count": 50}],
        ]
        result = await list_knowledge_bases()
        assert len(result) == 2
        assert result[0]["name"] == "personal"
        assert result[0]["articles"] == 2
        assert result[0]["words"] == 300
        assert result[1]["articles"] == 1

    @patch("app.mcp_server.get_articles")
    @patch("app.mcp_server.storage.list_kbs")
    async def test_list_kbs_empty(self, mock_list, mock_articles):
        from app.mcp_server import list_knowledge_bases

        mock_list.return_value = []
        result = await list_knowledge_bases()
        assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# list_articles tool
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpListArticles:
    @patch("app.mcp_server.get_articles")
    async def test_list_articles(self, mock_get):
        from app.mcp_server import list_articles

        mock_get.return_value = [
            {"slug": "k8s", "title": "Kubernetes", "tags": ["k8s"], "word_count": 500,
             "status": "published", "updated": "2026-01-01"},
        ]
        result = await list_articles("personal")
        assert len(result) == 1
        assert result[0]["slug"] == "k8s"
        assert result[0]["word_count"] == 500


# ──────────────────────────────────────────────────────────────────────────────
# research_topic tool (lines 93-128)
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpResearchTopic:
    @patch("app.db.create_job", new_callable=AsyncMock, return_value=42)
    @patch("app.db.check_cooldown", new_callable=AsyncMock, return_value=None)
    async def test_successful_enqueue(self, mock_cooldown, mock_create):
        from app.mcp_server import research_topic

        result = await research_topic("A sufficiently long research topic")
        assert result["status"] == "queued"
        assert result["job_id"] == 42

    @patch("app.db.check_cooldown", new_callable=AsyncMock)
    async def test_topic_too_short(self, mock_cooldown):
        from app.mcp_server import research_topic

        result = await research_topic("short")
        assert "error" in result

    @patch("app.db.check_cooldown", new_callable=AsyncMock)
    async def test_empty_topic(self, mock_cooldown):
        from app.mcp_server import research_topic

        result = await research_topic("")
        assert "error" in result

    @patch("app.db.check_cooldown", new_callable=AsyncMock)
    async def test_cooldown_active(self, mock_cooldown):
        from app.mcp_server import research_topic

        mock_cooldown.return_value = {"id": 99}
        result = await research_topic("A sufficiently long research topic")
        assert result["status"] == "cooldown"
        assert result["existing_job_id"] == 99


# ──────────────────────────────────────────────────────────────────────────────
# chat_ask tool (lines 150-163)
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpChatAsk:
    @patch("app.llm.llm_chat", new_callable=AsyncMock, return_value="Here is the answer")
    @patch("app.chat.build_chat_prompt", return_value=("system msg", "user msg"))
    @patch("app.chat.retrieve_context", new_callable=AsyncMock)
    async def test_successful_chat(self, mock_ctx, mock_prompt, mock_llm):
        from app.mcp_server import chat_ask

        mock_ctx.return_value = [
            {"title": "Source 1", "url": "http://a.com", "score": 0.9},
        ]
        result = await chat_ask("What is Docker?")
        assert result["answer"] == "Here is the answer"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Source 1"

    async def test_empty_question(self):
        from app.mcp_server import chat_ask

        result = await chat_ask("")
        assert "error" in result

    @patch("app.llm.llm_chat", new_callable=AsyncMock, side_effect=Exception("API timeout"))
    @patch("app.chat.build_chat_prompt", return_value=("sys", "usr"))
    @patch("app.chat.retrieve_context", new_callable=AsyncMock, return_value=[])
    async def test_llm_failure(self, mock_ctx, mock_prompt, mock_llm):
        from app.mcp_server import chat_ask

        result = await chat_ask("What is Docker?")
        assert "error" in result
        assert "LLM call failed" in result["error"]


# ──────────────────────────────────────────────────────────────────────────────
# get_graph_neighbors tool (lines 188-195)
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpGetGraphNeighbors:
    @patch("app.knowledge_graph.get_related_by_graph", new_callable=AsyncMock)
    async def test_successful_graph_query(self, mock_graph):
        from app.mcp_server import get_graph_neighbors

        mock_graph.return_value = [
            {"slug": "related-1", "kb": "personal", "score": 0.8, "hop": 1, "connections": ["entity-a"]},
        ]
        result = await get_graph_neighbors("personal", "my-article")
        assert len(result) == 1
        assert result[0]["slug"] == "related-1"

    @patch("app.knowledge_graph.get_related_by_graph", new_callable=AsyncMock, side_effect=Exception("graph error"))
    async def test_graph_exception(self, mock_graph):
        from app.mcp_server import get_graph_neighbors

        result = await get_graph_neighbors("personal", "my-article")
        assert len(result) == 1
        assert "error" in result[0]


# ──────────────────────────────────────────────────────────────────────────────
# enqueue_auto_discovery tool (lines 217-239)
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpEnqueueAutoDiscovery:
    @patch("app.mcp_server.storage.list_kbs", return_value=["personal"])
    @patch("app.auto_discovery.run_discovery_for_kb", new_callable=AsyncMock, return_value={"found": 5})
    async def test_discovery_runs(self, mock_discovery, mock_list):
        from app.mcp_server import enqueue_auto_discovery

        # The enqueue path tries to import arq + connect to Redis, which will
        # fail in test — that's fine, it falls through to the except branch.
        result = await enqueue_auto_discovery("personal")

        assert result["discovery"] == {"found": 5}
        assert "error" in result["enqueue"] or "skipped" in result["enqueue"]

    @patch("app.mcp_server.storage.list_kbs", return_value=["personal"])
    async def test_unknown_kb(self, mock_list):
        from app.mcp_server import enqueue_auto_discovery

        result = await enqueue_auto_discovery("nonexistent")
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────────────
# list_topic_candidates tool
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpListTopicCandidates:
    @patch("app.db.get_pending_candidates", new_callable=AsyncMock)
    async def test_list_candidates(self, mock_pending):
        from app.mcp_server import list_topic_candidates

        mock_pending.return_value = [
            {"id": 1, "topic": "New Topic", "source": "kg_entity",
             "source_ref": "ref", "score": 0.9, "created_at": "2026-01-01"},
        ]
        result = await list_topic_candidates("personal", limit=10)
        assert len(result) == 1
        assert result[0]["topic"] == "New Topic"


# ──────────────────────────────────────────────────────────────────────────────
# get_article_versions tool
# ──────────────────────────────────────────────────────────────────────────────


class TestMcpGetArticleVersions:
    @patch("app.db.get_article_versions", new_callable=AsyncMock)
    async def test_list_versions(self, mock_versions):
        from app.mcp_server import get_article_versions

        mock_versions.return_value = [
            {"id": 1, "created_at": "2026-01-01", "change_type": "create",
             "content_hash": "abc123", "full_content": "word " * 50,
             "job_id": 5, "note": "initial"},
        ]
        result = await get_article_versions("personal", "my-article")
        assert len(result) == 1
        assert result[0]["change_type"] == "create"
        assert result[0]["word_count"] == 50

    @patch("app.db.get_article_versions", new_callable=AsyncMock, return_value=[])
    async def test_no_versions(self, mock_versions):
        from app.mcp_server import get_article_versions

        result = await get_article_versions("personal", "nonexistent")
        assert result == []
