"""Tests for the MCP server tools.

Skipped wholesale when the optional `fastmcp` dependency isn't
installed — app.mcp_server imports it at module load time and the
patches here wouldn't be resolvable otherwise.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

pytest.importorskip("fastmcp")

# Pre-import so `patch("app.mcp_server.X", ...)` paths resolve even
# when collection order doesn't naturally walk mcp_server first.
import app.mcp_server  # noqa: F401,E402


@pytest.fixture
def mock_search():
    with patch("app.mcp_server.search_fts", new_callable=AsyncMock) as mock:
        mock.return_value = [
            {"slug": "k8s", "title": "Kubernetes", "kb": "personal", "snippet": "test", "tags": ["k8s"]},
        ]
        yield mock


@pytest.fixture
def mock_articles():
    with patch("app.mcp_server.get_articles") as mock:
        mock.return_value = [
            {"slug": "k8s", "title": "Kubernetes", "tags": ["k8s"], "word_count": 500, "status": "published", "updated": "2026-04-01"},
            {"slug": "docker", "title": "Docker", "tags": ["docker"], "word_count": 300, "status": "draft", "updated": "2026-03-15"},
        ]
        yield mock


@pytest.fixture
def mock_get_article():
    with patch("app.mcp_server.get_article") as mock:
        mock.return_value = {
            "title": "Kubernetes",
            "slug": "k8s",
            "kb": "personal",
            "raw_markdown": "# Kubernetes\n\nContent here.",
            "summary": "K8s overview",
            "tags": ["k8s"],
            "status": "published",
            "updated": "2026-04-01",
            "word_count": 500,
        }
        yield mock


@pytest.fixture
def mock_kb_dirs(tmp_path):
    import app.config as _config, app.storage as _storage
    personal = tmp_path / "personal"
    work = tmp_path / "work"
    (personal / "wiki").mkdir(parents=True)
    (work / "wiki").mkdir(parents=True)
    prev = dict(_config.KB_DIRS)
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update({"personal": personal, "work": work})
    prev_backend = _storage._default
    _storage._default = _storage.LocalStorage()
    yield
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(prev)
    _storage._default = prev_backend


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_search):
        from app.mcp_server import search
        results = await search("kubernetes")
        assert len(results) == 1
        assert results[0]["title"] == "Kubernetes"
        assert results[0]["slug"] == "k8s"
        mock_search.assert_called_once_with("kubernetes", limit=10)

    @pytest.mark.asyncio
    async def test_search_custom_limit(self, mock_search):
        from app.mcp_server import search
        await search("test", limit=5)
        mock_search.assert_called_once_with("test", limit=5)

    @pytest.mark.asyncio
    async def test_search_limit_capped(self, mock_search):
        from app.mcp_server import search
        await search("test", limit=100)
        mock_search.assert_called_once_with("test", limit=30)

    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_search):
        mock_search.return_value = []
        from app.mcp_server import search
        results = await search("nonexistent")
        assert results == []


class TestGetArticleTool:
    @pytest.mark.asyncio
    async def test_get_existing_article(self, mock_get_article):
        from app.mcp_server import get_article_content
        result = await get_article_content("personal", "k8s")
        assert result["title"] == "Kubernetes"
        assert result["markdown"] == "# Kubernetes\n\nContent here."
        assert result["word_count"] == 500

    @pytest.mark.asyncio
    async def test_get_missing_article(self, mock_get_article):
        mock_get_article.return_value = None
        from app.mcp_server import get_article_content
        result = await get_article_content("personal", "missing")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_article_includes_metadata(self, mock_get_article):
        from app.mcp_server import get_article_content
        result = await get_article_content("personal", "k8s")
        assert result["tags"] == ["k8s"]
        assert result["status"] == "published"
        assert result["updated"] == "2026-04-01"


class TestListKBsTool:
    @pytest.mark.asyncio
    async def test_list_kbs(self, mock_kb_dirs, mock_articles):
        from app.mcp_server import list_knowledge_bases
        result = await list_knowledge_bases()
        assert len(result) == 2
        names = [kb["name"] for kb in result]
        assert "personal" in names
        assert "work" in names

    @pytest.mark.asyncio
    async def test_list_kbs_includes_counts(self, mock_kb_dirs, mock_articles):
        from app.mcp_server import list_knowledge_bases
        result = await list_knowledge_bases()
        personal = next(kb for kb in result if kb["name"] == "personal")
        assert personal["articles"] == 2
        assert personal["words"] == 800


class TestListArticlesTool:
    @pytest.mark.asyncio
    async def test_list_articles(self, mock_articles):
        from app.mcp_server import list_articles
        result = await list_articles("personal")
        assert len(result) == 2
        assert result[0]["slug"] == "k8s"
        assert result[1]["slug"] == "docker"

    @pytest.mark.asyncio
    async def test_list_articles_includes_metadata(self, mock_articles):
        from app.mcp_server import list_articles
        result = await list_articles("personal")
        assert result[0]["word_count"] == 500
        assert result[0]["status"] == "published"


class TestResearchTopicTool:
    @pytest.mark.asyncio
    async def test_research_creates_job(self):
        with patch("app.db.check_cooldown", new_callable=AsyncMock, return_value=None), \
             patch("app.db.create_job", new_callable=AsyncMock, return_value=42):
            from app.mcp_server import research_topic
            result = await research_topic("kubernetes best practices for production")
            assert result["status"] == "queued"
            assert result["job_id"] == 42

    @pytest.mark.asyncio
    async def test_research_short_topic_rejected(self):
        from app.mcp_server import research_topic
        result = await research_topic("short")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_research_cooldown(self):
        with patch("app.db.check_cooldown", new_callable=AsyncMock, return_value={"id": 99}):
            from app.mcp_server import research_topic
            result = await research_topic("already researched topic here")
            assert result["status"] == "cooldown"
            assert result["existing_job_id"] == 99

    @pytest.mark.asyncio
    async def test_research_empty_topic(self):
        from app.mcp_server import research_topic
        result = await research_topic("")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_research_whitespace_topic(self):
        from app.mcp_server import research_topic
        result = await research_topic("   ")
        assert "error" in result
