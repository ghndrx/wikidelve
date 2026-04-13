"""Unit tests for app.github_monitor."""

import base64
from unittest.mock import AsyncMock, patch

import pytest

from app import github_monitor


class TestTrackedLists:
    def test_tracked_repos_well_formed(self):
        for r in github_monitor.TRACKED_REPOS:
            assert "/" in r
            assert not r.startswith("/")
            assert not r.endswith("/")

    def test_own_repos_well_formed(self):
        for r in github_monitor.OWN_REPOS:
            assert "/" in r
            assert not r.startswith("/")


class TestGithubGet:
    @pytest.mark.asyncio
    async def test_success_returns_json(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 200
        fake_resp.json = lambda: {"ok": True}
        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            result = await github_monitor._github_get("/repos/x/y")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 404
        fake_resp.headers = {}
        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            result = await github_monitor._github_get("/missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        client = AsyncMock()
        client.get = AsyncMock(side_effect=RuntimeError("boom"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            result = await github_monitor._github_get("/boom")
        assert result is None


class TestGetLatestRelease:
    @pytest.mark.asyncio
    async def test_release_found(self):
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={
                "tag_name": "v1.0.0",
                "name": "Release 1.0",
                "body": "notes",
                "published_at": "2026-01-01",
                "html_url": "https://example/r",
            }),
        ):
            out = await github_monitor.get_latest_release("x/y")
        assert out["tag_name"] == "v1.0.0"
        assert out["published_at"] == "2026-01-01"

    @pytest.mark.asyncio
    async def test_falls_back_to_tags(self):
        calls = {"n": 0}

        async def fake_get(path, token=""):
            calls["n"] += 1
            if "releases/latest" in path:
                return None
            return [{"name": "v0.1.0"}]

        with patch("app.github_monitor._github_get", new=fake_get):
            out = await github_monitor.get_latest_release("x/y")
        assert out is not None
        assert out["tag_name"] == "v0.1.0"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_returns_none_when_no_tags(self):
        async def fake_get(path, token=""):
            return None

        with patch("app.github_monitor._github_get", new=fake_get):
            out = await github_monitor.get_latest_release("x/y")
        assert out is None


class TestGetRepoReadme:
    @pytest.mark.asyncio
    async def test_decodes_base64(self):
        content = ("# demo\n\n" + "word " * 20).encode("utf-8")
        encoded = base64.b64encode(content).decode("ascii")
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={"content": encoded}),
        ):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is not None
        assert out["repo"] == "x/y"
        assert "demo" in out["content"]
        assert out["word_count"] > 0

    @pytest.mark.asyncio
    async def test_short_readme_returns_none(self):
        encoded = base64.b64encode(b"tiny").decode("ascii")
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={"content": encoded}),
        ):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is None

    @pytest.mark.asyncio
    async def test_missing_readme_returns_none(self):
        with patch("app.github_monitor._github_get", new=AsyncMock(return_value=None)):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is None

    @pytest.mark.asyncio
    async def test_no_content_field_returns_none(self):
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={"content": None}),
        ):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is None

    @pytest.mark.asyncio
    async def test_content_truncated_to_10k(self):
        long_content = ("word " * 5000).encode("utf-8")
        encoded = base64.b64encode(long_content).decode("ascii")
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={"content": encoded}),
        ):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is not None
        assert len(out["content"]) <= 10000


class TestGithubGetRateLimit:
    @pytest.mark.asyncio
    async def test_403_returns_none(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 403
        fake_resp.headers = {"X-RateLimit-Reset": "12345"}
        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            result = await github_monitor._github_get("/repos/x/y")
        assert result is None

    @pytest.mark.asyncio
    async def test_token_included_in_headers(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 200
        fake_resp.json = lambda: {"ok": True}
        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            await github_monitor._github_get("/repos/x/y", token="my-token")
        call_kwargs = client.get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-token"


class TestCheckAllReleases:
    @pytest.mark.asyncio
    async def test_collects_releases(self):
        async def fake_release(repo, token=""):
            return {
                "tag_name": "v1.0.0", "name": "Release",
                "body": "notes", "published_at": "2026-01-01",
                "html_url": "https://example.com",
            }

        with patch("app.github_monitor.get_latest_release", new=fake_release), \
             patch("app.github_monitor.asyncio.sleep", new=AsyncMock()):
            results = await github_monitor.check_all_releases()
        assert len(results) == len(github_monitor.TRACKED_REPOS)
        assert all(r["tag"] == "v1.0.0" for r in results)

    @pytest.mark.asyncio
    async def test_skips_none_releases(self):
        async def fake_release(repo, token=""):
            if "tailscale" in repo:
                return None
            return {
                "tag_name": "v1.0.0", "name": "Release",
                "body": "notes", "published_at": "2026-01-01",
                "html_url": "https://example.com",
            }

        with patch("app.github_monitor.get_latest_release", new=fake_release), \
             patch("app.github_monitor.asyncio.sleep", new=AsyncMock()):
            results = await github_monitor.check_all_releases()
        assert len(results) == len(github_monitor.TRACKED_REPOS) - 1

    @pytest.mark.asyncio
    async def test_body_truncated_to_500(self):
        async def fake_release(repo, token=""):
            return {
                "tag_name": "v1.0.0", "name": "Release",
                "body": "x" * 3000, "published_at": "2026-01-01",
                "html_url": "https://example.com",
            }

        with patch("app.github_monitor.get_latest_release", new=fake_release), \
             patch("app.github_monitor.asyncio.sleep", new=AsyncMock()):
            results = await github_monitor.check_all_releases()
        for r in results:
            assert len(r["body"]) <= 500


class TestGetRepoChangelog:
    @pytest.mark.asyncio
    async def test_finds_changelog(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 200
        fake_resp.text = "# Changelog\n\n" + "- Entry\n" * 30

        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            out = await github_monitor.get_repo_changelog("x/y")
        assert out is not None
        assert out["repo"] == "x/y"
        assert "Changelog" in out["title"]

    @pytest.mark.asyncio
    async def test_no_changelog_returns_none(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 404
        fake_resp.text = ""

        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            out = await github_monitor.get_repo_changelog("x/y")
        assert out is None

    @pytest.mark.asyncio
    async def test_short_changelog_returns_none(self):
        fake_resp = AsyncMock()
        fake_resp.status_code = 200
        fake_resp.text = "tiny"

        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            out = await github_monitor.get_repo_changelog("x/y")
        assert out is None

    @pytest.mark.asyncio
    async def test_exception_during_fetch_returns_none(self):
        client = AsyncMock()
        client.get = AsyncMock(side_effect=RuntimeError("network error"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            out = await github_monitor.get_repo_changelog("x/y")
        assert out is None


class TestIndexOwnRepos:
    @pytest.mark.asyncio
    async def test_indexes_readmes_and_changelogs(self):
        async def fake_readme(repo, token=""):
            return {"repo": repo, "title": f"{repo} README", "content": "data", "url": "u", "word_count": 10}

        async def fake_changelog(repo, token=""):
            return None  # no changelogs

        with patch("app.github_monitor.get_repo_readme", new=fake_readme), \
             patch("app.github_monitor.get_repo_changelog", new=fake_changelog), \
             patch("app.github_monitor.asyncio.sleep", new=AsyncMock()):
            results = await github_monitor.index_own_repos()
        assert len(results) == len(github_monitor.OWN_REPOS)

    @pytest.mark.asyncio
    async def test_includes_changelogs_when_available(self):
        async def fake_readme(repo, token=""):
            return {"repo": repo, "title": f"{repo} README", "content": "data", "url": "u", "word_count": 10}

        async def fake_changelog(repo, token=""):
            return {"repo": repo, "title": f"{repo} Changelog", "content": "log", "url": "u", "word_count": 5}

        with patch("app.github_monitor.get_repo_readme", new=fake_readme), \
             patch("app.github_monitor.get_repo_changelog", new=fake_changelog), \
             patch("app.github_monitor.asyncio.sleep", new=AsyncMock()):
            results = await github_monitor.index_own_repos()
        # Both readme + changelog for each repo
        assert len(results) == len(github_monitor.OWN_REPOS) * 2
