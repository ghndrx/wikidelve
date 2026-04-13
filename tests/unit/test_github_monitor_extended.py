"""Extended unit tests for app/github_monitor.py — targeting uncovered lines.

Covers:
  - find_new_releases: full pipeline with KB version comparison (lines 154-185)
  - get_repo_readme: base64 decode exception (lines 202-203)
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import github_monitor


# ===========================================================================
# find_new_releases (lines 154-185)
# ===========================================================================


class TestFindNewReleases:
    @pytest.mark.asyncio
    async def test_finds_releases_not_in_kb(self):
        """Releases whose version isn't mentioned in KB articles are returned."""
        # Mock get_articles to return one article about tailscale
        fake_articles = [{"slug": "tailscale-vpn", "title": "Tailscale VPN"}]

        # That article mentions v1.50.0 but not v1.52.0
        fake_article_text = (
            "---\ntitle: Tailscale VPN\n---\n\n"
            "Tailscale v1.50.0 was released with new features."
        )

        fake_releases = [
            {
                "repo": "tailscale/tailscale",
                "tag": "v1.52.0",
                "name": "Release 1.52.0",
                "body": "bug fixes",
                "published": "2026-04-01",
                "url": "https://github.com/tailscale/tailscale/releases/v1.52.0",
            },
        ]

        with patch("app.wiki.get_articles", return_value=fake_articles), \
             patch("app.wiki.read_article_text", return_value=fake_article_text), \
             patch("app.wiki.parse_frontmatter", return_value=({}, fake_article_text.split("---\n\n")[-1])), \
             patch("app.github_monitor.check_all_releases", AsyncMock(return_value=fake_releases)):
            result = await github_monitor.find_new_releases()

        assert len(result) >= 1
        assert result[0]["tag"] == "v1.52.0"

    @pytest.mark.asyncio
    async def test_skips_releases_already_in_kb(self):
        """Releases whose version IS mentioned in a matching KB article are skipped."""
        fake_articles = [{"slug": "tailscale-vpn", "title": "Tailscale VPN"}]

        fake_article_text = (
            "---\ntitle: Tailscale VPN\n---\n\n"
            "Tailscale v1.52.0 was released with new features."
        )

        fake_releases = [
            {
                "repo": "tailscale/tailscale",
                "tag": "v1.52.0",
                "name": "Release 1.52.0",
                "body": "bug fixes",
                "published": "2026-04-01",
                "url": "https://github.com/tailscale/tailscale/releases/v1.52.0",
            },
        ]

        with patch("app.wiki.get_articles", return_value=fake_articles), \
             patch("app.wiki.read_article_text", return_value=fake_article_text), \
             patch("app.wiki.parse_frontmatter", return_value=({}, "Tailscale v1.52.0 was released.")), \
             patch("app.github_monitor.check_all_releases", AsyncMock(return_value=fake_releases)):
            result = await github_monitor.find_new_releases()

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_empty_tag_skipped(self):
        """Releases with empty tag should be skipped."""
        fake_articles = [{"slug": "art", "title": "Art"}]
        fake_article_text = "---\ntitle: Art\n---\n\nSome content."

        fake_releases = [
            {
                "repo": "some/repo",
                "tag": "",
                "name": "Unnamed",
                "body": "",
                "published": "",
                "url": "",
            },
        ]

        with patch("app.wiki.get_articles", return_value=fake_articles), \
             patch("app.wiki.read_article_text", return_value=fake_article_text), \
             patch("app.wiki.parse_frontmatter", return_value=({}, "Some content.")), \
             patch("app.github_monitor.check_all_releases", AsyncMock(return_value=fake_releases)):
            result = await github_monitor.find_new_releases()

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_no_articles_all_releases_new(self):
        """When KB is empty, all releases with non-empty tags are returned."""
        fake_releases = [
            {
                "repo": "some/repo",
                "tag": "v2.0.0",
                "name": "v2",
                "body": "notes",
                "published": "2026-01-01",
                "url": "https://example.com",
            },
        ]

        with patch("app.wiki.get_articles", return_value=[]), \
             patch("app.github_monitor.check_all_releases", AsyncMock(return_value=fake_releases)):
            result = await github_monitor.find_new_releases()

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_read_article_text_returns_none_skipped(self):
        """Articles where read_article_text returns None are gracefully skipped."""
        fake_articles = [{"slug": "missing-content", "title": "Missing"}]
        fake_releases = [
            {
                "repo": "some/repo",
                "tag": "v1.0.0",
                "name": "v1",
                "body": "",
                "published": "",
                "url": "",
            },
        ]

        with patch("app.wiki.get_articles", return_value=fake_articles), \
             patch("app.wiki.read_article_text", return_value=None), \
             patch("app.github_monitor.check_all_releases", AsyncMock(return_value=fake_releases)):
            result = await github_monitor.find_new_releases()

        # With no article text matched, all releases with tags should be returned
        assert len(result) == 1


# ===========================================================================
# get_repo_readme — base64 decode failure (lines 202-203)
# ===========================================================================


class TestGetRepoReadmeExtended:
    @pytest.mark.asyncio
    async def test_invalid_base64_returns_none(self):
        """If base64 decode fails, should return None."""
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={"content": "!!!not-valid-base64!!!"}),
        ):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is None

    @pytest.mark.asyncio
    async def test_empty_content_field_returns_none(self):
        with patch(
            "app.github_monitor._github_get",
            new=AsyncMock(return_value={"content": ""}),
        ):
            out = await github_monitor.get_repo_readme("x/y")
        assert out is None


# ===========================================================================
# get_repo_changelog — content truncated to 8000
# ===========================================================================


class TestGetRepoChangelogExtended:
    @pytest.mark.asyncio
    async def test_content_truncated_to_8000(self):
        long_text = "x" * 20000
        fake_resp = AsyncMock()
        fake_resp.status_code = 200
        fake_resp.text = long_text

        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch("app.github_monitor.httpx.AsyncClient", return_value=client):
            out = await github_monitor.get_repo_changelog("x/y")
        assert out is not None
        assert len(out["content"]) <= 8000


# ===========================================================================
# index_own_repos — skip repos where readme is None
# ===========================================================================


class TestIndexOwnReposExtended:
    @pytest.mark.asyncio
    async def test_skips_repos_without_readme(self):
        """Repos where get_repo_readme returns None should be skipped."""
        async def fake_readme(repo, token=""):
            if "hearth" in repo:
                return None
            return {"repo": repo, "title": f"{repo} README", "content": "c", "url": "u", "word_count": 10}

        async def fake_changelog(repo, token=""):
            return None

        with patch("app.github_monitor.get_repo_readme", new=fake_readme), \
             patch("app.github_monitor.get_repo_changelog", new=fake_changelog), \
             patch("app.github_monitor.asyncio.sleep", new=AsyncMock()):
            results = await github_monitor.index_own_repos()

        # hearth, hearth-mobile, hearth-desktop should be skipped
        hearth_repos = [r for r in github_monitor.OWN_REPOS if "hearth" in r]
        expected = len(github_monitor.OWN_REPOS) - len(hearth_repos)
        assert len(results) == expected
