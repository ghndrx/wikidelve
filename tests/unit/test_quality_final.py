"""Final coverage tests for app.quality -- covers _serper_search, freshness_audit,
freshness_audit with auto_update, run_freshness_audit batch, and run_quality_pass
edge cases.
"""

import json
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.quality import (
    _quality_cache,
    _serper_search,
    fact_check_article,
    freshness_audit,
    run_freshness_audit,
    run_quality_pass,
    invalidate_quality_cache,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    _quality_cache.clear()
    yield
    _quality_cache.clear()


# ---------------------------------------------------------------------------
# _serper_search
# ---------------------------------------------------------------------------


class TestSerperSearch:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_api_key(self):
        with patch("app.quality.SERPER_API_KEY", ""):
            result = await _serper_search("test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_results_on_success(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "organic": [
                {"title": "Result 1", "snippet": "Snippet 1", "link": "https://example.com/1"},
                {"title": "Result 2", "snippet": "Snippet 2", "link": "https://example.com/2"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.quality.SERPER_API_KEY", "fake-key"):
            with patch("app.quality.httpx.AsyncClient", return_value=mock_client):
                result = await _serper_search("test query", num=2)

        assert len(result) == 2
        assert result[0]["title"] == "Result 1"
        assert result[0]["url"] == "https://example.com/1"

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("network error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.quality.SERPER_API_KEY", "fake-key"):
            with patch("app.quality.httpx.AsyncClient", return_value=mock_client):
                result = await _serper_search("test query")
        assert result == []


# ---------------------------------------------------------------------------
# freshness_audit
# ---------------------------------------------------------------------------


class TestFreshnessAudit:
    @pytest.mark.asyncio
    async def test_returns_fresh_when_no_outdated(self):
        """When fact_check finds no outdated claims, status=fresh."""
        with patch("app.quality.fact_check_article", new_callable=AsyncMock) as mock_fc:
            mock_fc.return_value = {
                "slug": "test-article",
                "title": "Test Article",
                "claims_checked": 3,
                "verified": 3,
                "outdated": 0,
                "results": [
                    {"claim": "c1", "status": "verified"},
                    {"claim": "c2", "status": "verified"},
                    {"claim": "c3", "status": "verified"},
                ],
            }
            result = await freshness_audit("personal", "test-article")

        assert result["status"] == "fresh"
        assert result["outdated"] == 0

    @pytest.mark.asyncio
    async def test_returns_stale_when_auto_update_false(self):
        """Outdated claims + auto_update=False returns stale without writing."""
        with patch("app.quality.fact_check_article", new_callable=AsyncMock) as mock_fc:
            mock_fc.return_value = {
                "slug": "test-article",
                "title": "Test Article",
                "claims_checked": 2,
                "results": [
                    {"claim": "v1.0 released", "status": "outdated",
                     "correction": "v2.0 released"},
                ],
            }
            result = await freshness_audit("personal", "test-article", auto_update=False)

        assert result["status"] == "stale"
        assert result["outdated"] == 1
        assert result["updates_applied"] == 0

    @pytest.mark.asyncio
    async def test_auto_update_applies_corrections(self):
        """Outdated claims + auto_update=True triggers LLM rewrite and write."""
        with patch("app.quality.fact_check_article", new_callable=AsyncMock) as mock_fc:
            mock_fc.return_value = {
                "slug": "test-article",
                "title": "Test Article",
                "claims_checked": 1,
                "results": [
                    {"claim": "Python 3.11", "status": "outdated",
                     "correction": "Python 3.13", "sources": ["https://x.com"]},
                ],
            }
            with patch("app.quality.read_article_text") as mock_read:
                mock_read.return_value = '---\ntitle: "Test"\n---\n\nPython 3.11 is the latest.'
                with patch("app.quality._chat", new_callable=AsyncMock) as mock_chat:
                    mock_chat.return_value = "Python 3.13 is the latest version available. " * 5 + "[Updated 2026-04-12]"
                    with patch("app.quality.write_article_text") as mock_write:
                        result = await freshness_audit("personal", "test-article", auto_update=True)

        assert result["status"] == "updated"
        assert result["updates_applied"] == 1
        mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_update_returns_error_on_short_llm_response(self):
        """When the LLM returns insufficient content, return error."""
        with patch("app.quality.fact_check_article", new_callable=AsyncMock) as mock_fc:
            mock_fc.return_value = {
                "slug": "test-article",
                "title": "Test Article",
                "claims_checked": 1,
                "results": [
                    {"claim": "old fact", "status": "outdated",
                     "correction": "new fact", "sources": ["https://x.com"]},
                ],
            }
            with patch("app.quality.read_article_text") as mock_read:
                mock_read.return_value = '---\ntitle: "Test"\n---\n\nOld fact.'
                with patch("app.quality._chat", new_callable=AsyncMock) as mock_chat:
                    mock_chat.return_value = "short"
                    result = await freshness_audit("personal", "test-article")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_fact_check_error_propagates(self):
        with patch("app.quality.fact_check_article", new_callable=AsyncMock) as mock_fc:
            mock_fc.return_value = {"error": "Article not found: x"}
            result = await freshness_audit("personal", "x")
        assert "error" in result


# ---------------------------------------------------------------------------
# run_freshness_audit (batch)
# ---------------------------------------------------------------------------


class TestRunFreshnessAudit:
    @pytest.mark.asyncio
    async def test_skips_articles_without_time_sensitive_content(self):
        """Articles with no versions/prices/dates score 0 and are skipped."""
        with patch("app.quality.get_articles") as mock_ga:
            mock_ga.return_value = [{"slug": "plain-article", "title": "Plain"}]
            with patch("app.quality.read_article_text") as mock_read:
                mock_read.return_value = '---\ntitle: "Plain"\n---\n\nJust text.'
                with patch("app.quality.parse_frontmatter") as mock_pf:
                    mock_pf.return_value = ({"title": "Plain"}, "Just text.")
                    result = await run_freshness_audit("personal")

        assert result["articles_audited"] == 0

    @pytest.mark.asyncio
    async def test_audits_articles_with_versions(self):
        with patch("app.quality.get_articles") as mock_ga:
            mock_ga.return_value = [{"slug": "versioned", "title": "Versioned"}]
            with patch("app.quality.read_article_text") as mock_read:
                mock_read.return_value = '---\ntitle: "V"\n---\n\nUses v1.9.3 and costs $5.'
                with patch("app.quality.parse_frontmatter") as mock_pf:
                    mock_pf.return_value = ({"title": "V"}, "Uses v1.9.3 and costs $5.")
                    with patch("app.quality.freshness_audit", new_callable=AsyncMock) as mock_fa:
                        mock_fa.return_value = {
                            "slug": "versioned", "title": "Versioned",
                            "status": "fresh", "claims_checked": 2,
                            "outdated": 0, "updates_applied": 0,
                        }
                        result = await run_freshness_audit("personal", max_articles=5)

        assert result["articles_audited"] == 1

    @pytest.mark.asyncio
    async def test_counts_updates(self):
        with patch("app.quality.get_articles") as mock_ga:
            mock_ga.return_value = [{"slug": "stale", "title": "Stale"}]
            with patch("app.quality.read_article_text") as mock_read:
                mock_read.return_value = '---\ntitle: "S"\n---\n\nv2.0 costs $10.'
                with patch("app.quality.parse_frontmatter") as mock_pf:
                    mock_pf.return_value = ({"title": "S"}, "v2.0 costs $10.")
                    with patch("app.quality.freshness_audit", new_callable=AsyncMock) as mock_fa:
                        mock_fa.return_value = {
                            "slug": "stale", "title": "Stale",
                            "status": "updated", "claims_checked": 1,
                            "outdated": 1, "updates_applied": 1,
                        }
                        result = await run_freshness_audit("personal")

        assert result["articles_updated"] == 1
        assert result["total_corrections"] == 1

    @pytest.mark.asyncio
    async def test_handles_exception_in_audit(self):
        with patch("app.quality.get_articles") as mock_ga:
            mock_ga.return_value = [{"slug": "err", "title": "Err"}]
            with patch("app.quality.read_article_text") as mock_read:
                mock_read.return_value = '---\ntitle: "E"\n---\n\nv1.0 released 2025-01.'
                with patch("app.quality.parse_frontmatter") as mock_pf:
                    mock_pf.return_value = ({"title": "E"}, "v1.0 released 2025-01.")
                    with patch("app.quality.freshness_audit", new_callable=AsyncMock) as mock_fa:
                        mock_fa.side_effect = RuntimeError("boom")
                        result = await run_freshness_audit("personal")

        assert result["articles_audited"] == 0
        assert any("error" in d for d in result["details"])


# ---------------------------------------------------------------------------
# run_quality_pass edge cases
# ---------------------------------------------------------------------------


class TestRunQualityPass:
    @pytest.mark.asyncio
    async def test_quality_pass_no_shallow(self):
        """When no shallow articles exist, nothing is enriched."""
        with patch("app.quality.find_shallow_articles") as mock_fs:
            mock_fs.return_value = []
            result = await run_quality_pass("personal")
        assert result["shallow_found"] == 0
        assert result["articles_enriched"] == 0

    @pytest.mark.asyncio
    async def test_quality_pass_enrich_error(self):
        """When enrich raises, the error is logged in details."""
        with patch("app.quality.find_shallow_articles") as mock_fs:
            mock_fs.return_value = [{"slug": "bad", "title": "Bad", "issues": ["short"]}]
            with patch("app.quality.enrich_article", new_callable=AsyncMock) as mock_enrich:
                mock_enrich.side_effect = RuntimeError("LLM down")
                result = await run_quality_pass("personal", max_articles=1)
        assert result["articles_enriched"] == 0
        assert any("error" in d for d in result["details"])

    @pytest.mark.asyncio
    async def test_quality_pass_crosslink_failure(self):
        """When crosslink fails, enrichment still counts."""
        with patch("app.quality.find_shallow_articles") as mock_fs:
            mock_fs.return_value = [{"slug": "ok", "title": "Ok", "issues": ["few links"]}]
            with patch("app.quality.enrich_article", new_callable=AsyncMock) as mock_enrich:
                mock_enrich.return_value = {
                    "slug": "ok", "title": "Ok", "old_words": 100, "new_words": 500, "status": "enriched",
                }
                with patch("app.quality.add_crosslinks", new_callable=AsyncMock) as mock_cl:
                    mock_cl.side_effect = RuntimeError("crosslink boom")
                    result = await run_quality_pass("personal", max_articles=1)
        assert result["articles_enriched"] == 1
        assert result["articles_crosslinked"] == 0

    @pytest.mark.asyncio
    async def test_quality_pass_enrich_returns_error_dict(self):
        """When enrich returns an error dict, it goes into details but doesn't count."""
        with patch("app.quality.find_shallow_articles") as mock_fs:
            mock_fs.return_value = [{"slug": "x", "title": "X", "issues": ["short"]}]
            with patch("app.quality.enrich_article", new_callable=AsyncMock) as mock_enrich:
                mock_enrich.return_value = {"error": "LLM returned insufficient content"}
                result = await run_quality_pass("personal", max_articles=1)
        assert result["articles_enriched"] == 0
        assert len(result["details"]) == 1
