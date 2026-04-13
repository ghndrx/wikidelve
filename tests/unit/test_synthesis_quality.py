"""Unit tests for synthesis quality helpers.

Covers:
  - Adaptive per-tier source truncation (``_truncate_for_tier``,
    ``_format_sources_for_prompt``)
  - Self-citation lint (``_lint_citations``)
  - Per-source extraction pre-pass
  - Two-pass critique parser + Confidence Notes renderer
  - ``claims_json`` round-trip + ``worker._persist_critique_claims``
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "phase2_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            yield


# ===========================================================================
# 2.2 — Adaptive source truncation
# ===========================================================================


class TestTruncateForTier:
    def test_short_content_passthrough(self):
        from app.research import _truncate_for_tier
        assert _truncate_for_tier("short", 1) == "short"
        assert _truncate_for_tier("short", 3) == "short"

    def test_tier_1_gets_8000_chars(self):
        from app.research import _truncate_for_tier
        big = "a" * 10000
        out = _truncate_for_tier(big, 1)
        # 8000 cap + " ..." marker
        assert len(out) <= 8010
        assert out.endswith("...")

    def test_tier_2_gets_3000_chars(self):
        from app.research import _truncate_for_tier
        big = "b" * 5000
        out = _truncate_for_tier(big, 2)
        assert len(out) <= 3010
        assert out.endswith("...")

    def test_tier_3_gets_1000_chars(self):
        from app.research import _truncate_for_tier
        big = "c" * 5000
        out = _truncate_for_tier(big, 3)
        assert len(out) <= 1010
        assert out.endswith("...")

    def test_unknown_tier_defaults_to_smallest(self):
        from app.research import _truncate_for_tier
        assert len(_truncate_for_tier("d" * 5000, 99)) <= 1010

    def test_empty_content(self):
        from app.research import _truncate_for_tier
        assert _truncate_for_tier("", 1) == ""
        assert _truncate_for_tier(None, 1) == ""


class TestFormatSourcesForPrompt:
    def test_applies_per_tier_truncation(self):
        from app.research import _format_sources_for_prompt
        results = [
            {"title": "Tier1Doc", "url": "https://example.com/1", "tier": 1, "content": "x" * 10000},
            {"title": "Tier3Forum", "url": "https://example.com/3", "tier": 3, "content": "y" * 10000},
        ]
        out = _format_sources_for_prompt(results)
        # Tier 1 keeps a lot more content than Tier 3
        # The Tier-1 block precedes the Tier-3 block; comparing rough proportions
        assert "x" * 7000 in out  # tier 1 retains most of its content
        assert "y" * 7000 not in out  # tier 3 was truncated hard

    def test_marks_extracted_sources(self):
        from app.research import _format_sources_for_prompt
        results = [
            {
                "title": "DocA", "url": "https://example.com/a",
                "tier": 1, "content": "- fact 1\n- fact 2",
                "extracted": True,
            }
        ]
        out = _format_sources_for_prompt(results)
        assert "[EXTRACTED FACTS]" in out

    def test_skips_empty_content(self):
        from app.research import _format_sources_for_prompt
        results = [
            {"title": "Empty", "url": "https://example.com/e", "tier": 1, "content": ""},
            {"title": "Real", "url": "https://example.com/r", "tier": 1, "content": "real text"},
        ]
        out = _format_sources_for_prompt(results)
        assert "Empty" not in out
        assert "Real" in out


# ===========================================================================
# 2.5 — Self-citation lint
# ===========================================================================


class TestLintCitations:
    def test_keeps_real_links(self):
        from app.research import _lint_citations
        text = "See [Python](https://docs.python.org/3/) for details."
        cleaned, stripped = _lint_citations(text, ["https://docs.python.org/3/"])
        assert cleaned == text
        assert stripped == []

    def test_strips_hallucinated_link(self):
        from app.research import _lint_citations
        text = "According to [Fake](https://nonexistent.example/page) the answer is X."
        cleaned, stripped = _lint_citations(text, ["https://docs.python.org/3/"])
        assert "https://nonexistent.example/page" not in cleaned
        assert "Fake" in cleaned  # label preserved
        assert stripped == ["https://nonexistent.example/page"]

    def test_normalizes_protocol_and_www(self):
        from app.research import _lint_citations
        text = "[Doc](https://www.example.com/page/)"
        # Source uses http and no www and no trailing slash
        cleaned, stripped = _lint_citations(text, ["http://example.com/page"])
        assert stripped == []  # should match after normalization

    def test_handles_empty(self):
        from app.research import _lint_citations
        cleaned, stripped = _lint_citations("", ["https://x.com"])
        assert cleaned == ""
        assert stripped == []

    def test_multiple_mixed(self):
        from app.research import _lint_citations
        text = (
            "Real: [A](https://real.example.com/a). "
            "Fake: [B](https://fake.example.com/b). "
            "Real: [C](https://real.example.com/c)."
        )
        cleaned, stripped = _lint_citations(
            text,
            ["https://real.example.com/a", "https://real.example.com/c"],
        )
        assert "https://real.example.com/a" in cleaned
        assert "https://real.example.com/c" in cleaned
        assert "https://fake.example.com/b" not in cleaned
        assert "B" in cleaned  # label preserved
        assert stripped == ["https://fake.example.com/b"]


class TestNormalizeUrl:
    def test_strips_protocol(self):
        from app.research import _normalize_url_for_match
        assert _normalize_url_for_match("https://example.com/x") == "example.com/x"
        assert _normalize_url_for_match("http://example.com/x") == "example.com/x"

    def test_strips_www(self):
        from app.research import _normalize_url_for_match
        assert _normalize_url_for_match("https://www.example.com/x") == "example.com/x"

    def test_strips_trailing_slash(self):
        from app.research import _normalize_url_for_match
        assert _normalize_url_for_match("https://example.com/x/") == "example.com/x"

    def test_lowercases(self):
        from app.research import _normalize_url_for_match
        assert _normalize_url_for_match("https://EXAMPLE.COM/Page") == "example.com/page"


# ===========================================================================
# 2.1 — Per-source extraction pre-pass
# ===========================================================================


class TestExtractionPrePass:
    @pytest.mark.asyncio
    async def test_short_sources_skipped(self):
        from app.research import _extract_facts_from_source
        # Source under 1500 chars → returns None (no extraction worth doing)
        source = {"title": "T", "url": "u", "content": "short content"}
        result = await _extract_facts_from_source(None, "topic", source)
        assert result is None

    @pytest.mark.asyncio
    async def test_extracts_when_long_enough(self):
        from app import research
        source = {
            "title": "Long doc",
            "url": "https://example.com/doc",
            "content": "para. " * 400,  # ~2000 chars
        }
        with patch.object(research, "_chat", AsyncMock(return_value="- fact 1\n- fact 2\n- fact 3")):
            result = await research._extract_facts_from_source(None, "topic", source)
        assert "fact 1" in result

    @pytest.mark.asyncio
    async def test_pre_pass_mutates_results(self):
        from app import research
        results = [
            {"title": "A", "url": "https://example.com/a", "tier": 1, "content": "x" * 2000},
            {"title": "B", "url": "https://example.com/b", "tier": 3, "content": "y" * 2000},
            {"title": "C", "url": "https://example.com/c", "tier": 1, "content": "short"},
        ]
        with patch.object(
            research, "_chat",
            AsyncMock(return_value="- extracted bullet"),
        ):
            count = await research._extract_facts_for_tier1_sources(
                None, "topic", results,
            )
        assert count == 1  # Only A qualifies (tier 1 + long enough)
        assert results[0]["extracted"] is True
        assert "extracted bullet" in results[0]["content"]
        # Tier 3 source untouched
        assert "extracted" not in results[1]
        # Short tier-1 source untouched
        assert "extracted" not in results[2]


# ===========================================================================
# 2.4 — Two-pass critique parser + Confidence Notes
# ===========================================================================


class TestCritiqueParser:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self):
        from app import research

        fake_json = json.dumps({
            "claims": [
                {"text": "Tokio is async", "status": "supported", "note": "from official docs"},
                {"text": "Tokio is the only runtime", "status": "unsupported", "note": "smol exists"},
                {"text": "Spawning costs 50ns", "status": "missing", "note": "important perf detail"},
            ]
        })
        with patch.object(research, "_chat", AsyncMock(return_value=fake_json)):
            claims = await research._critique_draft(None, "Tokio", "draft", "sources")
        assert len(claims) == 3
        assert claims[0]["status"] == "supported"
        assert claims[1]["status"] == "unsupported"
        assert claims[2]["status"] == "missing"

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self):
        from app import research
        fenced = '```json\n{"claims": [{"text": "x", "status": "supported", "note": ""}]}\n```'
        with patch.object(research, "_chat", AsyncMock(return_value=fenced)):
            claims = await research._critique_draft(None, "t", "d", "s")
        assert len(claims) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_garbage(self):
        from app import research
        with patch.object(research, "_chat", AsyncMock(return_value="not json at all")):
            claims = await research._critique_draft(None, "t", "d", "s")
        assert claims == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_llm_failure(self):
        from app import research
        with patch.object(research, "_chat", AsyncMock(side_effect=RuntimeError("api down"))):
            claims = await research._critique_draft(None, "t", "d", "s")
        assert claims == []

    @pytest.mark.asyncio
    async def test_normalizes_unknown_status(self):
        from app import research
        fake = json.dumps({
            "claims": [{"text": "x", "status": "weird-status", "note": ""}]
        })
        with patch.object(research, "_chat", AsyncMock(return_value=fake)):
            claims = await research._critique_draft(None, "t", "d", "s")
        assert claims[0]["status"] == "unsupported"


class TestRenderConfidenceNotes:
    def test_empty_claims_returns_empty_string(self):
        from app.research import _render_confidence_notes
        assert _render_confidence_notes([]) == ""

    def test_renders_supported_unsupported_missing(self):
        from app.research import _render_confidence_notes
        claims = [
            {"text": "Claim A", "status": "supported", "note": ""},
            {"text": "Claim B", "status": "unsupported", "note": "no source"},
            {"text": "Claim C", "status": "missing", "note": "important"},
        ]
        out = _render_confidence_notes(claims)
        assert "## Confidence Notes" in out
        assert "1 supported" in out
        assert "1 unsupported" in out
        assert "1 missing" in out
        assert "Claim B" in out
        assert "Claim C" in out
        # Supported claims aren't listed (they're just counted)
        assert "Claim A" not in out


# ===========================================================================
# 2.6 — claims_json round-trip + worker._persist_critique_claims
# ===========================================================================


class TestClaimsJsonPersistence:
    @pytest.mark.asyncio
    async def test_claims_json_column_exists(self):
        """The claims_json migration adds the column to research_jobs."""
        from app import db
        await db.init_db()

        import aiosqlite
        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            cursor = await conn.execute("PRAGMA table_info(research_jobs)")
            cols = {row[1] for row in await cursor.fetchall()}
            assert "claims_json" in cols
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_persist_critique_claims_writes_to_article_claims(self):
        from app import db
        from app.worker import _persist_critique_claims

        await db.init_db()

        job = {
            "id": 1,
            "claims_json": json.dumps([
                {"text": "Tokio is async", "status": "supported", "note": ""},
                {"text": "Costs 50ns", "status": "unsupported", "note": "no source"},
                {"text": "Missing detail", "status": "missing", "note": ""},
            ]),
        }
        await _persist_critique_claims(job, "personal", "tokio")

        stored = await db.get_claims_for_article("personal", "tokio")
        assert len(stored) == 3
        # Status mapping: supported→verified(0.9), unsupported→unverified(0.3), missing→unverified(0.5)
        by_text = {c["claim_text"]: c for c in stored}
        assert by_text["Tokio is async"]["status"] == "verified"
        assert by_text["Tokio is async"]["confidence"] == 0.9
        assert by_text["Costs 50ns"]["status"] == "unverified"
        assert by_text["Costs 50ns"]["confidence"] == 0.3
        assert by_text["Missing detail"]["status"] == "unverified"
        assert by_text["Missing detail"]["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_persist_handles_empty_claims_json(self):
        from app import db
        from app.worker import _persist_critique_claims

        await db.init_db()

        # No claims_json — should be a no-op
        await _persist_critique_claims({"id": 1}, "personal", "x")
        assert await db.get_claims_for_article("personal", "x") == []

        # Empty list — also no-op
        await _persist_critique_claims(
            {"id": 1, "claims_json": "[]"}, "personal", "x",
        )
        assert await db.get_claims_for_article("personal", "x") == []

    @pytest.mark.asyncio
    async def test_persist_handles_malformed_json(self):
        from app import db
        from app.worker import _persist_critique_claims

        await db.init_db()
        await _persist_critique_claims(
            {"id": 1, "claims_json": "not valid json"},
            "personal", "x",
        )
        # Should not raise, no claims persisted
        assert await db.get_claims_for_article("personal", "x") == []
