"""Unit tests for app.knowledge_graph – entity/relationship extraction and graph building."""

import json

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.knowledge_graph import (
    _parse_json_array,
    extract_entities,
    extract_relationships,
    build_graph_for_article,
    build_full_graph,
)


# ---------------------------------------------------------------------------
# _parse_json_array  (supplements test_knowledge_graph_parse.py)
# ---------------------------------------------------------------------------


class TestParseJsonArrayExtended:
    """Additional edge-case coverage beyond test_knowledge_graph_parse.py."""

    def test_empty_array(self):
        assert _parse_json_array("[]", ["name", "type"]) == []

    def test_whitespace_only(self):
        assert _parse_json_array("   \n\t  ", ["name"]) == []

    def test_empty_string(self):
        assert _parse_json_array("", ["name"]) == []

    def test_code_fence_no_language_tag(self):
        text = "```\n[{\"name\": \"Go\", \"type\": \"language\"}]\n```"
        out = _parse_json_array(text, ["name", "type"])
        assert out == [{"name": "Go", "type": "language"}]

    def test_multiple_code_fence_lines(self):
        text = "```json\n[{\"name\": \"A\", \"type\": \"tool\"},\n{\"name\": \"B\", \"type\": \"concept\"}]\n```"
        out = _parse_json_array(text, ["name", "type"])
        assert len(out) == 2

    def test_filters_non_dict_items(self):
        out = _parse_json_array('[1, "string", {"name": "X", "type": "tool"}]', ["name", "type"])
        assert out == [{"name": "X", "type": "tool"}]

    def test_nested_brackets_in_surrounding_text(self):
        text = 'Here is your answer: [{"source": "A", "target": "B", "relationship": "uses"}] done.'
        out = _parse_json_array(text, ["source", "target", "relationship"])
        assert len(out) == 1
        assert out[0]["relationship"] == "uses"

    def test_relationship_keys(self):
        raw = json.dumps([
            {"source": "Django", "target": "Python", "relationship": "built-with"},
            {"source": "Flask", "target": "Python", "relationship": "built-with"},
        ])
        out = _parse_json_array(raw, ["source", "target", "relationship"])
        assert len(out) == 2

    def test_extra_keys_preserved(self):
        raw = '[{"name": "X", "type": "tool", "extra": 1}]'
        out = _parse_json_array(raw, ["name", "type"])
        assert len(out) == 1
        assert out[0]["extra"] == 1


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    @pytest.mark.asyncio
    async def test_returns_parsed_entities(self):
        llm_response = json.dumps([
            {"name": "Python", "type": "language"},
            {"name": "FastAPI", "type": "framework"},
        ])
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await extract_entities("Some article about Python and FastAPI", "Tech Article")

        assert len(result) == 2
        assert result[0]["name"] == "Python"
        assert result[1]["name"] == "FastAPI"

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self):
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value="[]"):
            result = await extract_entities("Some text", "Title")
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_empty(self):
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value="Sorry, I can't do that."):
            result = await extract_entities("Some text", "Title")
        assert result == []

    @pytest.mark.asyncio
    async def test_truncates_long_text(self):
        long_text = "x" * 10000
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value="[]") as mock_llm:
            await extract_entities(long_text, "Title")
            # The prompt should contain the truncated text (6000 chars), not the full 10000
            call_args = mock_llm.call_args
            user_msg = call_args.kwargs.get("user_msg", call_args[1].get("user_msg", ""))
            assert "x" * 6000 in user_msg
            assert "x" * 10000 not in user_msg

    @pytest.mark.asyncio
    async def test_filters_items_missing_required_keys(self):
        llm_response = json.dumps([
            {"name": "Python", "type": "language"},
            {"name": "Orphan"},  # missing "type"
        ])
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await extract_entities("text", "Title")
        assert len(result) == 1
        assert result[0]["name"] == "Python"

    @pytest.mark.asyncio
    async def test_llm_raises_propagates(self):
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")):
            with pytest.raises(RuntimeError, match="LLM down"):
                await extract_entities("text", "Title")

    @pytest.mark.asyncio
    async def test_code_fenced_llm_response(self):
        llm_response = '```json\n[{"name": "Redis", "type": "tool"}]\n```'
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await extract_entities("Article about Redis", "Redis Guide")
        assert len(result) == 1
        assert result[0]["name"] == "Redis"


# ---------------------------------------------------------------------------
# extract_relationships
# ---------------------------------------------------------------------------


class TestExtractRelationships:
    @pytest.mark.asyncio
    async def test_returns_parsed_relationships(self):
        entities = [
            {"name": "Django", "type": "framework"},
            {"name": "Python", "type": "language"},
        ]
        llm_response = json.dumps([
            {"source": "Django", "target": "Python", "relationship": "built-with"},
        ])
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await extract_relationships("Django is a Python framework.", entities)
        assert len(result) == 1
        assert result[0]["source"] == "Django"
        assert result[0]["relationship"] == "built-with"

    @pytest.mark.asyncio
    async def test_fewer_than_two_entities_returns_empty(self):
        # Should short-circuit without calling LLM
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock) as mock_llm:
            result = await extract_relationships("text", [{"name": "Solo", "type": "tool"}])
            assert result == []
            mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_entities_returns_empty(self):
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock) as mock_llm:
            result = await extract_relationships("text", [])
            assert result == []
            mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_response_returns_empty(self):
        entities = [{"name": "A", "type": "tool"}, {"name": "B", "type": "tool"}]
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value="not json"):
            result = await extract_relationships("text", entities)
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_incomplete_relationships(self):
        entities = [{"name": "A", "type": "tool"}, {"name": "B", "type": "tool"}]
        llm_response = json.dumps([
            {"source": "A", "target": "B", "relationship": "uses"},
            {"source": "A", "target": "B"},  # missing relationship
        ])
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, return_value=llm_response):
            result = await extract_relationships("text", entities)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_llm_raises_propagates(self):
        entities = [{"name": "A", "type": "tool"}, {"name": "B", "type": "tool"}]
        with patch("app.knowledge_graph.llm_chat", new_callable=AsyncMock, side_effect=RuntimeError("timeout")):
            with pytest.raises(RuntimeError, match="timeout"):
                await extract_relationships("text", entities)


# ---------------------------------------------------------------------------
# build_graph_for_article
# ---------------------------------------------------------------------------


class TestBuildGraphForArticle:
    @pytest.mark.asyncio
    async def test_article_not_found(self):
        with patch("app.knowledge_graph.get_article", return_value=None):
            result = await build_graph_for_article("kb", "missing-slug")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_body_skips(self):
        with patch("app.knowledge_graph.get_article", return_value={"title": "T", "raw_markdown": ""}):
            result = await build_graph_for_article("kb", "empty-slug")
        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_no_raw_markdown_key_skips(self):
        with patch("app.knowledge_graph.get_article", return_value={"title": "T"}):
            result = await build_graph_for_article("kb", "no-body")
        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_no_entities_found(self):
        with patch("app.knowledge_graph.get_article", return_value={"title": "T", "raw_markdown": "some text"}):
            with patch("app.knowledge_graph.extract_entities", new_callable=AsyncMock, return_value=[]):
                result = await build_graph_for_article("kb", "slug")
        assert result["status"] == "ok"
        assert result["entities"] == 0
        assert result["relationships"] == 0

    @pytest.mark.asyncio
    async def test_entity_extraction_failure(self):
        with patch("app.knowledge_graph.get_article", return_value={"title": "T", "raw_markdown": "text"}):
            with patch("app.knowledge_graph.extract_entities", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
                result = await build_graph_for_article("kb", "slug")
        assert result["status"] == "error"
        assert "extraction failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_successful_build(self):
        entities = [
            {"name": "Python", "type": "language"},
            {"name": "FastAPI", "type": "framework"},
        ]
        relationships = [
            {"source": "FastAPI", "target": "Python", "relationship": "built-with"},
        ]

        mock_db = AsyncMock()
        # _upsert_entity returns incrementing IDs
        entity_id_counter = iter([1, 2])

        async def fake_upsert_entity(db, name, etype):
            return next(entity_id_counter)

        with patch("app.knowledge_graph.get_article", return_value={"title": "T", "raw_markdown": "text"}):
            with patch("app.knowledge_graph.extract_entities", new_callable=AsyncMock, return_value=entities):
                with patch("app.knowledge_graph.extract_relationships", new_callable=AsyncMock, return_value=relationships):
                    with patch("app.knowledge_graph._get_kg_db", new_callable=AsyncMock, return_value=mock_db):
                        with patch("app.knowledge_graph._upsert_entity", side_effect=fake_upsert_entity):
                            with patch("app.knowledge_graph._upsert_edge", new_callable=AsyncMock):
                                result = await build_graph_for_article("kb", "slug")

        assert result["status"] == "ok"
        assert result["entities"] == 2
        assert result["relationships"] == 1
        assert result["slug"] == "slug"
        assert result["kb"] == "kb"

    @pytest.mark.asyncio
    async def test_relationship_extraction_failure_still_stores_entities(self):
        """If relationship extraction fails, entities should still be stored."""
        entities = [{"name": "Go", "type": "language"}, {"name": "Docker", "type": "tool"}]
        mock_db = AsyncMock()
        entity_id_counter = iter([1, 2])

        async def fake_upsert_entity(db, name, etype):
            return next(entity_id_counter)

        with patch("app.knowledge_graph.get_article", return_value={"title": "T", "raw_markdown": "text"}):
            with patch("app.knowledge_graph.extract_entities", new_callable=AsyncMock, return_value=entities):
                with patch("app.knowledge_graph.extract_relationships", new_callable=AsyncMock, side_effect=RuntimeError("llm fail")):
                    with patch("app.knowledge_graph._get_kg_db", new_callable=AsyncMock, return_value=mock_db):
                        with patch("app.knowledge_graph._upsert_entity", side_effect=fake_upsert_entity):
                            with patch("app.knowledge_graph._upsert_edge", new_callable=AsyncMock) as mock_edge:
                                result = await build_graph_for_article("kb", "slug")

        # Entities stored, relationships = 0 because extraction failed
        assert result["status"] == "ok"
        assert result["entities"] == 2
        assert result["relationships"] == 0
        mock_edge.assert_not_called()

    @pytest.mark.asyncio
    async def test_unmatched_relationship_entities_skipped(self):
        """Edges referencing entities not in entity_ids dict should be skipped."""
        entities = [{"name": "Python", "type": "language"}]
        relationships = [
            {"source": "Python", "target": "UNKNOWN", "relationship": "uses"},
        ]
        mock_db = AsyncMock()

        async def fake_upsert_entity(db, name, etype):
            return 1

        with patch("app.knowledge_graph.get_article", return_value={"title": "T", "raw_markdown": "text"}):
            with patch("app.knowledge_graph.extract_entities", new_callable=AsyncMock, return_value=entities):
                with patch("app.knowledge_graph.extract_relationships", new_callable=AsyncMock, return_value=relationships):
                    with patch("app.knowledge_graph._get_kg_db", new_callable=AsyncMock, return_value=mock_db):
                        with patch("app.knowledge_graph._upsert_entity", side_effect=fake_upsert_entity):
                            with patch("app.knowledge_graph._upsert_edge", new_callable=AsyncMock) as mock_edge:
                                result = await build_graph_for_article("kb", "slug")

        assert result["status"] == "ok"
        # The edge should not have been inserted since "UNKNOWN" is not in entity_ids
        mock_edge.assert_not_called()


# ---------------------------------------------------------------------------
# build_full_graph
# ---------------------------------------------------------------------------


class TestBuildFullGraph:
    @pytest.mark.asyncio
    async def test_no_articles(self):
        with patch("app.knowledge_graph.get_articles", return_value=[]):
            result = await build_full_graph("kb")
        assert result["status"] == "ok"
        assert result["processed"] == 0
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_none_articles(self):
        with patch("app.knowledge_graph.get_articles", return_value=None):
            result = await build_full_graph("kb")
        assert result["status"] == "ok"
        assert result["processed"] == 0

    @pytest.mark.asyncio
    async def test_processes_multiple_articles(self):
        articles = [{"slug": "a"}, {"slug": "b"}, {"slug": "c"}]

        async def fake_build(kb, slug):
            return {"status": "ok", "entities": 3, "relationships": 1}

        with patch("app.knowledge_graph.get_articles", return_value=articles):
            with patch("app.knowledge_graph.build_graph_for_article", side_effect=fake_build):
                result = await build_full_graph("kb")

        assert result["status"] == "ok"
        assert result["processed"] == 3
        assert result["total_entities"] == 9
        assert result["total_relationships"] == 3
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_counts_errors(self):
        articles = [{"slug": "good"}, {"slug": "bad"}]

        async def fake_build(kb, slug):
            if slug == "bad":
                return {"status": "error", "error": "something broke"}
            return {"status": "ok", "entities": 2, "relationships": 1}

        with patch("app.knowledge_graph.get_articles", return_value=articles):
            with patch("app.knowledge_graph.build_graph_for_article", side_effect=fake_build):
                result = await build_full_graph("kb")

        assert result["processed"] == 1
        assert result["errors"] == 1
        assert result["total_entities"] == 2

    @pytest.mark.asyncio
    async def test_exception_in_article_counted_as_error(self):
        articles = [{"slug": "explode"}]

        async def fake_build(kb, slug):
            raise RuntimeError("unexpected")

        with patch("app.knowledge_graph.get_articles", return_value=articles):
            with patch("app.knowledge_graph.build_graph_for_article", side_effect=fake_build):
                result = await build_full_graph("kb")

        assert result["processed"] == 0
        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_skipped_articles_not_counted_as_processed_or_error(self):
        articles = [{"slug": "empty"}]

        async def fake_build(kb, slug):
            return {"status": "skip", "reason": "empty body"}

        with patch("app.knowledge_graph.get_articles", return_value=articles):
            with patch("app.knowledge_graph.build_graph_for_article", side_effect=fake_build):
                result = await build_full_graph("kb")

        # "skip" is neither "ok" nor "error", so both counters stay at 0
        assert result["processed"] == 0
        assert result["errors"] == 0
