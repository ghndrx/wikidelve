"""Unit tests for app.knowledge_graph._parse_json_array."""

import pytest

from app.knowledge_graph import _parse_json_array


class TestParseJsonArray:
    def test_plain_array(self):
        out = _parse_json_array(
            '[{"name": "Python", "type": "language"}]',
            ["name", "type"],
        )
        assert out == [{"name": "Python", "type": "language"}]

    def test_drops_items_missing_keys(self):
        out = _parse_json_array(
            '[{"name": "X", "type": "tool"}, {"name": "Y"}]',
            ["name", "type"],
        )
        assert out == [{"name": "X", "type": "tool"}]

    def test_strips_code_fences(self):
        text = "```json\n[{\"name\": \"X\", \"type\": \"tool\"}]\n```"
        out = _parse_json_array(text, ["name", "type"])
        assert out == [{"name": "X", "type": "tool"}]

    def test_extracts_from_wrapped_text(self):
        text = 'Sure, here is the array: [{"name": "X", "type": "tool"}] hope that helps'
        out = _parse_json_array(text, ["name", "type"])
        assert out == [{"name": "X", "type": "tool"}]

    def test_malformed_returns_empty(self):
        out = _parse_json_array("not json at all", ["name", "type"])
        assert out == []

    def test_non_array_returns_empty(self):
        out = _parse_json_array('{"name": "X", "type": "tool"}', ["name", "type"])
        assert out == []
