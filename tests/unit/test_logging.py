"""Unit tests for logging configuration."""

import json
import logging
from unittest.mock import patch

from app.logging_config import JSONFormatter, TextFormatter, log_chat_interaction


class TestJSONFormatter:
    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "ts" in data

    def test_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="chat event", args=(), exc_info=None,
        )
        record.event = "user_message"
        record.session_id = "abc-123"
        record.user_input = "hello"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["event"] == "user_message"
        assert data["session_id"] == "abc-123"
        assert data["user_input"] == "hello"

    def test_missing_extras_not_included(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="simple", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "event" not in data
        assert "session_id" not in data


class TestTextFormatter:
    def test_basic_format(self):
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "hello" in output
        assert "test" in output

    def test_extras_in_output(self):
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="event", args=(), exc_info=None,
        )
        record.event = "search"
        output = formatter.format(record)
        assert "event=search" in output


class TestChatInteractionLogger:
    def test_log_chat_interaction(self, caplog):
        with caplog.at_level(logging.INFO, logger="kb-service.chat"):
            log_chat_interaction(
                session_id="test-session",
                event="user_message",
                user_input="hello world",
            )
        assert "event=user_message" in caplog.text
        assert "session=test-session" in caplog.text
