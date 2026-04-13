"""Unit tests for app.logging_config."""

import json
import logging

import pytest

from app.logging_config import JSONFormatter, TextFormatter, setup_logging, log_chat_interaction


def _make_record(**extras):
    record = logging.LogRecord(
        name="kb-service.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for k, v in extras.items():
        setattr(record, k, v)
    return record


class TestJSONFormatter:
    def test_base_fields(self):
        record = _make_record()
        data = json.loads(JSONFormatter().format(record))
        assert data["level"] == "INFO"
        assert data["logger"] == "kb-service.test"
        assert data["msg"] == "hello world"
        assert "ts" in data

    def test_extras_propagate(self):
        record = _make_record(event="click", session_id="s1", user_input="hi")
        data = json.loads(JSONFormatter().format(record))
        assert data["event"] == "click"
        assert data["session_id"] == "s1"
        assert data["user_input"] == "hi"

    def test_missing_extras_omitted(self):
        record = _make_record()
        data = json.loads(JSONFormatter().format(record))
        assert "event" not in data
        assert "session_id" not in data


class TestTextFormatter:
    def test_plain_record(self):
        record = _make_record()
        out = TextFormatter().format(record)
        assert "INFO" in out
        assert "kb-service.test" in out
        assert "hello world" in out

    def test_extras_appended(self):
        record = _make_record(event="click", action="save")
        out = TextFormatter().format(record)
        assert "event=click" in out
        assert "action=save" in out


class TestSetupLogging:
    def test_adds_handler_when_none_exist(self):
        root = logging.getLogger()
        saved = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging("INFO")
            assert len(root.handlers) >= 1
        finally:
            root.handlers = saved

    def test_does_not_duplicate_handlers(self):
        root = logging.getLogger()
        count_before = len(root.handlers)
        setup_logging("INFO")
        assert len(root.handlers) == count_before

    def test_quiets_noisy_loggers(self):
        setup_logging("DEBUG")
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_invalid_level_falls_back_to_info(self):
        setup_logging("BOGUS")
        assert logging.getLogger().level == logging.INFO


class TestLogChatInteraction:
    def test_emits_info_log(self, caplog):
        with caplog.at_level(logging.INFO, logger="kb-service.chat"):
            log_chat_interaction("s1", "msg", user_input="hi", result="ok")
        assert "session=s1" in caplog.text
        assert "event=msg" in caplog.text

    def test_skips_falsy_kwargs(self, caplog):
        with caplog.at_level(logging.INFO, logger="kb-service.chat"):
            log_chat_interaction("s2", "ev", user_input=None, result="")
        assert "user_input" not in caplog.text
        assert "result" not in caplog.text
