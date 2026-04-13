"""
Logging configuration for KB Service.
"""

import json
import logging
import sys
from datetime import datetime, timezone


_EXTRA_FIELDS = ("event", "session_id", "user_input", "command", "action", "result", "error")


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                data[field] = val
        return json.dumps(data)


class TextFormatter(logging.Formatter):
    """Human-readable formatter that appends extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<8} {record.name:<25} {record.getMessage()}"
        extras = " ".join(
            f"{f}={getattr(record, f)}"
            for f in _EXTRA_FIELDS
            if getattr(record, f, None) is not None
        )
        return f"{base} {extras}".rstrip()


def setup_logging(level: str = "INFO"):
    """Configure structured logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler with structured format
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)-25s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("arq").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def log_chat_interaction(session_id: str, event: str, **kwargs):
    """Log a chat interaction for observability."""
    logger = logging.getLogger("kb-service.chat")
    details = " ".join(f"{k}={v!r}" for k, v in kwargs.items() if v)
    logger.info("chat session=%s event=%s %s", session_id, event, details)
