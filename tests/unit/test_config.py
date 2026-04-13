"""Unit tests for app/config.py — environment-driven configuration."""

import os
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

import app.config as _config_mod


@pytest.fixture(autouse=True)
def _restore_config():
    """Reload config and re-patch all modules that imported KB_DIRS from it."""
    import app.wiki as _wiki_mod
    import app.storage as _storage_mod
    yield
    importlib.reload(_config_mod)
    # Re-bind KB_DIRS in modules that do `from app.config import KB_DIRS`
    if hasattr(_wiki_mod, "KB_DIRS"):
        _wiki_mod.KB_DIRS = _config_mod.KB_DIRS
    if hasattr(_storage_mod, "KB_DIRS"):
        _storage_mod.KB_DIRS = _config_mod.KB_DIRS


class TestConfig:
    def test_defaults(self):
        from app.config import KB_DIRS, DB_PATH, RESEARCH_DIR, REDIS_HOST
        assert "personal" in KB_DIRS
        assert DB_PATH is not None
        assert RESEARCH_DIR is not None
        assert REDIS_HOST  # should have a value

    def test_kb_dirs_default_personal(self):
        from app.config import KB_DIRS
        assert "personal" in KB_DIRS

    def test_cooldown_days(self):
        from app.config import COOLDOWN_DAYS
        assert COOLDOWN_DAYS == 7

    def test_tier1_domains_populated(self):
        from app.config import TIER1_DOMAINS
        assert "gov" in TIER1_DOMAINS
        assert "edu" in TIER1_DOMAINS

    def test_llm_provider_default(self):
        from app.config import LLM_PROVIDER
        assert LLM_PROVIDER in ("minimax", "bedrock")

    def test_synthesis_prompt_exists(self):
        from app.config import SYNTHESIS_SYSTEM_PROMPT
        assert len(SYNTHESIS_SYSTEM_PROMPT) > 100
        assert "research" in SYNTHESIS_SYSTEM_PROMPT.lower()


# ──────────────────────────────────────────────────────────────────────────────
# EXTRA_KB env var discovery (lines 49-53)
# ──────────────────────────────────────────────────────────────────────────────


class TestExtraKBEnvVars:
    def test_extra_kb_loaded_from_env(self, tmp_path, monkeypatch):
        """Lines 50-53: EXTRA_KB_WORK_PATH env var creates a KB entry."""
        kb_path = str(tmp_path / "work")
        monkeypatch.setenv("EXTRA_KB_WORK_PATH", kb_path)
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(tmp_path / "personal"))
        (tmp_path / "personal" / "wiki").mkdir(parents=True, exist_ok=True)

        import app.config as config_mod
        importlib.reload(config_mod)

        assert "work" in config_mod.KB_DIRS
        assert config_mod.KB_DIRS["work"] == Path(kb_path)

    def test_extra_kb_empty_value_skipped(self, tmp_path, monkeypatch):
        """Lines 52: empty value should be skipped."""
        monkeypatch.setenv("EXTRA_KB_EMPTY_PATH", "  ")
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(tmp_path / "personal"))
        (tmp_path / "personal" / "wiki").mkdir(parents=True, exist_ok=True)

        import app.config as config_mod
        importlib.reload(config_mod)

        assert "empty" not in config_mod.KB_DIRS


# ──────────────────────────────────────────────────────────────────────────────
# Auto-discover KB directories (lines 57-65)
# ──────────────────────────────────────────────────────────────────────────────


class TestAutoDiscoverKBs:
    def test_discovers_kb_with_wiki_subfolder(self, tmp_path, monkeypatch):
        """Lines 59-65: directories under KB_ROOT with wiki/ subfolder are auto-discovered."""
        (tmp_path / "science" / "wiki").mkdir(parents=True)
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(tmp_path / "personal"))
        (tmp_path / "personal" / "wiki").mkdir(parents=True, exist_ok=True)

        import app.config as config_mod
        importlib.reload(config_mod)

        assert "science" in config_mod.KB_DIRS

    def test_skips_reserved_directories(self, tmp_path, monkeypatch):
        """Lines 58-61: 'research' and 'downloads' are reserved and skipped."""
        (tmp_path / "research" / "wiki").mkdir(parents=True)
        (tmp_path / "downloads" / "wiki").mkdir(parents=True)
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(tmp_path / "personal"))
        (tmp_path / "personal" / "wiki").mkdir(parents=True, exist_ok=True)

        import app.config as config_mod
        importlib.reload(config_mod)

        assert "research" not in config_mod.KB_DIRS
        assert "downloads" not in config_mod.KB_DIRS

    def test_skips_hidden_directories(self, tmp_path, monkeypatch):
        """Lines 62: directories starting with '.' are skipped."""
        (tmp_path / ".hidden" / "wiki").mkdir(parents=True)
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(tmp_path / "personal"))
        (tmp_path / "personal" / "wiki").mkdir(parents=True, exist_ok=True)

        import app.config as config_mod
        importlib.reload(config_mod)

        assert ".hidden" not in config_mod.KB_DIRS

    def test_skips_directories_without_wiki(self, tmp_path, monkeypatch):
        """Lines 64: directories without a wiki/ subfolder are skipped."""
        (tmp_path / "nowikidir").mkdir(parents=True)
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(tmp_path / "personal"))
        (tmp_path / "personal" / "wiki").mkdir(parents=True, exist_ok=True)

        import app.config as config_mod
        importlib.reload(config_mod)

        assert "nowikidir" not in config_mod.KB_DIRS

    def test_does_not_override_explicit_kb(self, tmp_path, monkeypatch):
        """Lines 63: already-defined KB names (from env) are not overridden."""
        (tmp_path / "personal" / "wiki").mkdir(parents=True)
        explicit_path = tmp_path / "explicit_personal"
        explicit_path.mkdir(parents=True)
        monkeypatch.setenv("KB_ROOT", str(tmp_path))
        monkeypatch.setenv("PERSONAL_KB_PATH", str(explicit_path))

        import app.config as config_mod
        importlib.reload(config_mod)

        # personal should point to the explicit path, not the auto-discovered one
        assert config_mod.KB_DIRS["personal"] == explicit_path


# ──────────────────────────────────────────────────────────────────────────────
# register_kb (lines 76-92)
# ──────────────────────────────────────────────────────────────────────────────


class TestRegisterKB:
    def test_register_new_kb(self, tmp_path, monkeypatch):
        """Lines 82-91: registering a new KB creates dirs and adds to KB_DIRS."""
        import app.config as config_mod
        monkeypatch.setattr(config_mod, "KB_ROOT", tmp_path)
        config_mod.KB_DIRS.pop("newkb", None)

        result = config_mod.register_kb("newkb")
        assert result == tmp_path / "newkb"
        assert (tmp_path / "newkb" / "wiki").exists()
        assert (tmp_path / "newkb" / "raw").exists()
        assert "newkb" in config_mod.KB_DIRS

    def test_register_existing_kb_returns_path(self, tmp_path, monkeypatch):
        """Lines 85-86: registering an existing KB just returns its path."""
        import app.config as config_mod
        monkeypatch.setattr(config_mod, "KB_ROOT", tmp_path)

        existing_path = tmp_path / "existing"
        existing_path.mkdir()
        config_mod.KB_DIRS["existing"] = existing_path

        result = config_mod.register_kb("existing")
        assert result == existing_path

    def test_register_invalid_name_raises(self, tmp_path, monkeypatch):
        """Lines 83-84: invalid names (all special chars) raise ValueError."""
        import app.config as config_mod
        monkeypatch.setattr(config_mod, "KB_ROOT", tmp_path)

        with pytest.raises(ValueError, match="Invalid KB name"):
            config_mod.register_kb("!!!")

    def test_register_sanitizes_name(self, tmp_path, monkeypatch):
        """Line 82: name is sanitized (special chars removed, lowercased)."""
        import app.config as config_mod
        monkeypatch.setattr(config_mod, "KB_ROOT", tmp_path)
        config_mod.KB_DIRS.pop("mykb", None)

        result = config_mod.register_kb("My KB!")
        # Special chars and spaces removed, lowercased
        assert "mykb" in config_mod.KB_DIRS


# ──────────────────────────────────────────────────────────────────────────────
# Miscellaneous config values
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigMisc:
    def test_redis_url_format(self):
        from app.config import REDIS_URL
        assert REDIS_URL.startswith("redis://")

    def test_redis_port_is_int(self):
        from app.config import REDIS_PORT
        assert isinstance(REDIS_PORT, int)

    def test_research_constants(self):
        from app.config import RESEARCH_MAX_RETRIES, RESEARCH_RETRY_DELAY, MINIMAX_TIMEOUT
        assert RESEARCH_MAX_RETRIES == 3
        assert RESEARCH_RETRY_DELAY == 15
        assert MINIMAX_TIMEOUT == 300

    def test_auto_discovery_flag(self):
        from app.config import AUTO_DISCOVERY_ENABLED
        assert isinstance(AUTO_DISCOVERY_ENABLED, bool)

    def test_serper_calls_estimate(self):
        from app.config import SERPER_CALLS_PER_JOB_ESTIMATE
        assert isinstance(SERPER_CALLS_PER_JOB_ESTIMATE, int)

    def test_research_kb_constant(self):
        from app.config import RESEARCH_KB
        assert RESEARCH_KB == "_research"

    def test_db_path_is_path(self):
        from app.config import DB_PATH
        assert isinstance(DB_PATH, Path)

    def test_tier2_substrings(self):
        from app.config import TIER2_SUBSTRINGS
        assert isinstance(TIER2_SUBSTRINGS, set)
        assert "stackoverflow.com" in TIER2_SUBSTRINGS

    def test_rate_limit_defaults(self):
        from app.config import RATE_LIMIT_RESEARCH, RATE_LIMIT_SEARCH
        assert "minute" in RATE_LIMIT_RESEARCH
        assert "minute" in RATE_LIMIT_SEARCH

    def test_synthesis_prompt_has_english_rule(self):
        from app.config import SYNTHESIS_SYSTEM_PROMPT
        assert "English" in SYNTHESIS_SYSTEM_PROMPT

    def test_bedrock_defaults(self):
        from app.config import BEDROCK_REGION, BEDROCK_MODEL
        assert BEDROCK_REGION  # should have a default
        assert BEDROCK_MODEL  # should have a default
