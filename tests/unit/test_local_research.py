"""Unit tests for app.local_research — file discovery, reading, scoring, git context, pipeline."""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.local_research import (
    discover_files,
    read_file_content,
    score_content_relevance,
    get_git_context,
    _find_git_root,
    _file_info,
    MAX_CONTENT_PER_FILE,
    MAX_FILE_SIZE,
    run_local_research,
)


@pytest.fixture
def sample_tree(tmp_path):
    (tmp_path / "README.md").write_text("# Project\n\nAuth flow overview\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def login(): pass\n")
    (tmp_path / "src" / "utils.py").write_text("def noop(): pass\n")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignore.js").write_text("nope")
    binary = tmp_path / "image.bin"
    binary.write_bytes(b"\x00" * 10)
    return tmp_path


# ===========================================================================
# discover_files
# ===========================================================================


class TestDiscoverFiles:
    def test_returns_expected_files(self, sample_tree):
        files = discover_files(sample_tree, topic="auth login")
        names = {f["name"] for f in files}
        assert "README.md" in names
        assert "auth.py" in names
        assert "pyproject.toml" in names

    def test_skipped_dirs_are_pruned(self, sample_tree):
        files = discover_files(sample_tree, topic="auth")
        rel_paths = {f["rel_path"] for f in files}
        assert not any("node_modules" in p for p in rel_paths)
        assert not any("__pycache__" in p for p in rel_paths)

    def test_topic_keyword_boosts_score(self, sample_tree):
        files = discover_files(sample_tree, topic="auth login flow")
        scores = {f["name"]: f["score"] for f in files}
        assert scores.get("auth.py", 0) > scores.get("utils.py", 0)

    def test_readme_outranks_generic_files(self, sample_tree):
        files = discover_files(sample_tree, topic="generic")
        top = files[0]
        assert top["name"] == "README.md"

    def test_nonexistent_root_returns_empty(self, tmp_path):
        assert discover_files(tmp_path / "missing") == []

    def test_single_file_root(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# notes\n")
        result = discover_files(f, topic="notes")
        assert len(result) == 1
        assert result[0]["name"] == "notes.md"

    def test_no_topic_still_works(self, sample_tree):
        """Files are discovered and scored even without a topic."""
        files = discover_files(sample_tree, topic="")
        assert len(files) > 0

    def test_hidden_dotdirs_pruned(self, tmp_path):
        """Directories starting with '.' are pruned."""
        hidden = tmp_path / ".secret"
        hidden.mkdir()
        (hidden / "leak.py").write_text("secret = True")
        (tmp_path / "visible.py").write_text("ok = True")
        files = discover_files(tmp_path, topic="")
        names = {f["name"] for f in files}
        assert "leak.py" not in names
        assert "visible.py" in names


# ===========================================================================
# _file_info
# ===========================================================================


class TestFileInfo:
    def test_stat_error_returns_none(self, tmp_path):
        """Coverage: lines 126-127 — OSError on stat."""
        result = _file_info(tmp_path / "nonexistent.py", tmp_path)
        assert result is None

    def test_oversized_file_skipped(self, tmp_path):
        """Coverage: lines 130-131 — large file rejected."""
        big = tmp_path / "big.py"
        big.write_text("x" * (MAX_FILE_SIZE + 1))
        result = _file_info(big, tmp_path)
        assert result is None

    def test_empty_file_skipped(self, tmp_path):
        """Coverage: line 130 — empty file rejected."""
        empty = tmp_path / "empty.py"
        empty.write_text("")
        result = _file_info(empty, tmp_path)
        assert result is None

    def test_symlink_skipped(self, tmp_path):
        """Coverage: lines 132-133 — symlink rejected."""
        real = tmp_path / "real.py"
        real.write_text("content")
        link = tmp_path / "link.py"
        link.symlink_to(real)
        result = _file_info(link, tmp_path)
        assert result is None

    def test_extensionless_non_important_skipped(self, tmp_path):
        """Coverage: lines 143-145 — extensionless files without importance skipped."""
        f = tmp_path / "randomfile"
        f.write_text("content")
        result = _file_info(f, tmp_path)
        assert result is None

    def test_important_filename_gets_boost(self, tmp_path):
        """Coverage: lines 155-158 — important files get base boost."""
        readme = tmp_path / "README.md"
        readme.write_text("# hello world\n")
        info = _file_info(readme, tmp_path, topic="hello")
        assert info is not None
        assert info["score"] >= 50  # 20 (important) + 30 (readme) + 10 (doc)

    def test_topic_word_in_filename(self, tmp_path):
        """Coverage: lines 169-171 — topic word matching in filename."""
        f = tmp_path / "authentication.py"
        f.write_text("def auth(): pass")
        info = _file_info(f, tmp_path, topic="authentication", topic_words={"authentication"})
        assert info is not None
        assert info["score"] >= 15  # at least the filename match bonus

    def test_topic_word_in_path_not_filename(self, tmp_path):
        """Coverage: lines 172-173 — topic word in path gets smaller boost."""
        subdir = tmp_path / "authentication"
        subdir.mkdir()
        f = subdir / "handler.py"
        f.write_text("def handle(): pass")
        info = _file_info(f, tmp_path, topic="authentication", topic_words={"authentication"})
        assert info is not None
        assert info["score"] >= 8

    def test_short_topic_words_ignored(self, tmp_path):
        """Coverage: lines 168-169 — words < 3 chars skipped."""
        f = tmp_path / "ab.py"
        f.write_text("x = 1")
        info = _file_info(f, tmp_path, topic="ab cd", topic_words={"ab", "cd"})
        assert info is not None
        # Short words don't add to score; score should be 0 from topic matching

    def test_recency_bonus_recent(self, tmp_path):
        """Coverage: lines 177-178 — recently modified file gets bonus."""
        f = tmp_path / "recent.py"
        f.write_text("fresh = True")
        # File was just created, so it's < 7 days old — gets +5 recency
        info = _file_info(f, tmp_path)
        assert info is not None
        # Score includes recency bonus (5) minus depth penalty (0.5)
        assert info["score"] >= 4

    def test_config_file_boost(self, tmp_path):
        """Coverage: lines 183-184 — config extensions get a boost."""
        f = tmp_path / "config.yaml"
        f.write_text("key: value")
        info = _file_info(f, tmp_path)
        assert info is not None
        assert info["score"] >= 3

    def test_depth_penalty(self, tmp_path):
        """Coverage: lines 187-188 — deeper files get score penalty."""
        shallow = tmp_path / "top.py"
        shallow.write_text("x = 1")
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)
        deep = deep_dir / "bottom.py"
        deep.write_text("x = 1")
        info_shallow = _file_info(shallow, tmp_path)
        info_deep = _file_info(deep, tmp_path)
        assert info_shallow["score"] > info_deep["score"]

    def test_returns_correct_metadata(self, tmp_path):
        """Coverage: lines 190-198 — returned dict has expected keys."""
        f = tmp_path / "example.py"
        f.write_text("x = 42")
        info = _file_info(f, tmp_path)
        assert info["name"] == "example.py"
        assert info["suffix"] == ".py"
        assert info["size"] > 0
        assert info["rel_path"] == "example.py"
        assert "modified" in info
        assert info["score"] >= 0


# ===========================================================================
# read_file_content
# ===========================================================================


class TestReadFileContent:
    def test_reads_small_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        assert read_file_content(str(f)) == "hello"

    def test_truncates_large_file(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 200)
        out = read_file_content(str(f), max_chars=50)
        assert out.startswith("x" * 50)
        assert "truncated" in out

    def test_missing_file_returns_none(self, tmp_path):
        assert read_file_content(str(tmp_path / "missing.txt")) is None


# ===========================================================================
# score_content_relevance
# ===========================================================================


class TestScoreContentRelevance:
    def test_empty_inputs_return_zero(self):
        assert score_content_relevance("", "topic") == 0.0
        assert score_content_relevance("body", "") == 0.0

    def test_matching_terms_score_positive(self):
        score = score_content_relevance(
            "This document describes the authentication login flow in detail.",
            "authentication login",
        )
        assert score > 0

    def test_short_terms_ignored(self):
        score = score_content_relevance("a b c", "a b")
        assert score == 0.0

    def test_repeated_mentions_have_diminishing_returns(self):
        """Coverage: line 317 — min(count, 10) caps contribution."""
        text = "auth " * 100  # 100 mentions
        score = score_content_relevance(text, "auth")
        # Max contribution from one word is 10 * 2 = 20 (before normalization)
        assert score > 0
        assert score < 100  # capped, not linear

    def test_no_qualifying_words_returns_zero(self):
        """Coverage: lines 307-308 — all words < 3 chars."""
        assert score_content_relevance("ab cd ef", "ab cd") == 0.0


# ===========================================================================
# get_git_context
# ===========================================================================


class TestGetGitContext:
    def test_no_git_returns_none(self, tmp_path):
        """Coverage: lines 218-220."""
        assert get_git_context(tmp_path) is None

    def test_with_git_repo(self, tmp_path):
        """Coverage: lines 218-283 — full git context extraction."""
        # Create a real git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, capture_output=True,
        )
        (tmp_path / "hello.py").write_text("print('hi')")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path, capture_output=True,
        )

        ctx = get_git_context(tmp_path)
        assert ctx is not None
        assert "git_root" in ctx
        assert ctx.get("branch") is not None
        assert "recent_commits" in ctx

    def test_git_context_from_subdirectory(self, tmp_path):
        """Git context found from nested directory."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, capture_output=True,
        )
        (tmp_path / "file.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path, capture_output=True,
        )
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        ctx = get_git_context(nested)
        assert ctx is not None
        assert ctx["git_root"] == str(tmp_path)


# ===========================================================================
# _find_git_root
# ===========================================================================


class TestFindGitRoot:
    def test_returns_none_outside_repo(self, tmp_path):
        assert _find_git_root(tmp_path) is None

    def test_walks_up_to_root(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (nested / "file.txt").write_text("x")
        root = _find_git_root(nested / "file.txt")
        assert root == tmp_path

    def test_file_path_resolves_parent(self, tmp_path):
        """Coverage: line 288 — file path uses parent for start."""
        (tmp_path / ".git").mkdir()
        f = tmp_path / "code.py"
        f.write_text("x")
        root = _find_git_root(f)
        assert root == tmp_path


# ===========================================================================
# run_local_research (async pipeline)
# ===========================================================================


class TestRunLocalResearch:
    @pytest.fixture
    def research_tree(self, tmp_path):
        """A minimal project tree for pipeline tests."""
        (tmp_path / "README.md").write_text("# Demo Project\nThis is a demo.\n")
        (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
        (tmp_path / "config.yaml").write_text("key: value\ndb: postgres\n")
        return tmp_path

    @pytest.mark.asyncio
    async def test_path_not_found(self, tmp_path):
        """Coverage: lines 362-369 — nonexistent path."""
        mock_db = AsyncMock()
        with patch("app.local_research.db", mock_db):
            await run_local_research(
                topic="test",
                path=str(tmp_path / "nonexistent"),
                job_id=1,
            )
        mock_db.update_job.assert_any_call(
            1, status="error", error=f"Path not found: {str(tmp_path / 'nonexistent')}"
        )

    @pytest.mark.asyncio
    async def test_no_files_found(self, tmp_path):
        """Coverage: lines 385-390 — no readable files in path."""
        # Create a dir with only binary/empty files
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        mock_db = AsyncMock()
        with patch("app.local_research.db", mock_db):
            await run_local_research(
                topic="test",
                path=str(empty_dir),
                job_id=2,
            )
        # Check that error status was set
        calls = [str(c) for c in mock_db.update_job.call_args_list]
        assert any("error" in c and "No readable files" in c for c in calls)

    @pytest.mark.asyncio
    async def test_successful_pipeline(self, research_tree):
        """Coverage: lines 371-543 — full successful pipeline."""
        mock_db = AsyncMock()

        with patch("app.local_research.db", mock_db), \
             patch("app.local_research.llm_chat", new_callable=AsyncMock) as mock_llm, \
             patch("app.storage.get_storage") as mock_get_storage:
            mock_llm.return_value = "A" * 200  # sufficiently long content
            mock_store = MagicMock()
            mock_get_storage.return_value = mock_store
            await run_local_research(
                topic="demo project",
                path=str(research_tree),
                job_id=10,
            )

        # Verify status progression
        statuses = [c.kwargs.get("status", c.args[1] if len(c.args) > 1 else None)
                     for c in mock_db.update_job.call_args_list]
        assert "scanning_files" in statuses
        assert "reading_files" in statuses
        assert "synthesizing" in statuses
        assert "complete" in statuses

        # Verify sources were saved
        mock_db.save_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure(self, research_tree):
        """Coverage: lines 494-499 — LLM synthesis failure."""
        mock_db = AsyncMock()

        with patch("app.local_research.db", mock_db), \
             patch("app.local_research.llm_chat", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM timeout")
            await run_local_research(
                topic="demo project",
                path=str(research_tree),
                job_id=11,
            )

        calls = [str(c) for c in mock_db.update_job.call_args_list]
        assert any("LLM synthesis failed" in c for c in calls)

    @pytest.mark.asyncio
    async def test_llm_insufficient_content(self, research_tree):
        """Coverage: lines 501-505 — LLM returns too little content."""
        mock_db = AsyncMock()

        with patch("app.local_research.db", mock_db), \
             patch("app.local_research.llm_chat", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "short"  # < 100 chars
            await run_local_research(
                topic="demo project",
                path=str(research_tree),
                job_id=12,
            )

        calls = [str(c) for c in mock_db.update_job.call_args_list]
        assert any("insufficient content" in c for c in calls)

    @pytest.mark.asyncio
    async def test_storage_write_failure(self, research_tree):
        """Coverage: lines 521-522 — storage write failure."""
        mock_db = AsyncMock()

        with patch("app.local_research.db", mock_db), \
             patch("app.local_research.llm_chat", new_callable=AsyncMock) as mock_llm, \
             patch("app.storage.get_storage") as mock_get_storage:
            mock_llm.return_value = "B" * 200
            mock_store = MagicMock()
            mock_store.write_text.side_effect = RuntimeError("disk full")
            mock_get_storage.return_value = mock_store
            await run_local_research(
                topic="demo project",
                path=str(research_tree),
                job_id=13,
            )

        calls = [str(c) for c in mock_db.update_job.call_args_list]
        assert any("Failed to write output" in c for c in calls)

    @pytest.mark.asyncio
    async def test_file_pattern_filter(self, research_tree):
        """Coverage: lines 377-383 — file_pattern filters discovered files."""
        mock_db = AsyncMock()

        with patch("app.local_research.db", mock_db), \
             patch("app.local_research.llm_chat", new_callable=AsyncMock) as mock_llm, \
             patch("app.storage.get_storage") as mock_get_storage:
            mock_llm.return_value = "C" * 200
            mock_store = MagicMock()
            mock_get_storage.return_value = mock_store
            await run_local_research(
                topic="demo project",
                path=str(research_tree),
                job_id=14,
                file_pattern="*.py",
            )

        # Should have succeeded — there is a .py file
        calls = [str(c) for c in mock_db.update_job.call_args_list]
        assert any("complete" in c for c in calls)

    @pytest.mark.asyncio
    async def test_file_pattern_no_matches(self, research_tree):
        """Coverage: file_pattern that matches nothing triggers error."""
        mock_db = AsyncMock()

        with patch("app.local_research.db", mock_db):
            await run_local_research(
                topic="demo project",
                path=str(research_tree),
                job_id=15,
                file_pattern="*.nonexistent",
            )

        calls = [str(c) for c in mock_db.update_job.call_args_list]
        assert any("No readable files" in c for c in calls)
