"""Extended unit tests for app.worker — covers tasks not in test_worker_tasks.py.

Targets: local_research_task, youtube_task, ingest_document_task,
ingest_directory_task, palace tasks, github tasks, freshness_batch_task,
local_research_cron_task, _persist_critique_claims, startup/shutdown,
sitemap/podcast discovery, auto_discovery_single_kb_task, WorkerSettings.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import db, worker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "worker-ext-test.db"
    monkeypatch.setattr("app.config.DB_PATH", db_path)
    monkeypatch.setattr("app.db.DB_PATH", db_path, raising=False)
    await db.init_db()
    yield


@pytest.fixture
def fake_ctx():
    return {"redis": AsyncMock()}


# ---------------------------------------------------------------------------
# startup / shutdown
# ---------------------------------------------------------------------------


class TestStartupShutdown:
    @pytest.mark.asyncio
    async def test_startup_initializes_db(self, fresh_db, monkeypatch):
        fake_init = AsyncMock()
        monkeypatch.setattr(worker.db, "init_db", fake_init)
        await worker.startup({})
        fake_init.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_does_not_raise(self):
        # shutdown is a no-op logger call; just ensure it runs.
        await worker.shutdown({})


# ---------------------------------------------------------------------------
# local_research_task
# ---------------------------------------------------------------------------


class TestLocalResearchTask:
    @pytest.mark.asyncio
    async def test_happy_path(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("local:myproject")

        async def fake_run(topic, path, jid, file_pattern=None):
            await db.update_job(jid, status="complete", content="# Local findings")

        monkeypatch.setattr(worker, "run_local_research", fake_run)
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(return_value=("myproject", "created")),
        )
        monkeypatch.setattr(worker.db, "log_article_update", AsyncMock())
        monkeypatch.setattr(worker, "embed_article", AsyncMock())
        monkeypatch.setattr(worker, "build_graph_for_article", AsyncMock())

        result = await worker.local_research_task(
            fake_ctx, "myproject", "/code/myproject", job_id, file_pattern="*.py",
        )
        assert result == {"job_id": job_id, "status": "complete"}

    @pytest.mark.asyncio
    async def test_crash_marks_error(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("local:crash")

        async def boom(*a, **kw):
            raise RuntimeError("disk full")

        monkeypatch.setattr(worker, "run_local_research", boom)

        result = await worker.local_research_task(
            fake_ctx, "crash", "/tmp/x", job_id,
        )
        assert result["status"] == "error"
        assert "disk full" in result["error"]

    @pytest.mark.asyncio
    async def test_non_complete_status(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("local:partial")

        async def fake_run(topic, path, jid, file_pattern=None):
            await db.update_job(jid, status="partial")

        monkeypatch.setattr(worker, "run_local_research", fake_run)

        result = await worker.local_research_task(
            fake_ctx, "partial", "/tmp/x", job_id,
        )
        assert result["status"] == "partial"

    @pytest.mark.asyncio
    async def test_wiki_merge_failure(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("local:mergefail")

        async def fake_run(topic, path, jid, file_pattern=None):
            await db.update_job(jid, status="complete", content="# Content")

        monkeypatch.setattr(worker, "run_local_research", fake_run)
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(side_effect=Exception("no space")),
        )
        monkeypatch.setattr(worker, "embed_article", AsyncMock())
        monkeypatch.setattr(worker, "build_graph_for_article", AsyncMock())

        result = await worker.local_research_task(
            fake_ctx, "mergefail", "/tmp/x", job_id,
        )
        assert result["status"] == "complete"
        job = await db.get_job(job_id)
        assert job["added_to_wiki"] == 0


# ---------------------------------------------------------------------------
# youtube_task
# ---------------------------------------------------------------------------


class TestYoutubeTask:
    @pytest.mark.asyncio
    async def test_happy_path(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("youtube:test")
        from app import media as media_mod
        monkeypatch.setattr(
            media_mod, "ingest_youtube",
            AsyncMock(return_value={"slug": "vid", "status": "ok"}),
        )
        result = await worker.youtube_task(fake_ctx, "https://youtu.be/x", job_id)
        assert result == {"slug": "vid", "status": "ok"}

    @pytest.mark.asyncio
    async def test_crash_marks_error(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("youtube:crash")
        from app import media as media_mod
        monkeypatch.setattr(
            media_mod, "ingest_youtube",
            AsyncMock(side_effect=Exception("yt-dlp segfault")),
        )
        result = await worker.youtube_task(fake_ctx, "https://youtu.be/x", job_id)
        assert result == {"error": "yt-dlp segfault"}
        job = await db.get_job(job_id)
        assert job["status"] == "error"


# ---------------------------------------------------------------------------
# ingest_document_task / ingest_directory_task
# ---------------------------------------------------------------------------


class TestIngestTasks:
    @pytest.mark.asyncio
    async def test_ingest_document_happy(self, fake_ctx, monkeypatch):
        from app import ingest
        monkeypatch.setattr(
            ingest, "ingest_document_url",
            AsyncMock(return_value={"slug": "doc-1", "pages": 10}),
        )
        result = await worker.ingest_document_task(
            fake_ctx, "https://example.com/paper.pdf", kb_name="personal",
        )
        assert result["slug"] == "doc-1"

    @pytest.mark.asyncio
    async def test_ingest_document_crash(self, fake_ctx, monkeypatch):
        from app import ingest
        monkeypatch.setattr(
            ingest, "ingest_document_url",
            AsyncMock(side_effect=Exception("bad pdf")),
        )
        result = await worker.ingest_document_task(
            fake_ctx, "https://example.com/bad.pdf",
        )
        assert result == {"error": "bad pdf"}

    @pytest.mark.asyncio
    async def test_ingest_directory_happy(self, fake_ctx, monkeypatch):
        from app import ingest
        monkeypatch.setattr(
            ingest, "ingest_open_directory",
            AsyncMock(return_value={"files": 5}),
        )
        result = await worker.ingest_directory_task(
            fake_ctx, "https://example.com/dir/", max_files=5,
        )
        assert result["files"] == 5

    @pytest.mark.asyncio
    async def test_ingest_directory_crash(self, fake_ctx, monkeypatch):
        from app import ingest
        monkeypatch.setattr(
            ingest, "ingest_open_directory",
            AsyncMock(side_effect=Exception("timeout")),
        )
        result = await worker.ingest_directory_task(
            fake_ctx, "https://example.com/dir/",
        )
        assert result == {"error": "timeout"}


# ---------------------------------------------------------------------------
# Palace tasks
# ---------------------------------------------------------------------------


class TestPalaceTasks:
    @pytest.mark.asyncio
    async def test_palace_classify_happy(self, fake_ctx, monkeypatch):
        from app import palace
        monkeypatch.setattr(
            palace, "classify_all_articles",
            AsyncMock(return_value={"classified": 20}),
        )
        result = await worker.palace_classify_task(fake_ctx, "personal")
        assert result["classified"] == 20

    @pytest.mark.asyncio
    async def test_palace_classify_crash(self, fake_ctx, monkeypatch):
        from app import palace
        monkeypatch.setattr(
            palace, "classify_all_articles",
            AsyncMock(side_effect=Exception("llm down")),
        )
        result = await worker.palace_classify_task(fake_ctx, "personal")
        assert result == {"error": "llm down"}

    @pytest.mark.asyncio
    async def test_palace_cluster_happy(self, fake_ctx, monkeypatch):
        from app import palace
        monkeypatch.setattr(
            palace, "cluster_rooms",
            AsyncMock(return_value={"rooms": 5}),
        )
        result = await worker.palace_cluster_task(fake_ctx, "personal")
        assert result["rooms"] == 5

    @pytest.mark.asyncio
    async def test_palace_cluster_crash(self, fake_ctx, monkeypatch):
        from app import palace
        monkeypatch.setattr(
            palace, "cluster_rooms",
            AsyncMock(side_effect=Exception("graph empty")),
        )
        result = await worker.palace_cluster_task(fake_ctx)
        assert result == {"error": "graph empty"}


# ---------------------------------------------------------------------------
# GitHub tasks
# ---------------------------------------------------------------------------


class TestGithubTasks:
    @pytest.mark.asyncio
    async def test_github_releases_no_new(self, fresh_db, fake_ctx, monkeypatch):
        from app import github_monitor
        monkeypatch.setattr(
            github_monitor, "find_new_releases",
            AsyncMock(return_value=[]),
        )
        result = await worker.github_releases_task(fake_ctx)
        assert result["new_releases"] == 0
        assert result["researched"] == 0

    @pytest.mark.asyncio
    async def test_github_releases_crash(self, fake_ctx, monkeypatch):
        from app import github_monitor
        monkeypatch.setattr(
            github_monitor, "find_new_releases",
            AsyncMock(side_effect=Exception("rate limited")),
        )
        result = await worker.github_releases_task(fake_ctx)
        assert result == {"error": "rate limited"}

    @pytest.mark.asyncio
    async def test_github_index_happy(self, fake_ctx, monkeypatch):
        from app import github_monitor
        monkeypatch.setattr(
            github_monitor, "index_own_repos",
            AsyncMock(return_value=[
                {"title": "Repo A", "content": "# Readme"},
            ]),
        )
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(return_value=("repo-a", "created")),
        )
        result = await worker.github_index_task(fake_ctx)
        assert result["repos_indexed"] == 1
        assert result["articles_added"] == 1

    @pytest.mark.asyncio
    async def test_github_index_crash(self, fake_ctx, monkeypatch):
        from app import github_monitor
        monkeypatch.setattr(
            github_monitor, "index_own_repos",
            AsyncMock(side_effect=Exception("auth error")),
        )
        result = await worker.github_index_task(fake_ctx)
        assert result == {"error": "auth error"}

    @pytest.mark.asyncio
    async def test_github_index_individual_merge_failure(self, fake_ctx, monkeypatch):
        from app import github_monitor
        monkeypatch.setattr(
            github_monitor, "index_own_repos",
            AsyncMock(return_value=[
                {"title": "Repo Fail", "content": "# Content"},
            ]),
        )
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(side_effect=Exception("merge error")),
        )
        result = await worker.github_index_task(fake_ctx)
        assert result["repos_indexed"] == 1
        assert result["articles_added"] == 0


# ---------------------------------------------------------------------------
# freshness_batch_task
# ---------------------------------------------------------------------------


class TestFreshnessBatchTask:
    @pytest.mark.asyncio
    async def test_happy_path(self, fake_ctx, monkeypatch):
        monkeypatch.setattr(
            worker, "run_freshness_audit",
            AsyncMock(return_value={"audited": 5, "updated": 2}),
        )
        result = await worker.freshness_batch_task(
            fake_ctx, "personal", max_articles=5, auto_update=True,
        )
        assert result["audited"] == 5

    @pytest.mark.asyncio
    async def test_crash(self, fake_ctx, monkeypatch):
        monkeypatch.setattr(
            worker, "run_freshness_audit",
            AsyncMock(side_effect=Exception("serper down")),
        )
        result = await worker.freshness_batch_task(fake_ctx, "personal")
        assert result == {"error": "serper down"}


# ---------------------------------------------------------------------------
# _persist_critique_claims
# ---------------------------------------------------------------------------


class TestPersistCritiqueClaims:
    @pytest.mark.asyncio
    async def test_no_claims_json_noop(self):
        job = {"id": 1}
        # Should not raise
        await worker._persist_critique_claims(job, "personal", "slug")

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_raise(self):
        job = {"id": 1, "claims_json": "not valid json {{{"}
        await worker._persist_critique_claims(job, "personal", "slug")

    @pytest.mark.asyncio
    async def test_empty_list_noop(self):
        job = {"id": 1, "claims_json": "[]"}
        await worker._persist_critique_claims(job, "personal", "slug")

    @pytest.mark.asyncio
    async def test_saves_supported_claim(self, monkeypatch):
        claims = [{"text": "Python is fast", "status": "supported"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        fake_save = AsyncMock()
        monkeypatch.setattr(worker.db, "save_claim", fake_save)

        await worker._persist_critique_claims(job, "personal", "test-slug")

        fake_save.assert_awaited_once_with(
            "personal", "test-slug", "Python is fast",
            claim_type="supported", confidence=0.9, status="verified",
        )

    @pytest.mark.asyncio
    async def test_saves_unsupported_claim(self, monkeypatch):
        claims = [{"text": "Claim X", "status": "unsupported"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        fake_save = AsyncMock()
        monkeypatch.setattr(worker.db, "save_claim", fake_save)

        await worker._persist_critique_claims(job, "kb", "slug")

        fake_save.assert_awaited_once_with(
            "kb", "slug", "Claim X",
            claim_type="unsupported", confidence=0.3, status="unverified",
        )

    @pytest.mark.asyncio
    async def test_saves_missing_claim(self, monkeypatch):
        claims = [{"text": "Missing info", "status": "missing"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        fake_save = AsyncMock()
        monkeypatch.setattr(worker.db, "save_claim", fake_save)

        await worker._persist_critique_claims(job, "kb", "slug")

        fake_save.assert_awaited_once_with(
            "kb", "slug", "Missing info",
            claim_type="missing", confidence=0.5, status="unverified",
        )

    @pytest.mark.asyncio
    async def test_skips_non_dict_claims(self, monkeypatch):
        claims = ["just a string", 42, {"text": "Real claim", "status": "supported"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        fake_save = AsyncMock()
        monkeypatch.setattr(worker.db, "save_claim", fake_save)

        await worker._persist_critique_claims(job, "kb", "slug")

        # Only the dict claim should have been saved
        assert fake_save.await_count == 1

    @pytest.mark.asyncio
    async def test_skips_empty_text_claims(self, monkeypatch):
        claims = [{"text": "", "status": "supported"}, {"text": "  ", "status": "supported"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        fake_save = AsyncMock()
        monkeypatch.setattr(worker.db, "save_claim", fake_save)

        await worker._persist_critique_claims(job, "kb", "slug")

        fake_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_failure_does_not_propagate(self, monkeypatch):
        claims = [{"text": "Claim A", "status": "supported"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        monkeypatch.setattr(
            worker.db, "save_claim",
            AsyncMock(side_effect=Exception("db locked")),
        )

        # Should not raise
        await worker._persist_critique_claims(job, "kb", "slug")

    @pytest.mark.asyncio
    async def test_unknown_status_defaults(self, monkeypatch):
        claims = [{"text": "Weird claim", "status": "banana"}]
        job = {"id": 1, "claims_json": json.dumps(claims)}

        fake_save = AsyncMock()
        monkeypatch.setattr(worker.db, "save_claim", fake_save)

        await worker._persist_critique_claims(job, "kb", "slug")

        fake_save.assert_awaited_once_with(
            "kb", "slug", "Weird claim",
            claim_type="banana", confidence=0.5, status="unverified",
        )


# ---------------------------------------------------------------------------
# local_research_cron_task
# ---------------------------------------------------------------------------


class TestLocalResearchCronTask:
    @pytest.mark.asyncio
    async def test_no_watches_configured(self, fake_ctx, monkeypatch):
        monkeypatch.delenv("LOCAL_RESEARCH_WATCHES", raising=False)
        result = await worker.local_research_cron_task(fake_ctx)
        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_invalid_json(self, fake_ctx, monkeypatch):
        monkeypatch.setenv("LOCAL_RESEARCH_WATCHES", "not json!")
        result = await worker.local_research_cron_task(fake_ctx)
        assert result["status"] == "error"
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_skips_entries_without_topic_or_path(self, fresh_db, fake_ctx, monkeypatch):
        watches = [{"topic": "", "path": "/x"}, {"topic": "t", "path": ""}]
        monkeypatch.setenv("LOCAL_RESEARCH_WATCHES", json.dumps(watches))
        result = await worker.local_research_cron_task(fake_ctx)
        assert result["watched"] == 2
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_respects_cooldown(self, fresh_db, fake_ctx, monkeypatch):
        watches = [{"topic": "proj", "path": "/code/proj"}]
        monkeypatch.setenv("LOCAL_RESEARCH_WATCHES", json.dumps(watches))
        monkeypatch.setattr(
            worker.db, "check_cooldown",
            AsyncMock(return_value={"id": 99}),
        )
        result = await worker.local_research_cron_task(fake_ctx)
        assert result["results"][0]["status"] == "cooldown"
        assert result["results"][0]["job_id"] == 99

    @pytest.mark.asyncio
    async def test_happy_path_runs_research(self, fresh_db, fake_ctx, monkeypatch):
        watches = [{"topic": "proj", "path": "/code/proj", "pattern": "*.py"}]
        monkeypatch.setenv("LOCAL_RESEARCH_WATCHES", json.dumps(watches))
        monkeypatch.setattr(
            worker.db, "check_cooldown", AsyncMock(return_value=None),
        )

        async def fake_run(topic, path, jid, file_pattern=None):
            await db.update_job(jid, status="complete", content="# Results")

        monkeypatch.setattr(worker, "run_local_research", fake_run)
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(return_value=("proj", "created")),
        )

        result = await worker.local_research_cron_task(fake_ctx)
        assert result["results"][0]["status"] == "complete"
        assert result["results"][0]["slug"] == "proj"


# ---------------------------------------------------------------------------
# Discovery cron tasks
# ---------------------------------------------------------------------------


class TestDiscoveryCronTasks:
    @pytest.mark.asyncio
    async def test_sitemap_discovery_happy(self, fake_ctx, monkeypatch):
        from app.sources import sitemap
        monkeypatch.setattr(
            sitemap, "run_sitemap_discovery",
            AsyncMock(return_value={"urls_found": 10}),
        )
        result = await worker.sitemap_discovery_cron_task(fake_ctx)
        assert result["urls_found"] == 10

    @pytest.mark.asyncio
    async def test_sitemap_discovery_crash(self, fake_ctx, monkeypatch):
        from app.sources import sitemap
        monkeypatch.setattr(
            sitemap, "run_sitemap_discovery",
            AsyncMock(side_effect=Exception("parse error")),
        )
        result = await worker.sitemap_discovery_cron_task(fake_ctx)
        assert result == {"error": "parse error"}

    @pytest.mark.asyncio
    async def test_podcast_discovery_happy(self, fake_ctx, monkeypatch):
        from app.sources import podcast
        monkeypatch.setattr(
            podcast, "run_podcast_discovery",
            AsyncMock(return_value={"episodes": 3}),
        )
        result = await worker.podcast_discovery_cron_task(fake_ctx)
        assert result["episodes"] == 3

    @pytest.mark.asyncio
    async def test_podcast_discovery_crash(self, fake_ctx, monkeypatch):
        from app.sources import podcast
        monkeypatch.setattr(
            podcast, "run_podcast_discovery",
            AsyncMock(side_effect=Exception("feed timeout")),
        )
        result = await worker.podcast_discovery_cron_task(fake_ctx)
        assert result == {"error": "feed timeout"}

    @pytest.mark.asyncio
    async def test_auto_discovery_single_kb_happy(self, fake_ctx, monkeypatch):
        from app import auto_discovery
        monkeypatch.setattr(
            auto_discovery, "run_discovery_for_kb",
            AsyncMock(return_value={"candidates": 5}),
        )
        monkeypatch.setattr(
            auto_discovery, "enqueue_next_candidates_for_kb",
            AsyncMock(return_value={"enqueued": 2}),
        )
        result = await worker.auto_discovery_single_kb_task(fake_ctx, "mykb")
        assert result["discovery"]["candidates"] == 5
        assert result["enqueue"]["enqueued"] == 2

    @pytest.mark.asyncio
    async def test_auto_discovery_single_kb_crash(self, fake_ctx, monkeypatch):
        from app import auto_discovery
        monkeypatch.setattr(
            auto_discovery, "run_discovery_for_kb",
            AsyncMock(side_effect=Exception("no kb")),
        )
        result = await worker.auto_discovery_single_kb_task(fake_ctx, "bad")
        assert result == {"error": "no kb", "kb": "bad"}


# ---------------------------------------------------------------------------
# WorkerSettings validation
# ---------------------------------------------------------------------------


class TestWorkerSettings:
    def test_has_all_task_functions(self):
        func_names = {f.__name__ for f in worker.WorkerSettings.functions}
        expected = {
            "research_task", "research_collect_task", "research_synthesize_task",
            "local_research_task", "quality_task", "enrich_task", "crosslink_task",
            "fact_check_task", "freshness_task", "freshness_batch_task",
            "github_releases_task", "github_index_task", "youtube_task",
            "embed_all_task", "build_graph_task", "embed_article_task",
            "ingest_document_task", "ingest_directory_task",
            "palace_classify_task", "palace_cluster_task",
            "auto_discovery_refill_task", "auto_discovery_enqueue_task",
            "auto_discovery_single_kb_task",
            "rss_discovery_cron_task", "sitemap_discovery_cron_task",
            "podcast_discovery_cron_task", "media_audio_task",
        }
        assert expected.issubset(func_names)

    def test_has_cron_jobs(self):
        assert len(worker.WorkerSettings.cron_jobs) > 0

    def test_has_lifecycle_hooks(self):
        assert worker.WorkerSettings.on_startup is worker.startup
        assert worker.WorkerSettings.on_shutdown is worker.shutdown

    def test_job_timeout(self):
        assert worker.WorkerSettings.job_timeout == 600

    def test_max_jobs(self):
        assert worker.WorkerSettings.max_jobs == 15
