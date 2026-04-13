"""Unit tests for app.worker — every arq task driven with a fake ctx
and mocked externals.

The worker tasks are thin orchestration over:
    - app.research.run_research / run_research_collect / run_research_synthesize
    - app.wiki.create_or_update_article
    - app.quality.{enrich_article,fact_check_article,freshness_audit,...}
    - app.embeddings.embed_article / embed_all_articles
    - app.knowledge_graph.build_graph_for_article / build_full_graph
    - app.db.*

We mock each of those at the worker-module namespace (post-import) so
we can assert side effects without running the real pipelines.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app import db, worker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    """Initialize a clean sqlite DB for each test.

    Points DB_PATH at a tmp file so nothing leaks between tests.
    """
    db_path = tmp_path / "worker-test.db"
    monkeypatch.setattr("app.config.DB_PATH", db_path)
    monkeypatch.setattr("app.db.DB_PATH", db_path, raising=False)
    await db.init_db()
    yield
    # No teardown — tmp_path goes away automatically.


@pytest.fixture
def fake_ctx():
    """Minimal arq ctx dict with a mock redis pool."""
    return {"redis": AsyncMock()}


# ---------------------------------------------------------------------------
# research_task — full-pipeline happy/error/partial paths
# ---------------------------------------------------------------------------


class TestResearchTask:
    @pytest.mark.asyncio
    async def test_happy_path_updates_job_and_wiki(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        job_id = await db.create_job("kubernetes networking")

        async def fake_run_research(topic, jid, kb_name="personal"):
            # Simulate a successful pipeline run
            await db.update_job(
                jid, status="complete", content="# Networking\n\nBody.",
            )

        fake_create = AsyncMock(return_value=("kubernetes-networking", "created"))
        fake_log = AsyncMock()
        fake_embed = AsyncMock()
        fake_graph = AsyncMock()

        monkeypatch.setattr(worker, "run_research", fake_run_research)
        monkeypatch.setattr(worker, "create_or_update_article", fake_create)
        monkeypatch.setattr(worker.db, "log_article_update", fake_log)
        monkeypatch.setattr(worker, "embed_article", fake_embed)
        monkeypatch.setattr(worker, "build_graph_for_article", fake_graph)

        result = await worker.research_task(
            fake_ctx, "kubernetes networking", job_id, "personal",
        )

        assert result == {"job_id": job_id, "status": "complete"}
        fake_create.assert_awaited_once()
        fake_log.assert_awaited_once()
        fake_embed.assert_awaited_once_with("personal", "kubernetes-networking")
        fake_graph.assert_awaited_once_with("personal", "kubernetes-networking")

        job = await db.get_job(job_id)
        assert job["added_to_wiki"] == 1

    @pytest.mark.asyncio
    async def test_pipeline_crash_marks_job_error(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        job_id = await db.create_job("doomed topic")

        async def boom(*a, **kw):
            raise RuntimeError("minimax down")

        monkeypatch.setattr(worker, "run_research", boom)

        result = await worker.research_task(fake_ctx, "doomed topic", job_id)
        assert result["status"] == "error"
        assert "minimax down" in result["error"]

        job = await db.get_job(job_id)
        assert job["status"] == "error"
        assert "Pipeline crash" in (job.get("error") or "")

    @pytest.mark.asyncio
    async def test_pipeline_returns_non_complete(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        job_id = await db.create_job("awaiting review topic")

        async def fake_run(topic, jid, kb_name="personal"):
            await db.update_job(jid, status="awaiting_review")

        monkeypatch.setattr(worker, "run_research", fake_run)
        # Should not attempt to create the wiki article.
        fake_create = AsyncMock()
        monkeypatch.setattr(worker, "create_or_update_article", fake_create)

        result = await worker.research_task(fake_ctx, "awaiting review topic", job_id)
        assert result["status"] == "awaiting_review"
        fake_create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wiki_merge_failure_sets_added_to_wiki_zero(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        job_id = await db.create_job("merge-fail topic")

        async def fake_run(topic, jid, kb_name="personal"):
            await db.update_job(jid, status="complete", content="# Doc\n")

        monkeypatch.setattr(worker, "run_research", fake_run)
        # create_or_update_article raises — merge path should catch it
        # and set added_to_wiki=0 without re-raising.
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(side_effect=Exception("disk full")),
        )
        monkeypatch.setattr(worker, "embed_article", AsyncMock())
        monkeypatch.setattr(worker, "build_graph_for_article", AsyncMock())

        result = await worker.research_task(fake_ctx, "merge-fail topic", job_id)
        # Task still reports complete — wiki merge is best-effort.
        assert result["status"] == "complete"
        job = await db.get_job(job_id)
        assert job["added_to_wiki"] == 0

    @pytest.mark.asyncio
    async def test_embed_failure_does_not_fail_task(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        """Embed is best-effort after a successful wiki merge — a failure
        should log and move on, not flip the task to error."""
        job_id = await db.create_job("embed-fail topic")

        async def fake_run(topic, jid, kb_name="personal"):
            await db.update_job(jid, status="complete", content="# Body")

        monkeypatch.setattr(worker, "run_research", fake_run)
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(return_value=("embed-fail-topic", "created")),
        )
        monkeypatch.setattr(worker.db, "log_article_update", AsyncMock())
        monkeypatch.setattr(
            worker, "embed_article",
            AsyncMock(side_effect=Exception("embedding circuit open")),
        )
        monkeypatch.setattr(worker, "build_graph_for_article", AsyncMock())

        result = await worker.research_task(fake_ctx, "embed-fail topic", job_id)
        assert result["status"] == "complete"
        job = await db.get_job(job_id)
        assert job["added_to_wiki"] == 1  # still counted as published


# ---------------------------------------------------------------------------
# research_collect_task + research_synthesize_task
# ---------------------------------------------------------------------------


class TestResearchCollectTask:
    @pytest.mark.asyncio
    async def test_collect_happy_path(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("collect topic")

        async def fake_collect(topic, jid, kb_name="personal"):
            await db.update_job(jid, status="awaiting_review")

        monkeypatch.setattr(worker, "run_research_collect", fake_collect)

        result = await worker.research_collect_task(
            fake_ctx, "collect topic", job_id,
        )
        assert result == {"job_id": job_id, "status": "awaiting_review"}

    @pytest.mark.asyncio
    async def test_collect_crash_marks_error(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("collect crash")

        async def boom(*a, **kw):
            raise ValueError("serper 429")

        monkeypatch.setattr(worker, "run_research_collect", boom)
        result = await worker.research_collect_task(fake_ctx, "x", job_id)
        assert result["status"] == "error"
        assert "serper 429" in result["error"]


class TestResearchSynthesizeTask:
    @pytest.mark.asyncio
    async def test_synthesize_happy_path(self, fresh_db, fake_ctx, monkeypatch):
        job_id = await db.create_job("synth topic")

        async def fake_synth(topic, jid):
            await db.update_job(
                jid, status="complete", content="# Synth output",
            )

        monkeypatch.setattr(worker, "run_research_synthesize", fake_synth)
        monkeypatch.setattr(
            worker, "create_or_update_article",
            AsyncMock(return_value=("synth-topic", "created")),
        )
        monkeypatch.setattr(worker.db, "log_article_update", AsyncMock())
        monkeypatch.setattr(worker, "embed_article", AsyncMock())
        monkeypatch.setattr(worker, "build_graph_for_article", AsyncMock())

        result = await worker.research_synthesize_task(
            fake_ctx, "synth topic", job_id,
        )
        assert result == {"job_id": job_id, "status": "complete"}

    @pytest.mark.asyncio
    async def test_synthesize_crash_marks_error(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        job_id = await db.create_job("synth crash")

        async def boom(topic, jid):
            raise TimeoutError("llm timeout")

        monkeypatch.setattr(worker, "run_research_synthesize", boom)

        result = await worker.research_synthesize_task(fake_ctx, "x", job_id)
        assert result["status"] == "error"
        assert "llm timeout" in result["error"]


# ---------------------------------------------------------------------------
# Quality tasks
# ---------------------------------------------------------------------------


class TestQualityTasks:
    @pytest.mark.asyncio
    async def test_enrich_task_calls_enrich_article(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        fake_enrich = AsyncMock(return_value={"ok": True, "slug": "x"})
        monkeypatch.setattr(worker, "enrich_article", fake_enrich)
        result = await worker.enrich_task(fake_ctx, "personal", "slug-a")
        fake_enrich.assert_awaited_once_with("personal", "slug-a")
        assert result == {"ok": True, "slug": "x"}

    @pytest.mark.asyncio
    async def test_crosslink_task_calls_add_crosslinks(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        fake = AsyncMock(return_value={"added": 3})
        monkeypatch.setattr(worker, "add_crosslinks", fake)
        result = await worker.crosslink_task(fake_ctx, "personal", "slug-a")
        fake.assert_awaited_once_with("personal", "slug-a")
        assert result["added"] == 3

    @pytest.mark.asyncio
    async def test_fact_check_task_calls_fact_checker(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        fake = AsyncMock(return_value={"verified": 5, "unverified": 1})
        monkeypatch.setattr(worker, "fact_check_article", fake)
        result = await worker.fact_check_task(fake_ctx, "personal", "slug-a")
        fake.assert_awaited_once_with("personal", "slug-a")
        assert result["verified"] == 5

    @pytest.mark.asyncio
    async def test_freshness_task_calls_freshness_audit(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        fake = AsyncMock(return_value={"stale": 2})
        monkeypatch.setattr(worker, "freshness_audit", fake)
        result = await worker.freshness_task(
            fake_ctx, "personal", "slug-a", auto_update=True,
        )
        fake.assert_awaited_once_with("personal", "slug-a", auto_update=True)
        assert result["stale"] == 2

    @pytest.mark.asyncio
    async def test_quality_task_calls_run_quality_pass(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        # quality_task pulls these 3 keys off the result dict for
        # logging — the test mock must provide them or the task
        # catches the KeyError and returns {"error": ...}.
        fake = AsyncMock(return_value={
            "articles_enriched": 5,
            "articles_crosslinked": 3,
            "total_words_added": 1200,
        })
        monkeypatch.setattr(worker, "run_quality_pass", fake)
        result = await worker.quality_task(fake_ctx, "personal", max_articles=10)
        fake.assert_awaited_once()
        assert result["articles_enriched"] == 5
        assert result["articles_crosslinked"] == 3

    @pytest.mark.asyncio
    async def test_quality_task_wraps_errors(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        monkeypatch.setattr(
            worker, "run_quality_pass",
            AsyncMock(side_effect=Exception("pass crashed")),
        )
        result = await worker.quality_task(fake_ctx, "personal")
        assert result == {"error": "pass crashed"}


# ---------------------------------------------------------------------------
# Embedding / graph tasks
# ---------------------------------------------------------------------------


class TestEmbedTasks:
    @pytest.mark.asyncio
    async def test_embed_article_task(self, fresh_db, fake_ctx, monkeypatch):
        fake = AsyncMock(return_value={"embedded": 1})
        monkeypatch.setattr(worker, "embed_article", fake)
        result = await worker.embed_article_task(fake_ctx, "personal", "slug-a")
        fake.assert_awaited_once_with("personal", "slug-a")

    @pytest.mark.asyncio
    async def test_embed_all_task(self, fresh_db, fake_ctx, monkeypatch):
        fake = AsyncMock(return_value={"count": 42})
        monkeypatch.setattr(worker, "embed_all_articles", fake)
        result = await worker.embed_all_task(fake_ctx, "personal")
        fake.assert_awaited_once_with("personal")
        assert result["count"] == 42

    @pytest.mark.asyncio
    async def test_build_graph_task(self, fresh_db, fake_ctx, monkeypatch):
        fake = AsyncMock(return_value={"entities": 100, "edges": 250})
        monkeypatch.setattr(worker, "build_full_graph", fake)
        result = await worker.build_graph_task(fake_ctx, "personal")
        fake.assert_awaited_once_with("personal")
        assert result["entities"] == 100


# ---------------------------------------------------------------------------
# Media tasks — subprocess-heavy, we only assert the orchestration
# ---------------------------------------------------------------------------


class TestMediaAudioTask:
    @pytest.mark.asyncio
    async def test_missing_audio_url_returns_error(self, fake_ctx):
        result = await worker.media_audio_task(fake_ctx, "", title="t")
        assert result == {"error": "audio_url is required"}

    @pytest.mark.asyncio
    async def test_transcription_failure_returns_error(
        self, fake_ctx, monkeypatch,
    ):
        from app import transcribe
        monkeypatch.setattr(
            transcribe, "transcribe_audio_url",
            AsyncMock(return_value=""),
        )
        result = await worker.media_audio_task(
            fake_ctx, "https://example.com/ep.mp3", title="Episode",
        )
        assert "transcription failed" in result["error"]

    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_error(
        self, fake_ctx, monkeypatch,
    ):
        from app import transcribe, media
        monkeypatch.setattr(
            transcribe, "transcribe_audio_url",
            AsyncMock(return_value="hello world"),
        )
        monkeypatch.setattr(
            media, "synthesize_transcript",
            AsyncMock(side_effect=Exception("llm down")),
        )
        result = await worker.media_audio_task(
            fake_ctx, "https://example.com/ep.mp3", title="Episode",
        )
        assert "synthesis failed" in result["error"]

    @pytest.mark.asyncio
    async def test_happy_path_returns_slug(
        self, fresh_db, fake_ctx, monkeypatch,
    ):
        from app import transcribe, media
        monkeypatch.setattr(
            transcribe, "transcribe_audio_url",
            AsyncMock(return_value="hello podcast"),
        )
        monkeypatch.setattr(
            media, "synthesize_transcript",
            AsyncMock(return_value="# Podcast Summary\n\nBody"),
        )
        # The media task re-imports create_or_update_article locally, so
        # patch it at the wiki module and also at worker's namespace.
        fake_create = AsyncMock(return_value=("podcast-ep-1", "created"))
        monkeypatch.setattr("app.wiki.create_or_update_article", fake_create)

        result = await worker.media_audio_task(
            fake_ctx, "https://example.com/ep.mp3", title="Podcast Ep 1",
        )
        assert result == {
            "slug": "podcast-ep-1",
            "change_type": "created",
            "title": "Podcast Ep 1",
        }


# ---------------------------------------------------------------------------
# Cron / discovery tasks — all return-on-error wrapped
# ---------------------------------------------------------------------------


class TestCronTasks:
    @pytest.mark.asyncio
    async def test_rss_discovery_wraps_errors(self, fake_ctx, monkeypatch):
        from app.sources import rss
        monkeypatch.setattr(
            rss, "run_rss_discovery",
            AsyncMock(side_effect=RuntimeError("feed down")),
        )
        result = await worker.rss_discovery_cron_task(fake_ctx)
        assert result == {"error": "feed down"}

    @pytest.mark.asyncio
    async def test_rss_discovery_happy_path(self, fake_ctx, monkeypatch):
        from app.sources import rss
        monkeypatch.setattr(
            rss, "run_rss_discovery",
            AsyncMock(return_value={"new": 5}),
        )
        result = await worker.rss_discovery_cron_task(fake_ctx)
        assert result == {"new": 5}

    @pytest.mark.asyncio
    async def test_auto_discovery_refill_wraps_errors(
        self, fake_ctx, monkeypatch,
    ):
        from app import auto_discovery
        monkeypatch.setattr(
            auto_discovery, "run_discovery_all",
            AsyncMock(side_effect=Exception("no KBs configured")),
        )
        result = await worker.auto_discovery_refill_task(fake_ctx)
        assert result == {"error": "no KBs configured"}

    @pytest.mark.asyncio
    async def test_auto_discovery_enqueue_passes_redis(
        self, fake_ctx, monkeypatch,
    ):
        from app import auto_discovery
        fake = AsyncMock(return_value={"enqueued": 3})
        monkeypatch.setattr(auto_discovery, "enqueue_next_candidates_all", fake)
        result = await worker.auto_discovery_enqueue_task(fake_ctx)
        fake.assert_awaited_once_with(fake_ctx["redis"])
        assert result == {"enqueued": 3}
