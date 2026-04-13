"""
arq worker: task definitions and worker settings.

The worker connects to Redis, picks up research jobs, runs the pipeline,
and on completion writes results into the wiki via the merge logic.
"""

import logging
from app.logging_config import setup_logging
setup_logging()
from arq import cron
from arq.connections import RedisSettings

from app.config import REDIS_HOST, REDIS_PORT, ARQ_QUEUE_NAME
from app.research import run_research, run_research_collect, run_research_synthesize
from app.local_research import run_local_research
from app.wiki import create_or_update_article
from app.quality import (
    run_quality_pass, enrich_article, add_crosslinks,
    fact_check_article, freshness_audit, run_freshness_audit,
)
from app.embeddings import embed_article, embed_all_articles
from app.knowledge_graph import build_graph_for_article, build_full_graph
from app import db

logger = logging.getLogger("kb-service.worker")


async def startup(ctx: dict) -> None:
    """Called once when the worker starts."""
    await db.init_db()
    logger.info("Worker started, DB initialized")


async def shutdown(ctx: dict) -> None:
    """Called once when the worker shuts down."""
    logger.info("Worker shutting down")


async def agent_research_task(ctx: dict, topic: str, job_id: int, kb_name: str = "personal") -> dict:
    """Run the DeepAgents research agent for a single topic.

    The agent adaptively decides how many rounds of search to do, which
    sources to read deeply, and whether to fact-check — replacing the
    hardcoded 3-round pipeline.  It calls write_article directly, so
    wiki merge happens inside the agent loop.
    """
    logger.info("Starting AGENT research: job_id=%d, topic='%s', kb='%s'", job_id, topic, kb_name)
    try:
        from app.agent import run_agent_research
        await run_agent_research(topic, job_id, kb=kb_name)
    except Exception as exc:
        logger.exception("Agent research crashed for job %d", job_id)
        await db.update_job(job_id, status="error", error=f"Agent crash: {exc}")
        return {"job_id": job_id, "status": "error", "error": str(exc)}

    job = await db.get_job(job_id)
    if job and job["status"] == "complete":
        # Auto-embed the article the agent wrote
        try:
            from app.wiki import find_related_article
            from app.embeddings import embed_article
            from app.knowledge_graph import build_graph_for_article
            existing = find_related_article(kb_name, topic)
            if existing:
                slug = existing["slug"]
                await embed_article(kb_name, slug)
                await build_graph_for_article(kb_name, slug)
        except Exception as embed_exc:
            logger.warning("Auto-embed after agent research failed: %s", embed_exc)

    return {"job_id": job_id, "status": job["status"] if job else "unknown"}


async def research_task(ctx: dict, topic: str, job_id: int, kb_name: str = "personal") -> dict:
    """Run the full research pipeline for a single topic (legacy scripted pipeline).

    Steps:
      1. Execute 3-round search + synthesis via run_research()
      2. On success, merge the result into the wiki via create_or_update_article()
      3. Log the article update in SQLite
    """
    logger.info("Starting research task: job_id=%d, topic='%s', kb='%s'", job_id, topic, kb_name)

    try:
        await run_research(topic, job_id, kb_name=kb_name)
    except Exception as exc:
        logger.exception("Research pipeline crashed for job %d", job_id)
        await db.update_job(job_id, status="error", error=f"Pipeline crash: {exc}")
        return {"job_id": job_id, "status": "error", "error": str(exc)}

    # Check if the job completed successfully
    job = await db.get_job(job_id)
    if not job or job["status"] != "complete":
        return {"job_id": job_id, "status": job["status"] if job else "unknown"}

    # Auto-add to wiki with merge logic
    content = job.get("content", "")
    if content:
        try:
            slug, change_type = await create_or_update_article(kb_name, topic, content, source_type="web")
            await db.update_job(job_id, added_to_wiki=1)
            await db.log_article_update(
                article_slug=slug,
                kb_name=kb_name,
                job_id=job_id,
                change_type=change_type,
            )
            logger.info(
                "Wiki %s: %s/%s (job %d)", change_type, kb_name, slug, job_id,
            )

            # Persist critique claims to article_claims now that we know
            # the article slug. Failures are logged-only — claims are a
            # quality signal, not a publishing blocker.
            await _persist_critique_claims(job, kb_name, slug)

            # Auto-embed the new/updated article
            try:
                await embed_article(kb_name, slug)
                await build_graph_for_article(kb_name, slug)
            except Exception as embed_exc:
                logger.warning("Auto-embed failed for %s: %s", slug, embed_exc)
        except Exception as exc:
            logger.warning("Auto-add to wiki failed for job %d: %s", job_id, exc)
            await db.update_job(job_id, added_to_wiki=0)

    return {"job_id": job_id, "status": "complete"}


async def _persist_critique_claims(job: dict, kb_name: str, slug: str) -> None:
    """Write the synthesis critique claims to article_claims.

    The critique pass in app/research.py:_synthesize stashes a JSON list
    on the job row's claims_json column. Once the wiki merge has resolved
    the final slug, we can persist each claim. Status mapping:
        supported   → status='verified',  confidence=0.9
        unsupported → status='unverified', confidence=0.3
        missing     → status='unverified', confidence=0.5  (with a note marker)
    """
    raw = job.get("claims_json")
    if not raw:
        return
    try:
        import json as _json
        claims = _json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to parse claims_json for job %d: %s", job.get("id"), exc)
        return
    if not isinstance(claims, list) or not claims:
        return

    status_to_confidence = {
        "supported": (0.9, "verified"),
        "unsupported": (0.3, "unverified"),
        "missing": (0.5, "unverified"),
    }
    saved = 0
    for c in claims:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        crit_status = (c.get("status") or "unsupported").lower()
        confidence, claim_status = status_to_confidence.get(
            crit_status, (0.5, "unverified"),
        )
        try:
            await db.save_claim(
                kb_name, slug, text,
                claim_type=crit_status,
                confidence=confidence,
                status=claim_status,
            )
            saved += 1
        except Exception as exc:
            logger.warning("Failed to save claim for %s/%s: %s", kb_name, slug, exc)
    if saved:
        logger.info("Persisted %d critique claims for %s/%s", saved, kb_name, slug)


async def research_collect_task(ctx: dict, topic: str, job_id: int, kb_name: str = "personal") -> dict:
    """Collect sources only, pause for user review."""
    logger.info("Starting source collection: job_id=%d, topic='%s', kb='%s'", job_id, topic, kb_name)
    try:
        await run_research_collect(topic, job_id, kb_name=kb_name)
    except Exception as exc:
        logger.exception("Source collection crashed for job %d", job_id)
        await db.update_job(job_id, status="error", error=f"Collection crash: {exc}")
        return {"job_id": job_id, "status": "error", "error": str(exc)}

    job = await db.get_job(job_id)
    return {"job_id": job_id, "status": job["status"] if job else "unknown"}


async def research_synthesize_task(ctx: dict, topic: str, job_id: int, kb_name: str = "personal") -> dict:
    """Synthesize an article from the user-reviewed source set."""
    logger.info("Starting synthesis from reviewed sources: job_id=%d, topic='%s', kb='%s'", job_id, topic, kb_name)
    try:
        await run_research_synthesize(topic, job_id)
    except Exception as exc:
        logger.exception("Synthesis crashed for job %d", job_id)
        await db.update_job(job_id, status="error", error=f"Synthesis crash: {exc}")
        return {"job_id": job_id, "status": "error", "error": str(exc)}

    job = await db.get_job(job_id)
    if not job or job["status"] != "complete":
        return {"job_id": job_id, "status": job["status"] if job else "unknown"}

    # Auto-add to wiki
    content = job.get("content", "")
    if content:
        try:
            slug, change_type = await create_or_update_article(kb_name, topic, content, source_type="web")
            await db.update_job(job_id, added_to_wiki=1)
            await db.log_article_update(
                article_slug=slug, kb_name=kb_name,
                job_id=job_id, change_type=change_type,
            )
            logger.info("Wiki %s: %s/%s (job %d, reviewed)", change_type, kb_name, slug, job_id)
            # Persist critique claims (same path as research_task).
            await _persist_critique_claims(job, kb_name, slug)
            try:
                await embed_article(kb_name, slug)
                await build_graph_for_article(kb_name, slug)
            except Exception as embed_exc:
                logger.warning("Auto-embed failed for %s: %s", slug, embed_exc)
        except Exception as exc:
            logger.warning("Auto-add to wiki failed for job %d: %s", job_id, exc)
            await db.update_job(job_id, added_to_wiki=0)

    return {"job_id": job_id, "status": "complete"}


async def local_research_task(
    ctx: dict, topic: str, path: str, job_id: int,
    file_pattern: str | None = None,
) -> dict:
    """Run local research: scan files/folders/git repos and synthesize findings."""
    logger.info("Starting local research: job_id=%d, topic='%s', path='%s'", job_id, topic, path)

    try:
        await run_local_research(topic, path, job_id, file_pattern=file_pattern)
    except Exception as exc:
        logger.exception("Local research crashed for job %d", job_id)
        await db.update_job(job_id, status="error", error=f"Pipeline crash: {exc}")
        return {"job_id": job_id, "status": "error", "error": str(exc)}

    job = await db.get_job(job_id)
    if not job or job["status"] != "complete":
        return {"job_id": job_id, "status": job["status"] if job else "unknown"}

    # Auto-add to wiki
    content = job.get("content", "")
    if content:
        try:
            slug, change_type = await create_or_update_article("personal", topic, content, source_type="local")
            await db.update_job(job_id, added_to_wiki=1)
            await db.log_article_update(
                article_slug=slug, kb_name="personal",
                job_id=job_id, change_type=change_type,
            )
            logger.info("Wiki %s: %s (local research job %d)", change_type, slug, job_id)
            try:
                await embed_article("personal", slug)
                await build_graph_for_article("personal", slug)
            except Exception as embed_exc:
                logger.warning("Auto-embed failed for %s: %s", slug, embed_exc)
        except Exception as exc:
            logger.warning("Auto-add to wiki failed for local job %d: %s", job_id, exc)
            await db.update_job(job_id, added_to_wiki=0)

    return {"job_id": job_id, "status": "complete"}


async def media_audio_task(ctx: dict, audio_url: str, title: str = "") -> dict:
    """Download an audio URL, transcribe it, and turn the transcript
    into a wiki article via the existing media-synthesis pipeline.

    Used by the podcast feed source and as a fallback for YouTube
    videos with no captions. Returns the slug of the resulting article
    or an error dict.
    """
    from app.transcribe import transcribe_audio_url
    from app.media import synthesize_transcript
    from app.wiki import create_or_update_article

    if not audio_url:
        return {"error": "audio_url is required"}

    logger.info("Transcribing audio: %s", audio_url[:80])
    transcript = await transcribe_audio_url(audio_url)
    if not transcript:
        return {"error": "transcription failed or returned empty"}

    article_title = title.strip() or audio_url.rsplit("/", 1)[-1] or "Audio transcript"

    # Synthesize the transcript into a structured article via the same
    # helper used by the YouTube task.
    try:
        content = await synthesize_transcript(
            article_title, "(unknown)", transcript, audio_url,
        )
    except Exception as exc:
        logger.exception("Audio transcript synthesis failed")
        return {"error": f"synthesis failed: {exc}"}

    try:
        slug, change_type = await create_or_update_article(
            "personal", article_title, content, source_type="media",
        )
    except Exception as exc:
        logger.exception("Audio transcript merge failed")
        return {"error": f"wiki merge failed: {exc}"}

    return {"slug": slug, "change_type": change_type, "title": article_title}


async def rss_discovery_cron_task(ctx: dict) -> dict:
    """Cron task: passive RSS / Atom feed discovery.

    Walks every feed in ``RSS_WATCHES`` once a day, matches entries
    against existing articles, and queues unmatched titles as
    ``topic_candidate`` rows with ``source='rss'``.
    """
    from app.sources.rss import run_rss_discovery
    try:
        return await run_rss_discovery()
    except Exception as exc:
        logger.exception("RSS discovery crashed")
        return {"error": str(exc)}


async def podcast_discovery_cron_task(ctx: dict) -> dict:
    """Cron task: weekly podcast feed discovery.

    Walks every feed in ``PODCAST_WATCHES``, finds new episodes, and
    enqueues a ``media_audio_task`` for each so the worker downloads
    + transcribes the audio off the cron path.
    """
    from app.sources.podcast import run_podcast_discovery
    try:
        return await run_podcast_discovery(arq_pool=ctx.get("redis"))
    except Exception as exc:
        logger.exception("Podcast discovery crashed")
        return {"error": str(exc)}


async def sitemap_discovery_cron_task(ctx: dict) -> dict:
    """Cron task: weekly sitemap.xml walks.

    Walks every sitemap in ``SITEMAP_WATCHES``, finds URLs not yet in
    ``research_sources``, and queues them as ``topic_candidate`` rows
    with ``source='sitemap'``.
    """
    from app.sources.sitemap import run_sitemap_discovery
    try:
        return await run_sitemap_discovery()
    except Exception as exc:
        logger.exception("Sitemap discovery crashed")
        return {"error": str(exc)}


async def auto_discovery_refill_task(ctx: dict) -> dict:
    """Cron task: regenerate topic candidates for every enabled KB."""
    from app.auto_discovery import run_discovery_all
    try:
        return await run_discovery_all()
    except Exception as exc:
        logger.exception("auto_discovery refill crashed")
        return {"error": str(exc)}


async def auto_discovery_enqueue_task(ctx: dict) -> dict:
    """Cron task: drain candidate queues into research_task jobs.

    Runs every hour. Respects per-KB Serper budgets and max-per-hour caps.
    """
    from app.auto_discovery import enqueue_next_candidates_all
    try:
        return await enqueue_next_candidates_all(ctx["redis"])
    except Exception as exc:
        logger.exception("auto_discovery enqueue crashed")
        return {"error": str(exc)}


async def auto_discovery_single_kb_task(ctx: dict, kb: str) -> dict:
    """One-shot discovery + enqueue for a single KB, triggered by the
    web handler. Kept on the worker (not the API) because the discovery
    pipeline can run for minutes with heavy DynamoDB scans."""
    from app.auto_discovery import (
        run_discovery_for_kb, enqueue_next_candidates_for_kb,
    )
    try:
        discovery = await run_discovery_for_kb(kb)
        enqueue = await enqueue_next_candidates_for_kb(ctx["redis"], kb)
        return {"discovery": discovery, "enqueue": enqueue}
    except Exception as exc:
        logger.exception("auto_discovery single-kb task crashed (kb=%s)", kb)
        return {"error": str(exc), "kb": kb}


async def local_research_cron_task(ctx: dict) -> dict:
    """Cron task: run local research on all configured watch paths.

    Configure via LOCAL_RESEARCH_WATCHES env var as JSON:
    [{"topic": "my project", "path": "/code/myproject", "pattern": "*.py"}]
    """
    import json as _json
    import os

    watches_raw = os.getenv("LOCAL_RESEARCH_WATCHES", "").strip()
    if not watches_raw:
        logger.debug("No LOCAL_RESEARCH_WATCHES configured, skipping cron")
        return {"status": "skipped", "reason": "no watches configured"}

    try:
        watches = _json.loads(watches_raw)
    except _json.JSONDecodeError as exc:
        logger.warning("Invalid LOCAL_RESEARCH_WATCHES JSON: %s", exc)
        return {"status": "error", "error": f"Invalid JSON: {exc}"}

    results = []
    for watch in watches:
        topic = watch.get("topic", "")
        path = watch.get("path", "")
        pattern = watch.get("pattern")

        if not topic or not path:
            continue

        # Check cooldown — skip if researched recently
        existing = await db.check_cooldown(f"local:{topic}")
        if existing:
            results.append({"topic": topic, "status": "cooldown", "job_id": existing["id"]})
            continue

        job_id = await db.create_job(
            f"local:{topic}",
            job_type="local",
            source_params=_json.dumps({"path": path, "pattern": pattern}),
        )
        try:
            await run_local_research(topic, path, job_id, file_pattern=pattern)
            job = await db.get_job(job_id)
            if job and job.get("content"):
                slug, change_type = await create_or_update_article("personal", topic, job["content"], source_type="local")
                await db.update_job(job_id, added_to_wiki=1)
                results.append({"topic": topic, "status": "complete", "slug": slug})
            else:
                results.append({"topic": topic, "status": job["status"] if job else "unknown"})
        except Exception as exc:
            logger.warning("Local research cron failed for %s: %s", topic, exc)
            results.append({"topic": topic, "status": "error", "error": str(exc)})

    return {"watched": len(watches), "results": results}


async def quality_task(ctx: dict, kb_name: str, max_articles: int = 10) -> dict:
    """Run a quality improvement pass on a KB.

    Finds shallow articles, enriches them with Minimax, and adds crosslinks.
    """
    logger.info("Starting quality pass: kb=%s, max=%d", kb_name, max_articles)
    try:
        result = await run_quality_pass(kb_name, max_articles=max_articles)
        logger.info(
            "Quality pass complete: %d enriched, %d crosslinked, +%d words",
            result["articles_enriched"],
            result["articles_crosslinked"],
            result["total_words_added"],
        )
        return result
    except Exception as exc:
        logger.exception("Quality pass crashed for %s", kb_name)
        return {"error": str(exc)}


async def enrich_task(ctx: dict, kb_name: str, slug: str) -> dict:
    """Enrich a single article."""
    logger.info("Enriching article: %s/%s", kb_name, slug)
    try:
        return await enrich_article(kb_name, slug)
    except Exception as exc:
        logger.exception("Enrich failed for %s/%s", kb_name, slug)
        return {"error": str(exc)}


async def crosslink_task(ctx: dict, kb_name: str, slug: str) -> dict:
    """Add crosslinks to a single article."""
    logger.info("Crosslinking article: %s/%s", kb_name, slug)
    try:
        return await add_crosslinks(kb_name, slug)
    except Exception as exc:
        logger.exception("Crosslink failed for %s/%s", kb_name, slug)
        return {"error": str(exc)}


async def fact_check_task(ctx: dict, kb_name: str, slug: str) -> dict:
    """Fact-check a single article against current Serper results."""
    logger.info("Fact-checking: %s/%s", kb_name, slug)
    try:
        return await fact_check_article(kb_name, slug)
    except Exception as exc:
        logger.exception("Fact check failed for %s/%s", kb_name, slug)
        return {"error": str(exc)}


async def freshness_task(ctx: dict, kb_name: str, slug: str, auto_update: bool = True) -> dict:
    """Freshness audit + auto-update for a single article."""
    logger.info("Freshness audit: %s/%s (auto_update=%s)", kb_name, slug, auto_update)
    try:
        return await freshness_audit(kb_name, slug, auto_update=auto_update)
    except Exception as exc:
        logger.exception("Freshness audit failed for %s/%s", kb_name, slug)
        return {"error": str(exc)}


async def freshness_batch_task(ctx: dict, kb_name: str, max_articles: int = 10, auto_update: bool = True) -> dict:
    """Batch freshness audit across stalest articles."""
    logger.info("Batch freshness audit: kb=%s, max=%d", kb_name, max_articles)
    try:
        return await run_freshness_audit(kb_name, max_articles=max_articles, auto_update=auto_update)
    except Exception as exc:
        logger.exception("Batch freshness audit crashed for %s", kb_name)
        return {"error": str(exc)}


async def github_releases_task(ctx: dict) -> dict:
    """Check all tracked repos for new releases and auto-research them."""
    from app.github_monitor import find_new_releases
    logger.info("Checking GitHub releases...")
    try:
        new = await find_new_releases()
        logger.info("Found %d repos with new releases", len(new))

        # Auto-dispatch research for each new release
        researched = []
        for rel in new[:10]:  # max 10 per run
            topic = f"{rel['repo'].split('/')[-1]} {rel['tag']} release notes new features"
            job_id = await db.create_job(topic)
            # Run research inline (we're already in a worker)
            try:
                await run_research(topic, job_id)
                job = await db.get_job(job_id)
                if job and job.get("content"):
                    slug, change_type = await create_or_update_article("personal", topic, job["content"])
                    await db.update_job(job_id, added_to_wiki=1)
                    researched.append({"repo": rel["repo"], "tag": rel["tag"], "slug": slug})
            except Exception as exc:
                logger.warning("Release research failed for %s: %s", rel["repo"], exc)

        return {"new_releases": len(new), "researched": len(researched), "details": researched}
    except Exception as exc:
        logger.exception("GitHub releases check failed")
        return {"error": str(exc)}


async def github_index_task(ctx: dict) -> dict:
    """Index READMEs and changelogs from your own repos."""
    from app.github_monitor import index_own_repos
    logger.info("Indexing own GitHub repos...")
    try:
        docs = await index_own_repos()
        added = 0
        for doc in docs:
            try:
                slug, change_type = await create_or_update_article("personal", doc["title"], doc["content"])
                added += 1
            except Exception as exc:
                logger.warning("Failed to add %s: %s", doc["title"], exc)

        return {"repos_indexed": len(docs), "articles_added": added}
    except Exception as exc:
        logger.exception("Repo indexing failed")
        return {"error": str(exc)}


async def youtube_task(ctx: dict, url: str, job_id: int) -> dict:
    """Ingest a YouTube video: extract transcript → synthesize → wiki."""
    from app.media import ingest_youtube
    logger.info("YouTube ingest: job=%d url=%s", job_id, url)
    try:
        return await ingest_youtube(url, job_id=job_id)
    except Exception as exc:
        logger.exception("YouTube ingest failed for %s", url)
        await db.update_job(job_id, status="error", error=f"YouTube ingest: {exc}")
        return {"error": str(exc)}


async def embed_all_task(ctx: dict, kb_name: str) -> dict:
    """Embed all articles in a KB."""
    logger.info("Embedding all articles: kb=%s", kb_name)
    try:
        return await embed_all_articles(kb_name)
    except Exception as exc:
        logger.exception("Embed all failed for %s", kb_name)
        return {"error": str(exc)}


async def build_graph_task(ctx: dict, kb_name: str) -> dict:
    """Build knowledge graph for all articles in a KB."""
    logger.info("Building knowledge graph: kb=%s", kb_name)
    try:
        return await build_full_graph(kb_name)
    except Exception as exc:
        logger.exception("Graph build failed for %s", kb_name)
        return {"error": str(exc)}


async def embed_article_task(ctx: dict, kb_name: str, slug: str) -> dict:
    """Embed a single article (called on article create/update)."""
    logger.info("Embedding article: %s/%s", kb_name, slug)
    try:
        result = await embed_article(kb_name, slug)
        # Also update graph for this article
        try:
            await build_graph_for_article(kb_name, slug)
        except Exception as graph_exc:
            logger.warning("Graph update failed for %s/%s: %s", kb_name, slug, graph_exc)
        return result
    except Exception as exc:
        logger.exception("Embed article failed for %s/%s", kb_name, slug)
        return {"error": str(exc)}


async def ingest_document_task(ctx: dict, url: str, kb_name: str = "personal") -> dict:
    """Ingest a single document URL (PDF/ePub)."""
    from app.ingest import ingest_document_url
    logger.info("Ingesting document: %s", url)
    try:
        return await ingest_document_url(url, kb_name)
    except Exception as exc:
        logger.exception("Document ingest failed: %s", url)
        return {"error": str(exc)}


async def ingest_directory_task(ctx: dict, base_url: str, kb_name: str = "personal", max_files: int = 10) -> dict:
    """Crawl and ingest documents from an open directory."""
    from app.ingest import ingest_open_directory
    logger.info("Ingesting directory: %s", base_url)
    try:
        return await ingest_open_directory(base_url, kb_name, max_files)
    except Exception as exc:
        logger.exception("Directory ingest failed: %s", base_url)
        return {"error": str(exc)}


async def palace_classify_task(ctx: dict, kb_name: str = "personal") -> dict:
    """Classify all articles in a KB into palace halls."""
    from app.palace import classify_all_articles
    logger.info("Palace classify: %s", kb_name)
    try:
        return await classify_all_articles(kb_name)
    except Exception as exc:
        logger.exception("Palace classify failed: %s", kb_name)
        return {"error": str(exc)}


async def palace_cluster_task(ctx: dict, kb_name: str = "personal") -> dict:
    """Build palace rooms from knowledge graph data."""
    from app.palace import cluster_rooms
    logger.info("Palace cluster: %s", kb_name)
    try:
        return await cluster_rooms(kb_name)
    except Exception as exc:
        logger.exception("Palace cluster failed: %s", kb_name)
        return {"error": str(exc)}


class WorkerSettings:
    """arq worker configuration."""

    functions = [
        agent_research_task,
        research_task, research_collect_task, research_synthesize_task,
        local_research_task, quality_task,
        enrich_task, crosslink_task,
        fact_check_task, freshness_task, freshness_batch_task,
        github_releases_task, github_index_task,
        youtube_task,
        embed_all_task, build_graph_task, embed_article_task,
        ingest_document_task, ingest_directory_task,
        palace_classify_task, palace_cluster_task,
        auto_discovery_refill_task, auto_discovery_enqueue_task,
        auto_discovery_single_kb_task,
        rss_discovery_cron_task, sitemap_discovery_cron_task,
        podcast_discovery_cron_task, media_audio_task,
    ]

    cron_jobs = [
        # Run local research watches daily at 4am UTC
        cron(local_research_cron_task, hour=4, minute=0, unique=True),
        # Refill candidate topics daily at 3:30am UTC (before the 04:00 run)
        cron(auto_discovery_refill_task, hour=3, minute=30, unique=True),
        # Drain the candidate queue every hour at :07
        cron(auto_discovery_enqueue_task, minute=7, unique=True),
        # Passive RSS feed discovery, daily at 5am UTC
        cron(rss_discovery_cron_task, hour=5, minute=0, unique=True),
        # Sitemap crawl, weekly on Mondays at 5:30am UTC
        cron(sitemap_discovery_cron_task, weekday="mon", hour=5, minute=30, unique=True),
        # Podcast feed discovery, weekly on Tuesdays at 5:30am UTC
        cron(podcast_discovery_cron_task, weekday="tues", hour=5, minute=30, unique=True),
    ]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
    queue_name = ARQ_QUEUE_NAME

    # Allow long-running research jobs (up to 10 minutes)
    job_timeout = 600
    max_jobs = 15
    poll_delay = 0.3
