"""
WikiDelve research agent — powered by DeepAgents (LangChain/LangGraph).

Replaces the hardcoded 3-round research pipeline with an adaptive agent
that decides how many rounds of search, which sources to read deeply,
and whether to fact-check based on topic complexity.

Usage:
    from app.agent import run_agent_research
    await run_agent_research("Kubernetes networking", job_id=42, kb="personal")

The agent uses WikiDelve's existing functions directly (no HTTP sidecar).
"""

import logging
from typing import Optional

from deepagents import create_deep_agent

from app.agent_tools import (
    ALL_TOOLS, search_web, read_webpage,
    search_kb, get_article,
    get_document_version, list_document_versions,
    propose_document_version, save_document_version,
    add_pinned_fact as _add_pinned_fact_tool,
    ask_user,
)
from app.agent_prompts import (
    RESEARCH_AGENT_PROMPT, FACT_CHECKER_PROMPT, ARTICLE_IMPROVE_PROMPT,
    SCAFFOLD_AGENT_PROMPT, SCAFFOLD_EXTEND_PROMPT,
)
from app.config import KB_DIRS

logger = logging.getLogger("kb-service.agent")


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_model(purpose: Optional[str] = None):
    """Return a LangChain chat model identifier or instance for the
    agent to drive its loop.

    Per-purpose override: setting e.g. ``SCAFFOLD_AGENT_PROVIDER=kimi``
    routes scaffold runs through Kimi while keeping everything else
    on the global ``LLM_PROVIDER``. This lets us pick the right model
    per task — scaffolds + code-gen → Kimi K2.5 (strong on HTML/CSS
    + tool use), conversational doc agent → Minimax M2 (cheaper, fine
    for chat), research agent → whatever LLM_PROVIDER says.

    Providers exposed via OpenAI-compatible endpoints (Minimax, Kimi)
    return an instantiated ``ChatOpenAI`` so the agent loop talks to
    them through the same wrapper. Bedrock / Anthropic return the
    standard LangChain string form for create_deep_agent to resolve.
    """
    from app.config import (
        LLM_PROVIDER, BEDROCK_MODEL,
        MINIMAX_API_KEY, MINIMAX_BASE, MINIMAX_MODEL, MINIMAX_TIMEOUT,
        KIMI_API_KEY, KIMI_BASE, KIMI_MODEL,
    )
    import os as _os

    # Per-agent-purpose override: <PURPOSE>_AGENT_PROVIDER env var
    # wins over global LLM_PROVIDER. Lets us route scaffolds to Kimi
    # without disturbing anything else.
    provider = None
    if purpose:
        env_name = f"{purpose.upper()}_AGENT_PROVIDER"
        provider = (_os.getenv(env_name, "").strip() or None)
        if provider:
            logger.info("Agent for purpose=%r using override provider=%r", purpose, provider)
    if not provider:
        provider = (LLM_PROVIDER or "minimax").lower()
    provider = provider.lower()

    if provider == "minimax":
        if not MINIMAX_API_KEY:
            raise RuntimeError(
                "Minimax provider selected but MINIMAX_API_KEY is not set"
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MINIMAX_MODEL,
            base_url=MINIMAX_BASE,
            api_key=MINIMAX_API_KEY,
            temperature=0.2,
            timeout=MINIMAX_TIMEOUT,
            max_retries=4,
        )

    if provider == "kimi":
        if not KIMI_API_KEY:
            raise RuntimeError(
                "Kimi provider selected but KIMI_API_KEY is not set"
            )
        from langchain_openai import ChatOpenAI
        # Kimi K2.5 is OpenAI-compatible (Moonshot's endpoint).
        # Same wrapper, different base URL + model. Strong at code
        # generation + tool calling — well-suited for scaffolds.
        return ChatOpenAI(
            model=KIMI_MODEL,
            base_url=KIMI_BASE,
            api_key=KIMI_API_KEY,
            temperature=0.2,
            timeout=300,
            max_retries=4,
        )

    if provider == "bedrock":
        return f"bedrock:{BEDROCK_MODEL}"

    if provider == "anthropic":
        return "anthropic:claude-sonnet-4-6"

    # Unknown — fail loud rather than silently picking a wrong provider.
    raise RuntimeError(f"Unsupported LLM provider for agent: {provider!r}")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

async def create_research_agent(kb: str = "personal"):
    """Create a DeepAgents research agent with WikiDelve tools.

    Loads KB context (semantic memory) and recent episodes (episodic memory)
    to give the agent awareness of what's already in the KB and what has
    worked well in the past.

    Returns a compiled LangGraph graph that can be invoked or streamed.
    """
    from app.agent_memory import get_kb_context

    kb_list = ", ".join(KB_DIRS.keys())
    model = _resolve_model()

    # Load semantic + episodic memory for this KB
    kb_context = await get_kb_context(kb)

    # Build system prompt with memory context
    system_prompt = RESEARCH_AGENT_PROMPT.format(kb_list=kb_list, kb=kb)
    system_prompt += f"\n\n## KB Context (from memory)\n{kb_context}"

    fact_checker = {
        "name": "fact-checker",
        "description": "Verify claims by searching for supporting or contradicting evidence. Use for controversial or surprising claims.",
        "system_prompt": FACT_CHECKER_PROMPT,
        "tools": [search_web, read_webpage],
        "model": model,
    }

    agent = create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        subagents=[fact_checker],
        system_prompt=system_prompt,
    )

    return agent


# ---------------------------------------------------------------------------
# Research execution
# ---------------------------------------------------------------------------

async def run_agent_research(
    topic: str, job_id: int, kb: str = "personal",
) -> dict:
    """Run the research agent on a topic — replacement for the hardcoded pipeline.

    Updates the job record with status changes and final content.
    Returns the agent's result dict.
    """
    from app import db
    from app.agent_memory import record_research_episode

    agent = await create_research_agent(kb=kb)

    await db.update_job(job_id, status="agent_researching")
    logger.info("Agent research started: job=%d topic=%r kb=%s", job_id, topic, kb)

    try:
        result = await agent.ainvoke(
            {
                "messages": [{
                    "role": "user",
                    "content": f"Research this topic and write a comprehensive wiki article in the '{kb}' knowledge base: {topic}",
                }],
            },
            config={"recursion_limit": 150},
        )

        # The agent should have called write_article as its final step.
        # Update job as complete.
        final_message = result["messages"][-1]
        content = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )

        await db.update_job(job_id, status="complete", content=content)
        logger.info("Agent research complete: job=%d", job_id)

        # Record success episode
        await record_research_episode(
            kb=kb, topic=topic, job_id=job_id, outcome="complete",
            word_count=len(content.split()),
        )

        return result

    except Exception as exc:
        logger.exception("Agent research failed: job=%d error=%s", job_id, exc)
        await db.update_job(job_id, status="error", error=str(exc))

        # Record failure episode
        await record_research_episode(
            kb=kb, topic=topic, job_id=job_id, outcome="error",
            notes=str(exc)[:200],
        )

        raise


async def create_improve_agent(kb: str, slug: str):
    """Create an agent specialized for improving an existing article.

    Same tool surface as the research agent, but a different system
    prompt that steers it toward surgical patches rather than
    greenfield synthesis.
    """
    from app.agent_memory import get_kb_context

    kb_list = ", ".join(KB_DIRS.keys())
    model = _resolve_model()
    kb_context = await get_kb_context(kb)

    system_prompt = ARTICLE_IMPROVE_PROMPT.format(
        kb_list=kb_list, kb=kb, slug=slug,
    )
    system_prompt += f"\n\n## KB Context (from memory)\n{kb_context}"

    fact_checker = {
        "name": "fact-checker",
        "description": "Verify claims by searching for supporting or contradicting evidence.",
        "system_prompt": FACT_CHECKER_PROMPT,
        "tools": [search_web, read_webpage],
        "model": model,
    }

    return create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        subagents=[fact_checker],
        system_prompt=system_prompt,
    )


async def run_agent_improve(
    kb: str, slug: str, job_id: int,
) -> dict:
    """Run the improve agent against an existing wiki article.

    The agent reads the article, identifies weak spots, researches
    just those, and writes an updated version. Fuzzy-merge in
    create_or_update_article routes the output back into the same
    slug so we don't fork.
    """
    from app import db
    from app.wiki import get_article
    from app.agent_memory import record_research_episode

    article = get_article(kb, slug)
    if not article:
        await db.update_job(
            job_id, status="error",
            error=f"Article not found: {kb}/{slug}",
        )
        return {"error": f"Article not found: {kb}/{slug}"}

    # Remember the pre-run state so we can distinguish 'agent crashed
    # without writing' from 'agent wrote then crashed later'. The
    # second case is a partial success: the wiki IS improved, so
    # marking the job hard-errored makes the error counter lie.
    pre_updated = (article.get("updated") or "").strip()
    title = article.get("title") or slug.replace("-", " ").title()
    agent = await create_improve_agent(kb=kb, slug=slug)

    await db.update_job(job_id, status="agent_improving")
    logger.info("Agent improve started: job=%d kb=%s slug=%s", job_id, kb, slug)

    user_msg = (
        f"Improve the existing wiki article '{title}' (slug: {slug}) in "
        f"the '{kb}' knowledge base. Read the current version first, "
        f"identify its weakest sections, research only those, and "
        f"rewrite with citations. If the article is already solid, "
        f"stop and say so."
    )

    import asyncio as _asyncio
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            # 120 matches the research agent's ballpark. The pilot
            # showed occasional overruns at 80 on topics where the
            # agent does more sub-topic searches than our prompt
            # budget suggests; 120 absorbs that without costing much.
            config={"recursion_limit": 120},
        )
        final_message = result["messages"][-1]
        content = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )
        await db.update_job(job_id, status="complete", content=content)
        logger.info("Agent improve complete: job=%d slug=%s", job_id, slug)

        await record_research_episode(
            kb=kb, topic=f"improve:{slug}", job_id=job_id,
            outcome="complete", word_count=len(content.split()),
        )
        return result
    except (_asyncio.CancelledError, _asyncio.TimeoutError) as exc:
        # arq cancels the task with CancelledError when job_timeout
        # fires. That's BaseException, so the `except Exception`
        # below doesn't see it — previously left the DB row stuck at
        # status='agent_improving' forever. Catch it explicitly,
        # mark the job as errored so the admin UI + retry paths can
        # see the real state, THEN re-raise so arq still records the
        # task as failed (don't silently swallow cancellation).
        logger.warning(
            "Agent improve cancelled/timed-out: job=%d slug=%s", job_id, slug,
        )
        try:
            await db.update_job(
                job_id, status="error",
                error="Agent exceeded worker timeout (30min cap)",
            )
            await record_research_episode(
                kb=kb, topic=f"improve:{slug}", job_id=job_id,
                outcome="timeout", notes="worker job_timeout",
            )
        except Exception:
            pass  # never block the cancellation path
        raise
    except Exception as exc:
        logger.exception("Agent improve failed: job=%d slug=%s", job_id, slug)
        # Partial-success check: if the wiki file was actually touched
        # before the crash, treat it as complete-with-warning rather
        # than a hard error. The user's article IS improved; only the
        # post-write bookkeeping (fact-check, embed, etc.) failed.
        post = get_article(kb, slug)
        post_updated = (post.get("updated") if post else "") or ""
        wiki_was_touched = bool(post_updated) and post_updated != pre_updated
        if wiki_was_touched:
            logger.warning(
                "Agent improve partial success: job=%d slug=%s — wiki "
                "updated %s → %s but post-write step failed: %s",
                job_id, slug, pre_updated, post_updated, exc,
            )
            await db.update_job(
                job_id, status="complete",
                error=f"Partial: wiki written but post-step failed: {str(exc)[:150]}",
                wiki_slug=slug, wiki_kb=kb, added_to_wiki=1,
            )
            await record_research_episode(
                kb=kb, topic=f"improve:{slug}", job_id=job_id,
                outcome="partial", notes=str(exc)[:200],
            )
            return {"status": "partial", "slug": slug, "error": str(exc)}
        await db.update_job(job_id, status="error", error=str(exc))
        await record_research_episode(
            kb=kb, topic=f"improve:{slug}", job_id=job_id,
            outcome="error", notes=str(exc)[:200],
        )
        raise


async def create_scaffold_agent(kb: str, topic: str, scaffold_type: str):
    """Create an agent whose output contract is a multi-file scaffold.

    Same tool surface as the research agent, but steered toward
    calling ``write_scaffold_files`` instead of ``write_article``.
    """
    from app.scaffolds import SCAFFOLD_TYPES

    model = _resolve_model(purpose="scaffold")
    system_prompt = SCAFFOLD_AGENT_PROMPT.format(
        topic=topic, scaffold_type=scaffold_type,
        scaffold_types=", ".join(sorted(SCAFFOLD_TYPES)),
    )

    return create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=system_prompt,
    )


async def run_scaffold_agent(
    kb: str, topic: str, scaffold_type: str, job_id: int,
) -> dict:
    """Run the scaffold agent for a single topic.

    On success the agent will have called ``write_scaffold_files``
    which persists the manifest + files to storage. The job row's
    ``content`` gets the agent's final narrative (useful for audit /
    future refinement), and ``source_params`` is stamped with the
    resulting slug so the API can surface the preview URL directly.
    """
    from app import db
    import asyncio as _asyncio

    # Trust-but-verify the scaffold_type lives in the enum; fall
    # back to 'other' so the agent doesn't hard-fail on typos.
    from app.scaffolds import SCAFFOLD_TYPES
    if scaffold_type not in SCAFFOLD_TYPES:
        scaffold_type = "other"

    agent = await create_scaffold_agent(kb=kb, topic=topic, scaffold_type=scaffold_type)

    # Snapshot scaffold slugs BEFORE the agent runs so the
    # post-run diff tells us authoritatively which slug (if any)
    # this run produced — independent of any text-parsing on the
    # agent's reply.
    from app.scaffolds import list_scaffolds
    pre_slugs = {s["slug"] for s in list_scaffolds(kb)}

    await db.update_job(job_id, status="agent_scaffolding")
    logger.info(
        "Scaffold agent started: job=%d kb=%s topic=%r type=%s",
        job_id, kb, topic, scaffold_type,
    )

    user_msg = (
        f"Produce a '{scaffold_type}' scaffold for: {topic}\n\n"
        f"Research real reference implementations first, then call "
        f"write_scaffold_files with a complete, runnable set of "
        f"files. Use kb='{kb}'. Framework must be 'vanilla' for this "
        f"MVP — no frameworks, no bundlers. When you've called the "
        f"tool successfully, stop."
    )

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            # Recursion budget tuned through three real failures:
            # 80 → exhausted on simple multi-file
            # 200 → still exhausted on detailed-spec topics where
            #       the agent over-researched
            # 300 → headroom for stubborn over-researchers + paired
            #       with the v3 prompt that tells spec-heavy runs
            #       to skip search entirely
            config={"recursion_limit": 300},
        )
        final_message = result["messages"][-1]
        content = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )
        # Detect whether write_scaffold_files actually ran. Two
        # signals, in order of reliability:
        #   1. New slug in the store that wasn't there at kickoff.
        #      (Authoritative — if the file landed in S3, the work
        #      happened, regardless of what the agent's message
        #      trail says.)
        #   2. Regex on the message trail for a "slug": "..." token.
        #      Backup signal for when the storage backend lags.
        import json as _json, re as _re

        post_slugs = {s["slug"] for s in list_scaffolds(kb)}
        slug = None
        new_slugs = post_slugs - pre_slugs
        if len(new_slugs) == 1:
            slug = next(iter(new_slugs))
        elif len(new_slugs) > 1:
            # Multiple new slugs — pick the freshest by manifest's
            # 'created' or 'updated' field. Conservative.
            from app.scaffolds import get_manifest
            best, best_ts = None, ""
            for s in new_slugs:
                m = get_manifest(kb, s) or {}
                ts = m.get("updated") or m.get("created") or ""
                if ts > best_ts:
                    best, best_ts = s, ts
            slug = best

        if not slug:
            # Fallback: scan the agent's message trail. The
            # write_scaffold_files tool echoes JSON with both
            # "slug" and "preview_url"; finding both adjacent
            # confirms it's our envelope, not stray text.
            for msg in result.get("messages", []):
                text = msg.content if isinstance(msg.content, str) else str(getattr(msg, "content", ""))
                m = _re.search(r'"slug"\s*:\s*"([^"]+)"[^{}]*"preview_url"', text)
                if m:
                    slug = m.group(1)
                    break

        if not slug:
            logger.warning(
                "Scaffold agent finished WITHOUT producing a scaffold: "
                "job=%d kb=%s topic=%r — recursion limit exhausted or "
                "agent gave up. Marking as error.",
                job_id, kb, topic,
            )
            await db.update_job(
                job_id, status="error",
                error="Agent finished without calling write_scaffold_files (likely recursion-limit exhaustion).",
                content=content,
            )
            return result

        await db.update_job(
            job_id, status="complete", content=content,
            source_params=_json.dumps({
                "scaffold_slug": slug, "scaffold_type": scaffold_type,
                "kb": kb, "topic": topic,
            }),
        )
        logger.info(
            "Scaffold agent complete: job=%d slug=%s", job_id, slug,
        )
        return result
    except (_asyncio.CancelledError, _asyncio.TimeoutError):
        logger.warning("Scaffold agent cancelled: job=%d", job_id)
        try:
            await db.update_job(
                job_id, status="error",
                error="Scaffold agent exceeded worker timeout",
            )
        except Exception:
            pass
        raise
    except Exception as exc:
        logger.exception("Scaffold agent failed: job=%d", job_id)
        await db.update_job(job_id, status="error", error=str(exc))
        raise


async def run_scaffold_agent_kimi(
    kb: str, topic: str, scaffold_type: str, job_id: int,
) -> dict:
    """Kimi-backed scaffold runner.

    Hands the job to kimi-cli through the bridge sidecar: we allocate
    an ephemeral workdir, let kimi write files into it with its own
    file tools, then ingest the result via scaffolds.create_scaffold.
    No LangChain, no Minimax — used when KIMI_BRIDGE_ENABLED=true.
    """
    from app import db
    from app import kimi_bridge
    from app.scaffolds import SCAFFOLD_TYPES, create_scaffold
    import json as _json

    if scaffold_type not in SCAFFOLD_TYPES:
        scaffold_type = "other"

    await db.update_job(job_id, status="agent_scaffolding")
    logger.info(
        "Kimi scaffold agent started: job=%d kb=%s topic=%r type=%s",
        job_id, kb, topic, scaffold_type,
    )

    host_dir, container_dir = kimi_bridge.make_workdir(prefix=f"scaffold-{job_id}")
    prompt = (
        f"Build a '{scaffold_type}' static web scaffold for this topic: {topic}\n\n"
        f"Working directory: {container_dir} — write files directly here "
        f"with your built-in file tools.\n\n"
        f"You have six tools available. USE THEM — they are the single "
        f"biggest quality lever over just writing from memory:\n\n"
        f"  RESEARCH (do FIRST, before writing any code):\n"
        f"  1. search_kb({{query, kb}}) — search our internal wiki for "
        f"existing patterns. Start here — we may already have research on "
        f"this topic.\n"
        f"  2. search_web({{query}}) — Google via Serper. Find 2-3 "
        f"best-in-class reference sites for this scaffold type.\n"
        f"  3. read_webpage({{url}}) — pull the full readable text from a "
        f"URL. Use to study a specific reference in depth.\n\n"
        f"  VISUAL (use DURING build):\n"
        f"  4. browser_screenshot({{url}}) — screenshot the reference sites "
        f"you found so you can match their visual language.\n"
        f"  5. browser_snapshot({{url}}) — accessibility tree + links from a "
        f"reference page; understand its information architecture.\n\n"
        f"  SELF-VALIDATE (run AFTER writing files):\n"
        f"  6. browser_validate_local({{path}}) — render YOUR OWN output in "
        f"Chromium. Screenshot + console errors come back. If it looks bad "
        f"or has errors, fix and re-validate. Iterate until polished.\n\n"
        f"Rules:\n"
        f"- Vanilla HTML / CSS / JS only. No frameworks, no bundlers, no "
        f"build step. Must load by opening index.html directly.\n"
        f"- Minimum: index.html + styles.css + app.js. Add more pages if "
        f"the topic warrants it.\n"
        f"- Total size under 2 MB, under 50 files.\n"
        f"- No remote fonts/CDNs that break offline preview — inline or drop.\n"
        f"- Stop when the scaffold is complete and you've validated it "
        f"renders cleanly. Don't keep polishing forever."
    )

    try:
        bridge_result = await kimi_bridge.run_agent(
            prompt=prompt, workdir_container=container_dir,
        )
    except kimi_bridge.KimiBridgeError as exc:
        logger.exception("kimi-bridge call failed: job=%d", job_id)
        await db.update_job(job_id, status="error", error=f"kimi-bridge: {exc}")
        kimi_bridge.cleanup_workdir(host_dir)
        raise

    status = bridge_result.get("status", "unknown")
    transcript_parts = [
        p.get("text", "") for p in bridge_result.get("transcript", [])
        if p.get("kind") == "text"
    ]
    narrative = "\n".join(transcript_parts).strip()

    files = kimi_bridge.walk_workdir(host_dir)
    kimi_bridge.cleanup_workdir(host_dir)

    if not files:
        logger.warning(
            "Kimi scaffold finished without writing files: job=%d status=%s",
            job_id, status,
        )
        await db.update_job(
            job_id, status="error",
            error=f"kimi produced no files (turn status={status})",
            content=narrative or None,
        )
        return bridge_result

    # Build a minimal manifest; scaffolds.create_scaffold validates +
    # fills in the rest.
    manifest = {
        "topic": topic,
        "scaffold_type": scaffold_type,
        "framework": "vanilla",
        "description": narrative[:500] if narrative else f"{scaffold_type} scaffold for {topic}",
    }
    try:
        slug = create_scaffold(kb=kb, manifest=manifest, files=files)
    except Exception as exc:
        logger.exception("create_scaffold failed after kimi run: job=%d", job_id)
        await db.update_job(
            job_id, status="error", error=f"ingest failed: {exc}",
            content=narrative or None,
        )
        raise

    await db.update_job(
        job_id, status="complete", content=narrative,
        source_params=_json.dumps({
            "scaffold_slug": slug, "scaffold_type": scaffold_type,
            "kb": kb, "topic": topic, "backend": "kimi",
        }),
    )
    logger.info("Kimi scaffold complete: job=%d slug=%s files=%d", job_id, slug, len(files))
    return bridge_result


async def create_scaffold_extend_agent(kb: str, slug: str, page_path: str, brief: str):
    """Build an agent specialised for adding a single sibling page
    to an existing scaffold. Tighter tool surface than the full
    scaffold agent — only the read+write-page tools, no web search,
    so the agent stays focused on matching tokens rather than
    re-researching."""
    from app.agent_tools import get_scaffold_file, add_scaffold_page

    model = _resolve_model(purpose="scaffold")
    system_prompt = SCAFFOLD_EXTEND_PROMPT.format(
        kb=kb, slug=slug, page_path=page_path, page_brief=brief,
    )
    return create_deep_agent(
        model=model,
        tools=[get_scaffold_file, add_scaffold_page],
        system_prompt=system_prompt,
    )


async def run_scaffold_extend_agent(
    kb: str, slug: str, page_path: str, brief: str, job_id: int,
) -> dict:
    """Add a single sibling page to a scaffold via the extension agent.

    On success the page is in the scaffold's files/ directory and
    the manifest's planned_extensions list has the entry removed.
    """
    from app import db
    import asyncio as _asyncio

    agent = await create_scaffold_extend_agent(kb, slug, page_path, brief)
    await db.update_job(job_id, status="agent_scaffolding")
    logger.info(
        "Scaffold extend agent started: job=%d slug=%s page=%s",
        job_id, slug, page_path,
    )

    user_msg = (
        f"Add the page '{page_path}' to scaffold '{kb}/{slug}'. "
        f"Brief: {brief}\n\n"
        f"Read the existing scaffold's manifest.json, styles.css, "
        f"and index.html FIRST so you can match tokens + class "
        f"naming. Then call add_scaffold_page exactly once with "
        f"the new page's HTML."
    )

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config={"recursion_limit": 100},
        )
        # Verify the page actually got written.
        from app.scaffolds import get_manifest
        manifest = get_manifest(kb, slug)
        if not manifest or page_path not in (manifest.get("files") or []):
            await db.update_job(
                job_id, status="error",
                error=f"Extension agent did not call add_scaffold_page for {page_path}",
            )
            return result
        await db.update_job(job_id, status="complete")
        logger.info(
            "Scaffold extend complete: job=%d slug=%s page=%s",
            job_id, slug, page_path,
        )
        return result
    except (_asyncio.CancelledError, _asyncio.TimeoutError):
        await db.update_job(
            job_id, status="error",
            error="Extension agent exceeded worker timeout",
        )
        raise
    except Exception as exc:
        logger.exception("Scaffold extend failed: job=%d", job_id)
        await db.update_job(job_id, status="error", error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Document chat agent — chunk 4
# ---------------------------------------------------------------------------

# Tool surface for the doc chat agent. Deliberately tighter than the
# global ALL_TOOLS so the agent doesn't accidentally call e.g.
# write_article (which would create a new wiki article instead of a
# document version) or write_scaffold_files (would land a code
# template). All doc tools + safe research tools, nothing else.
_DOC_CHAT_TOOLS = [
    search_kb,
    search_web,
    read_webpage,
    get_article,
    get_document_version,
    list_document_versions,
    propose_document_version,
    save_document_version,
    _add_pinned_fact_tool,
    ask_user,
]


async def create_doc_chat_agent(kb: str, slug: str, manifest: dict):
    """Create an agent specialised for iterating a single document.

    System prompt is templated per-document so the agent always knows
    its autonomy mode, the brief, and the pinned facts that must
    survive every turn.
    """
    model = _resolve_model()

    seeds = manifest.get("seed_articles") or []
    seed_lines = "\n  ".join(
        f"- {s.get('kb','personal')}/{s.get('slug','?')}" for s in seeds
    ) or "(none — agent works from brief alone)"
    pinned = manifest.get("pinned_facts") or []
    pinned_lines = "\n  ".join(f"- {p}" for p in pinned) or "(none yet)"

    # Per-document system prompt: autonomy mode + brief + pinned facts
    # + seed articles all inlined so each turn knows the contract.
    from app.agent_prompts import DOC_CHAT_AGENT_PROMPT
    system_prompt = DOC_CHAT_AGENT_PROMPT.format(
        autonomy_mode=manifest.get("autonomy_mode", "propose"),
        title=manifest.get("title", slug),
        doc_type=manifest.get("doc_type", "pdf"),
        brief=manifest.get("brief", ""),
        current_version=manifest.get("current_version", 0),
        pinned_facts=pinned_lines,
        seed_articles=seed_lines,
    )

    return create_deep_agent(
        model=model,
        tools=_DOC_CHAT_TOOLS,
        system_prompt=system_prompt,
    )


async def run_doc_chat_turn(
    kb: str, slug: str, user_message: str,
) -> dict:
    """One conversational turn against the document drafting agent.

    The user's message + the recent history get fed to the agent;
    the agent decides whether to draft, ask, or simply answer.
    Returns ``{response, pending, version_after, error}``.

    Persists both the user turn and the agent turn to the document's
    history.jsonl so future turns see the conversation context.
    """
    from app import documents

    manifest = documents.get_manifest(kb, slug)
    if not manifest:
        return {"error": f"document not found: {kb}/{slug}"}

    # Persist the incoming user message FIRST so a mid-turn agent
    # crash doesn't lose what the user said.
    documents.append_chat_turn(kb, slug, "user", user_message)

    agent = await create_doc_chat_agent(kb, slug, manifest)

    # Replay recent history into the conversation so the agent has
    # context. We cap to the last 12 turns to keep the prompt budget
    # reasonable on long-running docs.
    history = documents.get_history(kb, slug, limit=24)
    msgs: list[dict] = []
    for evt in history:
        if evt.get("type") != "turn":
            continue
        role = evt.get("role")
        if role == "user":
            msgs.append({"role": "user", "content": evt.get("content", "")})
        elif role == "agent":
            msgs.append({"role": "assistant", "content": evt.get("content", "")})

    # Make sure the just-appended user message is the last entry.
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": user_message})

    try:
        result = await agent.ainvoke(
            {"messages": msgs[-12:]},
            config={"recursion_limit": 80},
        )
        final_message = result["messages"][-1]
        response_text = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )
        # Capture whether the agent staged a pending draft this turn.
        pending = documents.get_pending_draft(kb, slug)
        manifest_after = documents.get_manifest(kb, slug)
        version_after = manifest_after.get("current_version", 0) if manifest_after else 0

        documents.append_chat_turn(
            kb, slug, "agent", response_text,
            metadata={
                "pending": bool(pending),
                "version_after": version_after,
            },
        )
        return {
            "response": response_text,
            "pending": bool(pending),
            "version_after": version_after,
        }
    except Exception as exc:
        logger.exception("Doc chat turn failed: %s/%s", kb, slug)
        documents.append_chat_turn(
            kb, slug, "agent", f"(agent error: {exc})",
            metadata={"error": str(exc)},
        )
        return {"error": str(exc)}


async def run_agent_research_streaming(
    topic: str, job_id: int, kb: str = "personal",
):
    """Run agent research with streaming events for the SSE endpoint.

    Yields dicts with {event, data} matching WikiDelve's SSE format.
    """
    from app import db

    agent = await create_research_agent(kb=kb)

    await db.update_job(job_id, status="agent_researching")
    logger.info("Agent research (streaming) started: job=%d", job_id)

    try:
        async for event in agent.astream_events(
            {"messages": [{
                "role": "user",
                "content": f"Research this topic and write a comprehensive wiki article in the '{kb}' knowledge base: {topic}",
            }]},
            version="v2",
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                await db.update_job(job_id, status=f"using_{tool_name}")
                yield {"event": "status", "data": f"Using tool: {tool_name}"}

            elif kind == "on_tool_end":
                yield {"event": "progress", "data": "tool_complete"}

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {})
                if hasattr(chunk, "content") and chunk.content:
                    yield {"event": "delta", "data": chunk.content}

        await db.update_job(job_id, status="complete")
        yield {"event": "done", "data": "complete"}

    except Exception as exc:
        logger.exception("Agent research streaming failed: job=%d", job_id)
        await db.update_job(job_id, status="error", error=str(exc))
        yield {"event": "error", "data": str(exc)}
