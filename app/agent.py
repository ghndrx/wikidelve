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

from deepagents import create_deep_agent

from app.agent_tools import ALL_TOOLS, search_web, read_webpage
from app.agent_prompts import (
    RESEARCH_AGENT_PROMPT, FACT_CHECKER_PROMPT, ARTICLE_IMPROVE_PROMPT,
    SCAFFOLD_AGENT_PROMPT,
)
from app.config import KB_DIRS

logger = logging.getLogger("kb-service.agent")


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_model():
    """Return a LangChain chat model identifier or instance for the
    agent to drive its loop.

    Minimax is exposed via an OpenAI-compatible endpoint, so we
    return an instantiated ``ChatOpenAI`` pointed at MINIMAX_BASE —
    that matches what the rest of the app already uses (app.llm)
    and keeps the agent on the same quota/budget as synthesis.

    Bedrock / Anthropic providers return the standard LangChain
    string form, which create_deep_agent resolves internally.
    """
    from app.config import (
        LLM_PROVIDER, BEDROCK_MODEL,
        MINIMAX_API_KEY, MINIMAX_BASE, MINIMAX_MODEL, MINIMAX_TIMEOUT,
    )

    provider = (LLM_PROVIDER or "minimax").lower()

    if provider == "minimax":
        if not MINIMAX_API_KEY:
            raise RuntimeError(
                "LLM_PROVIDER=minimax but MINIMAX_API_KEY is not set"
            )
        from langchain_openai import ChatOpenAI
        # Minimax exposes /chat/completions in OpenAI format. The
        # agent loop uses temperature=0.2 (deterministic-ish) and a
        # generous per-call timeout because synthesis steps run long.
        return ChatOpenAI(
            model=MINIMAX_MODEL,
            base_url=MINIMAX_BASE,
            api_key=MINIMAX_API_KEY,
            temperature=0.2,
            timeout=MINIMAX_TIMEOUT,
            max_retries=4,
        )

    if provider == "bedrock":
        return f"bedrock:{BEDROCK_MODEL}"

    if provider == "anthropic":
        return "anthropic:claude-sonnet-4-6"

    # Unknown — fail loud rather than silently picking a wrong provider.
    raise RuntimeError(f"Unsupported LLM_PROVIDER for agent: {provider!r}")


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

    model = _resolve_model()
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
            config={"recursion_limit": 80},
        )
        final_message = result["messages"][-1]
        content = (
            final_message.content
            if isinstance(final_message.content, str)
            else str(final_message.content)
        )
        # Try to pull the slug off the last tool result — scaffolds
        # tool returns JSON {slug, kb, preview_url, file_count}.
        import json as _json, re as _re
        slug_match = _re.search(r'"slug"\s*:\s*"([^"]+)"', content)
        slug = slug_match.group(1) if slug_match else None
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
