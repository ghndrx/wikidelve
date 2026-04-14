"""
Tiered memory system for WikiDelve's DeepAgents research agent.

Three memory tiers:
  - Working memory: Current session context (managed by DeepAgents internally)
  - Episodic memory: Past research sessions — what worked, source reliability, failures
  - Semantic memory: KB-level facts — article count, topic coverage, style preferences

Episodic and semantic memories are persisted to DynamoDB and loaded at agent start.
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("kb-service.agent.memory")


# ---------------------------------------------------------------------------
# Episodic memory — records of past research sessions
# ---------------------------------------------------------------------------

async def record_research_episode(
    kb: str,
    topic: str,
    job_id: int,
    outcome: str,
    sources_used: int = 0,
    word_count: int = 0,
    quality_score: float | None = None,
    notes: str = "",
):
    """Record a research episode for future agent reference."""
    from app import db

    episode = {
        "kb": kb,
        "topic": topic,
        "job_id": job_id,
        "outcome": outcome,  # "complete", "error", "low_quality"
        "sources_used": sources_used,
        "word_count": word_count,
        "quality_score": quality_score,
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Store as a KB setting under a special key
        existing = await db.get_kb_settings(kb) or {}
        episodes = json.loads(existing.get("agent_episodes", "[]"))
        episodes.append(episode)
        # Keep last 50 episodes per KB
        episodes = episodes[-50:]
        await db.upsert_kb_settings(kb, agent_episodes=json.dumps(episodes))
        logger.debug("Recorded research episode: job=%d outcome=%s", job_id, outcome)
    except Exception as exc:
        logger.warning("Failed to record episode: %s", exc)


async def get_recent_episodes(kb: str, limit: int = 10) -> list[dict]:
    """Get recent research episodes for a KB."""
    from app import db

    try:
        settings = await db.get_kb_settings(kb) or {}
        episodes = json.loads(settings.get("agent_episodes", "[]"))
        return episodes[-limit:]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Semantic memory — KB-level knowledge and preferences
# ---------------------------------------------------------------------------

async def get_kb_context(kb: str) -> str:
    """Build a semantic memory context string for the agent.

    Summarizes what the agent should know about this KB before starting research.
    """
    from app.wiki import get_articles
    from app.config import KB_DIRS

    parts = []

    # KB basics
    articles = get_articles(kb)
    article_count = len(articles)
    total_words = sum(a.get("word_count", 0) for a in articles)
    parts.append(f"Knowledge Base: {kb}")
    parts.append(f"Articles: {article_count} ({total_words:,} total words)")
    parts.append(f"Available KBs: {', '.join(KB_DIRS.keys())}")

    # Top tags (what this KB covers)
    tag_counts: dict[str, int] = {}
    for a in articles:
        for tag in a.get("tags", []):
            if isinstance(tag, str) and len(tag) > 1:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:15]
    if top_tags:
        parts.append(f"Top topics: {', '.join(t[0] for t in top_tags)}")

    # Recent episodes
    episodes = await get_recent_episodes(kb, limit=5)
    if episodes:
        parts.append("\nRecent research history:")
        for ep in episodes[-5:]:
            score_str = f" (quality: {ep['quality_score']})" if ep.get("quality_score") else ""
            parts.append(f"  - {ep['topic'][:60]} → {ep['outcome']}{score_str}")

    # KB-specific settings/persona
    from app import db
    try:
        settings = await db.get_kb_settings(kb) or {}
        persona = settings.get("persona", "")
        if persona:
            parts.append(f"\nKB persona: {persona}")
    except Exception:
        pass

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Source reliability tracking
# ---------------------------------------------------------------------------

async def record_source_reliability(
    domain: str,
    reliable: bool,
    context: str = "",
):
    """Track whether a source domain was reliable or not.

    Over time this builds a profile of which domains produce good content.
    """
    from app import db

    try:
        settings = await db.get_kb_settings("_global") or {}
        reliability = json.loads(settings.get("source_reliability", "{}"))

        if domain not in reliability:
            reliability[domain] = {"good": 0, "bad": 0}

        if reliable:
            reliability[domain]["good"] += 1
        else:
            reliability[domain]["bad"] += 1

        # Keep only domains with 3+ data points
        await db.upsert_kb_settings(
            "_global", source_reliability=json.dumps(reliability),
        )
    except Exception as exc:
        logger.debug("Failed to record source reliability: %s", exc)


async def get_unreliable_domains(threshold: float = 0.5) -> list[str]:
    """Get domains that have been flagged as unreliable more than `threshold` of the time."""
    from app import db

    try:
        settings = await db.get_kb_settings("_global") or {}
        reliability = json.loads(settings.get("source_reliability", "{}"))

        unreliable = []
        for domain, counts in reliability.items():
            total = counts["good"] + counts["bad"]
            if total >= 3 and counts["bad"] / total > threshold:
                unreliable.append(domain)
        return unreliable
    except Exception:
        return []
