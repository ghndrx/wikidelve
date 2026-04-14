"""
Database layer: SQLite (default) or DynamoDB backend.

Set DB_BACKEND=dynamodb to use DynamoDB instead of SQLite.

Tables / entities:
  - research_jobs: tracks each research pipeline execution
  - research_sources: individual search results per job
  - article_updates: log of wiki article creates/updates
  - palace_classifications: hall assignment per article
  - palace_rooms: topic clusters from knowledge graph entities
  - palace_room_members: article-to-room assignments
"""

import os as _os

_DB_BACKEND = _os.getenv("DB_BACKEND", "sqlite").strip().lower()

if _DB_BACKEND not in ("sqlite", "dynamodb"):
    raise ValueError(f"Unknown DB_BACKEND: {_DB_BACKEND!r}. Use 'sqlite' or 'dynamodb'.")

# --- SQLite implementation --------------------------------------------------
# These definitions run unconditionally so the SQLite path is always
# available. When DB_BACKEND=dynamodb, a trailer at the END of this file
# re-binds every public name to the DynamoDB implementation, so callers
# of ``app.db.X`` always get the right backend.

import aiosqlite
import logging
from datetime import datetime, timezone, timedelta

from app.config import DB_PATH, COOLDOWN_DAYS

logger = logging.getLogger("kb-service.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS research_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    created_at  TEXT NOT NULL,
    completed_at TEXT,
    sources_count INTEGER DEFAULT 0,
    word_count  INTEGER DEFAULT 0,
    error       TEXT,
    added_to_wiki INTEGER DEFAULT 0,
    content     TEXT,
    wiki_slug   TEXT,
    wiki_kb     TEXT,
    job_type    TEXT NOT NULL DEFAULT 'web',
    source_params TEXT
);

CREATE TABLE IF NOT EXISTS research_sources (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  INTEGER NOT NULL REFERENCES research_jobs(id),
    url     TEXT,
    title   TEXT,
    content TEXT,
    tier    INTEGER DEFAULT 3,
    round   INTEGER DEFAULT 1,
    selected INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS article_updates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    article_slug  TEXT NOT NULL,
    kb_name       TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    job_id        INTEGER REFERENCES research_jobs(id),
    change_type   TEXT NOT NULL DEFAULT 'created'
);

CREATE TABLE IF NOT EXISTS article_embeddings (
    slug        TEXT NOT NULL,
    kb          TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (slug, kb)
);

CREATE TABLE IF NOT EXISTS kg_entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL UNIQUE,
    type           TEXT NOT NULL,
    aliases        TEXT,
    article_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id   INTEGER,
    target_entity_id   INTEGER,
    relationship       TEXT NOT NULL,
    article_slug       TEXT,
    kb                 TEXT,
    FOREIGN KEY (source_entity_id) REFERENCES kg_entities(id),
    FOREIGN KEY (target_entity_id) REFERENCES kg_entities(id)
);

CREATE TABLE IF NOT EXISTS palace_classifications (
    slug        TEXT NOT NULL,
    kb          TEXT NOT NULL,
    hall        TEXT NOT NULL,
    confidence  REAL DEFAULT 0.0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (slug, kb)
);

CREATE TABLE IF NOT EXISTS palace_rooms (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kb               TEXT NOT NULL,
    name             TEXT NOT NULL,
    anchor_entity_id INTEGER,
    article_count    INTEGER DEFAULT 0,
    updated_at       TEXT NOT NULL,
    UNIQUE(kb, name)
);

CREATE TABLE IF NOT EXISTS palace_room_members (
    room_id     INTEGER NOT NULL REFERENCES palace_rooms(id),
    slug        TEXT NOT NULL,
    kb          TEXT NOT NULL,
    relevance   REAL DEFAULT 1.0,
    PRIMARY KEY (room_id, slug, kb)
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    event       TEXT NOT NULL,
    user_input  TEXT,
    command     TEXT,
    action      TEXT,
    result      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS serper_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    day         TEXT NOT NULL,
    kb          TEXT,
    job_id      INTEGER,
    query       TEXT,
    num_results INTEGER,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_serper_usage_day_kb ON serper_usage(day, kb);

CREATE TABLE IF NOT EXISTS topic_candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kb           TEXT NOT NULL,
    topic        TEXT NOT NULL,
    source       TEXT NOT NULL,
    source_ref   TEXT,
    score        REAL DEFAULT 0.0,
    status       TEXT NOT NULL DEFAULT 'pending',
    reason       TEXT,
    created_at   TEXT NOT NULL,
    enqueued_at  TEXT,
    job_id       INTEGER,
    UNIQUE(kb, topic)
);
CREATE INDEX IF NOT EXISTS idx_topic_candidates_kb_status ON topic_candidates(kb, status, score DESC);

CREATE TABLE IF NOT EXISTS auto_discovery_config (
    kb            TEXT PRIMARY KEY,
    enabled       INTEGER NOT NULL DEFAULT 0,
    daily_budget  INTEGER NOT NULL DEFAULT 500,
    max_per_hour  INTEGER NOT NULL DEFAULT 3,
    strategy      TEXT NOT NULL DEFAULT 'hybrid',
    seed_topics   TEXT,
    llm_sample    INTEGER NOT NULL DEFAULT 5,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS article_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kb            TEXT NOT NULL,
    article_slug  TEXT NOT NULL,
    full_content  TEXT NOT NULL,
    content_hash  TEXT,
    change_type   TEXT NOT NULL DEFAULT 'updated',
    job_id        INTEGER,
    note          TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_article_versions_lookup
    ON article_versions(kb, article_slug, created_at DESC);

CREATE TABLE IF NOT EXISTS article_claims (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kb              TEXT NOT NULL,
    article_slug    TEXT NOT NULL,
    claim_text      TEXT NOT NULL,
    claim_type      TEXT DEFAULT 'general',
    sources_json    TEXT,
    confidence      REAL DEFAULT 0.5,
    status          TEXT DEFAULT 'unverified',
    last_checked_at TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(kb, article_slug, claim_text)
);
CREATE INDEX IF NOT EXISTS idx_article_claims_lookup
    ON article_claims(kb, article_slug, status);
CREATE INDEX IF NOT EXISTS idx_article_claims_audit
    ON article_claims(status, last_checked_at);

CREATE TABLE IF NOT EXISTS kb_settings (
    kb                  TEXT PRIMARY KEY,
    synthesis_provider  TEXT,
    synthesis_model     TEXT,
    query_provider      TEXT,
    query_model         TEXT,
    persona             TEXT,
    agent_episodes      TEXT,
    source_reliability  TEXT,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage_totals (
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    kind          TEXT NOT NULL,
    calls         INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    PRIMARY KEY (provider, model, kind)
);

CREATE TABLE IF NOT EXISTS llm_traces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id        TEXT NOT NULL,
    span_id         TEXT NOT NULL,
    parent_span_id  TEXT,
    timestamp       TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT 'llm.chat',
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL DEFAULT 'chat',
    prompt_hash     TEXT NOT NULL DEFAULT '',
    prompt_chars    INTEGER NOT NULL DEFAULT 0,
    completion_chars INTEGER NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'ok',
    error           TEXT,
    request_path    TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_traces_trace_id ON llm_traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON llm_traces(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traces_session ON llm_traces(session_id, timestamp DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _get_db() -> aiosqlite.Connection:
    """Open (or create) the database and return a connection.

    WAL mode + synchronous=NORMAL is the standard pairing for
    append-heavy workloads: one fsync per checkpoint instead of per
    transaction, readers never block writers. A 5s busy_timeout means
    concurrent writers queue instead of returning SQLITE_BUSY, and a
    larger cache reduces page churn for the job history scans.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA cache_size=-20000")  # ~20 MiB page cache
    await db.execute("PRAGMA temp_store=MEMORY")
    return db


async def init_db() -> None:
    """Create tables if they don't exist, and run migrations."""
    db = await _get_db()
    try:
        await db.executescript(SCHEMA)
        # Migration: add wiki_slug/wiki_kb columns if missing
        cursor = await db.execute("PRAGMA table_info(research_jobs)")
        columns = {row["name"] for row in await cursor.fetchall()}
        if "wiki_slug" not in columns:
            await db.execute("ALTER TABLE research_jobs ADD COLUMN wiki_slug TEXT")
            logger.info("Migration: added wiki_slug column to research_jobs")
        if "wiki_kb" not in columns:
            await db.execute("ALTER TABLE research_jobs ADD COLUMN wiki_kb TEXT")
            logger.info("Migration: added wiki_kb column to research_jobs")
        # Structured critique claims sidecar (JSON list of {text, status,
        # note} dicts) so the worker can persist them to article_claims
        # after the wiki merge knows the slug.
        if "claims_json" not in columns:
            await db.execute("ALTER TABLE research_jobs ADD COLUMN claims_json TEXT")
            logger.info("Migration: added claims_json column to research_jobs")
        # Migration: add job_type + source_params so we can re-run a
        # failed job with its original inputs (esp. local scans, where
        # path/pattern aren't recoverable from `topic` alone).
        if "job_type" not in columns:
            await db.execute(
                "ALTER TABLE research_jobs ADD COLUMN job_type TEXT NOT NULL DEFAULT 'web'"
            )
            logger.info("Migration: added job_type column to research_jobs")
        if "source_params" not in columns:
            await db.execute("ALTER TABLE research_jobs ADD COLUMN source_params TEXT")
            logger.info("Migration: added source_params column to research_jobs")
        # Migration: agent memory columns on kb_settings. Stores the
        # episodic log and per-domain reliability map as JSON text.
        cursor = await db.execute("PRAGMA table_info(kb_settings)")
        kb_cols = {row["name"] for row in await cursor.fetchall()}
        if "agent_episodes" not in kb_cols:
            await db.execute("ALTER TABLE kb_settings ADD COLUMN agent_episodes TEXT")
            logger.info("Migration: added agent_episodes column to kb_settings")
        if "source_reliability" not in kb_cols:
            await db.execute("ALTER TABLE kb_settings ADD COLUMN source_reliability TEXT")
            logger.info("Migration: added source_reliability column to kb_settings")
        # Migration: add selected column to research_sources
        cursor = await db.execute("PRAGMA table_info(research_sources)")
        src_columns = {row["name"] for row in await cursor.fetchall()}
        if "selected" not in src_columns:
            await db.execute("ALTER TABLE research_sources ADD COLUMN selected INTEGER DEFAULT 1")
            logger.info("Migration: added selected column to research_sources")
        await db.commit()
        logger.info("Database initialized at %s", DB_PATH)
    finally:
        await db.close()


# --- Research jobs -----------------------------------------------------------

async def create_job(
    topic: str,
    *,
    job_type: str = "web",
    source_params: str | None = None,
) -> int:
    """Insert a new queued research job. Returns the job id.

    ``job_type`` is ``'web'`` (default) or ``'local'``.
    ``source_params`` is a JSON string of the original inputs needed to
    re-run the job (e.g. ``{"path": "...", "pattern": "...", "kb": "..."}``
    for local scans). Callers should json-encode before passing.
    """
    db = await _get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO research_jobs (topic, status, created_at, job_type, source_params) "
            "VALUES (?, ?, ?, ?, ?)",
            (topic, "queued", _now_iso(), job_type, source_params),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


_JOB_UPDATE_FIELDS = frozenset({
    "status", "error", "sources_count", "word_count", "content",
    "completed_at", "added_to_wiki", "topic", "kb", "claims_json",
    "wiki_slug", "wiki_kb", "created_at", "source_params",
})


async def update_job(job_id: int, **fields) -> None:
    """Update whitelisted fields on a research job.

    Field names flow directly into the SQL string, so only pre-approved
    column names are allowed — anything else raises rather than risking
    injection. Values are always bound parameters.
    """
    if not fields:
        return
    bad = [k for k in fields if k not in _JOB_UPDATE_FIELDS]
    if bad:
        raise ValueError(f"update_job: disallowed field(s): {sorted(bad)}")
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values())
    vals.append(job_id)
    db = await _get_db()
    try:
        await db.execute(f"UPDATE research_jobs SET {cols} WHERE id = ?", vals)
        await db.commit()
    finally:
        await db.close()


async def get_job(job_id: int) -> dict | None:
    """Fetch a single research job by id."""
    db = await _get_db()
    try:
        cursor = await db.execute("SELECT * FROM research_jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def delete_job(job_id: int) -> None:
    """Delete a research job and its associated sources."""
    db = await _get_db()
    try:
        await db.execute("DELETE FROM research_sources WHERE job_id = ?", (job_id,))
        await db.execute("DELETE FROM research_jobs WHERE id = ?", (job_id,))
        await db.commit()
    finally:
        await db.close()


async def get_jobs(limit: int = 50, compact: bool = False) -> list[dict]:
    """Fetch recent research jobs, newest first.

    ``compact=True`` omits the large ``content`` column so list views
    don't pull megabytes of markdown per poll. The DynamoDB backend
    honours the same flag via ``ProjectionExpression``.
    """
    db = await _get_db()
    try:
        # Column list must stay in sync with the CREATE TABLE schema at
        # the top of this file — the `kb` alias was a typo that would
        # crash under `compact=True` on the sqlite backend. Caught by
        # tests/parity/test_db_jobs.py::test_compact_mode_excludes_content.
        columns = (
            "id, topic, status, created_at, completed_at, sources_count, "
            "word_count, error, added_to_wiki, wiki_slug, wiki_kb, job_type"
            if compact
            else "*"
        )
        cursor = await db.execute(
            f"SELECT {columns} FROM research_jobs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_stuck_jobs() -> list[dict]:
    """Find research jobs stuck in queued/searching status (lost from Redis on restart)."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM research_jobs WHERE status IN ('queued', 'searching', 'searching_round_1', 'searching_round_2', 'searching_round_3', 'synthesizing', 'reading_pages', 'downloading_docs', 'browser_reading') ORDER BY id"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_errored_jobs(limit: int = 50) -> list[dict]:
    """Find research jobs that errored (for retry)."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM research_jobs WHERE status = 'error' ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def reset_job_for_retry(job_id: int):
    """Reset an errored job back to queued for retry."""
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE research_jobs SET status = 'queued', error = NULL WHERE id = ? AND status = 'error'",
            (job_id,),
        )
        await db.commit()
    finally:
        await db.close()


async def get_job_stats() -> dict:
    """Get aggregate job statistics."""
    db = await _get_db()
    try:
        cursor = await db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as complete,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN status IN ('queued', 'searching', 'searching_round_1', 'searching_round_2', 'searching_round_3', 'synthesizing', 'reading_pages', 'downloading_docs', 'browser_reading') THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                SUM(COALESCE(word_count, 0)) as total_words,
                SUM(CASE WHEN added_to_wiki = 1 THEN 1 ELSE 0 END) as added_to_wiki
            FROM research_jobs
        """)
        row = await cursor.fetchone()
        return dict(row) if row else {}
    finally:
        await db.close()


async def check_cooldown(topic: str) -> dict | None:
    """Check if the same topic was researched within the cooldown window.

    Returns the existing job dict if still in cooldown, None otherwise.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=COOLDOWN_DAYS)).isoformat()
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM research_jobs "
            "WHERE topic = ? AND status = 'complete' AND created_at > ? "
            "ORDER BY id DESC LIMIT 1",
            (topic, cutoff),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# --- Research sources --------------------------------------------------------

async def save_sources(job_id: int, sources: list[dict], round_num: int) -> None:
    """Bulk-insert search result sources for a job."""
    if not sources:
        return
    db = await _get_db()
    try:
        await db.executemany(
            "INSERT INTO research_sources (job_id, url, title, content, tier, round) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    job_id,
                    s.get("url", ""),
                    s.get("title", ""),
                    s.get("content", ""),
                    s.get("tier", 3),
                    round_num,
                )
                for s in sources
            ],
        )
        await db.commit()
    finally:
        await db.close()


async def get_sources(job_id: int) -> list[dict]:
    """Fetch all sources for a research job."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM research_sources WHERE job_id = ? ORDER BY round, id",
            (job_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def update_source_selection(job_id: int, source_ids: list[int], selected: bool) -> int:
    """Set the selected flag for specific sources. Returns count updated."""
    if not source_ids:
        return 0
    db = await _get_db()
    try:
        placeholders = ",".join("?" * len(source_ids))
        cursor = await db.execute(
            f"UPDATE research_sources SET selected = ? WHERE job_id = ? AND id IN ({placeholders})",
            [int(selected), job_id] + source_ids,
        )
        await db.commit()
        return cursor.rowcount
    finally:
        await db.close()


async def get_selected_sources(job_id: int) -> list[dict]:
    """Fetch only selected sources for a research job."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM research_sources WHERE job_id = ? AND selected = 1 ORDER BY tier, round, id",
            (job_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def select_all_sources(job_id: int, selected: bool = True) -> int:
    """Select or deselect all sources for a job. Returns count updated."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "UPDATE research_sources SET selected = ? WHERE job_id = ?",
            (int(selected), job_id),
        )
        await db.commit()
        return cursor.rowcount
    finally:
        await db.close()


# --- Article updates ---------------------------------------------------------

async def log_article_update(
    article_slug: str,
    kb_name: str,
    job_id: int | None,
    change_type: str = "created",
) -> None:
    """Record that a wiki article was created or updated."""
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO article_updates (article_slug, kb_name, updated_at, job_id, change_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (article_slug, kb_name, _now_iso(), job_id, change_type),
        )
        await db.commit()
    finally:
        await db.close()


async def get_article_updates(limit: int = 50) -> list[dict]:
    """Fetch recent article updates."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM article_updates ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# --- Palace classifications --------------------------------------------------

async def upsert_classification(slug: str, kb: str, hall: str, confidence: float) -> None:
    """Insert or update an article's hall classification."""
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO palace_classifications (slug, kb, hall, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(slug, kb) DO UPDATE SET hall=excluded.hall, confidence=excluded.confidence, updated_at=excluded.updated_at",
            (slug, kb, hall, confidence, _now_iso()),
        )
        await db.commit()
    finally:
        await db.close()


async def get_classifications(kb: str | None = None) -> list[dict]:
    """Get all palace classifications, optionally filtered by KB."""
    db = await _get_db()
    try:
        if kb:
            cursor = await db.execute(
                "SELECT * FROM palace_classifications WHERE kb = ? ORDER BY hall, slug", (kb,)
            )
        else:
            cursor = await db.execute("SELECT * FROM palace_classifications ORDER BY kb, hall, slug")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_classifications_by_hall(kb: str, hall: str) -> list[dict]:
    """Get articles in a specific hall."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM palace_classifications WHERE kb = ? AND hall = ? ORDER BY confidence DESC",
            (kb, hall),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_article_classification(slug: str, kb: str) -> dict | None:
    """Get palace classification for a single article."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM palace_classifications WHERE slug = ? AND kb = ?", (slug, kb)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# --- Palace rooms ------------------------------------------------------------

async def upsert_room(kb: str, name: str, anchor_entity_id: int | None, article_count: int) -> int:
    """Insert or update a palace room. Returns the room id."""
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO palace_rooms (kb, name, anchor_entity_id, article_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kb, name) DO UPDATE SET anchor_entity_id=excluded.anchor_entity_id, "
            "article_count=excluded.article_count, updated_at=excluded.updated_at",
            (kb, name, anchor_entity_id, article_count, _now_iso()),
        )
        await db.commit()
        cursor = await db.execute(
            "SELECT id FROM palace_rooms WHERE kb = ? AND name = ?", (kb, name)
        )
        row = await cursor.fetchone()
        return row["id"]
    finally:
        await db.close()


async def add_room_member(room_id: int, slug: str, kb: str, relevance: float = 1.0) -> None:
    """Add an article to a palace room."""
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO palace_room_members (room_id, slug, kb, relevance) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(room_id, slug, kb) DO UPDATE SET relevance=excluded.relevance",
            (room_id, slug, kb, relevance),
        )
        await db.commit()
    finally:
        await db.close()


async def get_rooms(kb: str) -> list[dict]:
    """Get all rooms for a KB."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM palace_rooms WHERE kb = ? ORDER BY article_count DESC", (kb,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_room_members(room_id: int) -> list[dict]:
    """Get articles in a room."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM palace_room_members WHERE room_id = ? ORDER BY relevance DESC",
            (room_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_article_rooms(slug: str, kb: str) -> list[dict]:
    """Get all rooms an article belongs to."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT r.name, r.kb, m.relevance FROM palace_room_members m "
            "JOIN palace_rooms r ON m.room_id = r.id "
            "WHERE m.slug = ? AND m.kb = ? ORDER BY m.relevance DESC",
            (slug, kb),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def clear_rooms(kb: str) -> None:
    """Clear all room data for a KB (before rebuilding)."""
    db = await _get_db()
    try:
        room_ids = await db.execute("SELECT id FROM palace_rooms WHERE kb = ?", (kb,))
        ids = [row["id"] async for row in room_ids]
        if ids:
            placeholders = ",".join("?" * len(ids))
            await db.execute(f"DELETE FROM palace_room_members WHERE room_id IN ({placeholders})", ids)
        await db.execute("DELETE FROM palace_rooms WHERE kb = ?", (kb,))
        await db.commit()
    finally:
        await db.close()


# --- Chat sessions -----------------------------------------------------------

async def create_chat_session(session_id: str, title: str = "New Chat") -> dict:
    """Create a new chat session."""
    now = _now_iso()
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        await db.commit()
    finally:
        await db.close()
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_chat_sessions(limit: int = 30) -> list[dict]:
    """List recent chat sessions."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_chat_messages(session_id: str) -> list[dict]:
    """Get all messages for a chat session."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def add_chat_message(session_id: str, role: str, content: str) -> dict:
    """Add a message to a chat session. Creates the session if it doesn't exist."""
    now = _now_iso()
    db = await _get_db()
    try:
        # Ensure session exists
        cursor = await db.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
        if not await cursor.fetchone():
            # Auto-create session, title from first user message
            title = content[:60] if role == "user" else "New Chat"
            await db.execute(
                "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        else:
            await db.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
            # Update title from first user message if still "New Chat"
            if role == "user":
                await db.execute(
                    "UPDATE chat_sessions SET title = ? WHERE id = ? AND title = 'New Chat'",
                    (content[:60], session_id),
                )
        cursor = await db.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        await db.commit()
        return {"id": cursor.lastrowid, "session_id": session_id, "role": role, "content": content, "created_at": now}
    finally:
        await db.close()


async def delete_chat_session(session_id: str) -> None:
    """Delete a chat session and its messages."""
    db = await _get_db()
    try:
        await db.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        await db.commit()
    finally:
        await db.close()


# --- Chat events (observability) ---------------------------------------------

async def log_chat_event(
    event: str,
    session_id: str | None = None,
    user_input: str | None = None,
    command: str | None = None,
    action: str | None = None,
    result: str | None = None,
    error: str | None = None,
) -> None:
    """Log a chat interaction event for analytics."""
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO chat_events (session_id, event, user_input, command, action, result, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, event, user_input, command, action, result, error, _now_iso()),
        )
        await db.commit()
    finally:
        await db.close()


async def get_chat_events(limit: int = 100, event_type: str | None = None) -> list[dict]:
    """Fetch recent chat events for analytics."""
    db = await _get_db()
    try:
        if event_type:
            cursor = await db.execute(
                "SELECT * FROM chat_events WHERE event = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM chat_events ORDER BY id DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_chat_analytics() -> dict:
    """Aggregate chat analytics for observability dashboard."""
    db = await _get_db()
    try:
        # Event counts
        cursor = await db.execute(
            "SELECT event, COUNT(*) as count FROM chat_events GROUP BY event ORDER BY count DESC"
        )
        event_counts = {row["event"]: row["count"] async for row in cursor}

        # Failed commands (user inputs that didn't match any command)
        cursor = await db.execute(
            "SELECT user_input, COUNT(*) as count FROM chat_events "
            "WHERE event = 'command_unmatched' GROUP BY user_input ORDER BY count DESC LIMIT 20"
        )
        unmatched = [{"input": row["user_input"], "count": row["count"]} async for row in cursor]

        # Error patterns
        cursor = await db.execute(
            "SELECT error, COUNT(*) as count FROM chat_events "
            "WHERE error IS NOT NULL GROUP BY error ORDER BY count DESC LIMIT 10"
        )
        errors = [{"error": row["error"], "count": row["count"]} async for row in cursor]

        # Session count
        cursor = await db.execute("SELECT COUNT(*) as count FROM chat_sessions")
        session_count = (await cursor.fetchone())["count"]

        # Message count
        cursor = await db.execute("SELECT COUNT(*) as count FROM chat_messages")
        message_count = (await cursor.fetchone())["count"]

        return {
            "total_sessions": session_count,
            "total_messages": message_count,
            "event_counts": event_counts,
            "unmatched_commands": unmatched,
            "error_patterns": errors,
        }
    finally:
        await db.close()


# --- Auto-discovery: Serper usage -------------------------------------------

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def log_serper_call(
    query: str,
    num_results: int,
    kb: str | None,
    job_id: int | None,
) -> None:
    """Record a single Serper API call for per-KB daily budget tracking."""
    db = await _get_db()
    try:
        await db.execute(
            "INSERT INTO serper_usage (day, kb, job_id, query, num_results, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_today_utc(), kb, job_id, query, num_results, _now_iso()),
        )
        await db.commit()
    finally:
        await db.close()


async def serper_calls_today(kb: str) -> int:
    """Count today's Serper calls attributed to a given KB (UTC day)."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) AS c FROM serper_usage WHERE day = ? AND kb = ?",
            (_today_utc(), kb),
        )
        row = await cursor.fetchone()
        return int(row["c"]) if row else 0
    finally:
        await db.close()


# --- Auto-discovery: topic candidates ----------------------------------------

async def insert_topic_candidate(
    kb: str,
    topic: str,
    source: str,
    source_ref: str | None,
    score: float = 0.0,
) -> bool:
    """Insert a candidate topic. Returns True if inserted, False if duplicate."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "INSERT OR IGNORE INTO topic_candidates "
            "(kb, topic, source, source_ref, score, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (kb, topic, source, source_ref, float(score), _now_iso()),
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def get_pending_candidates(kb: str, limit: int) -> list[dict]:
    """Fetch pending candidates for a KB, highest score first."""
    if limit <= 0:
        return []
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM topic_candidates "
            "WHERE kb = ? AND status = 'pending' "
            "ORDER BY score DESC, created_at ASC LIMIT ?",
            (kb, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def mark_candidate_enqueued(candidate_id: int, job_id: int) -> None:
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE topic_candidates SET status = 'enqueued', enqueued_at = ?, job_id = ? "
            "WHERE id = ?",
            (_now_iso(), job_id, candidate_id),
        )
        await db.commit()
    finally:
        await db.close()


async def mark_candidate_skipped(candidate_id: int, reason: str) -> None:
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE topic_candidates SET status = 'skipped', reason = ? WHERE id = ?",
            (reason, candidate_id),
        )
        await db.commit()
    finally:
        await db.close()


async def count_pending_candidates(kb: str) -> int:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) AS c FROM topic_candidates WHERE kb = ? AND status = 'pending'",
            (kb,),
        )
        row = await cursor.fetchone()
        return int(row["c"]) if row else 0
    finally:
        await db.close()


# --- Auto-discovery: per-KB config -------------------------------------------

async def get_auto_discovery_config(kb: str) -> dict | None:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM auto_discovery_config WHERE kb = ?", (kb,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def list_enabled_auto_discovery_configs() -> list[dict]:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM auto_discovery_config WHERE enabled = 1"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def upsert_auto_discovery_config(kb: str, **fields) -> dict:
    """Upsert per-KB auto-discovery config. Only allowed fields are persisted."""
    allowed = {
        "enabled", "daily_budget", "max_per_hour",
        "strategy", "seed_topics", "llm_sample",
    }
    clean = {k: v for k, v in fields.items() if k in allowed}

    # Normalize types
    if "enabled" in clean:
        clean["enabled"] = 1 if clean["enabled"] else 0
    if "seed_topics" in clean and clean["seed_topics"] is not None and not isinstance(clean["seed_topics"], str):
        import json as _json
        clean["seed_topics"] = _json.dumps(clean["seed_topics"])

    db = await _get_db()
    try:
        # Ensure a row exists with defaults, then update allowed fields.
        await db.execute(
            "INSERT OR IGNORE INTO auto_discovery_config (kb, updated_at) VALUES (?, ?)",
            (kb, _now_iso()),
        )
        if clean:
            cols = ", ".join(f"{k} = ?" for k in clean) + ", updated_at = ?"
            vals = list(clean.values()) + [_now_iso(), kb]
            await db.execute(
                f"UPDATE auto_discovery_config SET {cols} WHERE kb = ?",
                vals,
            )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM auto_discovery_config WHERE kb = ?", (kb,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}
    finally:
        await db.close()


# --- Article version snapshots ---------------------------------------------

import hashlib as _hashlib


async def save_article_version(
    kb: str,
    article_slug: str,
    full_content: str,
    *,
    job_id: int | None = None,
    change_type: str = "updated",
    note: str | None = None,
) -> int:
    """Snapshot an article's full content before it's overwritten.

    Called by update_article in app/wiki.py before the file is rewritten.
    Stores a SHA256 content_hash for future dedup logic.
    """
    content_hash = _hashlib.sha256(full_content.encode("utf-8", errors="replace")).hexdigest()
    db = await _get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO article_versions "
            "(kb, article_slug, full_content, content_hash, change_type, job_id, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (kb, article_slug, full_content, content_hash, change_type, job_id, note, _now_iso()),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_article_versions(kb: str, article_slug: str, limit: int = 50) -> list[dict]:
    """Return version snapshots for an article, newest first."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM article_versions "
            "WHERE kb = ? AND article_slug = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (kb, article_slug, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_article_at_timestamp(
    kb: str, article_slug: str, before_iso: str,
) -> dict | None:
    """Return the latest version snapshot taken before a given timestamp."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM article_versions "
            "WHERE kb = ? AND article_slug = ? AND created_at <= ? "
            "ORDER BY created_at DESC LIMIT 1",
            (kb, article_slug, before_iso),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def get_article_version_by_id(version_id: int) -> dict | None:
    """Fetch a single ``article_versions`` row by its primary key."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM article_versions WHERE id = ?",
            (version_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


# --- Article claims (structured fact storage) ------------------------------


async def save_claim(
    kb: str,
    article_slug: str,
    claim_text: str,
    *,
    claim_type: str = "general",
    sources_json: str | None = None,
    confidence: float = 0.5,
    status: str = "unverified",
) -> int:
    """Insert or update a claim. Returns the claim id.

    Uses INSERT OR IGNORE on (kb, article_slug, claim_text) to dedupe identical
    claims; if a duplicate exists, updates the row's status/confidence/sources.
    """
    db = await _get_db()
    try:
        now = _now_iso()
        cursor = await db.execute(
            "INSERT OR IGNORE INTO article_claims "
            "(kb, article_slug, claim_text, claim_type, sources_json, confidence, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kb, article_slug, claim_text, claim_type, sources_json, confidence, status, now, now),
        )
        if cursor.rowcount == 0:
            # Existing claim — update its mutable fields
            await db.execute(
                "UPDATE article_claims SET claim_type=?, sources_json=?, confidence=?, status=?, updated_at=? "
                "WHERE kb=? AND article_slug=? AND claim_text=?",
                (claim_type, sources_json, confidence, status, now, kb, article_slug, claim_text),
            )
        await db.commit()
        # Fetch the (possibly newly-inserted, possibly pre-existing) row id
        cursor = await db.execute(
            "SELECT id FROM article_claims WHERE kb=? AND article_slug=? AND claim_text=?",
            (kb, article_slug, claim_text),
        )
        row = await cursor.fetchone()
        return int(row["id"]) if row else 0
    finally:
        await db.close()


async def get_claims_for_article(kb: str, article_slug: str) -> list[dict]:
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM article_claims WHERE kb = ? AND article_slug = ? "
            "ORDER BY id ASC",
            (kb, article_slug),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_stale_claims(days: int = 90) -> list[dict]:
    """Return claims whose last_checked_at is older than N days, or NULL."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM article_claims "
            "WHERE last_checked_at IS NULL OR last_checked_at < ? "
            "ORDER BY last_checked_at ASC NULLS FIRST LIMIT 500",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def update_claim_status(
    claim_id: int, status: str, confidence: float,
) -> None:
    """Update a claim's verification status and bump last_checked_at."""
    now = _now_iso()
    db = await _get_db()
    try:
        await db.execute(
            "UPDATE article_claims SET status=?, confidence=?, last_checked_at=?, updated_at=? "
            "WHERE id=?",
            (status, confidence, now, now, claim_id),
        )
        await db.commit()
    finally:
        await db.close()


# --- Per-KB settings (provider/model/persona overrides) --------------------


async def get_kb_settings(kb: str) -> dict | None:
    """Return per-KB LLM/persona settings, or None if not configured."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM kb_settings WHERE kb = ?", (kb,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def upsert_kb_settings(kb: str, **fields) -> dict:
    """Upsert per-KB settings. Only allowed fields are persisted.

    Mirrors upsert_auto_discovery_config: ensure-row-exists, then UPDATE.
    Empty strings are normalized to NULL so callers can clear a field.
    """
    allowed = {
        "synthesis_provider", "synthesis_model",
        "query_provider", "query_model", "persona",
        # Agent memory: episode log + source-reliability map (JSON text).
        "agent_episodes", "source_reliability",
    }
    clean = {k: v for k, v in fields.items() if k in allowed}
    # Normalize empty string → NULL for clearing
    for k in list(clean.keys()):
        if isinstance(clean[k], str) and clean[k].strip() == "":
            clean[k] = None

    db = await _get_db()
    try:
        await db.execute(
            "INSERT OR IGNORE INTO kb_settings (kb, updated_at) VALUES (?, ?)",
            (kb, _now_iso()),
        )
        if clean:
            cols = ", ".join(f"{k} = ?" for k in clean) + ", updated_at = ?"
            vals = list(clean.values()) + [_now_iso(), kb]
            await db.execute(
                f"UPDATE kb_settings SET {cols} WHERE kb = ?",
                vals,
            )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM kb_settings WHERE kb = ?", (kb,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Article metadata index
# ---------------------------------------------------------------------------
#
# SQLite stub implementations. The DynamoDB backend provides the real
# thing in ``db_dynamo.py``; callers on the SQLite path are happy to
# get a no-op (they'll just use the S3-scan read path). This keeps the
# module interface consistent across both backends so tests + app code
# can ``await db.upsert_article_meta(...)`` without caring which
# backend is active.


async def upsert_article_meta(kb: str, slug: str, meta: dict) -> None:
    """SQLite no-op — the metadata index is DynamoDB-only."""
    return None


async def get_article_meta(kb: str, slug: str) -> dict | None:
    return None


async def list_article_metas(kb: str) -> list[dict]:
    return []


async def delete_article_meta(kb: str, slug: str) -> None:
    return None


async def article_meta_count(kb: str) -> int:
    return 0


# ---------------------------------------------------------------------------
# LLM trace spans
# ---------------------------------------------------------------------------


async def save_trace_span(span: dict) -> None:
    """Insert a single trace span into llm_traces."""
    db = await _get_db()
    try:
        await db.execute(
            """INSERT INTO llm_traces
               (trace_id, span_id, parent_span_id, timestamp, name,
                provider, model, kind, prompt_hash, prompt_chars,
                completion_chars, input_tokens, output_tokens,
                duration_ms, status, error, request_path, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                span.get("trace_id", ""),
                span.get("span_id", ""),
                span.get("parent_span_id"),
                span.get("timestamp", ""),
                span.get("name", "llm.chat"),
                span.get("provider", ""),
                span.get("model", ""),
                span.get("kind", "chat"),
                span.get("prompt_hash", ""),
                int(span.get("prompt_chars", 0)),
                int(span.get("completion_chars", 0)),
                int(span.get("input_tokens", 0)),
                int(span.get("output_tokens", 0)),
                int(span.get("duration_ms", 0)),
                span.get("status", "ok"),
                span.get("error"),
                span.get("request_path", ""),
                span.get("session_id", ""),
            ),
        )
        await db.commit()
    finally:
        await db.close()


async def get_trace_spans(
    limit: int = 50,
    *,
    session_id: str | None = None,
    provider: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return recent trace spans, optionally filtered."""
    db = await _get_db()
    try:
        clauses = []
        params: list = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        cursor = await db.execute(
            f"SELECT * FROM llm_traces{where} ORDER BY id DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


async def get_trace_spans_by_trace_id(trace_id: str) -> list[dict]:
    """Return all spans for a single trace."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM llm_traces WHERE trace_id = ? ORDER BY id",
            (trace_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# LLM usage totals (all-time)
# ---------------------------------------------------------------------------


async def record_llm_usage_total(
    provider: str, model: str, kind: str,
    *, input_tokens: int, output_tokens: int,
) -> None:
    """Atomically accumulate per-(provider, model, kind) lifetime totals."""
    now = _now_iso()
    db = await _get_db()
    try:
        await db.execute(
            """
            INSERT INTO llm_usage_totals
                (provider, model, kind, calls, input_tokens, output_tokens,
                 first_seen, last_seen)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(provider, model, kind) DO UPDATE SET
                calls = calls + 1,
                input_tokens = input_tokens + excluded.input_tokens,
                output_tokens = output_tokens + excluded.output_tokens,
                last_seen = excluded.last_seen
            """,
            (provider, model, kind, int(input_tokens or 0),
             int(output_tokens or 0), now, now),
        )
        await db.commit()
    finally:
        await db.close()


async def get_llm_usage_totals() -> list[dict]:
    """Return every (provider, model, kind) row with lifetime counters."""
    db = await _get_db()
    try:
        cursor = await db.execute(
            """
            SELECT provider, model, kind, calls, input_tokens, output_tokens,
                   first_seen, last_seen
              FROM llm_usage_totals
             ORDER BY calls DESC
            """
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        await db.close()


# --- Backend trailer: override every helper when DB_BACKEND=dynamodb --------

if _DB_BACKEND == "dynamodb":
    from app.db_dynamo import (  # noqa: F401
        init_db, create_job, update_job, get_job, delete_job, get_jobs,
        get_stuck_jobs, get_errored_jobs, reset_job_for_retry, get_job_stats,
        check_cooldown, save_sources, get_sources,
        update_source_selection, get_selected_sources, select_all_sources,
        log_article_update,
        get_article_updates, upsert_classification, get_classifications,
        get_classifications_by_hall, get_article_classification,
        upsert_room, add_room_member, get_rooms, get_room_members,
        get_article_rooms, clear_rooms,
        create_chat_session, get_chat_sessions, get_chat_messages,
        add_chat_message, delete_chat_session,
        log_chat_event, get_chat_events, get_chat_analytics,
        log_serper_call, serper_calls_today,
        insert_topic_candidate, get_pending_candidates,
        mark_candidate_enqueued, mark_candidate_skipped,
        count_pending_candidates,
        get_auto_discovery_config, list_enabled_auto_discovery_configs,
        upsert_auto_discovery_config,
        save_article_version, get_article_versions, get_article_at_timestamp,
        get_article_version_by_id,
        save_claim, get_claims_for_article, get_stale_claims, update_claim_status,
        get_kb_settings, upsert_kb_settings,
        upsert_article_meta, get_article_meta, list_article_metas,
        delete_article_meta, article_meta_count,
        record_llm_usage_total, get_llm_usage_totals,
    )
