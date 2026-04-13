"""Unit tests for app/auto_discovery.py and its DB plumbing."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


# --- Isolated DB path for this module --------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    """Point the DB at a temp file and rebuild module-level DB_PATH refs."""
    global _test_db_path
    _test_db_path = tmp_path / "test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            with patch("app.auto_discovery.DB_PATH", _test_db_path):
                yield


@pytest.fixture
def tmp_wiki_kb(tmp_path, monkeypatch):
    """Create a KB directory with a couple of seed articles and route storage at it."""
    kb_dir = tmp_path / "personal"
    wiki = kb_dir / "wiki"
    wiki.mkdir(parents=True)

    (wiki / "tokio-runtime.md").write_text(
        '---\ntitle: "Tokio Runtime"\ntags: [rust, async]\n'
        "status: published\nupdated: 2026-04-01\n---\n\n"
        "## Overview\n\nTokio is an async runtime for Rust.\n"
    )
    (wiki / "postgres-basics.md").write_text(
        '---\ntitle: "PostgreSQL Basics"\ntags: [postgres, database]\n'
        "status: published\nupdated: 2026-04-01\n---\n\n"
        "## Overview\n\nPostgreSQL is a relational database.\n"
    )

    import app.config as _config
    import app.storage as _storage
    import app.wiki as _wiki

    dirs = {"personal": kb_dir}
    prev_kb_dirs = dict(_config.KB_DIRS)
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(dirs)

    fallback_root = tmp_path / "_fallback"
    fallback_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_storage, "KB_ROOT", fallback_root)

    prev_backend = _storage._default
    _storage._default = _storage.LocalStorage()
    _wiki.invalidate_articles_cache()

    try:
        yield kb_dir
    finally:
        _config.KB_DIRS.clear()
        _config.KB_DIRS.update(prev_kb_dirs)
        _storage._default = prev_backend
        _wiki.invalidate_articles_cache()


from app import db
from app import auto_discovery as ad


# --- DB helper smoke tests --------------------------------------------------


class TestSerperUsageTracking:
    @pytest.mark.asyncio
    async def test_log_and_count_per_kb(self):
        await db.init_db()
        await db.log_serper_call("q1", 8, "personal", 1)
        await db.log_serper_call("q2", 8, "personal", 1)
        await db.log_serper_call("q3", 8, "work", 2)

        assert await db.serper_calls_today("personal") == 2
        assert await db.serper_calls_today("work") == 1
        assert await db.serper_calls_today("other") == 0

    @pytest.mark.asyncio
    async def test_null_kb_is_not_counted(self):
        """Calls with kb=None (e.g. fact-check) shouldn't consume any budget."""
        await db.init_db()
        await db.log_serper_call("fact-check", 8, None, None)
        assert await db.serper_calls_today("personal") == 0


class TestTopicCandidates:
    @pytest.mark.asyncio
    async def test_insert_dedupes_on_topic(self):
        await db.init_db()
        assert await db.insert_topic_candidate("personal", "Tokio", "kg_entity", "Tokio", 5.0) is True
        assert await db.insert_topic_candidate("personal", "Tokio", "kg_entity", "Tokio", 5.0) is False
        assert await db.count_pending_candidates("personal") == 1

    @pytest.mark.asyncio
    async def test_get_pending_orders_by_score_desc(self):
        await db.init_db()
        await db.insert_topic_candidate("personal", "Low", "kg_entity", None, 1.0)
        await db.insert_topic_candidate("personal", "High", "kg_entity", None, 10.0)
        await db.insert_topic_candidate("personal", "Mid", "kg_entity", None, 5.0)
        pending = await db.get_pending_candidates("personal", 10)
        assert [p["topic"] for p in pending] == ["High", "Mid", "Low"]

    @pytest.mark.asyncio
    async def test_mark_enqueued_and_skipped_transitions(self):
        await db.init_db()
        await db.insert_topic_candidate("personal", "A", "kg_entity", None, 1.0)
        await db.insert_topic_candidate("personal", "B", "kg_entity", None, 1.0)
        pending = await db.get_pending_candidates("personal", 10)
        await db.mark_candidate_enqueued(pending[0]["id"], 42)
        await db.mark_candidate_skipped(pending[1]["id"], "cooldown")
        assert await db.count_pending_candidates("personal") == 0


class TestAutoDiscoveryConfig:
    @pytest.mark.asyncio
    async def test_upsert_roundtrip(self):
        await db.init_db()
        cfg = await db.upsert_auto_discovery_config(
            "personal",
            enabled=True,
            daily_budget=250,
            max_per_hour=2,
            strategy="hybrid",
            seed_topics=["Rust async", "Tokio"],
        )
        assert cfg["enabled"] == 1
        assert cfg["daily_budget"] == 250
        assert cfg["strategy"] == "hybrid"
        assert json.loads(cfg["seed_topics"]) == ["Rust async", "Tokio"]

    @pytest.mark.asyncio
    async def test_list_enabled_filters_disabled(self):
        await db.init_db()
        await db.upsert_auto_discovery_config("personal", enabled=True)
        await db.upsert_auto_discovery_config("work", enabled=False)
        enabled = await db.list_enabled_auto_discovery_configs()
        assert [c["kb"] for c in enabled] == ["personal"]

    @pytest.mark.asyncio
    async def test_upsert_rejects_unknown_fields(self):
        await db.init_db()
        # Unknown fields are silently dropped; defaults remain.
        cfg = await db.upsert_auto_discovery_config("personal", enabled=True, nonsense="x")
        assert "nonsense" not in cfg


# --- Discovery strategies ---------------------------------------------------


class TestKgEntitiesDiscovery:
    @pytest.mark.asyncio
    async def test_skips_entities_with_existing_article(self, tmp_wiki_kb):
        """Entities whose name already maps to a wiki article must be skipped."""
        await db.init_db()

        # Seed kg_entities with 5 entries. Two ("Tokio Runtime", "PostgreSQL Basics")
        # already have wiki articles — those must be skipped.
        import aiosqlite
        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            await conn.execute("INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)", ("Tokio Runtime", "framework", 3))
            await conn.execute("INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)", ("PostgreSQL Basics", "tool", 2))
            await conn.execute("INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)", ("Redis Streams", "tool", 4))
            await conn.execute("INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)", ("Kafka Partitions", "concept", 3))
            await conn.execute("INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)", ("Event Sourcing", "concept", 5))
            await conn.commit()
        finally:
            await conn.close()

        inserted = await ad.discover_from_kg_entities("personal", seed_topics=None, limit=50)
        assert inserted == 3

        pending = await db.get_pending_candidates("personal", 50)
        names = {p["topic"] for p in pending}
        assert "Tokio Runtime" not in names
        assert "PostgreSQL Basics" not in names
        assert names == {"Redis Streams", "Kafka Partitions", "Event Sourcing"}

    @pytest.mark.asyncio
    async def test_seed_topics_scopes_to_related_entities(self, tmp_wiki_kb):
        """When seed_topics is set, only entities linked to matching articles are considered."""
        await db.init_db()

        import aiosqlite
        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            # Two entities: one connected to the Tokio article, one to Postgres.
            c = await conn.execute(
                "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)",
                ("Async IO", "concept", 4),
            )
            async_id = c.lastrowid
            c = await conn.execute(
                "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)",
                ("B-Tree Index", "concept", 3),
            )
            btree_id = c.lastrowid
            c = await conn.execute(
                "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)",
                ("Futures Crate", "framework", 2),
            )
            futures_id = c.lastrowid

            await conn.execute(
                "INSERT INTO kg_edges (source_entity_id, target_entity_id, relationship, article_slug, kb) VALUES (?, ?, ?, ?, ?)",
                (async_id, futures_id, "uses", "tokio-runtime", "personal"),
            )
            await conn.execute(
                "INSERT INTO kg_edges (source_entity_id, target_entity_id, relationship, article_slug, kb) VALUES (?, ?, ?, ?, ?)",
                (btree_id, btree_id, "part-of", "postgres-basics", "personal"),
            )
            await conn.commit()
        finally:
            await conn.close()

        inserted = await ad.discover_from_kg_entities(
            "personal", seed_topics=["Tokio"], limit=50,
        )
        pending = await db.get_pending_candidates("personal", 50)
        names = {p["topic"] for p in pending}
        # B-Tree Index lives on the postgres article — must NOT appear.
        assert "B-Tree Index" not in names
        # Async IO and Futures Crate live on the tokio article — must appear.
        assert "Async IO" in names
        assert "Futures Crate" in names
        assert inserted == 2


class TestLlmFollowupDiscovery:
    @pytest.mark.asyncio
    async def test_generates_candidates_from_sampled_articles(self, tmp_wiki_kb):
        await db.init_db()

        fake_response = "Tokio Spawning Patterns\nRust Async Cancellation\nAsync Trait Methods\nStructured Concurrency"

        async def fake_llm_chat(*args, **kwargs):
            return fake_response

        with patch("app.auto_discovery.llm_chat", side_effect=fake_llm_chat):
            inserted = await ad.discover_from_llm_followups(
                "personal", seed_topics=None, sample_size=1, per_article=4,
            )

        assert inserted >= 3
        pending = await db.get_pending_candidates("personal", 50)
        names = {p["topic"] for p in pending}
        # At least one of the LLM-proposed topics should have landed.
        assert "Tokio Spawning Patterns" in names or "Structured Concurrency" in names
        # And the source should be 'llm_followup'
        assert any(p["source"] == "llm_followup" for p in pending)

    @pytest.mark.asyncio
    async def test_seed_topics_narrows_source_articles(self, tmp_wiki_kb):
        """If seed_topics matches nothing, no LLM calls happen and nothing is inserted."""
        await db.init_db()
        called = {"count": 0}

        async def fake_llm_chat(*args, **kwargs):
            called["count"] += 1
            return "Some Topic\nAnother Topic"

        with patch("app.auto_discovery.llm_chat", side_effect=fake_llm_chat):
            inserted = await ad.discover_from_llm_followups(
                "personal",
                seed_topics=["nonexistent-topic-xyz"],
                sample_size=5,
                per_article=2,
            )
        assert inserted == 0
        assert called["count"] == 0


# --- Enqueue loop -----------------------------------------------------------


class _FakeArqPool:
    """Minimal ArqRedis stand-in for enqueue tests."""

    def __init__(self):
        self.enqueued = []
        self.kv = {}

    async def enqueue_job(self, name, *args, **kwargs):
        self.enqueued.append((name, args, kwargs))

    async def set(self, key, value):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)


class TestEnqueueLoop:
    @pytest.mark.asyncio
    async def test_global_disabled_short_circuits(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config("personal", enabled=True, daily_budget=1000, max_per_hour=3)
        await db.insert_topic_candidate("personal", "Anything", "kg_entity", None, 1.0)

        pool = _FakeArqPool()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", False):
            result = await ad.enqueue_next_candidates_for_kb(pool, "personal")
        assert result == {"kb": "personal", "skipped": "global_disabled"}
        assert pool.enqueued == []

    @pytest.mark.asyncio
    async def test_kb_disabled_short_circuits(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config("personal", enabled=False)

        pool = _FakeArqPool()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True):
            result = await ad.enqueue_next_candidates_for_kb(pool, "personal")
        assert result["skipped"] == "disabled"

    @pytest.mark.asyncio
    async def test_budget_exhausted_returns_budget_skip(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config(
            "personal", enabled=True, daily_budget=10, max_per_hour=3,
        )
        # Pre-log 5 calls → remaining=5 < estimate(8) → skip
        for _ in range(5):
            await db.log_serper_call("q", 1, "personal", None)
        await db.insert_topic_candidate("personal", "A topic that is long enough", "kg_entity", None, 1.0)

        pool = _FakeArqPool()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True), \
             patch("app.auto_discovery.SERPER_CALLS_PER_JOB_ESTIMATE", 8):
            result = await ad.enqueue_next_candidates_for_kb(pool, "personal")
        assert result["skipped"] == "budget"
        assert result["remaining"] == 5
        assert pool.enqueued == []

    @pytest.mark.asyncio
    async def test_capacity_math_caps_at_max_per_hour(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config(
            "personal", enabled=True, daily_budget=100, max_per_hour=3,
        )
        # 5 candidates; only 3 should be enqueued per hour.
        for i in range(5):
            await db.insert_topic_candidate(
                "personal", f"Candidate Topic {i}", "kg_entity", None, float(10 - i),
            )

        pool = _FakeArqPool()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True), \
             patch("app.auto_discovery.SERPER_CALLS_PER_JOB_ESTIMATE", 8):
            result = await ad.enqueue_next_candidates_for_kb(pool, "personal")

        assert result["enqueued"] == 3
        assert len(pool.enqueued) == 3

        # Every enqueued job must target the 'research_task' function on the
        # wikidelve queue — missing _queue_name would send jobs into limbo.
        for name, args, kwargs in pool.enqueued:
            assert name == "research_task"
            assert kwargs.get("_queue_name") == "wikidelve"
            # args: (topic, job_id, kb)
            assert len(args) == 3
            assert args[2] == "personal"

        # Two candidates should still be pending.
        assert await db.count_pending_candidates("personal") == 2

    @pytest.mark.asyncio
    async def test_candidate_with_cooldown_is_skipped(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config(
            "personal", enabled=True, daily_budget=100, max_per_hour=3,
        )
        # Insert a completed recent job for the candidate topic → triggers cooldown.
        topic = "Advanced Cooldown Topic Name"
        job_id = await db.create_job(topic)
        await db.update_job(job_id, status="complete")
        await db.insert_topic_candidate("personal", topic, "kg_entity", None, 5.0)

        pool = _FakeArqPool()
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True), \
             patch("app.auto_discovery.SERPER_CALLS_PER_JOB_ESTIMATE", 8):
            result = await ad.enqueue_next_candidates_for_kb(pool, "personal")

        assert result["enqueued"] == 0
        assert result["skipped_cooldown"] == 1
        assert pool.enqueued == []

    @pytest.mark.asyncio
    async def test_per_kb_budget_isolation(self, tmp_wiki_kb):
        """Budget exhaustion on one KB must not affect the other."""
        await db.init_db()
        # Add a second KB so both can be exercised.
        (tmp_wiki_kb.parent / "work" / "wiki").mkdir(parents=True, exist_ok=True)
        dirs = {"personal": tmp_wiki_kb, "work": tmp_wiki_kb.parent / "work"}

        await db.upsert_auto_discovery_config(
            "personal", enabled=True, daily_budget=100, max_per_hour=1,
        )
        await db.upsert_auto_discovery_config(
            "work", enabled=True, daily_budget=8, max_per_hour=1,
        )
        await db.insert_topic_candidate(
            "personal", "Personal topic placeholder name", "kg_entity", None, 1.0,
        )
        await db.insert_topic_candidate(
            "work", "Work topic placeholder name", "kg_entity", None, 1.0,
        )
        # Pre-log 5 calls on work → remaining 3 < 8 estimate → budget skip.
        for _ in range(5):
            await db.log_serper_call("q", 1, "work", None)

        pool = _FakeArqPool()
        import app.config as _config
        _config.KB_DIRS.clear()
        _config.KB_DIRS.update(dirs)
        with patch("app.auto_discovery.AUTO_DISCOVERY_ENABLED", True), \
             patch("app.auto_discovery.SERPER_CALLS_PER_JOB_ESTIMATE", 8):
            p_result = await ad.enqueue_next_candidates_for_kb(pool, "personal")
            w_result = await ad.enqueue_next_candidates_for_kb(pool, "work")

        assert p_result["enqueued"] == 1
        assert w_result["skipped"] == "budget"
        # Only the personal job made it onto the queue.
        assert len(pool.enqueued) == 1
        assert pool.enqueued[0][1][2] == "personal"


# --- Serper instrumentation -------------------------------------------------


class TestSerperInstrumentation:
    @pytest.mark.asyncio
    async def test_search_serper_logs_against_current_kb(self):
        """_search_serper must record calls against the ContextVar-tracked KB."""
        await db.init_db()

        from app import research

        class FakeResp:
            def __init__(self):
                self._json = {"organic": [{"title": "t", "snippet": "s", "link": "u"}]}

            def raise_for_status(self):
                return None

            def json(self):
                return self._json

        class FakeClient:
            async def post(self, *args, **kwargs):
                return FakeResp()

        # Set ContextVars as run_research would
        tok_kb = research._current_kb.set("personal")
        tok_job = research._current_job_id.set(42)
        try:
            with patch("app.research.SERPER_API_KEY", "fake-key"):
                await research._search_serper(FakeClient(), "query-a", num=8)
                await research._search_serper(FakeClient(), "query-b", num=8)
                await research._search_serper(FakeClient(), "query-c", num=8)
        finally:
            research._current_kb.reset(tok_kb)
            research._current_job_id.reset(tok_job)

        assert await db.serper_calls_today("personal") == 3
        assert await db.serper_calls_today("work") == 0


# --- Helper function tests ---------------------------------------------------


class TestParseSeedTopics:
    def test_none_returns_none(self):
        assert ad._parse_seed_topics(None) is None

    def test_empty_string_returns_none(self):
        assert ad._parse_seed_topics("") is None

    def test_valid_json_list(self):
        result = ad._parse_seed_topics('["Rust", "Python"]')
        assert result == ["Rust", "Python"]

    def test_invalid_json_returns_none(self):
        assert ad._parse_seed_topics("not json") is None

    def test_non_list_json_returns_none(self):
        assert ad._parse_seed_topics('{"key": "val"}') is None

    def test_strips_whitespace(self):
        result = ad._parse_seed_topics('[" Rust ", "  Python  "]')
        assert result == ["Rust", "Python"]

    def test_empty_list_returns_none(self):
        assert ad._parse_seed_topics("[]") is None

    def test_filters_empty_strings(self):
        result = ad._parse_seed_topics('["Rust", "", "  "]')
        assert result == ["Rust"]


class TestWikilinkToSlugLocal:
    def test_basic_conversion(self):
        assert ad._wikilink_to_slug_local("My Article Title") == "my-article-title"

    def test_special_chars_stripped(self):
        assert ad._wikilink_to_slug_local("Hello! World?") == "hello-world"

    def test_preserves_hyphens(self):
        assert ad._wikilink_to_slug_local("pre-existing") == "pre-existing"

    def test_numbers_preserved(self):
        assert ad._wikilink_to_slug_local("Python 3.11") == "python-311"

    def test_empty_string(self):
        assert ad._wikilink_to_slug_local("") == ""


class TestQuestionRegex:
    def test_matches_question_sentence(self):
        text = "How does the garbage collector handle circular references?"
        matches = ad._QUESTION_RE.findall(text)
        assert len(matches) == 1

    def test_skips_short_questions(self):
        text = "What is it?"  # Under 15 chars after the capital
        matches = ad._QUESTION_RE.findall(text)
        assert len(matches) == 0

    def test_skips_url_like_patterns(self):
        text = "https://example.com/path?query=value"
        matches = ad._QUESTION_RE.findall(text)
        assert len(matches) == 0


class TestStubBulletRegex:
    def test_matches_bullet_items(self):
        text = "- Kubernetes Pod Scheduling\n- Container Networking Basics\n"
        matches = ad._STUB_BULLET_RE.findall(text)
        assert len(matches) == 2

    def test_matches_asterisk_bullets(self):
        text = "* Advanced Memory Management\n"
        matches = ad._STUB_BULLET_RE.findall(text)
        assert len(matches) == 1

    def test_skips_very_short_bullets(self):
        text = "- short\n"  # Under 8 chars
        matches = ad._STUB_BULLET_RE.findall(text)
        assert len(matches) == 0


class TestHighVelocityTags:
    def test_common_tags_present(self):
        for tag in ["ai", "ml", "kubernetes", "rust", "python", "docker"]:
            assert tag in ad._HIGH_VELOCITY_TAGS


# --- Self-mining strategy tests -----------------------------------------------


class TestDiscoverFromStale:
    @pytest.mark.asyncio
    async def test_queues_old_high_velocity_articles(self, tmp_wiki_kb):
        await db.init_db()
        # Patch get_articles to return an old article with a velocity tag
        old_articles = [
            {
                "slug": "old-k8s-article",
                "title": "Kubernetes Deep Dive",
                "tags": ["kubernetes", "devops"],
                "updated": "2025-01-01",
                "word_count": 500,
            },
        ]
        with patch("app.auto_discovery.get_articles", return_value=old_articles), \
             patch("app.auto_discovery.db.check_cooldown", new=AsyncMock(return_value=False)):
            inserted = await ad.discover_from_stale("personal", days=90)
        assert inserted == 1

    @pytest.mark.asyncio
    async def test_skips_recent_articles(self, tmp_wiki_kb):
        await db.init_db()
        recent_articles = [
            {
                "slug": "recent-article",
                "title": "Recent Article",
                "tags": ["kubernetes"],
                "updated": "2026-04-01",
                "word_count": 500,
            },
        ]
        with patch("app.auto_discovery.get_articles", return_value=recent_articles):
            inserted = await ad.discover_from_stale("personal", days=90)
        assert inserted == 0

    @pytest.mark.asyncio
    async def test_skips_non_velocity_tags(self, tmp_wiki_kb):
        await db.init_db()
        articles = [
            {
                "slug": "old-cooking",
                "title": "Cooking Tips",
                "tags": ["cooking", "recipes"],
                "updated": "2025-01-01",
                "word_count": 500,
            },
        ]
        with patch("app.auto_discovery.get_articles", return_value=articles):
            inserted = await ad.discover_from_stale("personal", days=90)
        assert inserted == 0

    @pytest.mark.asyncio
    async def test_empty_kb_returns_zero(self, tmp_wiki_kb):
        await db.init_db()
        with patch("app.auto_discovery.get_articles", return_value=[]):
            inserted = await ad.discover_from_stale("personal")
        assert inserted == 0


class TestDiscoverFromBrokenWikilinks:
    @pytest.mark.asyncio
    async def test_finds_broken_wikilinks(self, tmp_wiki_kb):
        await db.init_db()
        articles = [
            {"slug": "main-article", "title": "Main Article"},
        ]
        raw_body = "Check out [[Nonexistent Topic Article]] for more info.\n"
        with patch("app.auto_discovery.get_articles", return_value=articles), \
             patch("app.auto_discovery.storage.read_text", return_value=raw_body), \
             patch("app.auto_discovery.find_related_article", return_value=None), \
             patch("app.auto_discovery.db.check_cooldown", new=AsyncMock(return_value=False)):
            inserted = await ad.discover_from_broken_wikilinks("personal")
        assert inserted == 1

    @pytest.mark.asyncio
    async def test_skips_existing_slugs(self, tmp_wiki_kb):
        await db.init_db()
        articles = [
            {"slug": "main-article", "title": "Main Article"},
            {"slug": "existing-target", "title": "Existing Target"},
        ]
        raw_body = "See [[Existing Target]] for more.\n"
        with patch("app.auto_discovery.get_articles", return_value=articles), \
             patch("app.auto_discovery.storage.read_text", return_value=raw_body):
            inserted = await ad.discover_from_broken_wikilinks("personal")
        assert inserted == 0

    @pytest.mark.asyncio
    async def test_respects_limit(self, tmp_wiki_kb):
        await db.init_db()
        articles = [{"slug": "art", "title": "Art"}]
        raw_body = "\n".join(f"[[Broken Link Number {i} Here]]" for i in range(50))
        with patch("app.auto_discovery.get_articles", return_value=articles), \
             patch("app.auto_discovery.storage.read_text", return_value=raw_body), \
             patch("app.auto_discovery.find_related_article", return_value=None), \
             patch("app.auto_discovery.db.check_cooldown", new=AsyncMock(return_value=False)):
            inserted = await ad.discover_from_broken_wikilinks("personal", limit=5)
        assert inserted <= 5


class TestDiscoverFromQuestions:
    @pytest.mark.asyncio
    async def test_mines_questions_from_body(self, tmp_wiki_kb):
        await db.init_db()
        articles = [{"slug": "test-art", "title": "Test Article"}]
        full_article = {
            "slug": "test-art",
            "title": "Test Article",
            "raw_markdown": (
                "Some content here that is long enough.\n\n"
                "How does the Kubernetes scheduler handle pod affinity constraints?\n\n"
                "More content follows.\n" + "padding " * 50
            ),
        }
        with patch("app.auto_discovery.get_articles", return_value=articles), \
             patch("app.auto_discovery.get_article", return_value=full_article), \
             patch("app.auto_discovery.find_related_article", return_value=None), \
             patch("app.auto_discovery.db.check_cooldown", new=AsyncMock(return_value=False)):
            inserted = await ad.discover_from_questions("personal")
        assert inserted >= 1

    @pytest.mark.asyncio
    async def test_skips_rhetorical_questions(self, tmp_wiki_kb):
        await db.init_db()
        articles = [{"slug": "test-art", "title": "Test"}]
        full_article = {
            "slug": "test-art",
            "title": "Test",
            "raw_markdown": (
                "But what about this edge case thing?\n"
                "So how does this actually work in practice?\n"
                + "padding " * 50
            ),
        }
        with patch("app.auto_discovery.get_articles", return_value=articles), \
             patch("app.auto_discovery.get_article", return_value=full_article), \
             patch("app.auto_discovery.find_related_article", return_value=None), \
             patch("app.auto_discovery.db.check_cooldown", new=AsyncMock(return_value=False)):
            inserted = await ad.discover_from_questions("personal")
        # Both start with "But " and "So " which are filtered
        assert inserted == 0


class TestRunDiscoveryForKb:
    @pytest.mark.asyncio
    async def test_disabled_config_skips(self, tmp_wiki_kb):
        await db.init_db()
        with patch("app.auto_discovery.db.get_auto_discovery_config",
                    new=AsyncMock(return_value={"enabled": False})):
            result = await ad.run_discovery_for_kb("personal")
        assert result["skipped"] == "disabled"

    @pytest.mark.asyncio
    async def test_no_config_skips(self, tmp_wiki_kb):
        await db.init_db()
        with patch("app.auto_discovery.db.get_auto_discovery_config",
                    new=AsyncMock(return_value=None)):
            result = await ad.run_discovery_for_kb("personal")
        assert result["skipped"] == "disabled"

    @pytest.mark.asyncio
    async def test_single_mining_strategy(self, tmp_wiki_kb):
        await db.init_db()
        cfg = {"enabled": True, "strategy": "stale", "seed_topics": None, "llm_sample": 5}
        with patch("app.auto_discovery.db.get_auto_discovery_config",
                    new=AsyncMock(return_value=cfg)), \
             patch("app.auto_discovery.discover_from_stale",
                    new=AsyncMock(return_value=3)), \
             patch("app.auto_discovery.db.count_pending_candidates",
                    new=AsyncMock(return_value=3)):
            result = await ad.run_discovery_for_kb("personal")
        assert result["counts"]["stale"] == 3


class TestGetStatusForKb:
    @pytest.mark.asyncio
    async def test_returns_status_dict(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config(
            "personal", enabled=True, daily_budget=100, max_per_hour=3, strategy="hybrid",
        )
        result = await ad.get_status_for_kb("personal")
        assert result["kb"] == "personal"
        assert result["enabled"] is True
        assert result["daily_budget"] == 100
        assert result["max_per_hour"] == 3

    @pytest.mark.asyncio
    async def test_no_config_returns_defaults(self, tmp_wiki_kb):
        await db.init_db()
        result = await ad.get_status_for_kb("nonexistent")
        assert result["enabled"] is False
        assert result["daily_budget"] == 0

    @pytest.mark.asyncio
    async def test_with_arq_pool_reads_last_run(self, tmp_wiki_kb):
        await db.init_db()
        await db.upsert_auto_discovery_config("personal", enabled=True)

        pool = _FakeArqPool()
        import json
        await pool.set("auto_discovery:last_run:personal", json.dumps({"enqueued": 2}))

        result = await ad.get_status_for_kb("personal", arq_pool=pool)
        assert result["last_run"] == {"enqueued": 2}
