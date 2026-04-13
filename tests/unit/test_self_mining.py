"""Unit tests for the self-mining discovery strategies.

Covers:
  - Contradiction harvester (article pair finding + LLM verdict parsing)
  - Stale article rerun (90-day-old high-velocity tags)
  - Orphan entity mining (entities in ≥3 articles, no own page)
  - Question miner (?-ending sentences)
  - Research history clustering (sparse cluster detection)
  - TOC stub miner (short bullet phrases)
  - Broken wikilink miner ([[X]] without resolving slug)
  - Dispatcher: strategy=all runs every self-mining strategy
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "phase4_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            with patch("app.auto_discovery.DB_PATH", _test_db_path):
                yield


@pytest.fixture
def tmp_kb_with_articles(tmp_path, monkeypatch):
    """A temp KB with two tagged articles + one stale article."""
    kb_dir = tmp_path / "personal"
    wiki = kb_dir / "wiki"
    wiki.mkdir(parents=True)

    (wiki / "tokio.md").write_text(
        '---\ntitle: "Tokio Runtime"\ntags: [rust, async]\n'
        'updated: 2026-04-01\n---\n\n'
        "## Overview\n\n"
        "Tokio is an async runtime for Rust. See [[Async Cancellation Patterns]] "
        "for more.\n\n"
        "How does Tokio handle structured concurrency?\n\n"
        "- Spawning tasks\n"
        "- Channels\n"
    )

    (wiki / "rust-channels.md").write_text(
        '---\ntitle: "Rust Channels"\ntags: [rust, concurrency]\n'
        'updated: 2026-04-02\n---\n\n'
        "Rust has crossbeam, std::mpsc, and tokio::sync::mpsc.\n"
    )

    (wiki / "llms.md").write_text(
        '---\ntitle: "Large Language Models"\ntags: [ai, llm]\n'
        'updated: 2025-01-01\n---\n\n'
        "Overview of LLMs as of last year.\n"
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


# ===========================================================================
# 4.2 — Stale article rerun
# ===========================================================================


class TestStaleArticleRerun:
    @pytest.mark.asyncio
    async def test_picks_old_high_velocity_articles(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        inserted = await ad.discover_from_stale("personal", days=90)
        # llms.md has tag "ai" + updated=2025-01-01 → stale
        assert inserted == 1

        pending = await db.get_pending_candidates("personal", 10)
        assert any(p["topic"] == "Large Language Models" for p in pending)
        assert any(p["source"] == "stale" for p in pending)

    @pytest.mark.asyncio
    async def test_skips_recent_articles(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        inserted = await ad.discover_from_stale("personal", days=90)
        pending = await db.get_pending_candidates("personal", 10)
        # Recent articles (tokio, rust-channels) should NOT be queued
        assert not any(p["topic"] == "Tokio Runtime" for p in pending)
        assert not any(p["topic"] == "Rust Channels" for p in pending)

    @pytest.mark.asyncio
    async def test_skips_low_velocity_tags(self, tmp_path):
        """Old article with non-velocity tag should not be queued."""
        from app import db
        from app import auto_discovery as ad

        kb_dir = tmp_path / "personal"
        wiki = kb_dir / "wiki"
        wiki.mkdir(parents=True)
        (wiki / "old.md").write_text(
            '---\ntitle: "Latin Etymology"\ntags: [linguistics]\n'
            'updated: 2024-01-01\n---\n\nOld content.\n'
        )

        import app.config as _config, app.storage as _storage, app.wiki as _wiki
        dirs = {"personal": kb_dir}
        _config.KB_DIRS.clear()
        _config.KB_DIRS.update(dirs)
        _storage._default = _storage.LocalStorage()
        _wiki.invalidate_articles_cache()
        await db.init_db()
        inserted = await ad.discover_from_stale("personal", days=90)
        assert inserted == 0


# ===========================================================================
# 4.3 — Orphan entity mining
# ===========================================================================


class TestOrphanEntityMining:
    @pytest.mark.asyncio
    async def test_picks_entities_in_3plus_articles(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()

        # Seed kg_entities + kg_edges so that "Async Trait" appears in 3 articles
        # (none of which are slugged "async-trait")
        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            cursor = await conn.execute(
                "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)",
                ("Async Trait", "concept", 3),
            )
            ent_id = cursor.lastrowid
            for slug in ("tokio", "rust-channels", "third-article"):
                await conn.execute(
                    "INSERT INTO kg_edges (source_entity_id, target_entity_id, "
                    "relationship, article_slug, kb) VALUES (?, ?, 'uses', ?, 'personal')",
                    (ent_id, ent_id, slug),
                )

            # Also seed an entity in only 1 article — should NOT qualify
            cursor = await conn.execute(
                "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)",
                ("Niche Concept", "concept", 1),
            )
            niche_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO kg_edges (source_entity_id, target_entity_id, "
                "relationship, article_slug, kb) VALUES (?, ?, 'uses', 'tokio', 'personal')",
                (niche_id, niche_id),
            )
            await conn.commit()
        finally:
            await conn.close()

        inserted = await ad.discover_from_orphan_mentions("personal", min_articles=3)
        assert inserted == 1

        pending = await db.get_pending_candidates("personal", 10)
        topics = {p["topic"] for p in pending}
        assert "Async Trait" in topics
        assert "Niche Concept" not in topics

    @pytest.mark.asyncio
    async def test_skips_entities_with_existing_article(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        # Entity name matches an existing article — should be skipped
        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            cursor = await conn.execute(
                "INSERT INTO kg_entities (name, type, article_count) VALUES (?, ?, ?)",
                ("Tokio Runtime", "framework", 5),
            )
            ent_id = cursor.lastrowid
            for slug in ("a", "b", "c"):
                await conn.execute(
                    "INSERT INTO kg_edges (source_entity_id, target_entity_id, "
                    "relationship, article_slug, kb) VALUES (?, ?, 'uses', ?, 'personal')",
                    (ent_id, ent_id, slug),
                )
            await conn.commit()
        finally:
            await conn.close()

        inserted = await ad.discover_from_orphan_mentions("personal", min_articles=3)
        assert inserted == 0  # find_related_article matched "Tokio Runtime"


# ===========================================================================
# 4.4 — Question miner
# ===========================================================================


class TestQuestionMiner:
    @pytest.mark.asyncio
    async def test_finds_question_in_body(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        inserted = await ad.discover_from_questions("personal")
        # tokio.md contains "How does Tokio handle structured concurrency?"
        assert inserted >= 1

        pending = await db.get_pending_candidates("personal", 10)
        assert any("structured concurrency" in p["topic"].lower() for p in pending)
        assert any(p["source"] == "question" for p in pending)


# ===========================================================================
# 4.6 — TOC stub miner
# ===========================================================================


class TestStubMiner:
    @pytest.mark.asyncio
    async def test_finds_short_bullet_stubs(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        inserted = await ad.discover_from_stubs("personal")
        # tokio.md has bullets "Spawning tasks" and "Channels"
        assert inserted >= 1

        pending = await db.get_pending_candidates("personal", 20)
        assert any("Spawning tasks" in p["topic"] or "Channels" in p["topic"] for p in pending)
        assert any(p["source"] == "stub" for p in pending)


# ===========================================================================
# 4.7 — Broken wikilink miner
# ===========================================================================


class TestBrokenWikilinkMiner:
    @pytest.mark.asyncio
    async def test_finds_unresolved_wikilink(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        inserted = await ad.discover_from_broken_wikilinks("personal")
        # tokio.md contains [[Async Cancellation Patterns]] which isn't a slug
        assert inserted == 1

        pending = await db.get_pending_candidates("personal", 10)
        assert any(p["topic"] == "Async Cancellation Patterns" for p in pending)
        assert any(p["source"] == "wikilink" for p in pending)

    @pytest.mark.asyncio
    async def test_skips_resolved_wikilinks(self, tmp_path):
        from app import db
        from app import auto_discovery as ad

        kb_dir = tmp_path / "personal"
        wiki = kb_dir / "wiki"
        wiki.mkdir(parents=True)
        # tokio.md exists; the wikilink resolves
        (wiki / "tokio.md").write_text(
            '---\ntitle: "Tokio"\ntags: []\nupdated: 2026-04-01\n---\n\nbody'
        )
        (wiki / "asyncio.md").write_text(
            '---\ntitle: "Asyncio"\ntags: []\nupdated: 2026-04-01\n---\n\n'
            "See [[Tokio]] for the Rust equivalent."
        )

        import app.config as _config, app.storage as _storage, app.wiki as _wiki
        dirs = {"personal": kb_dir}
        _config.KB_DIRS.clear()
        _config.KB_DIRS.update(dirs)
        _storage._default = _storage.LocalStorage()
        _wiki.invalidate_articles_cache()
        await db.init_db()
        inserted = await ad.discover_from_broken_wikilinks("personal")
        assert inserted == 0


class TestWikilinkSlugifier:
    def test_basic(self):
        from app.auto_discovery import _wikilink_to_slug_local
        assert _wikilink_to_slug_local("Async Cancellation Patterns") == "async-cancellation-patterns"

    def test_strips_punctuation(self):
        from app.auto_discovery import _wikilink_to_slug_local
        assert _wikilink_to_slug_local("Rust: Async Trait") == "rust-async-trait"


# ===========================================================================
# 4.1 — Contradiction harvester
# ===========================================================================


class TestContradictionHarvester:
    @pytest.mark.asyncio
    async def test_finds_overlapping_pairs(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()

        # Seed kg_edges so tokio + rust-channels share an entity
        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            cursor = await conn.execute(
                "INSERT INTO kg_entities (name, type) VALUES (?, ?)",
                ("Channel", "concept"),
            )
            ent_id = cursor.lastrowid
            for slug in ("tokio", "rust-channels"):
                await conn.execute(
                    "INSERT INTO kg_edges (source_entity_id, target_entity_id, "
                    "relationship, article_slug, kb) VALUES (?, ?, 'uses', ?, 'personal')",
                    (ent_id, ent_id, slug),
                )
            await conn.commit()
        finally:
            await conn.close()

        pairs = await ad._find_article_pairs_with_overlap("personal")
        # Both tokio and rust-channels share tag 'rust' AND entity 'Channel'
        assert len(pairs) == 1
        slug_set = {pairs[0][0]["slug"], pairs[0][1]["slug"]}
        assert slug_set == {"tokio", "rust-channels"}

    @pytest.mark.asyncio
    async def test_inserts_candidate_on_contradiction(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()

        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            cursor = await conn.execute(
                "INSERT INTO kg_entities (name, type) VALUES (?, ?)",
                ("Channel", "concept"),
            )
            ent_id = cursor.lastrowid
            for slug in ("tokio", "rust-channels"):
                await conn.execute(
                    "INSERT INTO kg_edges (source_entity_id, target_entity_id, "
                    "relationship, article_slug, kb) VALUES (?, ?, 'uses', ?, 'personal')",
                    (ent_id, ent_id, slug),
                )
            await conn.commit()
        finally:
            await conn.close()

        # Mock LLM to claim a contradiction
        fake_response = json.dumps({
            "contradicts": True,
            "topic": "Resolution: Tokio vs std channels semantics",
            "reason": "They disagree on closing behavior."
        })
        with patch.object(ad, "llm_chat", AsyncMock(return_value=fake_response)):
            inserted = await ad.discover_from_contradictions("personal")

        assert inserted == 1
        pending = await db.get_pending_candidates("personal", 10)
        assert any(p["source"] == "contradiction" for p in pending)
        # source_ref encodes the pair
        contradictions = [p for p in pending if p["source"] == "contradiction"]
        assert "|" in contradictions[0]["source_ref"]

    @pytest.mark.asyncio
    async def test_skips_when_llm_says_no_contradiction(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()

        conn = await aiosqlite.connect(str(_test_db_path))
        try:
            cursor = await conn.execute(
                "INSERT INTO kg_entities (name, type) VALUES (?, ?)",
                ("Channel", "concept"),
            )
            ent_id = cursor.lastrowid
            for slug in ("tokio", "rust-channels"):
                await conn.execute(
                    "INSERT INTO kg_edges (source_entity_id, target_entity_id, "
                    "relationship, article_slug, kb) VALUES (?, ?, 'uses', ?, 'personal')",
                    (ent_id, ent_id, slug),
                )
            await conn.commit()
        finally:
            await conn.close()

        fake_response = json.dumps({"contradicts": False, "topic": "", "reason": ""})
        with patch.object(ad, "llm_chat", AsyncMock(return_value=fake_response)):
            inserted = await ad.discover_from_contradictions("personal")
        assert inserted == 0


# ===========================================================================
# 4.5 — Research history clustering (smoke test)
# ===========================================================================


class TestResearchHistoryClustering:
    @pytest.mark.asyncio
    async def test_too_few_jobs_returns_zero(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        # Only 2 jobs — under the 5-job minimum
        for topic in ("Tokio internals", "Rust ownership"):
            jid = await db.create_job(topic)
            await db.update_job(
                jid, status="complete", content="content", wiki_kb="personal",
            )
        result = await ad.cluster_research_history("personal")
        assert result == 0

    @pytest.mark.asyncio
    async def test_clusters_with_mocked_embeddings(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        # Seed 6 completed jobs in this KB
        for topic in (
            "Tokio internals",
            "Rust ownership",
            "Async cancellation",
            "Channels deep dive",
            "Lifetimes guide",
            "Lonely outlier topic",
        ):
            jid = await db.create_job(topic)
            await db.update_job(
                jid, status="complete", content="content", wiki_kb="personal",
            )

        # Mock embeddings: 5 similar vectors + 1 outlier
        fake_vectors = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # The outlier
        ]
        with patch(
            "app.embeddings.embed_texts",
            AsyncMock(return_value=fake_vectors),
        ):
            inserted = await ad.cluster_research_history("personal", limit=2)

        assert inserted >= 1
        pending = await db.get_pending_candidates("personal", 20)
        assert any(p["source"] == "blind_spot" for p in pending)


# ===========================================================================
# Dispatcher: strategy=all runs every self-mining strategy
# ===========================================================================


class TestDispatcherAll:
    @pytest.mark.asyncio
    async def test_all_strategy_calls_every_function(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        await db.upsert_auto_discovery_config("personal", enabled=True, strategy="all")

        called: dict[str, int] = {}

        async def fake_kg(*args, **kwargs):
            called["kg"] = called.get("kg", 0) + 1
            return 0

        async def fake_llm(*args, **kwargs):
            called["llm"] = called.get("llm", 0) + 1
            return 0

        async def fake_contra(*args, **kwargs):
            called["contradiction"] = 1
            return 0

        async def fake_stale(*args, **kwargs):
            called["stale"] = 1
            return 0

        async def fake_orphan(*args, **kwargs):
            called["orphan"] = 1
            return 0

        async def fake_question(*args, **kwargs):
            called["question"] = 1
            return 0

        async def fake_cluster(*args, **kwargs):
            called["blind_spot"] = 1
            return 0

        async def fake_stubs(*args, **kwargs):
            called["stub"] = 1
            return 0

        async def fake_wikilink(*args, **kwargs):
            called["wikilink"] = 1
            return 0

        with patch.object(ad, "discover_from_kg_entities", side_effect=fake_kg), \
             patch.object(ad, "discover_from_llm_followups", side_effect=fake_llm), \
             patch.object(ad, "discover_from_contradictions", side_effect=fake_contra), \
             patch.object(ad, "discover_from_stale", side_effect=fake_stale), \
             patch.object(ad, "discover_from_orphan_mentions", side_effect=fake_orphan), \
             patch.object(ad, "discover_from_questions", side_effect=fake_question), \
             patch.object(ad, "cluster_research_history", side_effect=fake_cluster), \
             patch.object(ad, "discover_from_stubs", side_effect=fake_stubs), \
             patch.object(ad, "discover_from_broken_wikilinks", side_effect=fake_wikilink):
            result = await ad.run_discovery_for_kb("personal")

        # All 9 discovery functions should have been called
        assert "kg" in called
        assert "llm" in called
        assert "contradiction" in called
        assert "stale" in called
        assert "orphan" in called
        assert "question" in called
        assert "blind_spot" in called
        assert "stub" in called
        assert "wikilink" in called
        assert result["strategy"] == "all"
        assert "counts" in result

    @pytest.mark.asyncio
    async def test_single_strategy_dispatches_only_that_one(self, tmp_kb_with_articles):
        from app import db
        from app import auto_discovery as ad

        await db.init_db()
        await db.upsert_auto_discovery_config(
            "personal", enabled=True, strategy="wikilink",
        )

        called: list[str] = []

        async def fake_wiki(*args, **kwargs):
            called.append("wikilink")
            return 0

        async def fake_kg(*args, **kwargs):
            called.append("kg")
            return 0

        async def fake_llm(*args, **kwargs):
            called.append("llm")
            return 0

        with patch.object(ad, "discover_from_broken_wikilinks", side_effect=fake_wiki), \
             patch.object(ad, "discover_from_kg_entities", side_effect=fake_kg), \
             patch.object(ad, "discover_from_llm_followups", side_effect=fake_llm):
            await ad.run_discovery_for_kb("personal")

        assert "wikilink" in called
        # Only the wikilink strategy should be invoked, not the
        # KG/LLM baseline ones.
        assert "kg" not in called
        assert "llm" not in called
