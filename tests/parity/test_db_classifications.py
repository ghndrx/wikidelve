"""Palace classification parity — hall assignments per article."""

from __future__ import annotations

import pytest


class TestUpsertClassification:
    @pytest.mark.asyncio
    async def test_create_new_classification(self, parity_db):
        await parity_db.upsert_classification("kb-basics", "personal", "how-to", 0.92)
        row = await parity_db.get_article_classification("kb-basics", "personal")
        assert row is not None
        assert row["hall"] == "how-to"
        assert row["confidence"] == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, parity_db):
        await parity_db.upsert_classification("kb-basics", "personal", "how-to", 0.50)
        await parity_db.upsert_classification("kb-basics", "personal", "reference", 0.88)
        row = await parity_db.get_article_classification("kb-basics", "personal")
        assert row["hall"] == "reference"
        assert row["confidence"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, parity_db):
        assert await parity_db.get_article_classification("ghost", "personal") is None


class TestListClassifications:
    @pytest.mark.asyncio
    async def test_empty_kb_empty_list(self, parity_db):
        assert await parity_db.get_classifications("personal") == []

    @pytest.mark.asyncio
    async def test_filter_by_kb(self, parity_db):
        await parity_db.upsert_classification("a", "personal", "how-to", 0.9)
        await parity_db.upsert_classification("b", "work", "reference", 0.8)
        personal = await parity_db.get_classifications("personal")
        work = await parity_db.get_classifications("work")
        assert len(personal) == 1 and personal[0]["slug"] == "a"
        assert len(work) == 1 and work[0]["slug"] == "b"

    @pytest.mark.asyncio
    async def test_filter_by_hall(self, parity_db):
        await parity_db.upsert_classification("a", "personal", "how-to", 0.9)
        await parity_db.upsert_classification("b", "personal", "how-to", 0.8)
        await parity_db.upsert_classification("c", "personal", "reference", 0.8)
        how_to = await parity_db.get_classifications_by_hall("personal", "how-to")
        assert {r["slug"] for r in how_to} == {"a", "b"}
