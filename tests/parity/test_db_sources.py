"""Research source parity — save/get/select across backends."""

from __future__ import annotations

import pytest


@pytest.fixture
async def job_id(parity_db):
    """Create a fresh job for each source test."""
    return await parity_db.create_job("source fixture topic")


class TestSaveAndGetSources:
    @pytest.mark.asyncio
    async def test_empty_save_is_noop(self, parity_db, job_id):
        await parity_db.save_sources(job_id, [], round_num=1)
        assert await parity_db.get_sources(job_id) == []

    @pytest.mark.asyncio
    async def test_save_and_retrieve_single(self, parity_db, job_id):
        src = {
            "url": "https://example.com/a",
            "title": "Example A",
            "content": "body a",
            "tier": 1,
        }
        await parity_db.save_sources(job_id, [src], round_num=1)
        rows = await parity_db.get_sources(job_id)
        assert len(rows) == 1
        assert rows[0]["url"] == "https://example.com/a"
        assert rows[0]["title"] == "Example A"
        assert rows[0]["tier"] == 1
        assert rows[0]["round"] == 1

    @pytest.mark.asyncio
    async def test_save_batch(self, parity_db, job_id):
        batch = [
            {"url": f"https://example.com/{i}", "title": f"T{i}",
             "content": "x", "tier": (i % 3) + 1}
            for i in range(10)
        ]
        await parity_db.save_sources(job_id, batch, round_num=2)
        rows = await parity_db.get_sources(job_id)
        assert len(rows) == 10
        assert {r["round"] for r in rows} == {2}

    @pytest.mark.asyncio
    async def test_get_sources_missing_job_is_empty(self, parity_db):
        assert await parity_db.get_sources(99999) == []


class TestSourceSelection:
    @pytest.mark.asyncio
    async def test_default_all_selected(self, parity_db, job_id):
        await parity_db.save_sources(
            job_id,
            [{"url": "u", "title": "t", "content": "c", "tier": 1}],
            round_num=1,
        )
        rows = await parity_db.get_sources(job_id)
        # `selected` defaults to 1/True on both backends
        assert all(r.get("selected") in (1, True) for r in rows)

    @pytest.mark.asyncio
    async def test_get_selected_filters_out_deselected(self, parity_db, job_id):
        batch = [
            {"url": f"u{i}", "title": f"t{i}", "content": "c", "tier": 1}
            for i in range(3)
        ]
        await parity_db.save_sources(job_id, batch, round_num=1)
        all_sources = await parity_db.get_sources(job_id)
        # Deselect the first one
        target_id = all_sources[0]["id"]
        await parity_db.update_source_selection(
            job_id, [target_id], selected=False,
        )
        selected = await parity_db.get_selected_sources(job_id)
        assert len(selected) == 2
        assert target_id not in {s["id"] for s in selected}

    @pytest.mark.asyncio
    async def test_select_all_toggle(self, parity_db, job_id):
        await parity_db.save_sources(
            job_id,
            [{"url": f"u{i}", "title": f"t{i}", "content": "c", "tier": 1}
             for i in range(5)],
            round_num=1,
        )
        await parity_db.select_all_sources(job_id, selected=False)
        assert await parity_db.get_selected_sources(job_id) == []

        await parity_db.select_all_sources(job_id, selected=True)
        assert len(await parity_db.get_selected_sources(job_id)) == 5
