"""Unit tests for source selection DB functions (source review feature)."""

import pytest
from unittest.mock import patch

from app import db


@pytest.fixture(autouse=True)
def _set_db_path(tmp_path):
    test_db = tmp_path / "test.db"
    with patch("app.config.DB_PATH", test_db), \
         patch("app.db.DB_PATH", test_db):
        yield


@pytest.fixture
async def init_db():
    await db.init_db()
    yield


@pytest.fixture
async def job_with_sources(init_db):
    """Create a job with 5 sources (all selected by default)."""
    job_id = await db.create_job("Source review test topic")
    sources = [
        {"title": f"Source {i}", "url": f"http://example.com/{i}", "content": f"Content {i}"}
        for i in range(5)
    ]
    await db.save_sources(job_id, sources, round_num=1)
    return job_id


class TestUpdateSourceSelection:
    @pytest.mark.asyncio
    async def test_deselect_sources(self, job_with_sources):
        job_id = job_with_sources
        sources = await db.get_sources(job_id)
        ids_to_deselect = [sources[0]["id"], sources[1]["id"]]

        count = await db.update_source_selection(job_id, ids_to_deselect, selected=False)
        assert count == 2

        # Check that only 3 remain selected
        selected = await db.get_selected_sources(job_id)
        assert len(selected) == 3

    @pytest.mark.asyncio
    async def test_reselect_sources(self, job_with_sources):
        job_id = job_with_sources
        sources = await db.get_sources(job_id)

        # Deselect all, then reselect two
        await db.select_all_sources(job_id, selected=False)
        ids_to_select = [sources[2]["id"], sources[4]["id"]]
        count = await db.update_source_selection(job_id, ids_to_select, selected=True)
        assert count == 2

        selected = await db.get_selected_sources(job_id)
        assert len(selected) == 2

    @pytest.mark.asyncio
    async def test_empty_source_ids_returns_zero(self, job_with_sources):
        count = await db.update_source_selection(job_with_sources, [], selected=False)
        assert count == 0

    @pytest.mark.asyncio
    async def test_wrong_job_id_updates_nothing(self, job_with_sources):
        job_id = job_with_sources
        sources = await db.get_sources(job_id)
        count = await db.update_source_selection(99999, [sources[0]["id"]], selected=False)
        assert count == 0


class TestGetSelectedSources:
    @pytest.mark.asyncio
    async def test_all_selected_by_default(self, job_with_sources):
        selected = await db.get_selected_sources(job_with_sources)
        assert len(selected) == 5

    @pytest.mark.asyncio
    async def test_returns_only_selected(self, job_with_sources):
        job_id = job_with_sources
        sources = await db.get_sources(job_id)
        await db.update_source_selection(job_id, [sources[0]["id"]], selected=False)

        selected = await db.get_selected_sources(job_id)
        assert len(selected) == 4
        selected_ids = {s["id"] for s in selected}
        assert sources[0]["id"] not in selected_ids

    @pytest.mark.asyncio
    async def test_empty_when_all_deselected(self, job_with_sources):
        await db.select_all_sources(job_with_sources, selected=False)
        selected = await db.get_selected_sources(job_with_sources)
        assert len(selected) == 0


class TestSelectAllSources:
    @pytest.mark.asyncio
    async def test_deselect_all(self, job_with_sources):
        count = await db.select_all_sources(job_with_sources, selected=False)
        assert count == 5

        selected = await db.get_selected_sources(job_with_sources)
        assert len(selected) == 0

    @pytest.mark.asyncio
    async def test_select_all_after_deselect(self, job_with_sources):
        await db.select_all_sources(job_with_sources, selected=False)
        count = await db.select_all_sources(job_with_sources, selected=True)
        assert count == 5

        selected = await db.get_selected_sources(job_with_sources)
        assert len(selected) == 5


class TestSelectedColumnMigration:
    @pytest.mark.asyncio
    async def test_sources_have_selected_field(self, job_with_sources):
        sources = await db.get_sources(job_with_sources)
        for s in sources:
            assert "selected" in s
            assert s["selected"] == 1
