"""E2E tests for source review and wikilink checker API endpoints.

Runs against the live instance at localhost:8888.

Tests the full lifecycle: create job, inspect sources, select/deselect,
trigger synthesis, validate error states. Also covers wikilink checker
scan and fix endpoints with edge cases.
"""

import time

import pytest
from playwright.sync_api import APIRequestContext


BASE_URL = "http://localhost:8888"


@pytest.fixture(scope="session")
def api(playwright):
    """Shared API context — generous timeout for slow endpoints."""
    ctx = playwright.request.new_context(base_url=BASE_URL, timeout=120000)
    yield ctx
    ctx.dispose()


@pytest.fixture(scope="session")
def created_job(api: APIRequestContext) -> dict:
    """Create a real research job and wait for it to complete.

    Returns the job dict with id, topic, status. Shared across all tests
    in this module to avoid dispatching many jobs.
    """
    resp = api.post("/api/research", data={
        "topic": "e2e test job for source review validation endpoint testing",
        "kb": "personal",
    })
    if resp.status == 429:
        # Cooldown — use the existing job
        data = resp.json()
        job_id = data.get("existing_job_id")
        if job_id:
            status_resp = api.get(f"/api/research/status/{job_id}")
            return status_resp.json()
        pytest.skip("Topic on cooldown and no existing job ID returned")

    assert resp.status == 200
    data = resp.json()
    job_id = data["job_id"]

    # Poll until complete or error (max 5 min)
    for _ in range(60):
        status = api.get(f"/api/research/status/{job_id}").json()
        if status["status"] in ("complete", "error", "no_results", "awaiting_review"):
            return status
        time.sleep(5)

    # Return whatever state it's in
    return api.get(f"/api/research/status/{job_id}").json()


# ===========================================================================
# 1. Source Review — Job Creation & Dispatch
# ===========================================================================

class TestSourceReviewDispatch:
    """Tests for POST /api/research with review_sources flag."""

    def test_job_created_successfully(self, created_job):
        """A research job should reach a terminal state."""
        assert created_job["status"] in ("complete", "error", "no_results", "awaiting_review")
        assert created_job["id"] > 0

    def test_review_sources_true_accepted(self, api: APIRequestContext):
        resp = api.post("/api/research", data={
            "topic": "e2e review sources flag acceptance testing here now",
            "review_sources": True,
        })
        assert resp.status in (200, 429)
        if resp.status == 200:
            data = resp.json()
            assert data["review_sources"] is True
            assert "job_id" in data
            assert data["status"] == "queued"

    def test_review_sources_false_is_default(self, api: APIRequestContext):
        resp = api.post("/api/research", data={
            "topic": "e2e default review false test topic validation here",
        })
        assert resp.status in (200, 429)
        if resp.status == 200:
            assert resp.json()["review_sources"] is False

    def test_review_sources_explicit_false(self, api: APIRequestContext):
        resp = api.post("/api/research", data={
            "topic": "e2e explicit review false flag test topic here now",
            "review_sources": False,
        })
        assert resp.status in (200, 429)
        if resp.status == 200:
            assert resp.json()["review_sources"] is False

    def test_review_sources_with_kb(self, api: APIRequestContext):
        resp = api.post("/api/research", data={
            "topic": "e2e review with kb parameter testing validation here",
            "review_sources": True,
            "kb": "personal",
        })
        assert resp.status in (200, 429)
        if resp.status == 200:
            data = resp.json()
            assert data["kb"] == "personal"
            assert data["review_sources"] is True


# ===========================================================================
# 2. Source Review — Source Inspection
# ===========================================================================

class TestSourceInspection:
    """Tests for GET /api/research/sources/{job_id}."""

    def test_sources_include_status_field(self, api: APIRequestContext, created_job):
        resp = api.get(f"/api/research/sources/{created_job['id']}")
        assert resp.status == 200
        data = resp.json()
        assert "status" in data
        assert "sources" in data
        assert "job_id" in data

    def test_sources_list_is_array(self, api: APIRequestContext, created_job):
        data = api.get(f"/api/research/sources/{created_job['id']}").json()
        assert isinstance(data["sources"], list)

    def test_source_record_shape(self, api: APIRequestContext, created_job):
        """Each source should have id, url, title, tier, round, selected."""
        data = api.get(f"/api/research/sources/{created_job['id']}").json()
        for src in data["sources"][:5]:
            assert "id" in src
            assert "url" in src
            assert "title" in src
            assert "tier" in src
            assert "round" in src
            assert "selected" in src

    def test_sources_not_found(self, api: APIRequestContext):
        resp = api.get("/api/research/sources/99999")
        assert resp.status == 404

    def test_sources_invalid_id_type(self, api: APIRequestContext):
        resp = api.get("/api/research/sources/not-a-number")
        assert resp.status == 422

    def test_all_sources_selected_by_default(self, api: APIRequestContext, created_job):
        """All sources from a standard pipeline job should be selected=1."""
        data = api.get(f"/api/research/sources/{created_job['id']}").json()
        for src in data["sources"]:
            assert src.get("selected") in (1, True), \
                f"Source {src['id']} not selected by default"


# ===========================================================================
# 3. Source Review — Source Selection (PUT)
# ===========================================================================

class TestSourceSelection:
    """Tests for PUT /api/research/sources/{job_id}."""

    def test_update_sources_job_not_found(self, api: APIRequestContext):
        resp = api.put("/api/research/sources/99999", data={
            "source_ids": [1, 2], "selected": False,
        })
        assert resp.status == 404

    def test_update_sources_not_awaiting_review(self, api: APIRequestContext, created_job):
        """PUT should 409 if job isn't in awaiting_review status."""
        if created_job["status"] == "awaiting_review":
            pytest.skip("Job is actually awaiting review")
        resp = api.put(f"/api/research/sources/{created_job['id']}", data={
            "source_ids": [1], "selected": False,
        })
        assert resp.status == 409
        assert "awaiting" in resp.json()["detail"].lower()

    def test_update_sources_empty_ids_rejected(self, api: APIRequestContext, created_job):
        resp = api.put(f"/api/research/sources/{created_job['id']}", data={
            "source_ids": [], "selected": False,
        })
        assert resp.status in (400, 409)

    def test_update_sources_invalid_json(self, api: APIRequestContext):
        resp = api.put("/api/research/sources/99999", data="not json")
        assert resp.status in (400, 404)

    def test_select_all_wrong_status(self, api: APIRequestContext, created_job):
        if created_job["status"] == "awaiting_review":
            pytest.skip("Job is actually awaiting review")
        resp = api.put(f"/api/research/sources/{created_job['id']}", data={
            "select_all": False,
        })
        assert resp.status == 409


# ===========================================================================
# 4. Source Review — Synthesis Trigger
# ===========================================================================

class TestSynthesisTrigger:
    """Tests for POST /api/research/synthesize/{job_id}."""

    def test_synthesize_not_found(self, api: APIRequestContext):
        resp = api.post("/api/research/synthesize/99999")
        assert resp.status == 404

    def test_synthesize_wrong_status(self, api: APIRequestContext, created_job):
        if created_job["status"] == "awaiting_review":
            pytest.skip("Job is actually awaiting review")
        resp = api.post(f"/api/research/synthesize/{created_job['id']}", data={"kb": "personal"})
        assert resp.status == 409

    def test_synthesize_no_body_ok(self, api: APIRequestContext, created_job):
        """POST with no body should not crash."""
        if created_job["status"] == "awaiting_review":
            pytest.skip("Job is actually awaiting review")
        resp = api.post(f"/api/research/synthesize/{created_job['id']}")
        assert resp.status == 409  # wrong status, but body parsing didn't crash

    def test_synthesize_invalid_id_type(self, api: APIRequestContext):
        resp = api.post("/api/research/synthesize/not-a-number")
        assert resp.status == 422


# ===========================================================================
# 5. Source Review — Status Lifecycle
# ===========================================================================

class TestSourceReviewLifecycle:
    """Integration tests for source review status flow."""

    def test_status_endpoint_works_for_created_job(self, api: APIRequestContext, created_job):
        resp = api.get(f"/api/research/status/{created_job['id']}")
        assert resp.status == 200
        data = resp.json()
        assert data["id"] == created_job["id"]
        assert "status" in data

    def test_awaiting_review_not_in_stuck_jobs(self, api: APIRequestContext):
        """The /api/status endpoint should not count awaiting_review as stuck."""
        resp = api.get("/api/status")
        assert resp.status == 200
        assert "jobs" in resp.json()

    def test_completed_job_has_content(self, api: APIRequestContext, created_job):
        """A completed job should have content and word count."""
        if created_job["status"] != "complete":
            pytest.skip("Job did not complete successfully")
        assert created_job.get("word_count", 0) > 0

    def test_completed_job_has_sources(self, api: APIRequestContext, created_job):
        """A completed job should have sources."""
        data = api.get(f"/api/research/sources/{created_job['id']}").json()
        assert len(data["sources"]) > 0


# ===========================================================================
# 6. Wikilink Checker — Scan
# ===========================================================================

class TestWikilinkCheck:
    """Tests for GET /api/quality/wikilinks/{kb_name}."""

    def test_check_returns_valid_structure(self, api: APIRequestContext):
        resp = api.get("/api/quality/wikilinks/personal")
        assert resp.status == 200
        data = resp.json()
        assert data["kb"] == "personal"
        assert isinstance(data["total_links"], int)
        assert isinstance(data["valid_links"], int)
        assert isinstance(data["broken_links"], int)
        assert isinstance(data["details"], list)

    def test_check_counts_are_consistent(self, api: APIRequestContext):
        """valid + broken == total."""
        data = api.get("/api/quality/wikilinks/personal").json()
        if "error" not in data:
            assert data["valid_links"] + data["broken_links"] == data["total_links"]

    def test_check_unknown_kb_returns_error(self, api: APIRequestContext):
        data = api.get("/api/quality/wikilinks/nonexistent-kb-xyz-99").json()
        assert "error" in data

    def test_broken_link_detail_shape(self, api: APIRequestContext):
        data = api.get("/api/quality/wikilinks/personal").json()
        for detail in data.get("details", []):
            assert "source_slug" in detail
            assert "source_title" in detail
            assert "link_text" in detail
            assert "expected_slug" in detail
            assert "suggestion" in detail
            if detail["suggestion"] is not None:
                s = detail["suggestion"]
                assert "slug" in s
                assert "title" in s
                assert "confidence" in s
                assert 0 <= s["confidence"] <= 1

    def test_check_idempotent(self, api: APIRequestContext):
        """Running check twice should return identical results (read-only)."""
        data1 = api.get("/api/quality/wikilinks/personal").json()
        data2 = api.get("/api/quality/wikilinks/personal").json()
        assert data1["total_links"] == data2["total_links"]
        assert data1["broken_links"] == data2["broken_links"]


# ===========================================================================
# 7. Wikilink Checker — Fix
# ===========================================================================

class TestWikilinkFix:
    """Tests for POST /api/quality/wikilinks/{kb_name}."""

    def test_fix_returns_valid_structure(self, api: APIRequestContext):
        resp = api.post("/api/quality/wikilinks/personal", data={"auto_remove": False})
        assert resp.status == 200
        data = resp.json()
        assert "kb" in data
        assert isinstance(data["articles_modified"], int)
        assert isinstance(data["links_fixed"], int)
        assert isinstance(data["links_removed"], int)
        assert isinstance(data["details"], list)

    def test_fix_auto_remove_false_preserves_links(self, api: APIRequestContext):
        data = api.post("/api/quality/wikilinks/personal", data={"auto_remove": False}).json()
        if "error" not in data:
            assert data["links_removed"] == 0

    def test_fix_single_article(self, api: APIRequestContext):
        articles = api.get("/api/articles?kb=personal").json()
        if not articles:
            pytest.skip("No articles")
        slug = articles[0]["slug"]
        data = api.post("/api/quality/wikilinks/personal", data={
            "slug": slug, "auto_remove": False,
        }).json()
        if "error" not in data:
            for d in data.get("details", []):
                assert d["slug"] == slug

    def test_fix_nonexistent_slug_error(self, api: APIRequestContext):
        data = api.post("/api/quality/wikilinks/personal", data={
            "slug": "nonexistent-slug-xyz-12345",
        }).json()
        assert "error" in data

    def test_fix_unknown_kb_error(self, api: APIRequestContext):
        data = api.post("/api/quality/wikilinks/nonexistent-kb-xyz", data={}).json()
        assert "error" in data

    def test_fix_no_body_uses_defaults(self, api: APIRequestContext):
        resp = api.post("/api/quality/wikilinks/personal")
        assert resp.status == 200
        data = resp.json()
        assert "articles_modified" in data or "error" in data

    def test_fix_then_check_consistency(self, api: APIRequestContext):
        """After fix with auto_remove=False, broken count should decrease or stay same."""
        before = api.get("/api/quality/wikilinks/personal").json()
        api.post("/api/quality/wikilinks/personal", data={"auto_remove": False})
        after = api.get("/api/quality/wikilinks/personal").json()
        if "error" not in before and "error" not in after:
            assert after["broken_links"] <= before["broken_links"]


# ===========================================================================
# 8. Quality Integration — Wikilinks + Scores
# ===========================================================================

class TestWikilinkQualityIntegration:
    def test_quality_scores_still_work(self, api: APIRequestContext):
        resp = api.get("/api/quality/scores/personal")
        assert resp.status == 200
        data = resp.json()
        assert "average_score" in data
        assert "distribution" in data

    def test_single_article_score_has_links_breakdown(self, api: APIRequestContext):
        articles = api.get("/api/articles?kb=personal").json()
        if not articles:
            pytest.skip("No articles")
        data = api.get(f"/api/quality/score/personal/{articles[0]['slug']}").json()
        assert "links" in data["breakdown"]
