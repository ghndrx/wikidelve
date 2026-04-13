"""End-to-end API tests for endpoints not covered by test_api.py.

Runs against the live instance at localhost:8888. Tests here favour
read-only probes and validation paths so the suite does not enqueue
expensive background work on every run.
"""

import pytest
from playwright.sync_api import APIRequestContext


BASE_URL = "http://localhost:8888"


@pytest.fixture(scope="session")
def api(playwright):
    ctx = playwright.request.new_context(base_url=BASE_URL, timeout=180000)
    yield ctx
    ctx.dispose()


@pytest.fixture(scope="session")
def sample_article(api: APIRequestContext):
    """Return (kb, slug) for an existing article. Uses the default KB only."""
    resp = api.get("/api/articles?kb=personal")
    if resp.status != 200:
        pytest.skip("default KB not available")
    articles = resp.json()
    if not articles:
        pytest.skip("default KB contains no articles")
    a = articles[0]
    return a["kb"], a["slug"]


# --- Article versions / diff / suggestions / related ----------------------

class TestArticleVersions:
    def test_versions_list_shape(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.get(f"/api/articles/{kb}/{slug}/versions")
        assert resp.status == 200
        data = resp.json()
        assert data["kb"] == kb
        assert data["slug"] == slug
        assert isinstance(data["versions"], list)

    def test_versions_unknown_kb(self, api: APIRequestContext, sample_article):
        _, slug = sample_article
        resp = api.get(f"/api/articles/nonexistent-kb-xyz/{slug}/versions")
        assert resp.status == 404

    def test_version_detail_unknown_id(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.get(f"/api/articles/{kb}/{slug}/versions/999999")
        assert resp.status == 404

    def test_version_diff_unknown_id(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.get(f"/api/articles/{kb}/{slug}/versions/999999/diff")
        assert resp.status == 404


class TestArticleSuggestions:
    @pytest.mark.heavy
    def test_suggestions_shape(self, api: APIRequestContext, sample_article):
        # Calls hybrid search + graph neighbours + pending candidates; slow.
        kb, slug = sample_article
        resp = api.get(f"/api/articles/{kb}/{slug}/suggestions")
        assert resp.status == 200
        data = resp.json()
        assert data["kb"] == kb
        assert data["slug"] == slug
        assert "title" in data
        assert isinstance(data["suggestions"], list)

    def test_suggestions_unknown_kb(self, api: APIRequestContext, sample_article):
        _, slug = sample_article
        resp = api.get(f"/api/articles/nonexistent-kb-xyz/{slug}/suggestions")
        assert resp.status == 404

    def test_suggestions_unknown_slug(self, api: APIRequestContext):
        resp = api.get("/api/articles/personal/nonexistent-slug-xyz-99/suggestions")
        assert resp.status == 404


class TestArticleRelated:
    @pytest.mark.heavy
    def test_related_shape(self, api: APIRequestContext, sample_article):
        # Iterates every article in the KB reading each one; O(N) I/O.
        kb, slug = sample_article
        resp = api.get(f"/api/articles/{kb}/{slug}/related")
        assert resp.status == 200
        data = resp.json()
        assert data["article"] == slug
        assert isinstance(data["related"], list)
        for item in data["related"]:
            assert "slug" in item
            assert "title" in item
            assert "score" in item
            assert "reasons" in item

    def test_related_unknown_article(self, api: APIRequestContext):
        resp = api.get("/api/articles/personal/nonexistent-slug-xyz-99/related")
        assert resp.status == 404


# --- Build endpoints -------------------------------------------------------

class TestBuildEndpoints:
    @pytest.mark.heavy
    def test_search_reindex(self, api: APIRequestContext):
        # Rebuilds the full FTS index; walks every article in every KB.
        resp = api.post("/api/search/reindex")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "indexed"
        assert isinstance(data["articles"], int)

    @pytest.mark.heavy
    def test_embeddings_build_enqueues(self, api: APIRequestContext):
        resp = api.post("/api/embeddings/build", data={"kb": "personal"})
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["kb"] == "personal"
        assert data["action"] == "build_embeddings"

    @pytest.mark.heavy
    def test_graph_build_enqueues(self, api: APIRequestContext):
        resp = api.post("/api/graph/build", data={"kb": "personal"})
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "build_graph"

    def test_graph_related_endpoint(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.get(f"/api/graph/related/{kb}/{slug}?depth=1")
        assert resp.status == 200
        data = resp.json()
        assert data["slug"] == slug
        assert data["kb"] == kb
        assert isinstance(data["related"], list)


# --- Per-article quality enqueue endpoints --------------------------------

class TestQualityEnqueue:
    @pytest.mark.heavy
    def test_enrich_enqueue(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.post(f"/api/quality/enrich/{kb}/{slug}")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["slug"] == slug
        assert data["action"] == "enrich"

    @pytest.mark.heavy
    def test_crosslink_enqueue(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.post(f"/api/quality/crosslink/{kb}/{slug}")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "crosslink"

    @pytest.mark.heavy
    def test_factcheck_enqueue(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.post(f"/api/quality/factcheck/{kb}/{slug}")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "fact_check"

    @pytest.mark.heavy
    def test_freshness_enqueue(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.post(
            f"/api/quality/freshness/{kb}/{slug}",
            data={"auto_update": False},
        )
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "freshness_audit"
        assert data["auto_update"] is False

    @pytest.mark.heavy
    def test_freshness_batch(self, api: APIRequestContext):
        resp = api.post(
            "/api/quality/freshness-batch",
            data={"kb": "personal", "max_articles": 1, "auto_update": False},
        )
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["max_articles"] == 1
        assert data["auto_update"] is False

    @pytest.mark.heavy
    def test_quality_pass(self, api: APIRequestContext):
        resp = api.post(
            "/api/quality/pass",
            data={"kb": "personal", "max_articles": 1},
        )
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["max_articles"] == 1

    @pytest.mark.heavy
    def test_auto_enrich_no_op(self, api: APIRequestContext):
        # Scores every article before filtering; heavy on large KBs.
        resp = api.post(
            "/api/quality/auto-enrich",
            data={"kb": "personal", "threshold": 0, "max_articles": 0},
        )
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["threshold"] == 0
        assert data["queued"] == 0


# --- Auto-discovery --------------------------------------------------------

class TestAutoDiscovery:
    def test_status_shape(self, api: APIRequestContext):
        resp = api.get("/api/auto-discovery/status")
        assert resp.status == 200
        data = resp.json()
        assert "global_enabled" in data
        assert "calls_per_job_estimate" in data
        assert isinstance(data["kbs"], list)
        for entry in data["kbs"]:
            assert "kb" in entry

    def test_config_unknown_kb(self, api: APIRequestContext):
        resp = api.put(
            "/api/auto-discovery/config/nonexistent-kb-xyz",
            data={"enabled": True},
        )
        assert resp.status == 404

    def test_config_rejects_unknown_fields(self, api: APIRequestContext):
        resp = api.put(
            "/api/auto-discovery/config/personal",
            data={"totally_made_up": 1},
        )
        assert resp.status == 400

    def test_config_rejects_invalid_strategy(self, api: APIRequestContext):
        resp = api.put(
            "/api/auto-discovery/config/personal",
            data={"strategy": "not-a-strategy"},
        )
        assert resp.status == 400

    def test_config_rejects_negative_budget(self, api: APIRequestContext):
        resp = api.put(
            "/api/auto-discovery/config/personal",
            data={"daily_budget": -5},
        )
        assert resp.status == 400

    def test_config_rejects_non_int_budget(self, api: APIRequestContext):
        resp = api.put(
            "/api/auto-discovery/config/personal",
            data={"daily_budget": "many"},
        )
        assert resp.status == 400

    def test_config_rejects_bad_seed_topics(self, api: APIRequestContext):
        resp = api.put(
            "/api/auto-discovery/config/personal",
            data={"seed_topics": "not-a-list"},
        )
        assert resp.status == 400

    def test_config_round_trip(self, api: APIRequestContext):
        # Read current config so we can restore it.
        status = api.get("/api/auto-discovery/status").json()
        original = next(
            (k for k in status["kbs"] if k["kb"] == "personal"),
            None,
        )
        assert original is not None

        resp = api.put(
            "/api/auto-discovery/config/personal",
            data={
                "enabled": original.get("enabled", True),
                "daily_budget": int(original.get("daily_budget") or 200),
                "max_per_hour": int(original.get("max_per_hour") or 5),
                "strategy": original.get("strategy") or "all",
                "llm_sample": int(original.get("llm_sample") or 10),
            },
        )
        assert resp.status == 200
        data = resp.json()
        assert data["kb"] == "personal"
        assert "config" in data

    def test_run_unknown_kb(self, api: APIRequestContext):
        resp = api.post("/api/auto-discovery/run/nonexistent-kb-xyz")
        assert resp.status == 404

    @pytest.mark.heavy
    def test_run_enqueues(self, api: APIRequestContext):
        resp = api.post("/api/auto-discovery/run/personal")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["kb"] == "personal"


# --- Per-KB settings -------------------------------------------------------

class TestKbSettings:
    def test_get_unknown_kb(self, api: APIRequestContext):
        resp = api.get("/api/kb-settings/nonexistent-kb-xyz")
        assert resp.status == 404

    def test_get_shape(self, api: APIRequestContext):
        resp = api.get("/api/kb-settings/personal")
        assert resp.status == 200
        data = resp.json()
        assert data["kb"] == "personal"
        assert "config" in data

    def test_put_unknown_kb(self, api: APIRequestContext):
        resp = api.put(
            "/api/kb-settings/nonexistent-kb-xyz",
            data={"persona": "neutral"},
        )
        assert resp.status == 404

    def test_put_rejects_unknown_fields(self, api: APIRequestContext):
        resp = api.put(
            "/api/kb-settings/personal",
            data={"not_a_real_field": "x"},
        )
        assert resp.status == 400

    def test_put_rejects_invalid_provider(self, api: APIRequestContext):
        resp = api.put(
            "/api/kb-settings/personal",
            data={"synthesis_provider": "not-a-provider"},
        )
        assert resp.status == 400

    def test_put_accepts_empty_body(self, api: APIRequestContext):
        resp = api.put("/api/kb-settings/personal", data={})
        assert resp.status == 200
        assert resp.json()["kb"] == "personal"

    def test_put_rejects_non_object(self, api: APIRequestContext):
        resp = api.put("/api/kb-settings/personal", data="not-json")
        assert resp.status == 400


# --- Palace ----------------------------------------------------------------

class TestPalaceEndpoints:
    def test_article_context(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.get(f"/api/palace/article/{kb}/{slug}")
        assert resp.status == 200
        data = resp.json()
        assert data["slug"] == slug
        assert data["kb"] == kb
        assert "hall" in data
        assert "rooms" in data
        assert isinstance(data["rooms"], list)

    @pytest.mark.heavy
    def test_classify_enqueue(self, api: APIRequestContext):
        resp = api.post("/api/palace/classify", data={"kb": "personal"})
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["kb"] == "personal"

    @pytest.mark.heavy
    def test_cluster_enqueue(self, api: APIRequestContext):
        resp = api.post("/api/palace/cluster", data={"kb": "personal"})
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["kb"] == "personal"

    @pytest.mark.heavy
    def test_hall_shape(self, api: APIRequestContext):
        # Pulls every article in the KB to join against classifications.
        resp = api.get("/api/palace/hall/personal/reference")
        assert resp.status == 200
        data = resp.json()
        assert data["hall"] == "reference"
        assert data["kb"] == "personal"
        assert isinstance(data["articles"], list)
        assert isinstance(data["count"], int)


# --- GitHub ----------------------------------------------------------------

class TestGitHubEndpoints:
    @pytest.mark.heavy
    def test_check_releases_enqueue(self, api: APIRequestContext):
        resp = api.post("/api/github/check-releases")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "check_releases"

    @pytest.mark.heavy
    def test_index_repos_enqueue(self, api: APIRequestContext):
        resp = api.post("/api/github/index-repos")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["action"] == "index_repos"


# --- Chat session messages / ask ------------------------------------------

class TestChatSessionMessages:
    def test_add_message_requires_content(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/e2e-gap-test/messages",
            data={"role": "user"},
        )
        assert resp.status == 400

    def test_add_message_invalid_json(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/e2e-gap-test/messages",
            data="not-json",
        )
        assert resp.status == 400

    def test_add_then_list_messages(self, api: APIRequestContext):
        session_id = "e2e-gap-test-messages"
        try:
            resp = api.post(
                f"/api/chat/sessions/{session_id}/messages",
                data={"role": "user", "content": "hello from e2e"},
            )
            assert resp.status == 200
            listed = api.get(f"/api/chat/sessions/{session_id}")
            assert listed.status == 200
            messages = listed.json()
            assert isinstance(messages, list)
            assert any(m.get("content") == "hello from e2e" for m in messages)
        finally:
            api.delete(f"/api/chat/sessions/{session_id}")

    def test_ask_requires_question(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/e2e-gap-test-ask/ask",
            data={},
        )
        assert resp.status == 400

    def test_ask_rejects_short_question(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/e2e-gap-test-ask/ask",
            data={"question": "hi"},
        )
        assert resp.status == 400

    def test_ask_rejects_invalid_json(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/e2e-gap-test-ask/ask",
            data="not-json",
        )
        assert resp.status == 400

    def test_ask_rejects_blank_kb(self, api: APIRequestContext):
        resp = api.post(
            "/api/chat/sessions/e2e-gap-test-ask/ask",
            data={"question": "what is this project", "kb": "   "},
        )
        assert resp.status == 400


# --- Title refinement ------------------------------------------------------

class TestRefineTitles:
    @pytest.mark.heavy
    def test_dry_run(self, api: APIRequestContext):
        # Iterates every article in the KB; slow on large stores.
        resp = api.post(
            "/api/articles/refine-titles",
            data={"kb": "personal", "dry_run": True},
        )
        assert resp.status == 200
        data = resp.json()
        assert data["kb"] == "personal"
        assert data["dry_run"] is True
        assert isinstance(data["total"], int)
        assert isinstance(data["updated"], int)
        assert isinstance(data["changes"], list)

    def test_unknown_kb(self, api: APIRequestContext):
        resp = api.post(
            "/api/articles/refine-titles",
            data={"kb": "nonexistent-kb-xyz", "dry_run": True},
        )
        assert resp.status == 404


# --- Research: retry errors + delete file ---------------------------------

class TestResearchMisc:
    def test_retry_errors_shape(self, api: APIRequestContext):
        resp = api.post("/api/research/retry-errors", data={"limit": 0})
        assert resp.status == 200
        data = resp.json()
        assert "retried" in data
        assert "total_errors" in data
        assert data["retried"] == 0

    def test_delete_research_file_rejects_traversal(self, api: APIRequestContext):
        resp = api.delete("/api/research/..%2Fetc%2Fpasswd")
        assert resp.status in (400, 404)

    def test_delete_research_file_unknown(self, api: APIRequestContext):
        resp = api.delete("/api/research/nonexistent-file-xyz.md")
        assert resp.status == 404


# --- Research jobs listing (compact + filter + limit) --------------------

class TestResearchJobsListing:
    def test_default_returns_list(self, api: APIRequestContext):
        resp = api.get("/api/research/jobs")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            for key in ("id", "topic", "status", "created"):
                assert key in data[0]

    def test_compact_strips_content(self, api: APIRequestContext):
        resp = api.get("/api/research/jobs?compact=true&limit=5")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)
        for row in data:
            assert "content" not in row
            # The compact payload should still carry the fields the admin
            # dashboard renders.
            for key in ("id", "topic", "status", "created"):
                assert key in row

    def test_limit_is_capped(self, api: APIRequestContext):
        resp = api.get("/api/research/jobs?limit=5")
        assert resp.status == 200
        data = resp.json()
        assert len(data) <= 5

    def test_limit_upper_bound_cap(self, api: APIRequestContext):
        resp = api.get("/api/research/jobs?limit=999999&compact=true")
        assert resp.status == 200
        data = resp.json()
        # Route caps the incoming limit at 500.
        assert len(data) <= 500

    def test_status_filter(self, api: APIRequestContext):
        resp = api.get("/api/research/jobs?compact=true&limit=200&status=complete")
        assert resp.status == 200
        data = resp.json()
        for row in data:
            assert row["status"] == "complete"

    def test_status_filter_no_matches(self, api: APIRequestContext):
        # A made-up status should yield an empty list, not a 400.
        resp = api.get("/api/research/jobs?status=definitely-not-a-status")
        assert resp.status == 200
        assert resp.json() == []


# --- Storage introspection -------------------------------------------------

class TestStorageStatus:
    def test_status_shape(self, api: APIRequestContext):
        resp = api.get("/api/storage/status")
        assert resp.status == 200
        data = resp.json()
        for key in ("storage_backend", "db_backend", "vector_backend", "kbs"):
            assert key in data
        assert isinstance(data["kbs"], list)


# --- Quality score per-article --------------------------------------------

class TestQualityScorePerArticle:
    @pytest.mark.heavy
    def test_score_single_shape(self, api: APIRequestContext, sample_article):
        kb, slug = sample_article
        resp = api.get(f"/api/quality/score/{kb}/{slug}")
        assert resp.status == 200
        data = resp.json()
        assert data["slug"] == slug
        assert isinstance(data["score"], int)
        assert isinstance(data["breakdown"], dict)
        assert isinstance(data["word_count"], int)
