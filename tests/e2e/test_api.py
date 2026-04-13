"""End-to-end API tests for WikiDelve.

Runs against the live instance at localhost:8888.
Tests the full request/response cycle including database state.
"""

import pytest
from playwright.sync_api import APIRequestContext


BASE_URL = "http://localhost:8888"


@pytest.fixture(scope="session")
def api(playwright):
    """Create a shared API request context."""
    ctx = playwright.request.new_context(base_url=BASE_URL, timeout=60000)
    yield ctx
    ctx.dispose()


# --- Health & Status --------------------------------------------------------

class TestHealthEndpoints:
    def test_health_check(self, api: APIRequestContext):
        resp = api.get("/health")
        assert resp.status == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "kbs" in data

    def test_api_status(self, api: APIRequestContext):
        resp = api.get("/api/status")
        assert resp.status == 200
        data = resp.json()
        assert "grokmaxxing" in data
        assert "jobs" in data
        assert "wiki" in data
        assert "total_articles" in data["wiki"]
        assert "total_words" in data["wiki"]
        assert "kbs" in data["wiki"]

    def test_api_stats(self, api: APIRequestContext):
        resp = api.get("/api/stats")
        assert resp.status == 200
        data = resp.json()
        assert "kbs" in data
        assert "total_articles" in data
        assert "total_words" in data
        assert "research_jobs" in data


# --- Articles ---------------------------------------------------------------

class TestArticleEndpoints:
    def test_list_articles(self, api: APIRequestContext):
        resp = api.get("/api/articles")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_articles_filter_kb(self, api: APIRequestContext):
        resp = api.get("/api/articles?kb=personal")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)
        for article in data:
            assert article["kb"] == "personal"

    def test_list_articles_filter_source(self, api: APIRequestContext):
        resp = api.get("/api/articles?source=web")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)
        for article in data:
            assert article.get("source_type") == "web"

    def test_get_article_not_found(self, api: APIRequestContext):
        resp = api.get("/api/articles/personal/nonexistent-slug-xyz-12345")
        assert resp.status == 404

    def test_get_article_if_exists(self, api: APIRequestContext):
        # Get list first, then fetch first article
        articles = api.get("/api/articles?kb=personal").json()
        if articles:
            first = articles[0]
            resp = api.get(f"/api/articles/{first['kb']}/{first['slug']}")
            assert resp.status == 200
            data = resp.json()
            assert data["slug"] == first["slug"]
            assert "html" in data
            assert "raw_markdown" in data
            assert "word_count" in data

    def test_related_articles(self, api: APIRequestContext):
        articles = api.get("/api/articles?kb=personal").json()
        if articles:
            first = articles[0]
            resp = api.get(f"/api/articles/{first['kb']}/{first['slug']}/related")
            assert resp.status == 200
            data = resp.json()
            assert "article" in data
            assert "related" in data
            assert isinstance(data["related"], list)

    def test_delete_article_not_found(self, api: APIRequestContext):
        resp = api.delete("/api/articles/personal/nonexistent-slug-xyz-12345")
        assert resp.status == 404


# --- Search -----------------------------------------------------------------

class TestSearchEndpoints:
    def test_fts_search(self, api: APIRequestContext):
        resp = api.get("/api/search?q=test")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_fts_search_empty_query(self, api: APIRequestContext):
        resp = api.get("/api/search?q=")
        assert resp.status == 200

    def test_hybrid_search(self, api: APIRequestContext):
        resp = api.get("/api/search/hybrid?q=kubernetes&limit=5")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_hybrid_search_empty(self, api: APIRequestContext):
        resp = api.get("/api/search/hybrid?q=")
        assert resp.status == 200
        data = resp.json()
        assert data == []

    def test_hybrid_search_with_kb_filter(self, api: APIRequestContext):
        resp = api.get("/api/search/hybrid?q=test&kb=personal&limit=3")
        assert resp.status == 200


# --- Research ---------------------------------------------------------------

class TestResearchEndpoints:
    def test_list_jobs(self, api: APIRequestContext):
        resp = api.get("/api/research/jobs")
        assert resp.status == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_research_history(self, api: APIRequestContext):
        resp = api.get("/api/research/history")
        assert resp.status == 200
        data = resp.json()
        assert "jobs" in data
        assert "article_updates" in data

    def test_research_invalid_json(self, api: APIRequestContext):
        resp = api.post("/api/research", data="not json")
        assert resp.status == 400

    def test_research_topic_too_short(self, api: APIRequestContext):
        resp = api.post("/api/research", data={"topic": "hi"})
        assert resp.status == 400

    def test_research_missing_topic(self, api: APIRequestContext):
        resp = api.post("/api/research", data={})
        assert resp.status == 400

    def test_job_not_found(self, api: APIRequestContext):
        resp = api.get("/api/research/status/99999")
        assert resp.status == 404

    def test_sources_not_found(self, api: APIRequestContext):
        resp = api.get("/api/research/sources/99999")
        assert resp.status == 404

    def test_cancel_not_found(self, api: APIRequestContext):
        resp = api.post("/api/research/cancel/99999")
        assert resp.status == 404

    def test_delete_job_not_found(self, api: APIRequestContext):
        resp = api.delete("/api/research/job/99999")
        assert resp.status == 404

    def test_delete_research_file_invalid(self, api: APIRequestContext):
        resp = api.delete("/api/research/../etc/passwd")
        assert resp.status in (400, 404, 405)

    def test_local_research_missing_fields(self, api: APIRequestContext):
        resp = api.post("/api/research/local", data={"topic": "test"})
        assert resp.status == 400


# --- Quality ----------------------------------------------------------------

class TestQualityEndpoints:
    def test_quality_scores(self, api: APIRequestContext):
        resp = api.get("/api/quality/scores/personal")
        assert resp.status == 200
        data = resp.json()
        assert "average_score" in data
        assert "total_articles" in data
        assert "distribution" in data

    def test_shallow_articles(self, api: APIRequestContext):
        resp = api.get("/api/quality/shallow/personal")
        assert resp.status == 200
        assert isinstance(resp.json(), list)

    def test_duplicates(self, api: APIRequestContext):
        resp = api.get("/api/quality/duplicates/personal")
        assert resp.status == 200
        assert isinstance(resp.json(), list)

    def test_single_article_score(self, api: APIRequestContext):
        articles = api.get("/api/articles?kb=personal").json()
        if articles:
            slug = articles[0]["slug"]
            resp = api.get(f"/api/quality/score/personal/{slug}")
            assert resp.status == 200


# --- Knowledge Graph --------------------------------------------------------

class TestGraphEndpoints:
    def test_graph_data(self, api: APIRequestContext):
        resp = api.get("/api/graph/data")
        assert resp.status == 200
        data = resp.json()
        assert "nodes" in data or isinstance(data, dict)

    def test_entity_articles(self, api: APIRequestContext):
        resp = api.get("/api/graph/entity/kubernetes")
        assert resp.status == 200
        data = resp.json()
        assert "entity" in data
        assert "articles" in data


# --- Palace -----------------------------------------------------------------

class TestPalaceEndpoints:
    def test_palace_map(self, api: APIRequestContext):
        resp = api.get("/api/palace/map")
        assert resp.status == 200

    def test_palace_map_with_kb(self, api: APIRequestContext):
        resp = api.get("/api/palace/map?kb=personal")
        assert resp.status == 200

    def test_palace_room_not_found(self, api: APIRequestContext):
        resp = api.get("/api/palace/room/personal/nonexistent-room-xyz")
        assert resp.status == 404


# --- GitHub -----------------------------------------------------------------

class TestGitHubEndpoints:
    def test_tracked_repos(self, api: APIRequestContext):
        resp = api.get("/api/github/tracked")
        assert resp.status == 200
        data = resp.json()
        assert "tracked" in data
        assert "own" in data


# --- Storage ----------------------------------------------------------------

class TestStorageEndpoints:
    def test_storage_status(self, api: APIRequestContext):
        resp = api.get("/api/storage/status")
        assert resp.status == 200
        data = resp.json()
        assert "storage_backend" in data
        assert "kbs" in data
        assert isinstance(data["kbs"], list)

    def test_legacy_s3_route_removed(self, api: APIRequestContext):
        """The old /api/s3/* routes should be gone — use /api/storage/status."""
        assert api.get("/api/s3/status").status == 404
        assert api.post("/api/s3/push").status == 404
        assert api.post("/api/s3/pull").status == 404


# --- Ingest -----------------------------------------------------------------

class TestIngestEndpoints:
    def test_youtube_missing_url(self, api: APIRequestContext):
        resp = api.post("/api/media/youtube", data={})
        assert resp.status == 400

    def test_document_missing_url(self, api: APIRequestContext):
        resp = api.post("/api/ingest/document", data={})
        assert resp.status == 400

    def test_directory_missing_url(self, api: APIRequestContext):
        resp = api.post("/api/ingest/directory", data={})
        assert resp.status == 400


# --- Error handling ---------------------------------------------------------

class TestErrorHandling:
    def test_404_unknown_route(self, api: APIRequestContext):
        resp = api.get("/api/nonexistent-endpoint")
        assert resp.status == 404

    def test_article_unknown_kb(self, api: APIRequestContext):
        resp = api.get("/api/articles/nonexistent-kb/some-slug")
        assert resp.status == 404
