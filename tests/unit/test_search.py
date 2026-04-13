"""Tests for app.search (in-memory BM25 index)."""

import pytest

from app import search


@pytest.fixture(autouse=True)
def isolate_index():
    """Reset the module-level index around every test."""
    original = search._index
    search._index = search._Index()
    yield
    search._index = original


def _add(meta: dict, body: str) -> None:
    search._index.add_doc(meta, body)


def _finalize() -> None:
    search._index.finalize()


def _seed_corpus():
    _add(
        {
            "slug": "python-basics",
            "kb": "alpha",
            "title": "Python Basics",
            "summary": "Introduction to Python programming",
            "tags": ["python", "language"],
        },
        "Python is a versatile programming language used for web, data, and scripting.",
    )
    _add(
        {
            "slug": "rust-memory",
            "kb": "alpha",
            "title": "Rust Memory Safety",
            "summary": "How ownership ensures memory safety",
            "tags": ["rust", "systems"],
        },
        "Rust ownership rules prevent data races and dangling pointers at compile time.",
    )
    _add(
        {
            "slug": "docker-intro",
            "kb": "beta",
            "title": "Docker Intro",
            "summary": "Containerize your applications with Docker",
            "tags": ["docker", "containers"],
        },
        "Docker packages an application plus its dependencies into a portable image.",
    )
    _finalize()


class TestTokenize:
    def test_lowercases_and_filters_stopwords(self):
        tokens = search._tokenize("The quick brown Fox")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "the" not in tokens

    def test_short_tokens_dropped(self):
        tokens = search._tokenize("a b foo")
        assert tokens == ["foo"]

    def test_empty_input(self):
        assert search._tokenize("") == []
        assert search._tokenize(None) == []


class TestIndexQuery:
    def test_single_term(self):
        _seed_corpus()
        results = search._index.query("python")
        assert results
        top_doc_id = results[0][1]
        assert search._index.docs[top_doc_id]["slug"] == "python-basics"

    def test_kb_filter(self):
        _seed_corpus()
        results = search._index.query("ownership", kb="alpha")
        assert results
        for _, doc_id in results:
            assert search._index.docs[doc_id]["kb"] == "alpha"

    def test_kb_filter_excludes_others(self):
        _seed_corpus()
        results = search._index.query("docker", kb="alpha")
        assert results == []

    def test_no_match_returns_empty(self):
        _seed_corpus()
        assert search._index.query("quantum") == []

    def test_empty_query_returns_empty(self):
        _seed_corpus()
        assert search._index.query("") == []

    def test_title_match_outranks_body_only(self):
        _add(
            {"slug": "a", "kb": "x", "title": "Widgets", "summary": "", "tags": []},
            "generic body",
        )
        _add(
            {"slug": "b", "kb": "x", "title": "Generic", "summary": "", "tags": []},
            "widgets appear here once",
        )
        _finalize()
        hits = search._index.query("widgets")
        top_slug = search._index.docs[hits[0][1]]["slug"]
        assert top_slug == "a"


class TestSearchFts:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        _seed_corpus()
        assert await search.search_fts("") == []
        assert await search.search_fts("   ") == []

    @pytest.mark.asyncio
    async def test_shape_of_result_row(self):
        _seed_corpus()
        results = await search.search_fts("python")
        assert results
        row = results[0]
        for key in ("slug", "kb", "title", "summary", "tags", "snippet", "score"):
            assert key in row

    @pytest.mark.asyncio
    async def test_kb_filter(self):
        _seed_corpus()
        results = await search.search_fts("docker", kb="beta")
        assert any(r["slug"] == "docker-intro" for r in results)


class TestSearchAutocomplete:
    @pytest.mark.asyncio
    async def test_prefix_match(self):
        _seed_corpus()
        results = await search.search_autocomplete("Pyth")
        assert any(r["slug"] == "python-basics" for r in results)

    @pytest.mark.asyncio
    async def test_short_prefix_rejected(self):
        _seed_corpus()
        assert await search.search_autocomplete("P") == []

    @pytest.mark.asyncio
    async def test_limit(self):
        for i in range(10):
            _add(
                {
                    "slug": f"test-{i}",
                    "kb": "x",
                    "title": f"TestArticle{i}",
                    "summary": "",
                    "tags": [],
                },
                "body",
            )
        _finalize()
        results = await search.search_autocomplete("Test", limit=3)
        assert len(results) == 3


class TestSnippet:
    def test_short_body_returned_as_is(self):
        assert search._make_snippet("short text", "nothing") == "short text"

    def test_long_body_truncated(self):
        body = "word " * 100
        snippet = search._make_snippet(body, "nothing", length=40)
        assert len(snippet) <= 43
        assert snippet.endswith("...")

    def test_snippet_centered_on_match(self):
        body = "a " * 60 + "needle " + "b " * 60
        snippet = search._make_snippet(body, "needle", length=40)
        assert "needle" in snippet


class TestSearchKb:
    def test_empty_query(self):
        _seed_corpus()
        assert search.search_kb("") == []

    def test_empty_index(self):
        assert search.search_kb("python") == []

    def test_basic_match(self):
        _seed_corpus()
        results = search.search_kb("python")
        assert any(r["slug"] == "python-basics" for r in results)

    def test_search_kb_basic_alias(self):
        _seed_corpus()
        direct = search.search_kb("rust")
        alias = search.search_kb_basic("rust")
        assert [r["slug"] for r in direct] == [r["slug"] for r in alias]


class TestIndexSize:
    def test_reports_doc_count(self):
        assert search.index_size() == 0
        _seed_corpus()
        assert search.index_size() == 3
