"""Unit tests for the source provider implementations.

Covers:
  - ArxivProvider (Atom XML parsing, graceful failure)
  - HackerNewsProvider (JSON parsing, self-post fallback)
  - WikipediaProvider (two-step opensearch + extracts)
  - RedditProvider, CrossrefProvider, StackExchangeProvider, BlueskyProvider
  - ``_round1_search_all_providers`` iterating registered providers
  - RSS feed parser + ``run_rss_discovery``
  - Sitemap parser + ``run_sitemap_discovery``
  - Podcast feed enclosure extractor + discovery cron
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# --- Isolated DB path -------------------------------------------------------

_test_db_path = None


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    global _test_db_path
    _test_db_path = tmp_path / "phase3_test.db"
    with patch("app.config.DB_PATH", _test_db_path):
        with patch("app.db.DB_PATH", _test_db_path):
            with patch("app.sources.sitemap.DB_PATH", _test_db_path):
                yield


# ===========================================================================
# 3.1 — arXiv provider
# ===========================================================================


class _FakeAtomResponse:
    """Stand-in for httpx.Response that returns canned arXiv Atom XML."""

    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeArxivClient:
    def __init__(self, response: _FakeAtomResponse):
        self.response = response
        self.calls = []

    async def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.response


_ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Attention Is All You Need (revisited)</title>
    <summary>We present a new architecture based purely on attention mechanisms.</summary>
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Tokio: An Async Runtime for Rust</title>
    <summary>Tokio is a high-performance asynchronous runtime.</summary>
    <link href="http://arxiv.org/abs/2401.00002v1" rel="alternate"/>
  </entry>
</feed>"""


class TestArxivProvider:
    @pytest.mark.asyncio
    async def test_parses_atom_entries(self):
        from app.sources.arxiv import ArxivProvider
        client = _FakeArxivClient(_FakeAtomResponse(_ARXIV_FIXTURE))
        provider = ArxivProvider(client)
        results = await provider.search("attention", num=5)
        assert len(results) == 2
        assert results[0]["title"] == "Attention Is All You Need (revisited)"
        assert "attention mechanisms" in results[0]["content"]
        assert results[0]["url"] == "http://arxiv.org/abs/2401.00001v1"
        assert results[1]["title"] == "Tokio: An Async Runtime for Rust"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from app.sources.arxiv import ArxivProvider
        client = _FakeArxivClient(_FakeAtomResponse(_ARXIV_FIXTURE))
        provider = ArxivProvider(client)
        assert await provider.search("") == []
        assert await provider.search("   ") == []

    @pytest.mark.asyncio
    async def test_handles_http_failure_gracefully(self):
        from app.sources.arxiv import ArxivProvider

        class BadClient:
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("network down")

        provider = ArxivProvider(BadClient())
        assert await provider.search("query") == []

    @pytest.mark.asyncio
    async def test_provider_attributes(self):
        from app.sources.arxiv import ArxivProvider
        assert ArxivProvider.name == "arxiv"
        assert ArxivProvider.tier_default == 1
        assert ArxivProvider.budget_attribution is False


# ===========================================================================
# 3.2 — Hacker News provider
# ===========================================================================


class _FakeJsonResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class _FakeJsonClient:
    def __init__(self, response):
        self.response = response

    async def get(self, *args, **kwargs):
        return self.response


_HN_FIXTURE = {
    "hits": [
        {
            "title": "Show HN: My Rust async runtime",
            "url": "https://example.com/runtime",
            "story_text": None,
            "points": 245,
            "num_comments": 87,
            "objectID": "12345",
        },
        {
            "title": "Ask HN: What's the best way to learn Rust?",
            "url": None,
            "story_text": "I want to learn Rust but find it hard.",
            "points": 50,
            "num_comments": 30,
            "objectID": "67890",
        },
    ]
}


class TestHackerNewsProvider:
    @pytest.mark.asyncio
    async def test_parses_external_link_story(self):
        from app.sources.hackernews import HackerNewsProvider
        client = _FakeJsonClient(_FakeJsonResponse(_HN_FIXTURE))
        provider = HackerNewsProvider(client)
        results = await provider.search("rust async", num=5)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/runtime"
        assert "245 points" in results[0]["content"]
        assert "87 comments" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_self_post_links_to_hn_thread(self):
        from app.sources.hackernews import HackerNewsProvider
        client = _FakeJsonClient(_FakeJsonResponse(_HN_FIXTURE))
        provider = HackerNewsProvider(client)
        results = await provider.search("rust", num=5)
        # Second hit has no external URL — should fall back to HN item link
        ask_hn = results[1]
        assert "news.ycombinator.com/item?id=67890" in ask_hn["url"]
        assert "I want to learn Rust" in ask_hn["content"]

    @pytest.mark.asyncio
    async def test_handles_failure_gracefully(self):
        from app.sources.hackernews import HackerNewsProvider

        class BadClient:
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("dns")

        provider = HackerNewsProvider(BadClient())
        assert await provider.search("anything") == []

    @pytest.mark.asyncio
    async def test_provider_attributes(self):
        from app.sources.hackernews import HackerNewsProvider
        assert HackerNewsProvider.name == "hackernews"
        assert HackerNewsProvider.tier_default == 2
        assert HackerNewsProvider.budget_attribution is False


# ===========================================================================
# 3.3 — Wikipedia provider
# ===========================================================================


class _FakeWikipediaClient:
    """Two-step Wikipedia client: returns opensearch then extracts."""

    def __init__(self, opensearch_payload, extracts_payload):
        self.opensearch = opensearch_payload
        self.extracts = extracts_payload
        self.call_count = 0

    async def get(self, url, params=None, headers=None, timeout=None):
        self.call_count += 1
        if params and params.get("action") == "opensearch":
            return _FakeJsonResponse(self.opensearch)
        return _FakeJsonResponse(self.extracts)


_WIKI_OPENSEARCH = [
    "tokio",
    ["Tokio (software)", "Tokyo"],
    ["", ""],
    [
        "https://en.wikipedia.org/wiki/Tokio_(software)",
        "https://en.wikipedia.org/wiki/Tokyo",
    ],
]

_WIKI_EXTRACTS = {
    "query": {
        "pages": {
            "1": {
                "title": "Tokio (software)",
                "extract": "Tokio is an open-source asynchronous runtime for Rust.",
            },
            "2": {
                "title": "Tokyo",
                "extract": "Tokyo is the capital of Japan.",
            },
        }
    }
}


class TestWikipediaProvider:
    @pytest.mark.asyncio
    async def test_two_step_search(self):
        from app.sources.wikipedia import WikipediaProvider
        client = _FakeWikipediaClient(_WIKI_OPENSEARCH, _WIKI_EXTRACTS)
        provider = WikipediaProvider(client)
        results = await provider.search("tokio")
        assert client.call_count == 2  # opensearch + extracts
        assert len(results) == 2
        assert results[0]["title"] == "Tokio (software)"
        assert "asynchronous runtime" in results[0]["content"]
        assert results[0]["url"].startswith("https://en.wikipedia.org/wiki/")

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self):
        from app.sources.wikipedia import WikipediaProvider
        client = _FakeWikipediaClient(_WIKI_OPENSEARCH, _WIKI_EXTRACTS)
        provider = WikipediaProvider(client)
        assert await provider.search("") == []

    @pytest.mark.asyncio
    async def test_no_search_results(self):
        from app.sources.wikipedia import WikipediaProvider
        empty = ["", [], [], []]
        client = _FakeWikipediaClient(empty, {"query": {"pages": {}}})
        provider = WikipediaProvider(client)
        assert await provider.search("zzznoresults") == []

    @pytest.mark.asyncio
    async def test_handles_failure(self):
        from app.sources.wikipedia import WikipediaProvider

        class BadClient:
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("dns")

        provider = WikipediaProvider(BadClient())
        assert await provider.search("anything") == []

    @pytest.mark.asyncio
    async def test_provider_attributes(self):
        from app.sources.wikipedia import WikipediaProvider
        assert WikipediaProvider.name == "wikipedia"
        assert WikipediaProvider.tier_default == 1
        assert WikipediaProvider.budget_attribution is False


# ===========================================================================
# 3 — Round 1 multi-provider iteration
# ===========================================================================


class TestRound1MultiProvider:
    def test_all_five_providers_registered(self):
        from app.sources import get_provider_classes
        names = [p.name for p in get_provider_classes()]
        assert "serper" in names
        assert "tavily" in names
        assert "arxiv" in names
        assert "hackernews" in names
        assert "wikipedia" in names

    @pytest.mark.asyncio
    async def test_round1_iterates_all_providers(self):
        """A single mocked round 1 should yield results from at least 3 providers."""
        from app import research

        # Patch each provider class so its `search` returns canned results.
        # We use the actual class objects from the registry rather than
        # constructing fakes so the iteration path stays realistic.
        from app.sources import (
            SerperProvider, TavilyProvider, ArxivProvider,
            HackerNewsProvider, WikipediaProvider,
        )

        async def serper_search(self, query, num=8):
            return [{"title": "S1", "content": "c", "url": "https://s.example/1"}]

        async def tavily_search(self, query, num=5):
            return [{"title": "T1", "content": "c", "url": "https://t.example/1"}]

        async def arxiv_search(self, query, num=5):
            return [{"title": "A1", "content": "c", "url": "https://a.example/1"}]

        async def hn_search(self, query, num=5):
            return [{"title": "H1", "content": "c", "url": "https://h.example/1"}]

        async def wiki_search(self, query, num=5):
            return [{"title": "W1", "content": "c", "url": "https://w.example/1"}]

        with patch.object(SerperProvider, "search", serper_search), \
             patch.object(TavilyProvider, "search", tavily_search), \
             patch.object(ArxivProvider, "search", arxiv_search), \
             patch.object(HackerNewsProvider, "search", hn_search), \
             patch.object(WikipediaProvider, "search", wiki_search):
            client = MagicMock()
            results = await research._round1_search_all_providers(
                client, "test query", per_provider=5,
            )
        # Five providers, each returns one fake result → total 5
        urls = {r["url"] for r in results}
        assert len(urls) == 5
        assert "https://s.example/1" in urls
        assert "https://a.example/1" in urls
        assert "https://w.example/1" in urls

    @pytest.mark.asyncio
    async def test_failing_provider_does_not_block_others(self):
        """If one provider raises, the others still contribute."""
        from app import research
        from app.sources import (
            SerperProvider, TavilyProvider, ArxivProvider,
            HackerNewsProvider, WikipediaProvider,
        )

        async def good(self, query, num=5):
            return [{"title": "G", "content": "c", "url": "https://good.example/1"}]

        async def bad(self, query, num=5):
            raise RuntimeError("provider down")

        with patch.object(SerperProvider, "search", good), \
             patch.object(TavilyProvider, "search", bad), \
             patch.object(ArxivProvider, "search", good), \
             patch.object(HackerNewsProvider, "search", bad), \
             patch.object(WikipediaProvider, "search", good):
            client = MagicMock()
            results = await research._round1_search_all_providers(
                client, "test", per_provider=5,
            )
        # Three good providers, two bad → 3 results, no exception
        assert len(results) == 3


# ===========================================================================
# 3.5 — RSS feed parser + cron
# ===========================================================================


_RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Lobsters</title>
  <item>
    <title>Tokio: An Async Runtime for Rust</title>
    <link>https://example.com/posts/tokio</link>
    <description>A deep dive into the Tokio runtime.</description>
  </item>
  <item>
    <title>Brand New Topic Never Seen Before</title>
    <link>https://example.com/posts/new</link>
    <description><![CDATA[Some <b>HTML</b> content here.]]></description>
  </item>
</channel>
</rss>"""

_ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Rust Blog</title>
  <entry>
    <title>Rust 1.80 Released</title>
    <link href="https://blog.rust-lang.org/2024/07/25/Rust-1.80.0.html"/>
    <summary>The Rust team is happy to announce a new version.</summary>
  </entry>
</feed>"""


class TestRSSParser:
    def test_parses_rss_2(self):
        from app.sources.rss import _parse_feed
        entries = _parse_feed(_RSS_FIXTURE)
        assert len(entries) == 2
        assert entries[0]["title"] == "Tokio: An Async Runtime for Rust"
        assert entries[0]["url"] == "https://example.com/posts/tokio"
        assert "deep dive" in entries[0]["summary"]

    def test_parses_atom(self):
        from app.sources.rss import _parse_feed
        entries = _parse_feed(_ATOM_FIXTURE)
        assert len(entries) == 1
        assert entries[0]["title"] == "Rust 1.80 Released"
        assert "blog.rust-lang.org" in entries[0]["url"]

    def test_strips_cdata_and_html(self):
        from app.sources.rss import _parse_feed
        entries = _parse_feed(_RSS_FIXTURE)
        # The CDATA HTML wrapper should be unwrapped and tags stripped
        assert "<b>" not in entries[1]["summary"]
        assert "HTML" in entries[1]["summary"]

    def test_empty_feed_returns_empty(self):
        from app.sources.rss import _parse_feed
        assert _parse_feed("") == []
        assert _parse_feed("<rss></rss>") == []


class TestRSSDiscovery:
    @pytest.mark.asyncio
    async def test_no_watches_skips(self, monkeypatch):
        from app.sources.rss import run_rss_discovery
        monkeypatch.delenv("RSS_WATCHES", raising=False)
        result = await run_rss_discovery()
        assert result["feeds"] == 0
        assert result["candidates"] == 0

    @pytest.mark.asyncio
    async def test_invalid_json_handled(self, monkeypatch):
        from app.sources.rss import run_rss_discovery
        monkeypatch.setenv("RSS_WATCHES", "not valid json")
        result = await run_rss_discovery()
        assert result["feeds"] == 0

    @pytest.mark.asyncio
    async def test_inserts_unmatched_as_candidates(self, monkeypatch):
        from app import db
        from app.sources import rss as rss_mod

        await db.init_db()

        watches = json.dumps([
            {"kb": "personal", "url": "https://example.com/feed.xml"},
        ])
        monkeypatch.setenv("RSS_WATCHES", watches)

        # Mock httpx client to return our RSS fixture
        class FakeStream:
            def __init__(self, text):
                self.text = text
            def raise_for_status(self): pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return None
            async def get(self, url):
                return FakeStream(_RSS_FIXTURE)

        # Patch find_related_article to return None (no matches)
        with patch.object(rss_mod.httpx, "AsyncClient", FakeClient), \
             patch.object(rss_mod, "find_related_article", return_value=None):
            result = await rss_mod.run_rss_discovery()

        assert result["feeds"] == 1
        assert result["candidates"] == 2  # Both RSS entries inserted

        pending = await db.get_pending_candidates("personal", 10)
        topics = {p["topic"] for p in pending}
        assert "Tokio: An Async Runtime for Rust" in topics
        assert "Brand New Topic Never Seen Before" in topics
        assert all(p["source"] == "rss" for p in pending)


# ===========================================================================
# 3.6 — Sitemap parser + crawler
# ===========================================================================


_SITEMAP_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/docs/intro</loc></url>
  <url><loc>https://example.com/docs/advanced/topics</loc></url>
  <url><loc>https://example.com/docs/api/reference</loc></url>
</urlset>"""


class TestSitemapHelpers:
    def test_url_to_topic_label(self):
        from app.sources.sitemap import _url_to_topic_label
        label = _url_to_topic_label("https://example.com/docs/advanced/topics")
        assert "advanced" in label or "topics" in label
        assert "example.com" in label

    def test_normalize_url(self):
        from app.sources.sitemap import _normalize_url
        assert _normalize_url("https://Example.com/Path/") == "https://example.com/path"


class TestSitemapDiscovery:
    @pytest.mark.asyncio
    async def test_no_watches_skips(self, monkeypatch):
        from app.sources.sitemap import run_sitemap_discovery
        monkeypatch.delenv("SITEMAP_WATCHES", raising=False)
        result = await run_sitemap_discovery()
        assert result["sitemaps"] == 0

    @pytest.mark.asyncio
    async def test_inserts_new_urls_as_candidates(self, monkeypatch):
        from app import db
        from app.sources import sitemap as sitemap_mod

        await db.init_db()

        watches = json.dumps([
            {"kb": "personal", "url": "https://example.com/sitemap.xml"},
        ])
        monkeypatch.setenv("SITEMAP_WATCHES", watches)

        class FakeStream:
            def __init__(self, text):
                self.text = text
            def raise_for_status(self): pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return None
            async def get(self, url):
                return FakeStream(_SITEMAP_FIXTURE)

        with patch.object(sitemap_mod.httpx, "AsyncClient", FakeClient):
            result = await sitemap_mod.run_sitemap_discovery()

        assert result["sitemaps"] == 1
        assert result["candidates"] == 3

        pending = await db.get_pending_candidates("personal", 10)
        assert len(pending) == 3
        assert all(p["source"] == "sitemap" for p in pending)
        # source_ref should be the leaf URL
        urls_seen = {p["source_ref"] for p in pending}
        assert "https://example.com/docs/intro" in urls_seen


# ===========================================================================
# Reddit provider
# ===========================================================================


_REDDIT_FIXTURE = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "Why Rust traits are amazing",
                    "permalink": "/r/rust/comments/abc/why_rust_traits/",
                    "score": 432,
                    "num_comments": 78,
                    "subreddit": "rust",
                    "selftext": "I've been using Rust for two years and traits are the best part.",
                }
            },
            {
                "data": {
                    "title": "Show HN-style: my new tool",
                    "permalink": "/r/programming/comments/def/show_tool/",
                    "score": 12,
                    "num_comments": 4,
                    "subreddit": "programming",
                    "selftext": "",
                }
            },
        ]
    }
}


class TestRedditProvider:
    @pytest.mark.asyncio
    async def test_parses_self_post(self):
        from app.sources.reddit import RedditProvider
        client = _FakeJsonClient(_FakeJsonResponse(_REDDIT_FIXTURE))
        provider = RedditProvider(client)
        results = await provider.search("rust traits")
        assert len(results) == 2
        assert "rust" in results[0]["content"].lower()
        assert "432 points" in results[0]["content"]
        assert results[0]["url"].startswith("https://www.reddit.com/r/rust/")

    @pytest.mark.asyncio
    async def test_link_post_falls_back_message(self):
        from app.sources.reddit import RedditProvider
        client = _FakeJsonClient(_FakeJsonResponse(_REDDIT_FIXTURE))
        provider = RedditProvider(client)
        results = await provider.search("any")
        link_post = results[1]
        assert "see thread" in link_post["content"].lower()

    @pytest.mark.asyncio
    async def test_handles_failure(self):
        from app.sources.reddit import RedditProvider

        class BadClient:
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("dns")

        provider = RedditProvider(BadClient())
        assert await provider.search("anything") == []

    @pytest.mark.asyncio
    async def test_provider_attributes(self):
        from app.sources.reddit import RedditProvider
        assert RedditProvider.name == "reddit"
        assert RedditProvider.tier_default == 3
        assert RedditProvider.budget_attribution is False


# ===========================================================================
# Crossref provider
# ===========================================================================


_CROSSREF_FIXTURE = {
    "message": {
        "items": [
            {
                "title": ["Attention Is All You Need"],
                "abstract": "<p>We propose a new architecture based on attention.</p>",
                "DOI": "10.5555/1234.5678",
                "URL": "https://doi.org/10.5555/1234.5678",
                "author": [
                    {"given": "Ashish", "family": "Vaswani"},
                    {"given": "Noam", "family": "Shazeer"},
                ],
                "container-title": ["NeurIPS"],
                "published-print": {"date-parts": [[2017]]},
            }
        ]
    }
}


class TestCrossrefProvider:
    @pytest.mark.asyncio
    async def test_parses_paper_with_abstract(self):
        from app.sources.crossref import CrossrefProvider
        client = _FakeJsonClient(_FakeJsonResponse(_CROSSREF_FIXTURE))
        provider = CrossrefProvider(client)
        results = await provider.search("attention")
        assert len(results) == 1
        r = results[0]
        assert r["title"] == "Attention Is All You Need"
        # JATS XML tags should be stripped
        assert "<p>" not in r["content"]
        assert "attention" in r["content"].lower()
        assert "Vaswani" in r["content"]
        assert "NeurIPS" in r["content"]
        assert r["url"].startswith("https://doi.org/")

    @pytest.mark.asyncio
    async def test_handles_failure(self):
        from app.sources.crossref import CrossrefProvider

        class BadClient:
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("dns")

        provider = CrossrefProvider(BadClient())
        assert await provider.search("anything") == []

    @pytest.mark.asyncio
    async def test_provider_attributes(self):
        from app.sources.crossref import CrossrefProvider
        assert CrossrefProvider.name == "crossref"
        assert CrossrefProvider.tier_default == 1
        assert CrossrefProvider.budget_attribution is False


# ===========================================================================
# Stack Exchange provider
# ===========================================================================


_STACKEXCHANGE_FIXTURE = {
    "items": [
        {
            "title": "How do I borrow a mutable reference in Rust?",
            "link": "https://stackoverflow.com/questions/12345/borrow-mutable",
            "score": 88,
            "answer_count": 4,
            "is_answered": True,
            "tags": ["rust", "ownership", "borrowing"],
        }
    ]
}


class TestStackExchangeProvider:
    @pytest.mark.asyncio
    async def test_parses_answered_question(self):
        from app.sources.stackexchange import StackExchangeProvider
        client = _FakeJsonClient(_FakeJsonResponse(_STACKEXCHANGE_FIXTURE))
        provider = StackExchangeProvider(client)
        results = await provider.search("rust borrow")
        assert len(results) == 1
        r = results[0]
        assert "borrow" in r["title"].lower()
        assert "88 votes" in r["content"]
        assert "accepted" in r["content"]
        assert "rust" in r["content"]
        assert r["url"].startswith("https://stackoverflow.com/")

    @pytest.mark.asyncio
    async def test_default_site(self):
        from app.sources.stackexchange import StackExchangeProvider
        client = _FakeJsonClient(_FakeJsonResponse(_STACKEXCHANGE_FIXTURE))
        provider = StackExchangeProvider(client)
        assert provider.site == "stackoverflow"

    @pytest.mark.asyncio
    async def test_custom_site(self):
        from app.sources.stackexchange import StackExchangeProvider
        client = _FakeJsonClient(_FakeJsonResponse(_STACKEXCHANGE_FIXTURE))
        provider = StackExchangeProvider(client, site="security")
        assert provider.site == "security"

    @pytest.mark.asyncio
    async def test_handles_failure(self):
        from app.sources.stackexchange import StackExchangeProvider

        class BadClient:
            async def get(self, *args, **kwargs):
                raise httpx.ConnectError("dns")

        provider = StackExchangeProvider(BadClient())
        assert await provider.search("anything") == []


# ===========================================================================
# Bluesky provider
# ===========================================================================


_BLUESKY_FIXTURE = {
    "posts": [
        {
            "uri": "at://did:plc:abc123/app.bsky.feed.post/post1",
            "record": {"text": "Just discovered async traits in Rust 1.75!"},
            "author": {"handle": "rustlover.bsky.social", "displayName": "Rust Lover"},
            "likeCount": 42,
            "repostCount": 8,
            "replyCount": 5,
        },
        {
            "uri": "invalid",
            "record": {"text": "fallback handle path"},
            "author": {"handle": "fallback.bsky.social"},
            "likeCount": 1,
        },
    ]
}


class TestBlueskyProvider:
    @pytest.mark.asyncio
    async def test_parses_post_with_at_uri(self):
        from app.sources.bluesky import BlueskyProvider
        client = _FakeJsonClient(_FakeJsonResponse(_BLUESKY_FIXTURE))
        provider = BlueskyProvider(client)
        results = await provider.search("rust async")
        assert len(results) == 2
        first = results[0]
        assert "Rust Lover" in first["title"]
        assert "rustlover.bsky.social" in first["title"]
        assert first["url"].startswith("https://bsky.app/profile/")
        assert "/post/post1" in first["url"]
        assert "42 likes" in first["content"]

    @pytest.mark.asyncio
    async def test_falls_back_to_profile_url(self):
        from app.sources.bluesky import BlueskyProvider
        client = _FakeJsonClient(_FakeJsonResponse(_BLUESKY_FIXTURE))
        provider = BlueskyProvider(client)
        results = await provider.search("anything")
        # Second post had an invalid uri → falls back to profile URL
        assert results[1]["url"] == "https://bsky.app/profile/fallback.bsky.social"

    @pytest.mark.asyncio
    async def test_provider_attributes(self):
        from app.sources.bluesky import BlueskyProvider
        assert BlueskyProvider.name == "bluesky"
        assert BlueskyProvider.tier_default == 3


# ===========================================================================
# Whisper transcription pipeline
# ===========================================================================


class TestTranscribeAudioUrl:
    @pytest.mark.asyncio
    async def test_no_backend_returns_empty(self, monkeypatch):
        from app import transcribe
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("WHISPER_BACKEND", "groq")
        result = await transcribe.transcribe_audio_url("https://example.com/audio.mp3")
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_url_returns_empty(self):
        from app import transcribe
        assert await transcribe.transcribe_audio_url("") == ""
        assert await transcribe.transcribe_audio_url("   ") == ""

    @pytest.mark.asyncio
    async def test_unknown_backend_returns_empty(self, monkeypatch):
        from app import transcribe
        monkeypatch.setenv("WHISPER_BACKEND", "weird")
        monkeypatch.setenv("GROQ_API_KEY", "ignored")
        result = await transcribe.transcribe_audio_url("https://example.com/x.mp3")
        assert result == ""

    def test_resolve_backend_prefers_groq(self, monkeypatch):
        from app.transcribe import _resolve_backend
        monkeypatch.setenv("WHISPER_BACKEND", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        backend, url, key = _resolve_backend()
        assert backend == "groq"
        assert "groq.com" in url
        assert key == "test-key"

    def test_resolve_backend_openai(self, monkeypatch):
        from app.transcribe import _resolve_backend
        monkeypatch.setenv("WHISPER_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        backend, url, key = _resolve_backend()
        assert backend == "openai"
        assert "openai.com" in url
        assert key == "openai-key"


# ===========================================================================
# Podcast discovery
# ===========================================================================


_PODCAST_RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0">
<channel>
  <title>Test Podcast</title>
  <item>
    <title>Episode 1: Intro to Rust</title>
    <link>https://example.com/episodes/1</link>
    <description>The first episode about Rust.</description>
    <enclosure url="https://example.com/audio/ep1.mp3" type="audio/mpeg" length="123456"/>
  </item>
  <item>
    <title>Episode 2: Async Patterns</title>
    <link>https://example.com/episodes/2</link>
    <description>Async deep dive.</description>
    <enclosure url="https://example.com/audio/ep2.mp3" type="audio/mpeg"/>
  </item>
</channel>
</rss>"""


class TestPodcastEnclosureExtractor:
    def test_extracts_audio_urls_keyed_by_link(self):
        from app.sources.podcast import _extract_audio_urls_from_feed
        urls = _extract_audio_urls_from_feed(_PODCAST_RSS_FIXTURE)
        assert urls["https://example.com/episodes/1"] == "https://example.com/audio/ep1.mp3"
        assert urls["https://example.com/episodes/2"] == "https://example.com/audio/ep2.mp3"

    def test_no_enclosures_returns_empty(self):
        from app.sources.podcast import _extract_audio_urls_from_feed
        # Plain RSS without <enclosure> tags
        feed = "<rss><channel><item><title>x</title><link>https://x</link></item></channel></rss>"
        assert _extract_audio_urls_from_feed(feed) == {}


class TestPodcastDiscoveryCron:
    @pytest.mark.asyncio
    async def test_no_watches_skips(self, monkeypatch):
        from app.sources.podcast import run_podcast_discovery
        monkeypatch.delenv("PODCAST_WATCHES", raising=False)
        result = await run_podcast_discovery()
        assert result["feeds"] == 0
        assert result["queued"] == 0

    @pytest.mark.asyncio
    async def test_enqueues_new_episodes(self, monkeypatch):
        from app.sources import podcast as pod_mod

        watches = json.dumps([
            {"kb": "personal", "url": "https://example.com/feed.xml"},
        ])
        monkeypatch.setenv("PODCAST_WATCHES", watches)

        class FakeStream:
            def __init__(self, text):
                self.text = text
            def raise_for_status(self): pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return None
            async def get(self, url):
                return FakeStream(_PODCAST_RSS_FIXTURE)

        fake_pool = MagicMock()
        fake_pool.enqueue_job = AsyncMock(return_value=None)

        with patch.object(pod_mod.httpx, "AsyncClient", FakeClient), \
             patch.object(pod_mod, "find_related_article", return_value=None):
            result = await pod_mod.run_podcast_discovery(arq_pool=fake_pool)

        assert result["feeds"] == 1
        assert result["queued"] == 2
        assert fake_pool.enqueue_job.call_count == 2
        # Verify each call passed media_audio_task on the correct queue
        for call in fake_pool.enqueue_job.call_args_list:
            args, kwargs = call
            assert args[0] == "media_audio_task"
            assert kwargs.get("_queue_name") == "wikidelve"

    @pytest.mark.asyncio
    async def test_skips_existing_episodes(self, monkeypatch):
        from app.sources import podcast as pod_mod

        watches = json.dumps([
            {"kb": "personal", "url": "https://example.com/feed.xml"},
        ])
        monkeypatch.setenv("PODCAST_WATCHES", watches)

        class FakeStream:
            def __init__(self, text):
                self.text = text
            def raise_for_status(self): pass

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return None
            async def get(self, url):
                return FakeStream(_PODCAST_RSS_FIXTURE)

        fake_pool = MagicMock()
        fake_pool.enqueue_job = AsyncMock()

        # Pretend every episode already has a wiki article
        with patch.object(pod_mod.httpx, "AsyncClient", FakeClient), \
             patch.object(pod_mod, "find_related_article", return_value={"slug": "x"}):
            result = await pod_mod.run_podcast_discovery(arq_pool=fake_pool)

        assert result["queued"] == 0
        assert result["skipped_existing"] == 2
        fake_pool.enqueue_job.assert_not_called()


# ===========================================================================
# Provider registry — full set
# ===========================================================================


class TestFullProviderRegistry:
    def test_all_nine_providers_registered(self):
        from app.sources import get_provider_classes
        names = {p.name for p in get_provider_classes()}
        assert "serper" in names
        assert "tavily" in names
        assert "arxiv" in names
        assert "hackernews" in names
        assert "wikipedia" in names
        assert "reddit" in names
        assert "crossref" in names
        assert "stackexchange" in names
        assert "bluesky" in names
