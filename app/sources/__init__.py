"""
Pluggable source provider package.

Each search source (Serper, Tavily, arXiv, Hacker News, Wikipedia, etc.)
is a single file in this package implementing the ``SourceProvider``
protocol from ``base.py``. Background-discovery sources (RSS, sitemap)
expose ``run_*_discovery`` functions used by the worker cron tasks.

Providers self-register at import time. The research pipeline iterates
``get_provider_classes()`` instead of hard-coding HTTP calls.
"""

from app.sources.base import (
    SourceProvider,
    register,
    get_provider_classes,
    get_provider_class,
    clear_providers,
)
from app.sources.serper import SerperProvider
from app.sources.tavily import TavilyProvider
from app.sources.arxiv import ArxivProvider
from app.sources.hackernews import HackerNewsProvider
from app.sources.wikipedia import WikipediaProvider
from app.sources.reddit import RedditProvider
from app.sources.crossref import CrossrefProvider
from app.sources.stackexchange import StackExchangeProvider
from app.sources.bluesky import BlueskyProvider

# Register built-in providers at import time.
# Order matters for round-1 iteration in run_research: Serper/Tavily go
# first as the metered baseline, then the free providers.
register(SerperProvider)
register(TavilyProvider)
register(ArxivProvider)
register(HackerNewsProvider)
register(WikipediaProvider)
register(RedditProvider)
register(CrossrefProvider)
register(StackExchangeProvider)
register(BlueskyProvider)

__all__ = [
    "SourceProvider",
    "register",
    "get_provider_classes",
    "get_provider_class",
    "clear_providers",
    "SerperProvider",
    "TavilyProvider",
    "ArxivProvider",
    "HackerNewsProvider",
    "WikipediaProvider",
    "RedditProvider",
    "CrossrefProvider",
    "StackExchangeProvider",
    "BlueskyProvider",
]
