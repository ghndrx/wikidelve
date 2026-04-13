"""
SourceProvider protocol + global registry.

A pluggable provider package so search sources (Serper, Tavily, arXiv,
Hacker News, Wikipedia direct, etc.) can be added as one file each
without touching the research pipeline. Each provider implements the
``SourceProvider`` protocol and self-registers at import time.

The research pipeline iterates ``get_provider_classes()`` instead of
hard-coding HTTP calls.
"""

from typing import Protocol
import logging

logger = logging.getLogger("kb-service.sources")


class SourceProvider(Protocol):
    """Interface every search provider must implement.

    Attributes:
        name: stable identifier (e.g. ``"serper"``, ``"arxiv"``).
        tier_default: 1 (authoritative), 2 (reputable), or 3 (general).
        budget_attribution: True if calls should be logged to ``serper_usage``
            for the per-KB daily budget. Currently only Serper sets this;
            new providers can opt in if they hit a metered API.
    """

    name: str
    tier_default: int
    budget_attribution: bool

    async def search(self, query: str, num: int) -> list[dict]:
        """Run a search and return normalized results.

        Each result dict must have keys: ``title``, ``content``, ``url``.
        Returns an empty list if the provider isn't configured (e.g. missing
        API key) — graceful degradation, not an error.
        """
        ...


# Module-level registry — populated at import time by app/sources/__init__.py
_providers: dict[str, type] = {}


def register(provider_cls: type) -> None:
    """Register a provider class. Overwrites any previous registration.

    The provider class must have a non-empty ``name`` class attribute.
    """
    name = getattr(provider_cls, "name", None)
    if not name:
        raise ValueError("Provider class must define a non-empty 'name' attribute")
    if name in _providers:
        logger.warning("Overwriting source provider: %s", name)
    _providers[name] = provider_cls


def get_provider_classes() -> list[type]:
    """Return all registered provider classes in registration order."""
    return list(_providers.values())


def get_provider_class(name: str) -> type | None:
    """Look up a provider class by name. Returns None if not registered."""
    return _providers.get(name)


def clear_providers() -> None:
    """Clear the registry. Test-only helper."""
    _providers.clear()
