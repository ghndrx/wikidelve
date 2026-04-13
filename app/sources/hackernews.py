"""
Hacker News provider — search via the Algolia HN API.

No API key required. Endpoint: ``https://hn.algolia.com/api/v1/search``.

HN comments are often the most candid technical discussion of a topic.
The Algolia API returns top stories matching the query; we surface the
linked URL plus the story's headline and any text body. Tier 2 by default.
"""

import logging

import httpx

logger = logging.getLogger("kb-service.sources.hackernews")

HN_API_URL = "https://hn.algolia.com/api/v1/search"


class HackerNewsProvider:
    """Hacker News search via Algolia."""

    name = "hackernews"
    tier_default = 2
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        params = {
            "query": query,
            "tags": "story",
            "hitsPerPage": str(min(max(num, 1), 10)),
        }
        try:
            resp = await self.client.get(HN_API_URL, params=params, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Hacker News API request failed: %s", exc)
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Hacker News response was not JSON: %s", exc)
            return []

        results: list[dict] = []
        for hit in data.get("hits", []):
            title = (hit.get("title") or "").strip()
            url = (hit.get("url") or "").strip()
            if not url and hit.get("objectID"):
                # Self-post (no external URL) — link to the HN comment thread.
                url = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            if not title or not url:
                continue

            # Body text varies: external links have story_text=None;
            # self-posts have it set. Either way, prefix with the score so
            # the synthesis model can weight the result.
            story_text = (hit.get("story_text") or "").strip()
            points = hit.get("points") or 0
            comments = hit.get("num_comments") or 0
            preface = f"[HN: {points} points, {comments} comments]"
            if story_text:
                content = f"{preface} {story_text[:600]}"
            else:
                content = f"{preface} (linked story — see URL for content)"

            results.append({"title": title, "content": content, "url": url})

        return results
