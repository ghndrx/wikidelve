"""
Reddit provider — search via the public JSON API.

No OAuth required for read-only search; Reddit just needs a distinctive
``User-Agent`` header. Endpoint: ``https://www.reddit.com/search.json``
or ``https://www.reddit.com/r/{subreddit}/search.json``.

Reddit threads are usually Tier 2/3 (general discussion), occasionally
upgraded to Tier 2 when the subreddit is specifically authoritative
(e.g. ``r/rust`` for rust language questions).
"""

import logging

import httpx

logger = logging.getLogger("kb-service.sources.reddit")

REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"


class RedditProvider:
    """Reddit search provider."""

    name = "reddit"
    tier_default = 3
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        params = {
            "q": query,
            "limit": str(min(max(num, 1), 10)),
            "sort": "relevance",
            "type": "link",
        }
        try:
            resp = await self.client.get(
                REDDIT_SEARCH_URL,
                params=params,
                headers={"User-Agent": "wikidelve/1.0 (by /u/wikidelve)"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Reddit API request failed: %s", exc)
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Reddit response was not JSON: %s", exc)
            return []

        children = data.get("data", {}).get("children", [])
        results: list[dict] = []
        for item in children:
            d = item.get("data", {}) if isinstance(item, dict) else {}
            title = (d.get("title") or "").strip()
            permalink = d.get("permalink") or ""
            if not title or not permalink:
                continue

            url = f"https://www.reddit.com{permalink}"
            score = d.get("score") or 0
            comments = d.get("num_comments") or 0
            subreddit = d.get("subreddit") or "?"
            selftext = (d.get("selftext") or "").strip()

            preface = (
                f"[r/{subreddit} \u2022 {score} points \u2022 "
                f"{comments} comments]"
            )
            if selftext:
                content = f"{preface} {selftext[:500]}"
            else:
                content = f"{preface} (link post — see thread for discussion)"

            results.append({"title": title, "content": content, "url": url})

        return results
