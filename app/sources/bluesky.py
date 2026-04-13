"""
Bluesky provider — search via the AT Protocol public AppView.

No authentication needed for the public search endpoint:
``https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts``.

Bluesky has become a venue for technical discussion, especially among
researchers and engineers. Tier 3 by default — short-form posts are
useful as a discovery signal but rarely a primary source.
"""

import logging

import httpx

logger = logging.getLogger("kb-service.sources.bluesky")

BLUESKY_SEARCH_URL = (
    "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
)


class BlueskyProvider:
    """Bluesky post search."""

    name = "bluesky"
    tier_default = 3
    budget_attribution = False

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def search(self, query: str, num: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []

        params = {
            "q": query,
            "limit": str(min(max(num, 1), 25)),
        }
        try:
            resp = await self.client.get(
                BLUESKY_SEARCH_URL,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Bluesky API request failed: %s", exc)
            return []

        try:
            data = resp.json()
        except Exception as exc:
            logger.warning("Bluesky response was not JSON: %s", exc)
            return []

        posts = data.get("posts", [])
        results: list[dict] = []
        for post in posts:
            record = post.get("record", {}) or {}
            text = (record.get("text") or "").strip()
            if not text:
                continue

            author = post.get("author", {}) or {}
            handle = author.get("handle") or "?"
            display = author.get("displayName") or handle
            uri = post.get("uri") or ""
            # at:// URI → bsky.app web URL
            url = ""
            if uri.startswith("at://") and "/app.bsky.feed.post/" in uri:
                parts = uri.replace("at://", "").split("/app.bsky.feed.post/")
                if len(parts) == 2:
                    did_or_handle, post_id = parts
                    url = f"https://bsky.app/profile/{did_or_handle}/post/{post_id}"
            if not url:
                url = f"https://bsky.app/profile/{handle}"

            likes = post.get("likeCount") or 0
            reposts = post.get("repostCount") or 0
            replies = post.get("replyCount") or 0

            title = f"{display} (@{handle})"
            preface = (
                f"[Bluesky \u2022 {likes} likes \u2022 {reposts} reposts \u2022 "
                f"{replies} replies]"
            )
            content = f"{preface} {text[:500]}"
            results.append({"title": title, "content": content, "url": url})

        return results
