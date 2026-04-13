"""
GitHub release monitoring and repo indexing.

Monitors key GitHub repos for new releases and auto-triggers research.
Also indexes READMEs, changelogs, and code from your own repos.
"""

import asyncio
import logging
import re
from typing import Optional

import httpx


logger = logging.getLogger("kb-service.github")

GITHUB_API = "https://api.github.com"


# ─── Tracked Repos (tools you use) ──────────────────────────────────────────

TRACKED_REPOS = [
    # AI/ML
    "langchain-ai/langgraph",
    "langchain-ai/langchain",
    "ollama/ollama",
    "vllm-project/vllm",
    "unslothai/unsloth",

    # Infrastructure
    "tailscale/tailscale",
    "argoproj/argo-cd",
    "argoproj/argo-workflows",
    "cloudnative-pg/cloudnative-pg",
    "kubernetes/kubernetes",

    # Hearth stack
    "livekit/livekit",
    "nickvdyck/webtransport-go",
    "matrix-org/dendrite",
    "tauri-apps/tauri",

    # DevOps
    "hashicorp/terraform",
    "open-tofu/opentofu",
    "helm/helm",
    "derailed/k9s",
    "grafana/grafana",
    "prometheus/prometheus",

    # Self-hosted
    "immich-app/immich",
    "goauthentik/authentik",
    "dani-garcia/vaultwarden",
]

# Your own repos to index
OWN_REPOS = [
    "ghndrx/hearth",
    "ghndrx/hearth-mobile",
    "ghndrx/hearth-desktop",
    "ghndrx/homelab-gitops",
    "ghndrx/k8s-manifests",
    "ghndrx/terraform-foundation",
    "ghndrx/ai-kb",
    "ghndrx/monitoring-stack",
    "ghndrx/docker-templates",
    "ghndrx/github-actions-library",
    "ghndrx/pi-mono",
]


# ─── GitHub API Helpers ──────────────────────────────────────────────────────

async def _github_get(path: str, token: str = "") -> Optional[dict | list]:
    """Make a GitHub API request."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "WikiDelve/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{GITHUB_API}{path}", headers=headers)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 403:
                logger.warning("GitHub rate limited: %s", resp.headers.get("X-RateLimit-Reset", ""))
                return None
            else:
                logger.warning("GitHub API %d for %s", resp.status_code, path)
                return None
    except Exception as exc:
        logger.warning("GitHub API error: %s", exc)
        return None


# ─── Release Monitoring ──────────────────────────────────────────────────────

async def get_latest_release(repo: str, token: str = "") -> Optional[dict]:
    """Get the latest release for a repo."""
    data = await _github_get(f"/repos/{repo}/releases/latest", token)
    if not data:
        # Try tags if no releases
        tags = await _github_get(f"/repos/{repo}/tags", token)
        if tags and isinstance(tags, list) and tags:
            return {
                "tag_name": tags[0].get("name", ""),
                "name": tags[0].get("name", ""),
                "body": "",
                "published_at": "",
                "html_url": f"https://github.com/{repo}/releases",
            }
        return None

    return {
        "tag_name": data.get("tag_name", ""),
        "name": data.get("name", ""),
        "body": data.get("body", "")[:3000],
        "published_at": data.get("published_at", ""),
        "html_url": data.get("html_url", ""),
    }


async def check_all_releases(token: str = "") -> list[dict]:
    """Check latest releases for all tracked repos."""
    results = []

    for repo in TRACKED_REPOS:
        release = await get_latest_release(repo, token)
        if release:
            results.append({
                "repo": repo,
                "tag": release["tag_name"],
                "name": release["name"],
                "body": release["body"][:500],
                "published": release["published_at"],
                "url": release["html_url"],
            })
        await asyncio.sleep(0.5)  # Rate limit courtesy

    return results


async def find_new_releases(token: str = "") -> list[dict]:
    """Find releases newer than what's in the KB articles.

    Compares GitHub release versions against versions mentioned in wiki articles.
    Returns repos with potentially newer versions.
    """
    from app.wiki import get_articles

    # Get all versions mentioned in the KB
    articles = get_articles("personal")
    kb_versions = {}
    from app.wiki import read_article_text, parse_frontmatter
    for a in articles:
        text = read_article_text("personal", a["slug"])
        if text:
            _, body = parse_frontmatter(text)
            versions = re.findall(r'v?(\d+\.\d+(?:\.\d+)*)', body)
            for v in versions:
                kb_versions.setdefault(a["title"].lower(), set()).update(versions)

    new_releases = []
    releases = await check_all_releases(token)

    for rel in releases:
        tag = rel["tag"].lstrip("v")
        repo_name = rel["repo"].split("/")[-1].lower()

        # Check if this version is already in any KB article
        found = False
        for title, versions in kb_versions.items():
            if repo_name in title and tag in versions:
                found = True
                break

        if not found and tag:
            new_releases.append(rel)

    return new_releases


# ─── Repo Indexing ───────────────────────────────────────────────────────────

async def get_repo_readme(repo: str, token: str = "") -> Optional[dict]:
    """Get the README content for a repo."""
    data = await _github_get(f"/repos/{repo}/readme", token)
    if not data:
        return None

    # Decode base64 content
    import base64
    content = ""
    if data.get("content"):
        try:
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            return None

    if not content or len(content) < 50:
        return None

    return {
        "repo": repo,
        "title": f"{repo} README",
        "content": content[:10000],
        "url": f"https://github.com/{repo}",
        "word_count": len(content.split()),
    }


async def get_repo_changelog(repo: str, token: str = "") -> Optional[dict]:
    """Try to get CHANGELOG.md from a repo."""
    for filename in ["CHANGELOG.md", "CHANGES.md", "HISTORY.md", "changelog.md"]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"https://raw.githubusercontent.com/{repo}/main/{filename}",
                    headers={"User-Agent": "WikiDelve/1.0"},
                )
                if resp.status_code == 200 and len(resp.text) > 100:
                    return {
                        "repo": repo,
                        "title": f"{repo} Changelog",
                        "content": resp.text[:8000],
                        "url": f"https://github.com/{repo}/blob/main/{filename}",
                        "word_count": len(resp.text.split()),
                    }
        except Exception:
            continue

    return None


async def index_own_repos(token: str = "") -> list[dict]:
    """Index READMEs and changelogs from your own repos."""
    results = []

    for repo in OWN_REPOS:
        readme = await get_repo_readme(repo, token)
        if readme:
            results.append(readme)

        changelog = await get_repo_changelog(repo, token)
        if changelog:
            results.append(changelog)

        await asyncio.sleep(0.3)

    return results
