"""
Local research pipeline: scan files, folders, and git repos for knowledge.

No search API keys required — reads local filesystem and uses the LLM
to synthesize findings into structured wiki articles.

Pipeline:
  1. Discover files in the target path (filter by extension, skip binaries)
  2. Score files by relevance to the topic (filename, content matching)
  3. Read top-scoring files, extract meaningful content
  4. Gather git context if available (recent commits, contributors, structure)
  5. Synthesize into a structured wiki article via LLM
"""

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.llm import llm_chat
from app import db

logger = logging.getLogger("kb-service.local_research")

# File extensions we can meaningfully read
TEXT_EXTENSIONS = {
    # Code
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".graphql", ".proto",
    # Config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".env.example", ".editorconfig",
    # Docs
    ".md", ".rst", ".txt", ".adoc", ".org",
    # Web
    ".html", ".css", ".scss", ".less", ".svelte", ".vue",
    # Infra
    ".tf", ".hcl", ".dockerfile", ".containerfile",
    ".nginx", ".apache",
    # Data
    ".csv", ".xml", ".nix",
}

# Files always worth reading regardless of extension
IMPORTANT_FILENAMES = {
    "readme", "readme.md", "readme.rst", "readme.txt",
    "changelog", "changelog.md", "changes", "history.md",
    "contributing", "contributing.md",
    "makefile", "justfile", "taskfile.yml",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "package.json", "pyproject.toml", "cargo.toml", "go.mod",
    "requirements.txt", "gemfile", "build.gradle", "pom.xml",
    ".gitignore", "license", "license.md",
}

# Directories to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "target", ".next", ".nuxt",
    "vendor", "bower_components", ".terraform",
    ".idea", ".vscode", ".vs",
    "coverage", ".nyc_output", "htmlcov",
    ".eggs", "*.egg-info",
}

MAX_FILE_SIZE = 100_000  # 100KB per file
MAX_CONTENT_PER_FILE = 8000  # chars to include in synthesis prompt
MAX_FILES_TO_READ = 30  # top files to include


# --- File Discovery ---------------------------------------------------------

def discover_files(root: Path, topic: str = "") -> list[dict]:
    """Walk a directory tree and return scored file metadata.

    Each file gets a relevance score based on:
      - Filename match to topic keywords
      - File importance (README, config, etc.)
      - File extension (code, docs, config)
      - Recency (modification time)
    """
    if not root.exists():
        return []

    # If root is a single file, just return it
    if root.is_file():
        info = _file_info(root, root.parent, topic)
        return [info] if info else []

    topic_words = set(topic.lower().split()) if topic else set()
    files = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        dp = Path(dirpath)
        for fname in filenames:
            filepath = dp / fname
            info = _file_info(filepath, root, topic, topic_words)
            if info:
                files.append(info)

    # Sort by relevance score descending
    files.sort(key=lambda f: f["score"], reverse=True)
    return files


def _file_info(
    filepath: Path,
    root: Path,
    topic: str = "",
    topic_words: Optional[set] = None,
) -> Optional[dict]:
    """Build file info dict with relevance scoring."""
    try:
        stat = filepath.stat()
    except OSError:
        return None

    # Skip large files and symlinks
    if stat.st_size > MAX_FILE_SIZE or stat.st_size == 0:
        return None
    if filepath.is_symlink():
        return None

    fname_lower = filepath.name.lower()
    suffix = filepath.suffix.lower()

    # Check if we can read this file
    is_important = fname_lower in IMPORTANT_FILENAMES
    is_readable = suffix in TEXT_EXTENSIONS or is_important or suffix == ""

    # Dockerfiles, Makefiles, etc. have no extension
    if not suffix and not is_important:
        # Skip extensionless files unless they look like scripts or docs
        return None

    if not is_readable:
        return None

    # Calculate relevance score
    score = 0.0
    rel_path = str(filepath.relative_to(root))

    # Important files get a base boost
    if is_important:
        score += 20
    if fname_lower.startswith("readme"):
        score += 30

    # Doc files score higher
    if suffix in {".md", ".rst", ".txt", ".adoc"}:
        score += 10

    # Topic keyword matching in filename and path
    if topic_words:
        path_lower = rel_path.lower()
        for word in topic_words:
            if len(word) < 3:
                continue
            if word in fname_lower:
                score += 15
            elif word in path_lower:
                score += 8

    # Recency bonus (files modified in last 30 days)
    age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
    if age_days < 7:
        score += 5
    elif age_days < 30:
        score += 2

    # Config/infra files get a small boost
    if suffix in {".yaml", ".yml", ".toml", ".json", ".tf", ".hcl"}:
        score += 3

    # Depth penalty — deeper files are less important
    depth = len(Path(rel_path).parts)
    score -= depth * 0.5

    return {
        "path": str(filepath),
        "rel_path": rel_path,
        "name": filepath.name,
        "suffix": suffix,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "score": max(0, score),
    }


# --- File Reading -----------------------------------------------------------

def read_file_content(filepath: str, max_chars: int = MAX_CONTENT_PER_FILE) -> Optional[str]:
    """Read a file's text content, truncating if needed."""
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"
        return text
    except Exception:
        return None


# --- Git Context ------------------------------------------------------------

def get_git_context(path: Path) -> Optional[dict]:
    """Extract git metadata from a repo: recent commits, branch, contributors."""
    git_dir = _find_git_root(path)
    if not git_dir:
        return None

    context = {"git_root": str(git_dir)}

    # Current branch
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=git_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            context["branch"] = result.stdout.strip()
    except Exception:
        pass

    # Recent commits (last 20)
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-decorate", "-20"],
            cwd=git_dir, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            context["recent_commits"] = result.stdout.strip()
    except Exception:
        pass

    # File stats
    try:
        result = subprocess.run(
            ["git", "log", "--format=", "--name-only", "-50"],
            cwd=git_dir, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split("\n") if f.strip()]
            from collections import Counter
            context["recently_changed_files"] = [
                f for f, _ in Counter(files).most_common(10)
            ]
    except Exception:
        pass

    # Contributors
    try:
        result = subprocess.run(
            ["git", "shortlog", "-sn", "--no-merges", "-10"],
            cwd=git_dir, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            context["contributors"] = result.stdout.strip()
    except Exception:
        pass

    # Remote URL — prefer origin, fall back to upstream / any other
    # configured remote. Repos with no remotes return (None, None) and
    # the backlink section just omits the repo link.
    remote_name, remote_url = _pick_git_remote(git_dir)
    if remote_url:
        context["remote_url"] = remote_url
        context["remote_name"] = remote_name

    # Exact commit the scan sees — lets the wiki backlink to the
    # precise state the article was synthesized from, even if the
    # branch moves on later.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            context["commit"] = result.stdout.strip()
    except Exception:
        pass

    return context


_PREFERRED_REMOTES = ("origin", "upstream")


def _pick_git_remote(git_root: Path) -> tuple[Optional[str], Optional[str]]:
    """Return (remote_name, url) for the best available remote.

    Tries ``origin`` then ``upstream``, then falls back to the first
    remote in ``git remote``. Returns (None, None) if the repo has no
    remotes at all (common for fresh ``git init`` directories — the
    backlink just becomes a local-only marker in that case).
    """
    try:
        result = subprocess.run(
            ["git", "remote"], cwd=git_root, capture_output=True,
            text=True, timeout=5,
        )
        if result.returncode != 0:
            return None, None
        remotes = [r.strip() for r in result.stdout.splitlines() if r.strip()]
    except Exception:
        return None, None
    if not remotes:
        return None, None
    ordered = [r for r in _PREFERRED_REMOTES if r in remotes] + [
        r for r in remotes if r not in _PREFERRED_REMOTES
    ]
    for name in ordered:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", name],
                cwd=git_root, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return name, result.stdout.strip()
        except Exception:
            continue
    return None, None


def _normalize_git_remote(url: str) -> Optional[str]:
    """Convert a git remote URL to an HTTPS form suitable for linking.

    Handles the three common forms:
      git@github.com:owner/repo.git        → https://github.com/owner/repo
      ssh://git@github.com/owner/repo.git  → https://github.com/owner/repo
      https://github.com/owner/repo.git    → https://github.com/owner/repo

    Returns None if the URL doesn't parse into something we can link
    (file://, unknown scheme, malformed input). Strips the trailing
    ``.git`` suffix uniformly since neither GitHub, GitLab, nor
    Bitbucket require it for web views.
    """
    if not url:
        return None
    url = url.strip()

    # SCP-style SSH: git@host:path
    if url.startswith("git@") and ":" in url and "://" not in url:
        host_and_path = url[len("git@"):]
        host, _, path = host_and_path.partition(":")
        if not host or not path:
            return None
        url = f"https://{host}/{path}"
    # ssh:// form
    elif url.startswith("ssh://git@"):
        url = "https://" + url[len("ssh://git@"):]
    elif url.startswith("ssh://"):
        url = "https://" + url[len("ssh://"):]
    elif url.startswith("git://"):
        url = "https://" + url[len("git://"):]
    # file:// or other schemes — not linkable
    elif not (url.startswith("https://") or url.startswith("http://")):
        return None

    if url.endswith(".git"):
        url = url[:-4]
    return url


def _blob_url_template(remote_https: str, commit: Optional[str]) -> Optional[str]:
    """Return a format string for per-file deep links, or None if the
    host isn't recognised. Callers do ``template.format(path=rel_path)``.

    Only emits a template when we have a commit SHA — branch-based
    links are permanently broken once the branch moves.
    """
    if not remote_https or not commit:
        return None
    # github.com, gitlab.com, bitbucket.org, and common self-hosted
    # GitLab / Gitea mirrors all use /blob/<ref>/<path> or /src/<ref>/<path>.
    # We only emit for the three hosts where we're sure of the path
    # format — others get no per-file links but still get the repo link.
    if "github.com" in remote_https:
        return remote_https + "/blob/" + commit + "/{path}"
    if "gitlab" in remote_https:
        return remote_https + "/-/blob/" + commit + "/{path}"
    if "bitbucket.org" in remote_https:
        return remote_https + "/src/" + commit + "/{path}"
    return None


def _tree_url(remote_https: str, commit: str, subpath: str = "") -> Optional[str]:
    """Build a directory-tree URL for github/gitlab/bitbucket. Subpath
    makes a monorepo scan land in the right subdir, not the repo root.
    """
    subpath = subpath.strip("/")
    tail = f"/{subpath}" if subpath else ""
    if "github.com" in remote_https:
        return f"{remote_https}/tree/{commit}{tail}"
    if "gitlab" in remote_https:
        return f"{remote_https}/-/tree/{commit}{tail}"
    if "bitbucket.org" in remote_https:
        return f"{remote_https}/src/{commit}{tail}"
    return None


def _probe_manifest(scan_dir: Path) -> dict:
    """Probe language-specific manifest files for URLs that complement
    a git remote. These fill in homepage / docs / issue-tracker fields
    without overriding the git ground-truth for source_repo.

    Silent-best-effort: every parse failure is ignored, since a broken
    manifest isn't a research-blocker.
    """
    found: dict = {}

    # Node: package.json
    pkg_json = scan_dir / "package.json"
    if pkg_json.is_file():
        try:
            import json as _json
            data = _json.loads(pkg_json.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                if isinstance(data.get("homepage"), str):
                    found.setdefault("source_homepage", data["homepage"].strip())
                repo = data.get("repository")
                if isinstance(repo, str):
                    found.setdefault("source_repo_hint", repo.strip())
                elif isinstance(repo, dict) and isinstance(repo.get("url"), str):
                    found.setdefault("source_repo_hint", repo["url"].strip())
                bugs = data.get("bugs")
                if isinstance(bugs, str):
                    found.setdefault("source_issues", bugs.strip())
                elif isinstance(bugs, dict) and isinstance(bugs.get("url"), str):
                    found.setdefault("source_issues", bugs["url"].strip())
                name = data.get("name")
                if isinstance(name, str) and name:
                    # Scoped + unscoped both resolve on npmjs.
                    found.setdefault("source_npm", f"https://www.npmjs.com/package/{name}")
        except Exception:
            pass

    # Python: pyproject.toml (PEP 621 [project.urls])
    pyproject = scan_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            try:
                import tomllib  # py3.11+
            except ImportError:  # pragma: no cover — only on <3.11
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="ignore"))
            project = data.get("project", {}) if isinstance(data, dict) else {}
            urls = project.get("urls", {}) if isinstance(project, dict) else {}
            if isinstance(urls, dict):
                # Keys are case/naming-inconsistent in the wild — check
                # a few common spellings for each field.
                def _pick(keys: tuple[str, ...]) -> Optional[str]:
                    for k in urls:
                        if k.lower() in keys and isinstance(urls[k], str):
                            return urls[k].strip()
                    return None
                hp = _pick(("homepage", "home-page", "home"))
                if hp:
                    found.setdefault("source_homepage", hp)
                repo = _pick(("repository", "source", "source code"))
                if repo:
                    found.setdefault("source_repo_hint", repo)
                docs = _pick(("documentation", "docs"))
                if docs:
                    found.setdefault("source_docs", docs)
                issues = _pick(("issues", "issue tracker", "bug tracker", "bugs", "tracker"))
                if issues:
                    found.setdefault("source_issues", issues)
            name = project.get("name") if isinstance(project, dict) else None
            if isinstance(name, str) and name:
                found.setdefault("source_pypi", f"https://pypi.org/project/{name}/")
        except Exception:
            pass

    # Rust: Cargo.toml
    cargo = scan_dir / "Cargo.toml"
    if cargo.is_file():
        try:
            try:
                import tomllib
            except ImportError:  # pragma: no cover
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(cargo.read_text(encoding="utf-8", errors="ignore"))
            pkg = data.get("package", {}) if isinstance(data, dict) else {}
            if isinstance(pkg, dict):
                for key, field in (
                    ("repository", "source_repo_hint"),
                    ("homepage", "source_homepage"),
                    ("documentation", "source_docs"),
                ):
                    val = pkg.get(key)
                    if isinstance(val, str) and val.strip():
                        found.setdefault(field, val.strip())
                name = pkg.get("name")
                if isinstance(name, str) and name:
                    found.setdefault("source_crates", f"https://crates.io/crates/{name}")
        except Exception:
            pass

    # Go: go.mod — the first line's module path implies a source URL.
    go_mod = scan_dir / "go.mod"
    if go_mod.is_file():
        try:
            first = go_mod.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
            if first.startswith("module "):
                module = first[len("module "):].strip()
                if module:
                    found.setdefault("source_go_module", module)
                    # github.com/foo/bar → https://github.com/foo/bar
                    if module.startswith(("github.com/", "gitlab.com/", "bitbucket.org/")):
                        # Trim to owner/repo (strip submodule paths).
                        parts = module.split("/")
                        if len(parts) >= 3:
                            found.setdefault(
                                "source_repo_hint",
                                f"https://{parts[0]}/{parts[1]}/{parts[2]}",
                            )
        except Exception:
            pass

    return found


def _probe_override(scan_dir: Path) -> dict:
    """Read a ``.wikidelve.yml`` / ``.wikidelve.yaml`` override file.

    Lets users stamp canonical URLs on folders that aren't git repos
    (Obsidian vaults, shared doc directories, SMB mounts, etc.).
    Any top-level key starting with ``source_`` is accepted verbatim.
    Trusted because this file is on the user's own filesystem.
    """
    for name in (".wikidelve.yml", ".wikidelve.yaml"):
        f = scan_dir / name
        if not f.is_file():
            continue
        try:
            import yaml  # pyyaml is already a dep for frontmatter
            data = yaml.safe_load(f.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            k: str(v).strip()
            for k, v in data.items()
            if isinstance(k, str) and k.startswith("source_") and isinstance(v, (str, int, float)) and str(v).strip()
        }
    return {}


def build_source_meta(
    scan_path: str,
    git_context: Optional[dict],
) -> tuple[dict, str]:
    """Build (frontmatter_fields, markdown_appendix) for the source-of-record.

    Precedence (earlier entries win on key conflicts):
      1. ``.wikidelve.yml`` explicit override — user knows best
      2. Git context — ground-truth repo URL, branch, commit
      3. Language manifests — complementary URLs (homepage, docs, issues)

    Frontmatter keys are prefixed ``source_`` so they sort together.
    The appendix is deterministic markdown — does not go through the
    LLM, so the links are always present and always correct.
    """
    scan_dir = Path(scan_path)
    meta: dict = {"source_path": scan_path}
    lines = ["## Source", "", f"- **Scanned path**: `{scan_path}`"]

    # 1. Override file gets first dibs on every key.
    override = _probe_override(scan_dir) if scan_dir.is_dir() else {}
    for k, v in override.items():
        meta.setdefault(k, v)

    # 2. Git context — compute subpath-within-repo so monorepo scans
    # deep-link into the right subdir instead of the repo root.
    git_root_str = (git_context or {}).get("git_root")
    subpath = ""
    if git_root_str and scan_dir.is_dir():
        try:
            subpath = str(scan_dir.resolve().relative_to(Path(git_root_str).resolve()))
            if subpath == ".":
                subpath = ""
        except (ValueError, OSError):
            subpath = ""

    branch = (git_context or {}).get("branch")
    commit = (git_context or {}).get("commit")
    remote_raw = (git_context or {}).get("remote_url")
    remote_https = _normalize_git_remote(remote_raw) if remote_raw else None

    if branch:
        meta.setdefault("source_branch", branch)
    if commit:
        meta.setdefault("source_commit", commit)
    if subpath:
        meta.setdefault("source_repo_subpath", subpath)
    if remote_https:
        meta.setdefault("source_repo", remote_https)
    elif remote_raw:
        meta.setdefault("source_repo", remote_raw)

    # 3. Language manifests — only fill complementary fields.
    if scan_dir.is_dir():
        for k, v in _probe_manifest(scan_dir).items():
            meta.setdefault(k, v)
        # Walk up to the git root for manifests too — monorepos often
        # keep package.json / pyproject at the workspace root.
        if git_root_str and str(scan_dir.resolve()) != str(Path(git_root_str).resolve()):
            for k, v in _probe_manifest(Path(git_root_str)).items():
                meta.setdefault(k, v)

    # If git didn't give us a source_repo, fall back to any hint the
    # manifests picked up (e.g. package.json repository URL).
    hint = meta.pop("source_repo_hint", None)
    if hint:
        hint_https = _normalize_git_remote(hint) or hint
        meta.setdefault("source_repo", hint_https)

    # --- Build the appendix, in a fixed order for readability ------------
    if meta.get("source_branch"):
        lines.append(f"- **Branch**: `{meta['source_branch']}`")
    if meta.get("source_commit"):
        lines.append(f"- **Commit**: `{meta['source_commit'][:12]}`")
    if meta.get("source_repo_subpath"):
        lines.append(f"- **Repo subpath**: `{meta['source_repo_subpath']}`")
    repo = meta.get("source_repo")
    if repo and (repo.startswith("https://") or repo.startswith("http://")):
        link = repo
        if commit:
            tree = _tree_url(repo, commit, subpath)
            if tree:
                link = tree
        lines.append(f"- **Repo**: [{repo}]({link})")
    elif repo:
        lines.append(f"- **Repo**: `{repo}`")
    if meta.get("source_homepage"):
        lines.append(f"- **Homepage**: {meta['source_homepage']}")
    if meta.get("source_docs"):
        lines.append(f"- **Docs**: {meta['source_docs']}")
    if meta.get("source_issues"):
        lines.append(f"- **Issues**: {meta['source_issues']}")
    for label, key in (
        ("npm", "source_npm"),
        ("PyPI", "source_pypi"),
        ("crates.io", "source_crates"),
        ("Go module", "source_go_module"),
    ):
        if meta.get(key):
            lines.append(f"- **{label}**: {meta[key]}")

    lines.append("")
    return meta, "\n".join(lines) + "\n"


def _find_git_root(path: Path) -> Optional[Path]:
    """Walk up from path to find the .git directory."""
    current = path if path.is_dir() else path.parent
    for _ in range(20):
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# --- Content Matching -------------------------------------------------------

def score_content_relevance(content: str, topic: str) -> float:
    """Score how relevant a file's content is to the research topic."""
    if not content or not topic:
        return 0.0

    topic_words = [w.lower() for w in topic.split() if len(w) >= 3]
    if not topic_words:
        return 0.0

    content_lower = content.lower()
    score = 0.0

    for word in topic_words:
        count = content_lower.count(word)
        if count > 0:
            # Diminishing returns for repeated mentions
            score += min(count, 10) * 2

    # Normalize by content length
    word_count = len(content.split())
    if word_count > 0:
        score = score / (word_count / 100)  # per 100 words

    return round(score, 2)


# --- Synthesis Prompt -------------------------------------------------------

LOCAL_SYNTHESIS_PROMPT = (
    "You are a senior technical writer creating knowledge base articles from local "
    "source code, documentation, and configuration files.\n\n"
    "WRITING RULES:\n"
    "- Start with a 2-3 sentence summary of what this project/topic is about\n"
    "- Organize findings under clear ## headers\n"
    "- Reference specific files when citing information: `path/to/file.py`\n"
    "- Include code snippets for key patterns, configs, or important logic\n"
    "- Note the tech stack, dependencies, and architecture patterns found\n"
    "- Add a 'Key Findings' section with actionable bullet points\n"
    "- Add a 'Project Structure' section if researching a repo/directory\n"
    "- Be precise with version numbers, config values, and technical details\n"
    "- If the source material is insufficient, note gaps explicitly\n"
    "- Target 800-2000 words — comprehensive but not padded"
)


# --- Main Pipeline ----------------------------------------------------------

async def run_local_research(
    topic: str,
    path: str,
    job_id: int,
    file_pattern: Optional[str] = None,
) -> None:
    """Execute the local research pipeline.

    Args:
        topic: What to research (e.g. "authentication flow", "deployment config")
        path: Local path to scan (directory, git repo, or file)
        job_id: SQLite job ID for status tracking
        file_pattern: Optional glob pattern to filter files (e.g. "*.py", "**/*.yaml")
    """
    target = Path(path).expanduser().resolve()

    if not target.exists():
        await db.update_job(
            job_id, status="error",
            error=f"Path not found: {path}",
        )
        return

    await db.update_job(job_id, status="scanning_files")

    # --- Step 1: Discover files ---
    files = discover_files(target, topic)

    # Apply file pattern filter if specified
    if file_pattern:
        import fnmatch
        files = [
            f for f in files
            if fnmatch.fnmatch(f["rel_path"], file_pattern)
            or fnmatch.fnmatch(f["name"], file_pattern)
        ]

    if not files:
        await db.update_job(
            job_id, status="error",
            error=f"No readable files found in {path}",
        )
        return

    logger.info(
        "Local research [%d]: found %d files in %s",
        job_id, len(files), path,
    )

    # --- Step 2: Read and score content relevance ---
    await db.update_job(job_id, status="reading_files")

    file_contents: list[dict] = []
    for finfo in files[:MAX_FILES_TO_READ * 2]:  # read more than we need, filter by content
        content = read_file_content(finfo["path"])
        if not content:
            continue

        content_score = score_content_relevance(content, topic)
        finfo["content_score"] = content_score
        finfo["combined_score"] = finfo["score"] + content_score
        finfo["content"] = content
        file_contents.append(finfo)

    # Re-sort by combined score
    file_contents.sort(key=lambda f: f["combined_score"], reverse=True)
    top_files = file_contents[:MAX_FILES_TO_READ]

    logger.info(
        "Local research [%d]: read %d files, using top %d",
        job_id, len(file_contents), len(top_files),
    )

    # --- Step 3: Gather git context ---
    git_context = get_git_context(target)

    # --- Step 4: Build synthesis prompt ---
    await db.update_job(job_id, status="synthesizing", sources_count=len(top_files))

    # File listing
    file_listing = "\n".join(
        f"  {f['rel_path']} ({f['size']} bytes, score: {f['combined_score']:.1f})"
        for f in top_files
    )

    # File contents
    content_sections = []
    total_chars = 0
    max_total = 60000  # keep prompt under ~15k tokens

    for f in top_files:
        if total_chars >= max_total:
            break
        section = f"### File: {f['rel_path']}\n```{f['suffix'].lstrip('.')}\n{f['content']}\n```"
        if total_chars + len(section) > max_total:
            remaining = max_total - total_chars
            truncated_content = f['content'][:remaining - 200]
            section = f"### File: {f['rel_path']}\n```{f['suffix'].lstrip('.')}\n{truncated_content}\n[truncated]\n```"
        content_sections.append(section)
        total_chars += len(section)

    # Git context section
    git_section = ""
    if git_context:
        parts = []
        if git_context.get("branch"):
            parts.append(f"Branch: {git_context['branch']}")
        if git_context.get("remote_url"):
            parts.append(f"Remote: {git_context['remote_url']}")
        if git_context.get("recent_commits"):
            parts.append(f"Recent commits:\n{git_context['recent_commits']}")
        if git_context.get("contributors"):
            parts.append(f"Contributors:\n{git_context['contributors']}")
        if git_context.get("recently_changed_files"):
            parts.append(f"Recently changed:\n  " + "\n  ".join(git_context["recently_changed_files"]))
        git_section = "\n\n".join(parts)

    synthesis_prompt = f"""Create a comprehensive knowledge base article based on local source files.

Topic: {topic}
Source path: {path}
Files analyzed: {len(top_files)} of {len(files)} discovered

{'## Git Context' + chr(10) + git_section + chr(10) if git_section else ''}
## File Listing (by relevance)
{file_listing}

## File Contents
{chr(10).join(content_sections)}

Write a structured article covering:
1. What this project/code does and why it exists
2. Architecture and key patterns found in the source
3. Important configuration and dependencies
4. Notable implementation details relevant to "{topic}"
5. Key files and their roles
"""

    # --- Step 5: Synthesize ---
    try:
        content = await llm_chat(
            LOCAL_SYNTHESIS_PROMPT,
            synthesis_prompt,
            max_tokens=4000,
            temperature=0.2,
        )
    except Exception as exc:
        await db.update_job(
            job_id, status="error",
            error=f"LLM synthesis failed: {exc}",
        )
        return

    if not content or len(content) < 100:
        await db.update_job(
            job_id, status="error",
            error="LLM returned insufficient content",
        )
        return

    # Deterministic source appendix — built from git metadata, NOT the
    # LLM, so the repo backlink is always present and always correct.
    # Frontmatter fields ride along via source_params so the worker's
    # create_or_update_article call can stamp them onto the wiki article.
    source_frontmatter, source_markdown = build_source_meta(path, git_context)
    content = content.rstrip() + "\n\n" + source_markdown

    # --- Step 6: Write output ---
    await db.update_job(job_id, status="writing")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    safe_topic = "".join(c if c.isalnum() or c in " -" else "_" for c in topic)[:80]
    filename = f"local-{today}-{safe_topic.replace(' ', '-')}.md"

    try:
        from app import storage
        from app.config import RESEARCH_KB
        report = f"# Local Research: {topic}\n\n*{today} — Source: `{path}`*\n\n{content}\n"
        storage.write_text(RESEARCH_KB, filename, report)
    except Exception as exc:
        await db.update_job(job_id, status="error", error=f"Failed to write output: {exc}")
        return

    # Build source records for the job
    sources = [
        {
            "title": f["name"],
            "content": f"File: {f['rel_path']} ({f['size']} bytes)",
            "url": f"file://{f['path']}",
        }
        for f in top_files
    ]
    await db.save_sources(job_id, sources, round_num=1)

    # Extend the job's source_params with source_meta so the worker
    # (which only sees the DB row, not this function's locals) can
    # stamp the git fields onto the wiki article's frontmatter.
    import json as _json
    job_row = await db.get_job(job_id)
    existing_params = {}
    if job_row and job_row.get("source_params"):
        try:
            existing_params = _json.loads(job_row["source_params"]) or {}
        except (TypeError, ValueError):
            existing_params = {}
    existing_params["source_meta"] = source_frontmatter

    await db.update_job(
        job_id,
        status="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        sources_count=len(top_files),
        word_count=len(content.split()),
        content=content,
        source_params=_json.dumps(existing_params),
    )
    logger.info(
        "Local research [%d] complete: %d files, %d words",
        job_id, len(top_files), len(content.split()),
    )
