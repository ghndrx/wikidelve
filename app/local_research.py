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

    # Repo description from remote
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=git_dir, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            context["remote_url"] = result.stdout.strip()
    except Exception:
        pass

    return context


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

    await db.update_job(
        job_id,
        status="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        sources_count=len(top_files),
        word_count=len(content.split()),
        content=content,
    )
    logger.info(
        "Local research [%d] complete: %d files, %d words",
        job_id, len(top_files), len(content.split()),
    )
