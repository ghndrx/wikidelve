"""
Configuration: env vars, API endpoints, constants, prompts.

                    ╔═══════════════════════╗
                    ║  internal codename:    ║
                    ║  ✦ G R O K M A X X ✦  ║
                    ╚═══════════════════════╝
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider -----------------------------------------------------------
# "minimax" (default) or "bedrock"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "minimax").strip().lower()

# --- API Keys ---------------------------------------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "").strip()

# --- AWS Bedrock ------------------------------------------------------------
# Three auth methods (in priority order):
#   1. BEDROCK_API_KEY — Bearer token auth (no IAM/boto3 needed)
#   2. AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY — explicit IAM keys
#   3. boto3 credential chain — IAM roles, ~/.aws/credentials, AWS_PROFILE
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY", os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")).strip()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "").strip()
BEDROCK_REGION = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "us-east-1")).strip()
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0").strip()
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0").strip()

# --- Paths ------------------------------------------------------------------

KB_ROOT = Path(os.getenv("KB_ROOT", "/kb"))

KB_DIRS: dict[str, Path] = {
    "personal": Path(os.getenv("PERSONAL_KB_PATH", str(KB_ROOT / "personal"))),
}

# Optional: add extra KBs via EXTRA_KB_<NAME>_PATH env vars
# e.g. EXTRA_KB_WORK_PATH=/kb/work → KB_DIRS["work"] = Path("/kb/work")
for key, val in os.environ.items():
    if key.startswith("EXTRA_KB_") and key.endswith("_PATH"):
        name = key[len("EXTRA_KB_"):-len("_PATH")].lower()
        if name and val.strip():
            KB_DIRS[name] = Path(val.strip())

# Auto-discover KB directories under KB_ROOT that have a wiki/ subfolder
# This picks up dynamically-created KBs without needing env vars
if KB_ROOT.exists():
    _RESERVED = {"research", "downloads"}
    for child in KB_ROOT.iterdir():
        if (child.is_dir()
                and child.name not in _RESERVED
                and not child.name.startswith(".")
                and child.name not in KB_DIRS
                and (child / "wiki").exists()):
            KB_DIRS[child.name] = child

RESEARCH_DIR = Path(os.getenv("RESEARCH_PATH", str(KB_ROOT / "research")))
DB_PATH = Path(os.getenv("DB_PATH", str(KB_ROOT / "wikidelve.db")))

# Pseudo-KB name used by the storage backend to hold research output
# documents (not articles). Scratch files like media downloads stay under
# RESEARCH_DIR on disk; the .md outputs flow through storage.
RESEARCH_KB = "_research"


def register_kb(name: str) -> Path:
    """Create and register a new knowledge base at runtime.

    Creates the directory structure and adds it to KB_DIRS.
    Returns the KB path.
    """
    safe_name = "".join(c if c.isalnum() or c == "-" else "" for c in name.lower()).strip("-")
    if not safe_name:
        raise ValueError("Invalid KB name")
    if safe_name in KB_DIRS:
        return KB_DIRS[safe_name]

    kb_path = KB_ROOT / safe_name
    (kb_path / "wiki").mkdir(parents=True, exist_ok=True)
    (kb_path / "raw").mkdir(parents=True, exist_ok=True)
    KB_DIRS[safe_name] = kb_path
    return kb_path

# --- Redis ------------------------------------------------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
ARQ_QUEUE_NAME = "wikidelve"

# --- LLM / Search endpoints -------------------------------------------------

MINIMAX_BASE = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.7"

# Kimi / Moonshot AI — OpenAI-compatible API with 256K context
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "").strip()
# KIMI_BASE is now env-overridable. Moonshot has multiple regional
# endpoints (api.moonshot.ai international, api.moonshot.cn China),
# and some hosted-Kimi services use an OpenAI-compatible proxy with
# a different base URL. Keep the .ai default for backward compat.
KIMI_BASE = os.getenv("KIMI_BASE", "https://api.moonshot.ai/v1").strip()
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.5").strip()
TAVILY_URL = "https://api.tavily.com/search"
SERPER_URL = "https://google.serper.dev/search"

# --- Source quality tiers ----------------------------------------------------

TIER1_DOMAINS = {
    "gov", "edu", "mil",
}

TIER1_SUBSTRINGS = {
    "docs.python.org", "docs.microsoft.com", "developer.mozilla.org",
    "developer.apple.com", "cloud.google.com", "docs.aws.amazon.com",
    "learn.microsoft.com", "kubernetes.io/docs", "react.dev",
    "arxiv.org", "ieee.org", "acm.org", "nature.com", "sciencedirect.com",
    "wikipedia.org", "github.com",
    "martinfowler.com", "engineering.fb.com", "blog.google",
    "netflix.tech", "stripe.com/blog", "cloudflare.com/blog",
}

TIER2_SUBSTRINGS = {
    "stackoverflow.com", "stackexchange.com",
    "medium.com", "dev.to", "hashnode.dev",
    "bbc.com", "bbc.co.uk", "reuters.com", "nytimes.com",
    "theguardian.com", "arstechnica.com", "wired.com",
    "techcrunch.com", "theverge.com", "hackernews",
    "infoq.com", "dzone.com", "baeldung.com",
    "digitalocean.com/community", "freecodecamp.org",
}

# --- Research cooldown -------------------------------------------------------

COOLDOWN_DAYS = 7

# --- Synthesis prompt --------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a senior researcher writing for a personal knowledge base. "
    "Your role is to produce accurate, well-sourced, structured research documents "
    "that are useful for both human readers and AI agents pulling context. "
    "IMPORTANT: Write ONLY in English. Never include non-English text, characters, or translations.\n\n"
    "THINKING PROCESS (reason step by step before writing):\n"
    "1. Filter: Discard any sources that are irrelevant (wrong topic, spam, unrelated)\n"
    "2. Evaluate: Which sources are most authoritative? Where do they agree/disagree?\n"
    "3. Identify gaps: What obvious aspects of this topic did NO source cover?\n"
    "4. Prioritize: What are the 3-5 most important things a reader needs to know?\n"
    "5. Check yourself: For each key claim you plan to include, can you point to a specific source? If not, it goes in Limitations, not Key Findings\n"
    "6. Plan structure: executive summary, key findings organized by importance, recommendations\n\n"
    "STRICT SOURCE RULES:\n"
    "- Every factual claim MUST cite its source: [Source Name](url)\n"
    "- Do NOT fabricate sources. Only cite URLs from the search results\n"
    "- Do NOT use your training data to fill gaps. If a fact isn't in the sources, do NOT include it\n"
    "- If a comparison table would require data not in the sources, do NOT guess or approximate. "
    "Either skip the table or only include rows you can cite\n"
    "- Never write 'approximate based on general knowledge' or similar. If you can't cite it, leave it out\n"
    "- Ignore sources that are clearly irrelevant to the topic (wrong subject, spam, unrelated)\n\n"
    "WRITING RULES:\n"
    "- Start with a 2-3 sentence executive summary answering the core question\n"
    "- Use ## headers for major sections, ### for subsections\n"
    "- Include code examples where relevant (with language tags on code blocks)\n"
    "- If sources conflict, note both sides and which is more authoritative\n"
    "- If something cannot be determined from sources, say so in Limitations\n"
    "- Prefer Tier 1 sources over lower-tier sources\n"
    "- Be precise with numbers, dates, and version numbers\n"
    "- Target 1000-2000 words. Comprehensive but not padded\n\n"
    "CONTENT QUALITY:\n"
    "- Write about what matters, not everything you found. A city article should cover "
    "why someone would care about this place, not list every electoral district\n"
    "- Distinguish between high-value findings and reference data. Don't dump exhaustive "
    "lists (every hill elevation, every distance, every config option)\n"
    "- Tables should contain sourced data only. Never fill table cells with guesses\n"
    "- Practical recommendations should be specific and actionable, not generic\n"
    "- Adapt structure to the content. Non-technical topics skip code blocks. "
    "Geographic topics focus on what makes the place interesting, not raw statistics\n\n"
    "REQUIRED SECTIONS:\n"
    "- ## Executive Summary (2-3 sentences)\n"
    "- ## Key Findings (organized by sub-topic with ### headers)\n"
    "- ## Practical Recommendations (actionable bullet points)\n"
    "- ## Limitations (what the sources didn't cover, obvious gaps)\n"
    "- ## Key Takeaways (3-5 bullets)\n\n"
    "FORMATTING:\n"
    "- Clean markdown. No stray formatting artifacts (**, --, etc.)\n"
    "- Tables only when they genuinely clarify comparisons with sourced data\n"
    "- End cleanly. No trailing fragments or incomplete sections\n"
    "- Complete every sentence. If you're running out of tokens, close the current "
    "section cleanly rather than cutting off mid-sentence"
)

# --- Auth / Security --------------------------------------------------------

API_KEY = os.getenv("API_KEY", "").strip()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").strip()

# Rate-limit env knobs. slowapi strings like "30/minute" or "100/hour".
# Consumed when the research + search routes are decorated with
# @limiter.limit(...). Kept as config knobs so deployments can tune them
# without code changes — used by auto-discovery throttling + the search
# providers (Serper, Tavily) to back-pressure a busy KB.
RATE_LIMIT_RESEARCH = os.getenv("RATE_LIMIT_RESEARCH", "30/minute")
RATE_LIMIT_SEARCH = os.getenv("RATE_LIMIT_SEARCH", "100/minute")

# --- Retry configuration ----------------------------------------------------

RESEARCH_MAX_RETRIES = 3
RESEARCH_RETRY_DELAY = 15  # seconds (faster retry)
MINIMAX_TIMEOUT = 300  # seconds (5 min — Minimax M2.7 can be slow)

# Transient-error retry for the llm_chat path. When N workers hit the
# same provider concurrently (e.g. Bedrock on-demand) we see 429/503
# bursts; a single attempt fails a whole job. Retries are capped and
# use exponential backoff with jitter, plus Retry-After honour.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
LLM_RETRY_MAX_DELAY = float(os.getenv("LLM_RETRY_MAX_DELAY", "30.0"))

# --- Auto-discovery ---------------------------------------------------------
# Global kill switch. Per-KB settings (enabled flag, budget, strategy, seed
# topics, etc.) live in the auto_discovery_config table so they can be
# edited live without a restart.
AUTO_DISCOVERY_ENABLED = os.getenv("AUTO_DISCOVERY_ENABLED", "false").strip().lower() == "true"
# How many Serper calls a single research_task is expected to consume. Used
# to decide whether the remaining daily budget is enough for another job.
SERPER_CALLS_PER_JOB_ESTIMATE = int(os.getenv("SERPER_CALLS_PER_JOB_ESTIMATE", "8"))
