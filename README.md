# WikiDelve

**A self-hosted research knowledge base that researches itself.**

WikiDelve is a single-tenant FastAPI service that turns the open web (and your
own files) into a structured wiki you actually trust. You ask it about a topic,
it runs a multi-round research pipeline across nine different sources, the
critique pass strips out hallucinated citations, the article merges into your
wiki, and from then on its facts are reachable via a streaming RAG chat — and
the system goes looking for the next thing to research on its own.

It runs on your hardware. Your articles, your API keys, your infrastructure.

---

## What it does

### Knowledge base
- **Wikipedia-style browser** for markdown articles, with FTS5 full-text search,
  vector + knowledge-graph hybrid search, fuzzy autocomplete, and tag/category
  navigation across multiple isolated KBs.
- **Article version history** — every update writes a snapshot to
  `article_versions` before the file is overwritten, with on-demand unified
  diffs between any historical version and the current article.
- **Knowledge graph** — entity + relationship extraction per article, graph
  traversal for "find related articles," D3-rendered graph view, and a
  "Memory Palace" categorization layer that clusters articles into topical
  rooms (how-to / architecture / comparison / reference / etc.).
- **Inline confidence badges** — articles whose claims have been
  fact-checked show a colored confidence banner (green / amber / red) computed
  from the structured `article_claims` store.

### Research pipeline
- **Multi-source web research** — every research job hits **nine providers in
  parallel** (Serper, Tavily, arXiv, Hacker News, Wikipedia, Reddit, Crossref,
  Stack Exchange, Bluesky) plus deep page reads via headless browser
  (agent-browser → Jina → httpx fallback) and PDF/ePub text extraction.
- **Multi-round pipeline** — round 1 broad search → round 2 LLM-generated
  sub-question searches → round 3 verification searches → round 4 deep page
  reads + document downloads.
- **Per-source fact extraction** — Tier-1 sources go through a cheap parallel
  extraction pass that distills each page into bullet facts before synthesis.
- **Adaptive content budget** — Tier 1 sources get 8000 chars in the synthesis
  prompt, Tier 2 get 3000, Tier 3 get 1000.
- **Self-citation lint** — every `[label](url)` in the synthesized article is
  checked against the actual source set; hallucinated URLs are stripped while
  preserving the label text.
- **Two-pass critique** — after the draft, a critic LLM call returns structured
  JSON about which claims are supported / unsupported / missing. The result is
  appended to the article as a "Confidence Notes" section AND persisted to
  `article_claims` for periodic re-verification.
- **Reflection-aware fact-check** — periodic fact-check runs re-verify the
  stored claims instead of re-extracting them every time, dramatically reducing
  LLM cost.
- **Local research** — scan local files / folders / git repos and synthesize
  findings without needing any web search keys.

### Streaming chat (RAG)
- **Server-side RAG chat** at `POST /api/chat/sessions/{id}/ask` with hybrid
  retrieval, token-budgeted conversation memory, citation rules, and SSE
  streaming via `fetch()` + `Response.body.getReader()`.
- **Tool use** — the chat model can call `search_kb`, `get_article`,
  `find_related`, `get_graph_neighbors`, `enqueue_research`, and
  `refine_research` (which spawns a follow-up research job scoped to a
  specific section of an existing article).
- **Chat-gap feedback** — when the model emits a `[GAP] <topic>` sentinel,
  the topic is queued into auto-discovery so the KB grows around what users
  actually ask.

### Auto-discovery (continuous self-research)
- **Nine discovery strategies** — knowledge-graph entities, LLM follow-ups,
  contradictions between articles, stale articles in high-velocity tags,
  orphan entities, unanswered `?`-ending questions, sparse research-history
  clusters, TOC stubs, and broken `[[wikilinks]]`.
- **Per-KB Serper budget** — daily call cap, hourly enqueue cap, per-KB
  enabled flag, configurable strategy and seed-topic scoping. The enqueue
  loop runs hourly and only spawns research jobs while there's enough
  remaining budget for a full pipeline.
- **Background discovery from external feeds** — daily RSS feed sweeps,
  weekly sitemap crawls, weekly podcast feed scans (with audio →
  Whisper transcription → article synthesis).
- **Cooldown + dedup** — every queued candidate is checked against
  `find_related_article` (skip if a fuzzy match already exists) and a 7-day
  research cooldown.

### Quality engine
- **Enrichment passes** — find shallow articles and rewrite them with deeper
  content + better structure.
- **Crosslinking** — auto-detect mentions of other articles and convert them
  to `[[wikilinks]]`.
- **Freshness audits** — flag stale content (per article or batch) and
  optionally apply automatic updates.
- **Duplicate detection** + repair tools.
- **Wikilink integrity** — find and fix broken wikilinks across a KB.

### Per-KB configuration
- **Per-KB synthesis provider + model** — set `personal` to use Minimax and
  `work` to use Bedrock Claude Sonnet 4.5 (or any other combination) without
  restarts. NULL falls back to the global `LLM_PROVIDER` env var.
- **Per-KB persona** — prepend a custom system prompt to synthesis (e.g.
  "You are a senior security engineer writing for a SOC team") so different
  KBs speak in different voices.
- **Per-KB seed topics** — scope auto-discovery to specific topic areas so a
  work KB only mines its own subject matter.

### MCP server
A `fastmcp` server exposes ten tools to MCP-aware clients (Claude Code,
Cursor, etc.) so an external session can drive the wiki without the HTTP API:

- `search`, `get_article_content`, `list_knowledge_bases`, `list_articles`,
  `research_topic` (baseline)
- `chat_ask` — full RAG pipeline → grounded cited answer + sources
- `get_graph_neighbors` — walk the knowledge graph from any article
- `enqueue_auto_discovery` — trigger one-shot refill + enqueue cycle
- `list_topic_candidates` — inspect the auto-discovery pending queue
- `get_article_versions` — list historical snapshots

### GitHub + media ingestion
- **GitHub release tracking** — monitor configured repos, auto-research new
  releases, mirror READMEs and changelogs.
- **YouTube transcripts** — extract captions, synthesize structured articles.
- **Whisper transcription pipeline** — `app/transcribe.py` routes audio URLs
  through Groq (default) or OpenAI Whisper backends. Used by uncaptioned
  YouTube videos and the podcast feed importer.
- **Document ingestion** — direct PDF/ePub URLs, open directory crawls,
  Archive.org collections.

### Hosting
- **Tailscale sidecar** for automatic HTTPS over your tailnet — no port
  forwarding, no Caddy/nginx, no cert management.
- **Persistent queue** — Redis + arq for reliable async job processing.
- **Six worker replicas** by default, configurable via docker-compose.
- **SQLite or DynamoDB backend** (`DB_BACKEND` env var). Both are first-
  class: SQLite for single-host deployments, DynamoDB for cloud
  deployments with S3 article storage.
- **Local or S3 article storage** (`STORAGE_BACKEND` env var). The S3
  path keeps articles in `s3://{bucket}/{kb}/wiki/{slug}.md` with a
  matching DynamoDB metadata index (`ARTMETA#<kb>` rows) so list
  endpoints serve in milliseconds without scanning the bucket.
- **Hard CPU + memory caps** in `docker-compose.yml` — app gets 4 CPU
  cores + 2 GiB RAM, each worker gets 2 cores + 1.5 GiB, Redis gets
  1 core + 512 MiB.

### Security hardening
- **CSP with per-request nonce + strict-dynamic** — every `<script>`
  and `<style>` gets a fresh nonce, no `unsafe-inline`. Generated by
  the security-headers middleware and injected via a Jinja global so
  templates don't have to thread it through.
- **Full header set**: HSTS, X-Frame-Options, X-Content-Type-Options,
  Referrer-Policy, Permissions-Policy, cross-origin isolation headers.
  Server header replaced with `wikidelve`; uvicorn stock headers
  suppressed via `--no-server-header` / `--no-date-header`.
- **TrustedHostMiddleware** with an allowlist — only the configured
  hostnames + `localhost` + the docker-internal name are accepted.
- **Constant-time API key compare** via `hmac.compare_digest`.
- **Body size limit middleware** (4 MiB default, env override) to
  prevent memory-pressure DoS.
- **Rate limiting** — `@limiter.limit` decorators on `/api/research`,
  `/api/research/local`, `/api/search`, `/api/search/hybrid`, keyed on
  `(client_ip, kb)` so a hot KB can't starve other KBs.
- **Prompt-injection detector** in `retrieve_context` — scans every
  retrieved chunk for classic jailbreak markers (ignore previous,
  system role hijacks, reveal-prompt, chatml). Single-hit chunks are
  flagged + logged; multi-hit chunks are dropped outright.
- **SSE disconnect detection** — `sse_response()` polls
  `request.is_disconnected()` between yields so a closed browser tab
  cancels the upstream LLM stream immediately instead of burning
  tokens into the void.
- **Embedding circuit breaker** — hybrid search short-circuits the
  vector leg after 3 consecutive embedding-provider failures for a
  60 s cooldown, with a half-open probe on recovery. Search gracefully
  degrades to FTS-only instead of timing out on every query.

---

## Quick start

```bash
git clone https://github.com/ghndrx/wikidelve.git
cd wikidelve

cp .env.example .env
# Edit .env with your API keys (at minimum SERPER_API_KEY + MINIMAX_API_KEY)

# Optional: drop existing markdown into data/personal/wiki/
mkdir -p data/personal/wiki

docker compose up -d
```

Open `http://localhost:8888`. The wiki is ready, the FTS index is built, and
the worker is listening for research jobs.

### Trigger your first research

```bash
curl -X POST http://localhost:8888/api/research \
  -H 'Content-Type: application/json' \
  -d '{"topic": "Tokio async runtime architecture", "kb": "personal"}'
```

A job is queued; the worker picks it up, hits all nine source providers,
runs the synthesis + critique passes, and merges the resulting article into
`data/personal/wiki/` automatically.

### Try the chat

Visit `http://localhost:8888/chat`, click the "Chat: off" toggle in the
KB row to flip it on, and ask the wiki a question. The answer streams back
with cited sources from your articles.

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`
and edit; only the keys you actually need are required.

### Core API keys

| Variable | Required | Purpose |
|---|---|---|
| `SERPER_API_KEY` | yes (for web research) | Google search via serper.dev |
| `MINIMAX_API_KEY` | yes (default LLM) | Synthesis + entity extraction + chat |
| `TAVILY_API_KEY` | optional | Secondary search provider |
| `API_KEY` | optional | Bearer token gating POST/PUT/DELETE/PATCH endpoints |

### LLM provider

```bash
# Default: Minimax
LLM_PROVIDER=minimax
MINIMAX_API_KEY=...

# OR: AWS Bedrock (per-KB selection works for both)
LLM_PROVIDER=bedrock
BEDROCK_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0

# Auth: explicit keys, IAM role (EC2/ECS/EKS), or AWS_PROFILE — boto3 handles all three
```

### Auto-discovery

```bash
AUTO_DISCOVERY_ENABLED=true                # Global kill switch
SERPER_CALLS_PER_JOB_ESTIMATE=8            # How many Serper calls one full job costs
```

Per-KB configuration (enabled flag, daily budget, max-per-hour, strategy,
seed topics, llm sample size) lives in the `auto_discovery_config` table and
is editable live via `/admin#auto-discovery` or
`PUT /api/auto-discovery/config/{kb}`.

### Background discovery feeds

```bash
# Daily RSS sweep at 05:00 UTC
RSS_WATCHES='[
  {"kb": "personal", "url": "https://lobste.rs/rss"},
  {"kb": "personal", "url": "https://blog.rust-lang.org/feed.xml"}
]'

# Weekly sitemap crawl, Mondays at 05:30 UTC
SITEMAP_WATCHES='[
  {"kb": "personal", "url": "https://kubernetes.io/docs/sitemap.xml", "label": "k8s docs"}
]'

# Weekly podcast feed scan, Tuesdays at 05:30 UTC (transcribed via Whisper)
PODCAST_WATCHES='[
  {"kb": "personal", "url": "https://changelog.com/podcast/feed"}
]'

# Local file/git/code research, daily at 04:00 UTC
LOCAL_RESEARCH_WATCHES='[
  {"topic": "infra config", "path": "/code/infra"},
  {"topic": "api service", "path": "/code/api", "pattern": "*.py"}
]'
```

### Whisper transcription

```bash
WHISPER_BACKEND=groq         # or "openai"
GROQ_API_KEY=gsk_...         # required for groq backend
# OR
OPENAI_API_KEY=sk-...        # required for openai backend
```

### Database

```bash
DB_BACKEND=sqlite            # "sqlite" (default) or "dynamodb"
DB_PATH=/kb/wikidelve.db     # SQLite only
DB_DYNAMO_TABLE=wikidelve    # DynamoDB only — single-table name
```

The DynamoDB backend is a drop-in replacement — every helper in
`app/db.py` has a matching implementation in `app/db_dynamo.py` that
the module trailer swaps in when `DB_BACKEND=dynamodb`. Uses a single
table with composite keys:

- `PK=JOB#<id>` / `SK=META` — research jobs (GSI1 for newest-first).
- `PK=SRC#<job_id>` / `SK=ROUND#<n>#<source_id>` — sources for a job.
- `PK=ARTMETA#<kb>` / `SK=<slug>` — article metadata index (see below).
- `PK=CHAT#<session_id>` / `SK=MSG#<ts>#<id>` — chat history.
- Plus classifications, rooms, auto-discovery config, KB settings,
  topic candidates, audit events, Serper call log, claims, and
  knowledge-graph entities/edges.

### Storage backend (articles)

```bash
STORAGE_BACKEND=local        # "local" (default) or "s3"
S3_BUCKET=my-wikidelve-bucket
S3_PREFIX=wikidelve          # optional key prefix
```

When `STORAGE_BACKEND=s3`, articles live at
`s3://{bucket}/{prefix}/{kb}/wiki/{slug}.md`. List endpoints use the
DynamoDB metadata index (see `ARTICLE_META_INDEX` below) so they
don't pay the bucket-scan on every request.

### Article metadata index

```bash
ARTICLE_META_INDEX=true      # read list endpoints from the DynamoDB index
```

With this flag on, `get_articles(kb)` serves from the
`PK=ARTMETA#<kb>` rows in a single DynamoDB query instead of scanning
the S3 prefix and parsing every frontmatter block. List endpoints
(`/api/articles`, `/api/kbs`, `/api/status`) drop from multi-second
to milliseconds on a 700-article KB. Writes (create / update /
delete) fire a best-effort write-through to keep the index in sync
without blocking the primary S3 write.

To backfill the index on an existing deployment:

```bash
docker exec -w /app kb-service python3 -m scripts.backfill_article_meta
```

### Knowledge bases

The default KB is `personal`. Add more via env vars or by creating
directories under `KB_ROOT`:

```bash
KB_ROOT=/kb                          # base directory (default)
EXTRA_KB_WORK_PATH=/kb/work          # registers a "work" KB
EXTRA_KB_RESEARCH_PATH=/kb/research  # registers a "research" KB
```

Or just `mkdir /kb/<name>/wiki` — KBs with a `wiki/` subdirectory are
auto-discovered on startup.

---

## Architecture

```
                          ┌────────────────────┐
                          │   Tailscale (opt)  │
                          │  HTTPS sidecar     │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │   FastAPI (app)    │
                          │   Web UI + REST    │
                          │   + SSE streaming  │
                          │   + MCP server     │
                          └────┬──────────┬────┘
                               │          │
                ┌──────────────▼──┐    ┌──▼─────────────┐
                │   SQLite        │    │   Redis        │
                │   FTS5 + JSON   │    │   arq queue    │
                │   embeddings    │    └──┬─────────────┘
                │   KG + claims   │       │
                │   versions      │       │
                └────────▲────────┘       │
                         │       ┌────────▼────────┐
                         │       │  Workers (x3)   │
                         │       │  arq            │
                         │       └────┬──────┬─────┘
                         │            │      │
                         └────────────┘      ▼
                                        ┌─────────────────────────┐
                                        │  External services      │
                                        │  Serper, Tavily, arXiv, │
                                        │  HN, Wikipedia, Reddit, │
                                        │  Crossref, StackEx,     │
                                        │  Bluesky, Minimax,      │
                                        │  Bedrock, Whisper,      │
                                        │  agent-browser, GitHub  │
                                        └─────────────────────────┘
```

| Component | Module | Notes |
|---|---|---|
| HTTP + UI | `app/main.py` | FastAPI routes, Jinja templates, SSE streaming |
| Research pipeline | `app/research.py` | Rounds 1–4 + extraction + critique + lint + synthesis |
| Source providers | `app/sources/` | Pluggable; one file per provider |
| Auto-discovery | `app/auto_discovery.py` | Nine self-mining strategies + cron dispatch |
| Chat RAG | `app/chat.py` | Retrieval composer, prompt builder, tool dispatch |
| LLM abstraction | `app/llm.py` | Minimax + Bedrock; chat / streaming / tools / embeddings |
| Knowledge graph | `app/knowledge_graph.py` | Entity + relationship extraction, BFS traversal |
| Hybrid search | `app/hybrid_search.py` | FTS5 + vector + graph + palace, Reciprocal Rank Fusion |
| Embeddings | `app/embeddings.py` | Per-article vectors stored as JSON in SQLite |
| Quality engine | `app/quality.py` | Enrich, crosslink, fact-check, freshness |
| Workers | `app/worker.py` | arq tasks + cron registrations |
| Audio transcription | `app/transcribe.py` | Whisper (Groq / OpenAI) backend |
| MCP server | `app/mcp_server.py` | Ten tools for external Claude/Cursor sessions |
| Database | `app/db.py` | SQLite schema + helpers; `app/db_dynamo.py` stub |

---

## REST API (high-level)

### Wiki + search
- `GET /` — main page, article list with filters
- `GET /wiki/{kb}/{slug}` — render an article (with confidence banner +
  suggestions panel)
- `GET /api/search?q=...` — FTS5 search
- `GET /api/search/hybrid?q=...&kb=...` — hybrid FTS + vector + graph search
- `GET /api/articles/{kb}/{slug}` — article JSON
- `GET /api/articles/{kb}/{slug}/related` — related articles
- `GET /api/articles/{kb}/{slug}/suggestions` — "you might also research"
- `GET /api/articles/{kb}/{slug}/versions` — list snapshots
- `GET /api/articles/{kb}/{slug}/versions/{id}/diff` — unified diff
- `DELETE /api/articles/{kb}/{slug}` — delete an article

### Research
- `POST /api/research` — kick off web research (`{topic, kb, review_sources?}`)
- `POST /api/research/local` — kick off local file/git research
- `GET /api/research/status/{job_id}` — job status
- `GET /api/research/jobs` — recent jobs
- `GET /api/research/sources/{job_id}` — sources for a job
- `PUT /api/research/sources/{job_id}` — select/reject sources for review
- `POST /api/research/synthesize/{job_id}` — synthesize from selected sources
- `GET /research/review/{job_id}` — NotebookLM-style source-review UI

### Chat
- `GET /chat` — chat UI
- `GET /api/chat/sessions` — list sessions
- `GET /api/chat/sessions/{id}` — fetch session messages
- `POST /api/chat/sessions/{id}/messages` — append a message
- `POST /api/chat/sessions/{id}/ask` — **streaming RAG answer (SSE)**
- `DELETE /api/chat/sessions/{id}` — delete session

### Auto-discovery
- `GET /api/auto-discovery/status` — per-KB status JSON
- `PUT /api/auto-discovery/config/{kb}` — upsert per-KB config
- `POST /api/auto-discovery/run/{kb}` — manual one-shot refill + enqueue

### Per-KB settings
- `GET /api/kb-settings/{kb}` — get persona / synthesis provider / model
- `PUT /api/kb-settings/{kb}` — upsert

### Knowledge graph
- `POST /api/graph/build` — rebuild the graph for a KB
- `GET /api/graph/data` — graph data for D3 view
- `GET /api/graph/related/{kb}/{slug}` — graph-traversed neighbours

### Quality
- `POST /api/quality/pass` — quality pass on a KB
- `POST /api/quality/enrich/{kb}/{slug}` — enrich a single article
- `POST /api/quality/crosslink/{kb}/{slug}` — add crosslinks
- `POST /api/quality/factcheck/{kb}/{slug}` — fact-check an article
- `POST /api/quality/freshness-batch` — batch freshness audit
- `GET /api/quality/scores/{kb}` — per-article quality scores
- `POST /api/quality/wikilinks/{kb}` — fix broken wikilinks

### Admin
- `GET /admin` — tabbed admin dashboard (Jobs / Health / Quality /
  Duplicates / Actions / Auto-Discovery / KB Settings). Deep-link
  via `#fragment`. Only the active tab renders. Live job table with
  auto-refresh, filter, cursor paging, per-row "Smart retry".
- `GET /api/status` — job stats + KB totals
- `GET /api/research/jobs` — jobs list with `?limit=`, `?status=`,
  `?compact=` (drops the heavy content column), and `?before_id=`
  cursor paging
- `POST /api/research/smart-retry/{job_id}` — classify the failure
  and LLM-rewrite the topic before re-enqueuing as a new job
- `GET /api/_test/sse` — SSE smoke-test route

### Keyboard shortcuts
- **`⌘K` / `Ctrl-K`** — global command palette. Fuzzy search across
  every article title + all navigation pages. Arrow-key navigation,
  enter to jump, escape to dismiss. Articles are lazy-loaded on
  first open and cached for the session.

---

## Tailscale HTTPS sidecar (optional)

Automatic HTTPS over your tailnet with no port forwarding, no nginx, no
cert management.

1. Generate an auth key in
   [Tailscale Admin → Settings → Keys → Auth keys](https://login.tailscale.com/admin/settings/keys)
   with **Reusable: off**, **Pre-approved: on**, **Ephemeral: off**.
2. Add to `.env`:
   ```bash
   TS_AUTHKEY=tskey-auth-your-key-here
   ```
3. Start the sidecar:
   ```bash
   docker compose up -d tailscale
   ```

WikiDelve becomes available at `https://wikidelve.<your-tailnet>.ts.net`
from any device on your tailnet. To change the hostname, edit the
`hostname` field of the `tailscale` service in `docker-compose.yml`.

---

## Development

### Run the test suite

```bash
python3 -m pytest tests/unit/ -q         # unit tests, no server needed
python3 -m pytest tests/e2e/ -q          # e2e tests against localhost:8888
python3 -m pytest tests/e2e/ -q -m heavy # include the opt-in slow tests
```

The unit suite covers the LLM layer, sources package, synthesis
quality helpers, self-mining strategies, chat RAG primitives,
capability layer, auto-discovery, knowledge graph, hybrid search,
frontmatter parsing, wiki helpers, storage backends, prompt injection
detection, SSE disconnect handling, embedding circuit breaker, CSP
nonce rotation, rate-limit key composition, and more.

The e2e suite runs against a live `localhost:8888` and covers the
HTTP API surface end-to-end. Tests that would enqueue expensive
background work are tagged `@pytest.mark.heavy` and opt-in.

### Run a single test file

```bash
python3 -m pytest tests/unit/test_sources_providers.py -v
```

### End-to-end tests

```bash
python3 -m pytest tests/e2e/ -q
```

### Hot-reload during development

```bash
docker compose up -d redis
uvicorn app.main:app --reload --port 8888
```

In another shell:
```bash
arq app.worker.WorkerSettings
```

### Rebuild after code changes

```bash
docker compose up -d --build app worker
```

This rebuilds only the app + worker images without touching Redis or
Tailscale.

---

## License

MIT
