"""
KB Service -- FastAPI app with web UI + API routes.

All business logic lives in sibling modules (research, wiki, search, db).
This module handles HTTP routing, template rendering, and job dispatch via arq.
"""

import asyncio
import contextvars
import hmac
import json
import logging
import os
import secrets as _secrets
from contextlib import asynccontextmanager

import markdown
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import (
    REDIS_HOST,
    REDIS_PORT,
    ARQ_QUEUE_NAME,
    API_KEY,
    CORS_ORIGINS,
    RATE_LIMIT_RESEARCH,
    RATE_LIMIT_SEARCH,
)
from app import db, storage
from app.logging_config import setup_logging, log_chat_interaction

setup_logging()
from app.wiki import (
    get_articles,
    get_article,
    delete_article,
    parse_frontmatter,
    read_article_text,
)
from app.search import search_kb, search_fts, build_search_index
from app.hybrid_search import hybrid_search
from app.knowledge_graph import get_graph_data, get_entity_articles, get_related_by_graph

logger = logging.getLogger("kb-service")


# --- Lifespan ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + Redis pool + storage backend. Shutdown: close pool."""
    await db.init_db()
    app.state.redis = await create_pool(
        RedisSettings(host=REDIS_HOST, port=REDIS_PORT),
    )
    logger.info("App started, DB initialized, Redis pool ready")

    # Build FTS search index in the background so the HTTP port comes
    # up immediately. Cold-start rebuild from S3 takes ~20s for 700 docs
    # and we don't want docker-compose health checks to race it.
    async def _bg_build_index():
        try:
            logger.info("FTS build: starting background rebuild")
            count = await build_search_index()
            logger.info("FTS index built: %d articles", count)
        except Exception as exc:
            logger.exception("FTS index build failed: %s", exc)

    async def _bg_warm_palace():
        try:
            from app.palace import generate_palace_map
            logger.info("Palace: warming cache")
            await generate_palace_map()
            logger.info("Palace: cache warm")
        except Exception as exc:
            logger.warning("Palace warmup failed: %s", exc)

    # Hold references so tasks aren't GC'd before they start.
    app.state.fts_task = asyncio.create_task(_bg_build_index())
    app.state.palace_task = asyncio.create_task(_bg_warm_palace())

    # Re-queue stuck jobs that were lost when Redis restarted
    logger.info("Fetching stuck jobs...")
    try:
        stuck_jobs = await asyncio.wait_for(db.get_stuck_jobs(), timeout=20)
    except asyncio.TimeoutError:
        logger.warning("get_stuck_jobs() timed out after 20s, skipping")
        stuck_jobs = []
    logger.info("Stuck jobs: %d", len(stuck_jobs))
    if stuck_jobs:
        for job in stuck_jobs:
            try:
                await app.state.redis.enqueue_job(
                    "research_task", job["topic"], job["id"],
                    _queue_name=ARQ_QUEUE_NAME,
                )
                logger.info("Re-queued stuck job %d: %s", job["id"], job["topic"][:50])
            except Exception as exc:
                logger.warning("Failed to re-queue job %d: %s", job["id"], exc)
        logger.info("Re-queued %d stuck jobs", len(stuck_jobs))

    yield

    await app.state.redis.close()
    from app.http_client import close_http_client
    await close_http_client()


app = FastAPI(title="KB Service", version="2.0.0", lifespan=lifespan)

from app.internal_tools import router as _internal_tools_router
app.include_router(_internal_tools_router)

# --- Rate limiting ---
def _rate_limit_key(request: Request) -> str:
    """Compose the rate-limit bucket from (client IP, KB).

    Keying on IP alone collapses to a single bucket behind shared
    egress / Tailscale, so one busy tab can starve every other request
    against the same KB. Include the KB path/query so a hot KB can't
    block activity against a different one.
    """
    ip = get_remote_address(request) or "anon"
    kb = request.path_params.get("kb") if request.path_params else None
    if not kb and request.path_params:
        kb = request.path_params.get("kb_name")
    if not kb:
        kb = request.query_params.get("kb") or "_"
    return f"{ip}|{kb}"


limiter = Limiter(key_func=_rate_limit_key)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})


# --- Host allowlist ---
#
# Rejects requests whose Host header isn't on the allowlist, preventing
# Host-header injection and host-based cache poisoning. The default
# covers localhost (for curl + tests) and the docker-internal hostnames
# workers hit via the compose network. For Tailscale / public deploys,
# set ALLOWED_HOSTS="wikidelve.<your-tailnet>.ts.net,*.<your-tailnet>.ts.net,..."
# in .env.
_default_allowed_hosts = "localhost,127.0.0.1,kb-service,wikidelve,testserver"
ALLOWED_HOSTS = [
    h.strip()
    for h in os.getenv("ALLOWED_HOSTS", _default_allowed_hosts).split(",")
    if h.strip()
]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# --- CORS ---
#
# When an API_KEY is set we treat the deployment as production and
# collapse the CORS allowlist onto the Host allowlist so a stray "*"
# can't leak credentials cross-origin. Without an API_KEY (dev /
# open-mode) we honour CORS_ORIGINS as-is.
_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
if API_KEY and _cors_origins == ["*"]:
    _cors_origins = [
        f"https://{h}" for h in ALLOWED_HOSTS if not h.startswith("*") and h not in ("testserver",)
    ] + [f"http://{h}" for h in ("localhost", "127.0.0.1")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Body size limit --------------------------------------------------------
#
# Reject requests whose Content-Length exceeds MAX_REQUEST_BODY_BYTES
# (default 4 MiB). Prevents trivial memory-pressure DoS via huge JSON
# bodies and matches typical enterprise egress inspection rules.
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(4 * 1024 * 1024)))


@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length"},
            )
    return await call_next(request)


# --- API key auth middleware ---
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require Bearer token on mutating requests when API_KEY is set.

    Uses ``hmac.compare_digest`` for a constant-time comparison so the
    key length / contents can't be probed via a timing side-channel.
    """
    # Internal endpoints (kimi-bridge callbacks) authenticate via their
    # own x-kimi-bridge-secret header. Skip the Bearer-token gate so the
    # sidecar doesn't need both credentials.
    if request.url.path.startswith("/api/internal/"):
        return await call_next(request)
    if API_KEY and request.method in ("POST", "PUT", "DELETE", "PATCH"):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
        provided = auth_header[7:]
        if not hmac.compare_digest(provided.encode("utf-8"), API_KEY.encode("utf-8")):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


# --- Security headers --------------------------------------------------------
#
# A fresh per-request CSP nonce gets stamped into script / style tags by
# Jinja (see ``csp_nonce`` global registered below) and into the
# Content-Security-Policy header here. ``strict-dynamic`` means any
# script loaded by a trusted (nonced) script also runs — so the cdnjs
# highlight.js tag just needs a nonce in base.html, and bundlers can
# later load more modules without us having to whitelist their URLs.
_csp_nonce_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "csp_nonce", default=""
)


def _get_csp_nonce() -> str:
    """Jinja global: resolve the current request's CSP nonce on render."""
    return _csp_nonce_var.get()


_STATIC_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _build_csp(nonce: str) -> str:
    # script-src stays strict: nonce + strict-dynamic, no unsafe-inline,
    # so injected <script> blocks can't execute. The CSP inline-handler
    # delegation shim in base.html rewires onclick= attributes to real
    # addEventListener calls from a nonced block, so we still get
    # interactivity without loosening script policy.
    #
    # style-src is intentionally looser: we allow 'unsafe-inline' so
    # the hundreds of inline style="..." attributes across the
    # templates render. Inline styles can't execute code — the worst
    # case is layout / colour manipulation or very contrived CSS
    # exfiltration, which is out of scope for a single-user
    # Tailscale-gated deployment.
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}' 'strict-dynamic' https:; "
        f"style-src 'self' 'nonce-{nonce}' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' data: https://cdnjs.cloudflare.com; "
        "connect-src 'self'; "
        "frame-ancestors 'self'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "object-src 'none'"
    )


@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    """Set a per-request trace_id via contextvar so every LLM call in
    the request is automatically nested under the same root trace."""
    from app.tracing import start_trace, end_trace
    token = start_trace(request)
    try:
        return await call_next(request)
    finally:
        end_trace(token)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request latency + count per route template for /metrics."""
    import time as _time
    start = _time.monotonic()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        try:
            from app import metrics as _metrics
            # Use the matched route template (e.g. /api/articles/{kb_name}/{slug})
            # so we don't cardinality-explode on unique slugs.
            route = request.scope.get("route")
            path_template = getattr(route, "path", None) or request.url.path
            labels = {
                "method": request.method,
                "path": path_template,
                "status": status,
            }
            _metrics.inc_counter(
                "kb_http_requests_total", 1.0, labels,
                help="HTTP requests by method/path/status",
            )
            _metrics.observe_histogram(
                "kb_http_request_duration_seconds",
                _time.monotonic() - start,
                {"method": request.method, "path": path_template},
                help="HTTP request duration in seconds",
            )
        except Exception:
            pass


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Set hardening headers, generate a per-request CSP nonce, and strip
    disclosive headers from every response."""
    nonce = _secrets.token_urlsafe(18)
    token = _csp_nonce_var.set(nonce)
    request.state.csp_nonce = nonce
    try:
        response = await call_next(request)
    finally:
        _csp_nonce_var.reset(token)

    for key, value in _STATIC_SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    # Sandbox routes (/sandbox/...) set their own MUCH stricter CSP
    # (default-src 'self' data:; connect-src 'none'; ...) — overwriting
    # it with the global site CSP would let scaffolded HTML phone home
    # via connect-src 'self'. Skip the global CSP write for those
    # routes; the route-handler-set value flows through unchanged.
    if not request.url.path.startswith("/sandbox/"):
        response.headers["Content-Security-Policy"] = _build_csp(nonce)
    # Drop anything that leaks infra details. uvicorn emits ``server:
    # uvicorn`` by default — replacing it with a generic token stops
    # fingerprinting without breaking HTTP/1.1 compliance.
    response.headers["Server"] = "wikidelve"
    for leaky in ("X-Powered-By", "X-AspNet-Version", "X-AspNetMvc-Version"):
        if leaky in response.headers:
            del response.headers[leaky]
    return response


jinja_env = Environment(loader=FileSystemLoader("templates"), autoescape=True)
jinja_env.globals["csp_nonce"] = _get_csp_nonce

# Only mount /static if the directory actually exists. The app currently
# pulls its vendor assets from CDNs, so the dir is optional — drop files
# in later and restart to expose them.
from pathlib import Path as _Path
if _Path("static").is_dir():
    app.mount("/static", StaticFiles(directory="static"), name="static")


def render(template_name: str, **context) -> HTMLResponse:
    """Render a Jinja2 template without Starlette's broken cache."""
    tmpl = jinja_env.get_template(template_name)
    html = tmpl.render(**context)
    return HTMLResponse(html)


# --- Web UI Routes ----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    all_articles: list[dict] = []
    stats: dict[str, dict] = {}
    for kb_name in storage.list_kbs():
        articles = await asyncio.to_thread(get_articles, kb_name)
        all_articles.extend(articles)
        stats[kb_name] = {
            "count": len(articles),
            "words": sum(a["word_count"] for a in articles),
        }

    # Group by tags
    tag_groups: dict[str, list] = {}
    for a in all_articles:
        for tag in a.get("tags", [])[:2]:
            tag_groups.setdefault(tag, []).append(a)

    return render(
        "index.html",
        articles=all_articles,
        tag_groups=dict(sorted(tag_groups.items())),
        total_articles=len(all_articles),
    )


@app.get("/wiki/{kb_name}/{slug}", response_class=HTMLResponse)
async def view_article(request: Request, kb_name: str, slug: str):
    article = get_article(kb_name, slug)
    if not article:
        if kb_name not in storage.list_kbs():
            raise HTTPException(status_code=404, detail=f"Unknown KB: {kb_name}")
        pretty_title = slug.replace("-", " ").title()
        suggestions: list[dict] = []
        try:
            hits = await hybrid_search(pretty_title, kb_name=kb_name, limit=6)
            for h in hits:
                if h.get("slug") == slug:
                    continue
                suggestions.append({
                    "slug": h.get("slug"),
                    "title": h.get("title") or h.get("slug", "").replace("-", " ").title(),
                })
        except Exception as exc:
            logger.debug("hybrid_search for missing article failed: %s", exc)
        return render(
            "missing_article.html",
            kb=kb_name,
            slug=slug,
            pretty_title=pretty_title,
            suggestions=suggestions[:6],
        )

    # Confidence summary from article_claims. Empty / errored fetches
    # degrade gracefully — the template just hides the badge if confidence
    # is None.
    confidence = None
    try:
        claims = await db.get_claims_for_article(kb_name, slug)
        if claims:
            verified = sum(1 for c in claims if c.get("status") == "verified")
            outdated = sum(1 for c in claims if c.get("status") == "outdated")
            unverified = sum(
                1 for c in claims
                if c.get("status") in ("unverified", "partially_correct", None)
            )
            total = len(claims)
            avg_conf = (
                sum(float(c.get("confidence") or 0) for c in claims) / total
                if total else 0.0
            )
            if avg_conf >= 0.75:
                level = "high"
                emoji = "✓"
            elif avg_conf >= 0.5:
                level = "medium"
                emoji = "~"
            else:
                level = "low"
                emoji = "⚠"
            confidence = {
                "level": level,
                "emoji": emoji,
                "verified": verified,
                "unverified": unverified,
                "outdated": outdated,
                "total": total,
                "avg_confidence": round(avg_conf, 2),
            }
    except Exception as exc:
        logger.debug("Failed to load claims for %s/%s: %s", kb_name, slug, exc)

    return render("article.html", article=article, confidence=confidence)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, q: str = ""):
    results = search_kb(q) if q else []

    # Recent jobs for sidebar
    recent_jobs = await db.get_jobs(limit=20)
    jobs = _jobs_to_template_format(recent_jobs)

    # Dynamic suggestions — pick good titles (not raw filenames/dates)
    import random
    all_articles = []
    for kb_name in storage.list_kbs():
        all_articles.extend(get_articles(kb_name))
    # Filter to articles with clean titles (not slugs, not date-prefixed)
    good_articles = [
        a for a in all_articles
        if a.get("title")
        and len(a["title"]) > 10
        and not a["title"].startswith("Research 20")
        and not a["title"].startswith("Local ")
        and not a["title"].startswith("media-")
        and a["title"] != a["slug"]
        and a.get("word_count", 0) > 200
    ]
    if good_articles:
        sampled = random.sample(good_articles, min(3, len(good_articles)))
        # Send the full title — the template truncates the visible label
        # via `| truncate(25)` but the data-send payload stays complete so
        # clicking re-runs the exact article title the user sees.
        suggestions = [a["title"] for a in sampled]
    else:
        suggestions = ["Kubernetes networking", "Rust vs Go", "PostgreSQL tuning"]

    # Chat sessions for sidebar
    chat_sessions = await db.get_chat_sessions(limit=20)

    return render(
        "search.html",
        query=q, results=results, jobs=jobs,
        kb_names=storage.list_kbs(),
        suggestions=suggestions,
        chat_sessions=chat_sessions,
    )


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    """Interactive knowledge graph visualization."""
    return render("graph.html")


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs_page(request: Request):
    """Interactive API documentation."""
    return render("api_docs.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin dashboard: KB health, quality scores, actions."""
    from app.quality import find_shallow_articles, find_duplicates
    from app.auto_discovery import get_status_for_kb
    from app.config import AUTO_DISCOVERY_ENABLED

    # Stats
    stats = {}
    for kb_name in storage.list_kbs():
        articles = await asyncio.to_thread(get_articles, kb_name)
        words = sum(a["word_count"] for a in articles)
        stats[kb_name] = {
            "articles": len(articles),
            "words": words,
            "avg_words": words // max(len(articles), 1),
        }

    # Quality
    shallow = await asyncio.to_thread(find_shallow_articles, "personal")
    dupes = await asyncio.to_thread(find_duplicates, "personal")

    # Auto-discovery status (per KB)
    redis = request.app.state.redis
    auto_discovery = []
    for kb_name in storage.list_kbs():
        try:
            auto_discovery.append(await get_status_for_kb(kb_name, arq_pool=redis))
        except Exception as exc:
            auto_discovery.append({"kb": kb_name, "error": str(exc)})

    # Per-KB LLM/persona settings
    kb_settings = []
    for kb_name in storage.list_kbs():
        try:
            cfg = await db.get_kb_settings(kb_name) or {}
            kb_settings.append({"kb": kb_name, "config": cfg})
        except Exception as exc:
            kb_settings.append({"kb": kb_name, "config": {}, "error": str(exc)})

    return render(
        "admin.html",
        stats=stats,
        quality_summary={
            "shallow_count": len(shallow),
            "duplicate_count": len(dupes),
        },
        shallow_articles=shallow[:20],
        duplicates=dupes[:10],
        auto_discovery=auto_discovery,
        auto_discovery_global_enabled=AUTO_DISCOVERY_ENABLED,
        kb_settings=kb_settings,
    )


@app.get("/research/{filename}", response_class=HTMLResponse)
async def view_research(request: Request, filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    from app.config import RESEARCH_KB
    text = storage.read_text(RESEARCH_KB, filename)
    if text is None:
        raise HTTPException(status_code=404, detail="Research file not found")

    _, body = parse_frontmatter(text)
    try:
        html = markdown.markdown(body, extensions=["fenced_code", "tables", "codehilite"])
    except Exception:
        html = f"<pre>{body}</pre>"

    return render("research_view.html", filename=filename, html=html, raw=body)


# --- API Routes -------------------------------------------------------------

@app.get("/api/_test/sse")
async def api_test_sse(request: Request):
    """SSE smoke-test route. Yields 5 ticks then a 'done' event.

    Useful for verifying that the streaming layer + any nginx/proxy
    buffering tweaks are working end-to-end without depending on the
    chat endpoint.
    """
    import asyncio
    from app.sse import sse_response

    async def gen():
        for i in range(5):
            await asyncio.sleep(0.2)
            yield {"event": "tick", "data": {"i": i}}
        yield {"event": "done", "data": "complete"}

    return sse_response(gen(), request=request)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    """Liveness probe. Accepts HEAD so uptime monitors don't get a 405."""
    return {
        "status": "ok",
        "storage": storage.backend_name(),
        "kbs": storage.list_kbs(),
    }


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus text exposition for the in-process registry."""
    from app import metrics as _metrics
    from starlette.responses import PlainTextResponse
    # Snapshot queue + job counts as gauges at scrape time so Prometheus
    # sees live values, not the last value set by background code.
    try:
        stats = await db.get_job_stats()
        _metrics.set_gauge(
            "kb_jobs_queued", float(stats.get("queued", 0)),
            help="Jobs waiting to run",
        )
        _metrics.set_gauge(
            "kb_jobs_running", float(stats.get("running", 0)),
            help="Jobs currently executing",
        )
        _metrics.set_gauge(
            "kb_jobs_errored", float(stats.get("error", 0)),
            help="Jobs that ended with an error",
        )
    except Exception:
        pass
    try:
        from app.llm import _embedding_circuit
        _metrics.set_gauge(
            "kb_embedding_circuit_open",
            1.0 if _embedding_circuit.is_open else 0.0,
            help="Embedding circuit breaker state (1=open, 0=closed)",
        )
    except Exception:
        pass
    return PlainTextResponse(
        _metrics.prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/api/usage/summary")
async def api_usage_summary():
    """LLM token usage + rough cost estimate for the admin dashboard."""
    from app import metrics as _metrics
    return _metrics.get_usage_summary()


@app.get("/api/usage/all-time")
async def api_usage_all_time():
    """Lifetime LLM usage counters, persisted in the database."""
    from app import metrics as _metrics
    return await _metrics.get_all_time_totals()


@app.get("/api/traces")
async def api_traces(
    limit: int = 50,
    session_id: str | None = None,
    provider: str | None = None,
    status: str | None = None,
):
    """Recent LLM trace spans — for admin dashboard + playback tests."""
    from app import tracing
    return await tracing.get_recent_traces(
        limit=limit,
        session_id=session_id,
        provider=provider,
        status=status,
    )


@app.get("/api/traces/{trace_id}")
async def api_trace_detail(trace_id: str):
    """All spans for a single trace_id."""
    from app import tracing
    spans = await tracing.get_trace(trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "spans": spans}


@app.get("/api/articles")
async def api_articles(kb: str | None = None, source: str | None = None):
    """List articles. Optional filters: ?kb=personal&source=local"""
    if kb:
        articles = await asyncio.to_thread(get_articles, kb)
    else:
        articles = []
        for kb_name in storage.list_kbs():
            kb_articles = await asyncio.to_thread(get_articles, kb_name)
            articles.extend(kb_articles)
    if source:
        articles = [a for a in articles if a.get("source_type") == source]
    return articles


@app.get("/api/articles/{kb_name}/{slug}")
async def api_article(kb_name: str, slug: str):
    article = get_article(kb_name, slug)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@app.delete("/api/articles/{kb_name}/{slug}")
async def api_delete_article(kb_name: str, slug: str):
    """Delete a wiki article and its raw source file."""
    try:
        result = delete_article(kb_name, slug)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Article not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Remove embedding (vector store) and KG edges (sql)
    try:
        from app.vector_store import get_vector_store
        await get_vector_store().delete(kb_name, slug)
    except Exception:
        pass
    try:
        import aiosqlite
        from app.config import DB_PATH
        async with aiosqlite.connect(str(DB_PATH)) as conn:
            await conn.execute(
                "DELETE FROM kg_edges WHERE article_slug = ? AND kb = ?",
                (slug, kb_name),
            )
            await conn.commit()
    except Exception:
        pass

    return result


@app.delete("/api/research/{filename}")
async def api_delete_research_file(filename: str):
    """Delete a research output file."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    from app.config import RESEARCH_KB
    if not storage.exists(RESEARCH_KB, filename):
        raise HTTPException(status_code=404, detail="Research file not found")
    storage.delete(RESEARCH_KB, filename)
    return {"status": "deleted", "filename": filename}


@app.delete("/api/research/job/{job_id}")
async def api_delete_job(job_id: int):
    """Delete a research job record from the database."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    await db.delete_job(job_id)
    return {"status": "deleted", "job_id": job_id, "topic": job["topic"]}


@app.get("/api/search")
@limiter.limit(RATE_LIMIT_SEARCH)
async def api_search(request: Request, q: str = ""):
    """Search — uses FTS5 if available, falls back to basic string match."""
    results = await search_fts(q)
    return results


@app.post("/api/search/reindex")
async def api_reindex():
    """Rebuild the FTS search index."""
    count = await build_search_index()
    return {"status": "indexed", "articles": count}


@app.get("/api/search/hybrid")
@limiter.limit(RATE_LIMIT_SEARCH)
async def api_hybrid_search(request: Request, q: str = "", kb: str | None = None, limit: int = 15):
    """Hybrid search combining FTS5 + vector similarity + knowledge graph."""
    if not q or not q.strip():
        return []
    results = await hybrid_search(q, kb_name=kb, limit=limit)
    return results


@app.post("/api/embeddings/build")
async def api_build_embeddings(request: Request):
    """Build embedding index for all articles in a KB (async via worker)."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"

    redis = request.app.state.redis
    await redis.enqueue_job("embed_all_task", kb_name, _queue_name=ARQ_QUEUE_NAME)
    return {"status": "queued", "kb": kb_name, "action": "build_embeddings"}


@app.post("/api/graph/build")
async def api_build_graph(request: Request):
    """Build knowledge graph for all articles in a KB (async via worker)."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"

    redis = request.app.state.redis
    await redis.enqueue_job("build_graph_task", kb_name, _queue_name=ARQ_QUEUE_NAME)
    return {"status": "queued", "kb": kb_name, "action": "build_graph"}


@app.post("/api/articles/{kb_name}/{slug}/agent-improve")
async def api_agent_improve(request: Request, kb_name: str, slug: str):
    """Run the improvement agent against a single existing article.

    The agent reads the article, picks 3-5 weak sections, researches
    just those, and rewrites with citations. Returns the job id so
    the caller can poll /api/research/status.
    """
    from app.wiki import get_article
    article = get_article(kb_name, slug)
    if not article:
        raise HTTPException(
            status_code=404, detail=f"Article not found: {kb_name}/{slug}",
        )
    job_id = await db.create_job(
        f"improve:{kb_name}/{slug}", job_type="agent_improve",
    )
    redis = request.app.state.redis
    await redis.enqueue_job(
        "agent_improve_task", kb_name, slug, job_id,
        _queue_name=ARQ_QUEUE_NAME,
    )
    return {
        "job_id": job_id, "status": "queued",
        "kb": kb_name, "slug": slug,
    }


@app.get("/api/kb/{kb_name}/agent-resync/status")
async def api_agent_resync_status(kb_name: str):
    """Live progress snapshot for the in-flight agent-improve batch.

    Scans recent jobs with topic prefix 'improve:{kb}/' and reports
    counts. ETA is derived from completion rate over the window.
    """
    jobs = await db.get_jobs(limit=5000, compact=True)
    prefix = f"improve:{kb_name}/"
    improve = [j for j in jobs if (j.get("topic") or "").startswith(prefix)]
    total = len(improve)
    complete = sum(1 for j in improve if j.get("status") == "complete")
    errored = sum(1 for j in improve if j.get("status") == "error")
    queued = sum(1 for j in improve if j.get("status") == "queued")
    running = sum(
        1 for j in improve
        if j.get("status") in ("agent_improving", "writing")
    )

    # Rough ETA: completion rate over the finished set.
    import datetime as _dt
    completion_times = [
        j.get("completed_at") for j in improve
        if j.get("status") == "complete" and j.get("completed_at")
    ]
    eta_seconds = None
    rate_per_hour = None
    if len(completion_times) >= 5:
        try:
            times = sorted(_dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
                           for t in completion_times if t)
            span = (times[-1] - times[0]).total_seconds()
            if span > 0:
                rate = len(times) / span  # jobs per second
                rate_per_hour = round(rate * 3600, 1)
                remaining = queued + running
                if rate > 0:
                    eta_seconds = int(remaining / rate)
        except Exception:
            pass

    return {
        "kb": kb_name,
        "batch": {
            "total": total, "complete": complete,
            "errored": errored, "queued": queued, "running": running,
            "remaining": queued + running,
            "rate_per_hour": rate_per_hour,
            "eta_seconds": eta_seconds,
        },
    }


@app.post("/api/kb/{kb_name}/agent-triage")
async def api_agent_triage(request: Request, kb_name: str):
    """Score the KB and agent-improve only articles below threshold.

    Body: {"threshold": 70, "limit": N, "dry_run": bool}

    Running dry_run=true is the recommended first step — it returns
    how many articles are below/above the threshold so you can pick
    a sensible cutoff before actually queueing the improve batch.
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if not isinstance(body, dict):
        body = {}
    threshold = body.get("threshold", 70)
    try:
        threshold = int(threshold)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="threshold must be int")
    limit = body.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="limit must be int")
    dry_run = bool(body.get("dry_run", False))

    redis = request.app.state.redis
    await redis.enqueue_job(
        "agent_triage_kb_task", kb_name, threshold, limit, dry_run,
        _queue_name=ARQ_QUEUE_NAME,
    )
    return {
        "status": "queued", "kb": kb_name,
        "threshold": threshold, "limit": limit, "dry_run": dry_run,
        "action": "agent_triage",
    }


@app.post("/api/kb/{kb_name}/agent-resync")
async def api_agent_resync(request: Request, kb_name: str):
    """Queue an improvement agent run for every article in a KB.

    Body (optional JSON): {"limit": N, "dry_run": bool}. dry_run=true
    reports what would be queued without enqueueing anything.
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    if not isinstance(body, dict):
        body = {}
    limit = body.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="limit must be int")
    dry_run = bool(body.get("dry_run", False))

    redis = request.app.state.redis
    await redis.enqueue_job(
        "agent_resync_kb_task", kb_name, limit, dry_run,
        _queue_name=ARQ_QUEUE_NAME,
    )
    return {
        "status": "queued", "kb": kb_name,
        "limit": limit, "dry_run": dry_run,
        "action": "agent_resync",
    }


# ---------------------------------------------------------------------------
# Scaffolds — plug-and-play template packages
#
# A scaffold is a small multi-file code template (HTML/CSS/JS for MVP;
# React/Vue later) produced by the scaffold agent from a topic + type.
# It lives under ``scaffolds/<slug>/`` in the KB's storage and is
# rendered in a sandboxed iframe at /sandbox/<kb>/<slug>.
# ---------------------------------------------------------------------------

@app.post("/api/scaffolds/{kb_name}/create")
async def api_scaffold_create(request: Request, kb_name: str):
    """Queue a scaffold-agent run.

    Body: {"topic": "...", "scaffold_type": "landing-page"}
    Returns: {job_id, status: queued}
    """
    from app.scaffolds import SCAFFOLD_TYPES
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    topic = (body.get("topic") or "").strip()
    scaffold_type = (body.get("scaffold_type") or "other").strip()
    if not topic or len(topic) < 3:
        raise HTTPException(status_code=400, detail="topic is required (min 3 chars)")
    if scaffold_type not in SCAFFOLD_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"scaffold_type must be one of: {sorted(SCAFFOLD_TYPES)}",
        )

    job_id = await db.create_job(
        f"scaffold:{scaffold_type}:{topic[:80]}",
        job_type="scaffold",
    )
    redis = request.app.state.redis
    await redis.enqueue_job(
        "scaffold_create_task", kb_name, topic, scaffold_type, job_id,
        _queue_name=ARQ_QUEUE_NAME,
    )
    return {
        "job_id": job_id, "status": "queued",
        "kb": kb_name, "topic": topic, "scaffold_type": scaffold_type,
    }


@app.get("/api/scaffolds/{kb_name}")
async def api_scaffold_list(kb_name: str):
    """List all scaffolds in a KB."""
    from app.scaffolds import list_scaffolds
    return {"kb": kb_name, "scaffolds": list_scaffolds(kb_name)}


@app.get("/api/scaffolds/{kb_name}/{slug}")
async def api_scaffold_manifest(kb_name: str, slug: str):
    """Return the manifest for a single scaffold."""
    from app.scaffolds import get_manifest
    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="scaffold not found")
    return manifest


@app.delete("/api/scaffolds/{kb_name}/{slug}")
async def api_scaffold_delete(kb_name: str, slug: str):
    from app.scaffolds import get_manifest, delete_scaffold
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="scaffold not found")
    delete_scaffold(kb_name, slug)
    return {"status": "deleted", "kb": kb_name, "slug": slug}


# Common MIME types we serve from /sandbox/. Anything unknown → plain
# text so the browser can't execute it as a script by accident.
_SCAFFOLD_MIMES = {
    "html": "text/html; charset=utf-8",
    "htm": "text/html; charset=utf-8",
    "css": "text/css; charset=utf-8",
    "js": "application/javascript; charset=utf-8",
    "mjs": "application/javascript; charset=utf-8",
    "json": "application/json; charset=utf-8",
    "svg": "image/svg+xml",
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
}


@app.get("/sandbox/{kb_name}/{slug}/{rel_path:path}")
async def view_scaffold_file(kb_name: str, slug: str, rel_path: str):
    """Serve a scaffold file with strict CSP for iframe rendering.

    The sandbox route is deliberately the ONLY place scaffold file
    content leaves storage as executable HTML/JS. Callers should
    embed it in an ``<iframe sandbox="allow-scripts">`` — our CSP
    doubles up on the sandboxing in case the iframe attribute is
    dropped by a proxy.
    """
    from app.scaffolds import get_file, get_manifest
    from starlette.responses import Response

    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="scaffold not found")

    # Empty rel_path → redirect to entrypoint (lets <iframe src>
    # point at the slug root and Just Work).
    if not rel_path:
        from starlette.responses import RedirectResponse
        return RedirectResponse(
            url=f"/sandbox/{kb_name}/{slug}/{manifest['entrypoint']}",
            status_code=307,
        )

    content = get_file(kb_name, slug, rel_path)
    if content is None:
        raise HTTPException(status_code=404, detail=f"file not found: {rel_path}")

    ext = rel_path.rsplit(".", 1)[-1].lower() if "." in rel_path else ""
    mime = _SCAFFOLD_MIMES.get(ext, "text/plain; charset=utf-8")

    # Inject a tiny postMessage listener into every sandbox HTML
    # response so the parent viewer can live-override :root CSS custom
    # properties (design-token editor). Inert when no message arrives,
    # so it has zero effect for users who open the sandbox directly.
    if ext == "html":
        shim = (
            "<script>(function(){"
            "window.addEventListener('message',function(e){"
            "var d=e.data||{};"
            "if(d.type==='wd:token-override'&&d.tokens){"
            "var r=document.documentElement;"
            "Object.keys(d.tokens).forEach(function(k){"
            "r.style.setProperty('--'+k,d.tokens[k]);});"
            "}else if(d.type==='wd:token-reset'){"
            "var r=document.documentElement;"
            "(d.keys||[]).forEach(function(k){r.style.removeProperty('--'+k);});"
            "}});"
            "window.parent&&window.parent.postMessage({type:'wd:ready',path:"
            + json.dumps(rel_path) + "},'*');"
            "})();</script>"
        )
        # Inject before </head> if possible, else prepend to body.
        lower = content.lower()
        idx = lower.find("</head>")
        if idx >= 0:
            content = content[:idx] + shim + content[idx:]
        else:
            content = shim + content
    # CSP: no network, no parent, no inline-except-unsafe-for-scaffold.
    # 'unsafe-inline' is needed because scaffolds bundle small inline
    # <style>/<script> blocks; network is closed so this can't phone
    # home even if the agent emitted something janky.
    csp = (
        "default-src 'self' data:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' data:; "
        "connect-src 'none'; "
        "frame-ancestors 'self';"
    )
    headers = {
        "Content-Security-Policy": csp,
        "X-Frame-Options": "SAMEORIGIN",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }
    return Response(content, media_type=mime, headers=headers)


@app.get("/api/scaffolds/{kb_name}/{slug}/events")
async def scaffold_events_stream(kb_name: str, slug: str, request: Request):
    """Server-Sent Events stream for hot-reload.

    Emits `event: update` with the manifest's ``updated`` timestamp
    whenever storage changes (e.g. an extension page gets added by
    the scaffold-extend worker). Client reloads the iframe on each
    event. Poll loop on the server side — O(1) per connected viewer,
    no change-capture plumbing in the storage layer.
    """
    import asyncio
    from starlette.responses import StreamingResponse
    from app.scaffolds import get_manifest

    async def gen():
        last_key = None
        # Emit an initial "hello" so clients confirm the channel.
        yield "event: hello\ndata: {}\n\n"
        while True:
            if await request.is_disconnected():
                return
            manifest = get_manifest(kb_name, slug)
            if not manifest:
                yield "event: gone\ndata: {}\n\n"
                return
            # Version key: (updated, files count). Covers both manifest
            # timestamp bumps AND file-list growth (extension fan-out).
            key = (manifest.get("updated"), len(manifest.get("files") or []))
            if key != last_key:
                last_key = key
                payload = json.dumps({"updated": manifest.get("updated"), "files": len(manifest.get("files") or [])})
                yield f"event: update\ndata: {payload}\n\n"
            await asyncio.sleep(2.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/scaffolds/{kb_name}/{slug}", response_class=HTMLResponse)
async def view_scaffold(request: Request, kb_name: str, slug: str):
    """Scaffold viewer page — description + sandboxed iframe + file listing."""
    from app.scaffolds import get_manifest
    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="scaffold not found")
    return render(
        "scaffold_view.html",
        kb=kb_name, slug=slug, manifest=manifest,
    )


@app.get("/scaffolds", response_class=HTMLResponse)
async def browse_scaffolds(request: Request, kb: str = "personal"):
    from app.scaffolds import list_scaffolds, SCAFFOLD_TYPES
    items = list_scaffolds(kb)
    return render(
        "scaffolds_browse.html",
        kb=kb, items=items,
        types=sorted(SCAFFOLD_TYPES),
    )


# ---------------------------------------------------------------------------
# Documents — placeholder browse until chunk 4 of the Documents feature
# lands. Storage layer (app/documents.py) is wired but the agent +
# renderer + viewer aren't built yet. This stub lets the nav link
# resolve cleanly instead of 404'ing.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Documents — create, fetch, render, commit, export
# ---------------------------------------------------------------------------

@app.post("/api/documents/{kb_name}/create")
async def api_document_create(request: Request, kb_name: str):
    """Create a new document shell. Body: {title, brief, doc_type?,
    autonomy_mode?, seed_articles?, pinned_facts?}. The drafting
    agent fills v1 on the first chat turn."""
    from app.documents import (
        create_document, AUTONOMY_MODES, DOC_TYPES, DEFAULT_AUTONOMY,
    )
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be JSON object")

    title = (body.get("title") or "").strip()
    brief = (body.get("brief") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    if len(brief) < 5:
        raise HTTPException(status_code=400, detail="brief is required (min 5 chars)")

    doc_type = body.get("doc_type", "pdf")
    if doc_type not in DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"doc_type must be one of {sorted(DOC_TYPES)}",
        )
    autonomy_mode = body.get("autonomy_mode", DEFAULT_AUTONOMY)
    if autonomy_mode not in AUTONOMY_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"autonomy_mode must be one of {sorted(AUTONOMY_MODES)}",
        )

    try:
        slug = create_document(
            kb_name, title, brief,
            doc_type=doc_type, autonomy_mode=autonomy_mode,
            seed_articles=body.get("seed_articles"),
            pinned_facts=body.get("pinned_facts"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"kb": kb_name, "slug": slug, "status": "created"}


@app.get("/api/documents/{kb_name}")
async def api_document_list(kb_name: str):
    from app.documents import list_documents
    return {"kb": kb_name, "documents": list_documents(kb_name)}


@app.get("/api/documents/{kb_name}/{slug}")
async def api_document_manifest(kb_name: str, slug: str):
    from app.documents import get_manifest
    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="document not found")
    return manifest


@app.get("/api/documents/{kb_name}/{slug}/markdown")
async def api_document_markdown(kb_name: str, slug: str, v: int = 0):
    """Return the markdown source at version v (0 = current)."""
    from app.documents import get_markdown, get_manifest
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")
    md = get_markdown(kb_name, slug, version=v or None)
    if md is None:
        raise HTTPException(status_code=404, detail=f"version not found: {v}")
    return {"kb": kb_name, "slug": slug, "version": v, "markdown": md}


@app.get("/api/documents/{kb_name}/{slug}/pending")
async def api_document_pending(kb_name: str, slug: str):
    """Return the pending agent draft (if any) so the chat UI can
    diff it against current and offer ✓/✗."""
    from app.documents import get_pending_draft, get_manifest
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")
    pending = get_pending_draft(kb_name, slug)
    return {"pending": pending}


@app.post("/api/documents/{kb_name}/{slug}/commit")
async def api_document_commit(kb_name: str, slug: str):
    """Promote the pending agent draft to v+1 (renders PDF inline)."""
    from app.documents import commit_pending_draft, get_manifest
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")
    entry = commit_pending_draft(kb_name, slug)
    if not entry:
        raise HTTPException(status_code=409, detail="no pending draft to commit")
    return {"kb": kb_name, "slug": slug, "committed_version": entry["v"]}


@app.post("/api/documents/{kb_name}/{slug}/discard")
async def api_document_discard(kb_name: str, slug: str):
    from app.documents import discard_pending_draft
    dropped = discard_pending_draft(kb_name, slug)
    return {"kb": kb_name, "slug": slug, "discarded": dropped}


@app.get("/api/documents/{kb_name}/{slug}/export")
async def api_document_export(kb_name: str, slug: str, v: int = 0):
    """Download the rendered PDF (or whatever doc_type) at version v."""
    from app.documents import get_manifest, get_rendered, rerender_version
    from starlette.responses import Response
    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="document not found")
    version = v or manifest.get("current_version", 0)
    if version == 0:
        raise HTTPException(status_code=404, detail="document has no committed versions yet")
    rendered = get_rendered(kb_name, slug, version)
    if rendered is None:
        # Auto-recover: try a fresh render before giving up.
        rendered = rerender_version(kb_name, slug, version)
    if rendered is None:
        raise HTTPException(status_code=500, detail="render unavailable")
    doc_type = manifest.get("doc_type", "pdf")
    mime = {
        "pdf": "application/pdf",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "md-export": "text/markdown; charset=utf-8",
    }.get(doc_type, "application/octet-stream")
    safe_name = f"{slug}-v{version}.{doc_type if doc_type != 'md-export' else 'md'}"
    return Response(
        rendered, media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@app.delete("/api/documents/{kb_name}/{slug}")
async def api_document_delete(kb_name: str, slug: str):
    from app.documents import delete_document, get_manifest
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")
    delete_document(kb_name, slug)
    return {"status": "deleted", "kb": kb_name, "slug": slug}


@app.post("/api/documents/{kb_name}/{slug}/generate-output")
async def api_document_generate_output(kb_name: str, slug: str, body: dict):
    """Generate a specific output type from the document's sources.

    Output types (briefing / faq / study-guide / timeline) map to
    canned system instructions in ``DOC_OUTPUT_PROMPTS``. We reuse
    ``run_doc_chat_turn`` by wrapping the instruction as a user
    message, so the proposal / commit workflow stays identical.
    """
    from app.agent_prompts import DOC_OUTPUT_PROMPTS
    from app.documents import get_manifest
    from app.agent import run_doc_chat_turn

    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")

    output_type = (body.get("output_type") or "").strip()
    if output_type not in DOC_OUTPUT_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"output_type must be one of {sorted(DOC_OUTPUT_PROMPTS)}",
        )

    instruction = (
        f"Generate a '{output_type}' output from the current sources. "
        f"Replace the document body with this new output.\n\n"
        + DOC_OUTPUT_PROMPTS[output_type]
    )
    result = await run_doc_chat_turn(kb_name, slug, instruction)
    return {
        "kb": kb_name, "slug": slug, "output_type": output_type,
        **result,
    }


@app.patch("/api/documents/{kb_name}/{slug}/settings")
async def api_document_settings(kb_name: str, slug: str, body: dict):
    """Toggle per-document knobs (autonomy_mode, grounding_mode).

    Kept narrow on purpose — title/brief/seed edits would invalidate
    version history context, so we only accept the runtime knobs that
    affect the NEXT chat turn.
    """
    import json as _json
    from app import storage
    from app.documents import get_manifest, AUTONOMY_MODES, GROUNDING_MODES
    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="document not found")

    changed = False
    if "autonomy_mode" in body:
        mode = body["autonomy_mode"]
        if mode not in AUTONOMY_MODES:
            raise HTTPException(status_code=400, detail=f"autonomy_mode must be one of {sorted(AUTONOMY_MODES)}")
        manifest["autonomy_mode"] = mode
        changed = True
    if "grounding_mode" in body:
        mode = body["grounding_mode"]
        if mode not in GROUNDING_MODES:
            raise HTTPException(status_code=400, detail=f"grounding_mode must be one of {sorted(GROUNDING_MODES)}")
        manifest["grounding_mode"] = mode
        changed = True

    if changed:
        storage.write_text(
            kb_name, f"documents/{slug}/manifest.json",
            _json.dumps(manifest, indent=2, sort_keys=True),
        )
    return {
        "kb": kb_name, "slug": slug,
        "autonomy_mode": manifest.get("autonomy_mode"),
        "grounding_mode": manifest.get("grounding_mode"),
    }


@app.get("/api/documents/{kb_name}/{slug}/history")
async def api_document_history(kb_name: str, slug: str, limit: int = 50):
    """Return the chat-turn + version history log for a document."""
    from app.documents import get_history, get_manifest
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")
    return {
        "kb": kb_name, "slug": slug,
        "events": get_history(kb_name, slug, limit=limit),
    }


@app.post("/api/documents/{kb_name}/{slug}/chat")
async def api_document_chat(request: Request, kb_name: str, slug: str):
    """Send a user message to the document drafting agent.

    Body: {"message": "..."}.  Runs one agent turn synchronously
    (typical Minimax round-trip ~10-30s — fine for an interactive
    chat feel).  Returns {response, pending, version_after}.
    """
    from app.documents import get_manifest
    from app.agent import run_doc_chat_turn
    if not get_manifest(kb_name, slug):
        raise HTTPException(status_code=404, detail="document not found")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be JSON object")
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    result = await run_doc_chat_turn(kb_name, slug, message)
    if "error" in result:
        # Don't surface as 500 — the chat UI can render this as an
        # in-conversation agent error message.
        return {"error": result["error"], "ok": False}
    return {**result, "ok": True}


@app.get("/documents/{kb_name}/{slug}", response_class=HTMLResponse)
async def view_document(request: Request, kb_name: str, slug: str):
    from app.documents import get_manifest, get_history
    manifest = get_manifest(kb_name, slug)
    if not manifest:
        raise HTTPException(status_code=404, detail="document not found")
    history = get_history(kb_name, slug, limit=50)
    return render(
        "document_view.html",
        kb=kb_name, slug=slug,
        manifest=manifest, history=history,
    )


@app.get("/documents", response_class=HTMLResponse)
async def browse_documents(request: Request, kb: str = "personal"):
    from app.documents import list_documents, DOC_TYPES
    items = list_documents(kb)
    return render(
        "documents_browse.html",
        kb=kb, items=items, doc_types=sorted(DOC_TYPES),
    )


def render_inline(body_html: str, title: str = "WikiDelve") -> HTMLResponse:
    """Tiny helper: render an inline body inside base.html via a one-shot
    Jinja template literal. Used by stub pages that don't need their
    own .html file yet — csp_nonce + other globals come from
    jinja_env.globals (set up at module load).
    """
    tpl = jinja_env.from_string(
        "{% extends 'base.html' %}{% block title %}" + title +
        "{% endblock %}{% block content %}" + body_html + "{% endblock %}"
    )
    return HTMLResponse(tpl.render())


@app.get("/api/graph/data")
async def api_graph_data():
    """Return the full knowledge graph as nodes + edges for D3.js visualization."""
    return await get_graph_data()


@app.get("/api/graph/entity/{name}")
async def api_entity_articles(name: str):
    """Find all articles mentioning a specific entity."""
    articles = await get_entity_articles(name)
    return {"entity": name, "articles": articles}


@app.get("/api/graph/related/{kb}/{slug}")
async def api_graph_related(kb: str, slug: str, depth: int = 2):
    """Find articles related to a given article via knowledge graph traversal."""
    related = await get_related_by_graph(slug, kb, depth=min(depth, 3))
    return {"slug": slug, "kb": kb, "related": related}


@app.post("/api/quality/auto-enrich")
async def api_auto_enrich(request: Request):
    """Auto-enrich all articles below a quality threshold."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    threshold = body.get("threshold", 60) if isinstance(body, dict) else 60
    max_articles = body.get("max_articles", 20) if isinstance(body, dict) else 20

    from app.quality import score_all_articles
    scores = await asyncio.to_thread(score_all_articles, kb_name)
    below_threshold = [s for s in scores if s["score"] < threshold][:max_articles]

    redis = request.app.state.redis
    queued = 0
    for article in below_threshold:
        await redis.enqueue_job("enrich_task", kb_name, article["slug"], _queue_name="wikidelve")
        queued += 1

    return {
        "status": "queued",
        "kb": kb_name,
        "threshold": threshold,
        "below_threshold": len([s for s in scores if s["score"] < threshold]),
        "queued": queued,
    }


@app.post("/api/research/local")
@limiter.limit(RATE_LIMIT_RESEARCH)
async def api_local_research(request: Request):
    """Trigger local research: scan files/folders/git repos and synthesize.

    Body: {"topic": "auth flow", "path": "/code/myproject", "pattern": "*.py"}
    No search API keys required.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    topic = body.get("topic", "").strip()
    path = body.get("path", "").strip()
    pattern = body.get("pattern")  # optional glob filter

    if not topic or len(topic) < 3:
        raise HTTPException(status_code=400, detail="topic is required (min 3 chars)")
    if not path:
        raise HTTPException(status_code=400, detail="path is required")

    # Create job in DB. We persist the original inputs on the row so
    # smart-retry can re-run the exact same scan — the topic string
    # alone can't be reversed back into path/pattern.
    job_id = await db.create_job(
        f"local:{topic}",
        job_type="local",
        source_params=json.dumps({"path": path, "pattern": pattern}),
    )

    # Enqueue to arq
    redis = request.app.state.redis
    await redis.enqueue_job(
        "local_research_task",
        topic, path, job_id, pattern,
        _queue_name=ARQ_QUEUE_NAME,
    )

    return {"job_id": job_id, "status": "queued", "topic": topic, "path": path}


@app.post("/api/research")
@limiter.limit(RATE_LIMIT_RESEARCH)
async def api_research(request: Request):
    """Trigger research on a topic. Enqueues to Redis via arq.

    Returns job ID for status polling.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    topic = body.get("topic")
    if not topic or not isinstance(topic, str) or not topic.strip():
        raise HTTPException(
            status_code=400,
            detail="topic is required and must be a non-empty string",
        )
    if len(topic.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="topic must be at least 10 characters for meaningful research",
        )

    topic = topic.strip()
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    review_sources = body.get("review_sources", False) if isinstance(body, dict) else False
    use_agent = body.get("agent", False) if isinstance(body, dict) else False

    # Auto-create KB if it doesn't exist
    if kb_name not in storage.list_kbs():
        from app.config import register_kb
        try:
            register_kb(kb_name)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid KB name: {kb_name}")

    # Cooldown check
    existing = await db.check_cooldown(topic)
    if existing:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Topic was researched recently (job {existing['id']}). "
                          f"Cooldown is 7 days.",
                "existing_job_id": existing["id"],
            },
        )

    # Create job in DB
    job_id = await db.create_job(topic)

    # Enqueue to arq — choose agent, collect-only, or full pipeline
    redis = request.app.state.redis
    if use_agent:
        await redis.enqueue_job(
            "agent_research_task",
            topic,
            job_id,
            kb_name,
            _queue_name=ARQ_QUEUE_NAME,
        )
    elif review_sources:
        await redis.enqueue_job(
            "research_collect_task",
            topic,
            job_id,
            kb_name,
            _queue_name=ARQ_QUEUE_NAME,
        )
    else:
        await redis.enqueue_job(
            "research_task",
            topic,
            job_id,
            kb_name,
            _queue_name=ARQ_QUEUE_NAME,
        )

    return {
        "job_id": job_id,
        "status": "queued",
        "topic": topic,
        "kb": kb_name,
        "review_sources": review_sources,
        "agent": use_agent,
    }


@app.get("/api/research/status/{job_id}")
async def api_research_status(job_id: int):
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_api_format(job)


@app.get("/api/research/stream/{job_id}")
async def api_research_stream(job_id: int, request: Request):
    """Stream a research job's progress over SSE.

    Emits a ``status`` event every time the job's status or word count
    changes, a ``progress`` event with the sources count, and a final
    ``complete`` or ``error`` event when the job reaches a terminal
    state. Terminates the stream automatically; the browser's
    EventSource won't try to reconnect past a closed stream.

    Callers should open this as soon as they enqueue a job so they
    see every transition. The underlying poll cadence is 1.5s which
    is fine for research jobs that run for minutes.
    """
    from app.sse import sse_response

    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def gen():
        last_status: str | None = None
        last_words: int = -1
        last_sources: int = -1
        terminal = {"complete", "error", "no_results", "cancelled"}
        max_ticks = 600  # 1.5s × 600 = 15 minutes max
        for _ in range(max_ticks):
            if await request.is_disconnected():
                return
            try:
                current = await db.get_job(job_id)
            except Exception as exc:
                yield {"event": "error", "data": {"detail": str(exc)}}
                return
            if not current:
                yield {"event": "error", "data": {"detail": "Job vanished"}}
                return

            status = current.get("status") or "unknown"
            words = int(current.get("word_count") or 0)
            sources = int(current.get("sources_count") or 0)

            if status != last_status:
                yield {
                    "event": "status",
                    "data": {
                        "job_id": job_id,
                        "status": status,
                        "topic": current.get("topic"),
                        "sources": sources,
                        "word_count": words,
                    },
                }
                last_status = status
            elif sources != last_sources or words != last_words:
                yield {
                    "event": "progress",
                    "data": {
                        "job_id": job_id,
                        "sources": sources,
                        "word_count": words,
                    },
                }
            last_sources = sources
            last_words = words

            if status in terminal:
                yield {
                    "event": "complete" if status == "complete" else "done",
                    "data": {
                        "job_id": job_id,
                        "status": status,
                        "error": current.get("error"),
                        "word_count": words,
                        "added_to_wiki": bool(current.get("added_to_wiki")),
                    },
                }
                return

            await asyncio.sleep(1.5)

        yield {"event": "timeout", "data": {"job_id": job_id}}

    return sse_response(gen(), request=request)


@app.post("/api/research/cancel/{job_id}")
async def api_cancel_research(job_id: int):
    """Cancel a queued research job."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("queued", "searching", "searching_round_1", "searching_round_2"):
        return {"detail": "Job already completed or cannot be cancelled", "status": job["status"]}
    await db.update_job(job_id, status="cancelled", error="Cancelled by user")
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/api/research/retry-errors")
async def api_retry_errors(request: Request):
    """Re-queue all errored research jobs for retry."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    limit = body.get("limit", 50) if isinstance(body, dict) else 50

    errored = await db.get_errored_jobs(limit=limit)
    redis = request.app.state.redis
    retried = 0

    for job in errored:
        try:
            await db.reset_job_for_retry(job["id"])
            await redis.enqueue_job(
                "research_task", job["topic"], job["id"],
                _queue_name=ARQ_QUEUE_NAME,
            )
            retried += 1
        except Exception:
            pass

    return {"retried": retried, "total_errors": len(errored)}


@app.post("/api/research/smart-retry/{job_id}")
async def api_smart_retry(job_id: int, request: Request):
    """Smart-retry a failed job, optionally rewriting the topic.

    Examines the job's failure mode and asks the LLM to refactor the
    topic string based on the error class. Creates a **new** job with
    the rewritten topic (so the original failure trail is preserved in
    the audit log) and enqueues it to the full research pipeline.

    Failure-mode heuristics:
      - "no_results" / "Search failed" → topic too narrow; broaden it.
      - "timeout" / "Synthesis" / "Pipeline crash" → too wide or too
        expensive; tighten and scope it.
      - "MINIMAX_API_KEY not set" → infra, not topic; refuse.
      - anything else → keep original topic, just requeue.
    """
    from app.llm import llm_chat

    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") not in ("error", "no_results", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is not in a retryable state (status: {job.get('status')})",
        )

    original_topic = job.get("topic") or ""

    # Local research jobs scan files on disk — rewriting the topic
    # won't help (failure is usually transient: LLM timeout, file
    # permission, too-large input). Re-run the exact same scan using
    # the stored source_params instead of the web-search pipeline.
    # Back-compat: older rows without job_type but with a "local:"
    # topic prefix are still treated as local.
    is_local = (
        job.get("job_type") == "local"
        or original_topic.lower().startswith("local:")
    )
    if is_local:
        raw_params = job.get("source_params")
        try:
            params = json.loads(raw_params) if raw_params else {}
        except (TypeError, ValueError):
            params = {}
        path = params.get("path")
        if not path:
            raise HTTPException(
                status_code=400,
                detail="Cannot retry this local job: original path not "
                "recorded. Re-run manually via POST /api/research/local.",
            )
        pattern = params.get("pattern")
        # Strip the "local:" prefix to recover the user-facing topic.
        topic = original_topic.split(":", 1)[1].strip() if original_topic.lower().startswith("local:") else original_topic
        new_job_id = await db.create_job(
            f"local:{topic}",
            job_type="local",
            source_params=json.dumps({"path": path, "pattern": pattern}),
        )
        redis = request.app.state.redis
        await redis.enqueue_job(
            "local_research_task", topic, path, new_job_id, pattern,
            _queue_name=ARQ_QUEUE_NAME,
        )
        return {
            "original_job_id": job_id,
            "new_job_id": new_job_id,
            "failure_mode": "local_replay",
            "original_topic": original_topic,
            "new_topic": original_topic,
            "rewritten": False,
            "kb": job.get("kb") or "personal",
        }

    error = (job.get("error") or "").lower()
    kb_name = job.get("kb") or "personal"

    # Classify the failure mode — cheap string match, no LLM call.
    if "minimax_api_key" in error or "not set" in error:
        raise HTTPException(
            status_code=400,
            detail="Job failed due to missing API key — infra issue, not topic. Fix the env and retry normally.",
        )

    failure_mode = "unknown"
    rewrite_instruction = ""
    if "no_results" in error or "no results" in error or job.get("status") == "no_results":
        failure_mode = "too_narrow"
        rewrite_instruction = (
            "The previous search returned zero results. Rewrite the topic "
            "to be BROADER and more searchable: remove obscure jargon, "
            "expand abbreviations, and keep it under 15 words. Return ONLY "
            "the rewritten topic on a single line, no explanation."
        )
    elif any(k in error for k in ("timeout", "synthesis", "pipeline crash", "context length")):
        failure_mode = "too_wide"
        rewrite_instruction = (
            "The previous attempt failed due to scope or size. Rewrite "
            "the topic to be TIGHTER and more focused: pick the single "
            "most interesting angle, drop qualifiers, keep it under 12 "
            "words. Return ONLY the rewritten topic on a single line, no "
            "explanation."
        )

    new_topic = original_topic
    if rewrite_instruction:
        try:
            response = await llm_chat(
                system_msg=(
                    "You rewrite research topics that failed. "
                    "Return ONLY the rewritten topic, one line, no quotes, "
                    "no preamble, no explanation. 3 to 15 words."
                ),
                user_msg=(
                    f"Original topic: {original_topic}\n"
                    f"Failure: {job.get('error', 'unknown')}\n\n"
                    f"{rewrite_instruction}"
                ),
                max_tokens=200,
                temperature=0.4,
            )
            candidate = (response or "").strip().splitlines()[0].strip()
            # Strip common LLM preamble garbage.
            candidate = candidate.lstrip("•*-> ").strip('"').strip("'")
            if 10 <= len(candidate) <= 200 and candidate.lower() != original_topic.lower():
                new_topic = candidate
        except Exception as exc:
            logger.warning("smart-retry rewrite failed for job %s: %s", job_id, exc)

    new_job_id = await db.create_job(new_topic)
    redis = request.app.state.redis
    await redis.enqueue_job(
        "research_task", new_topic, new_job_id, kb_name,
        _queue_name=ARQ_QUEUE_NAME,
    )

    return {
        "original_job_id": job_id,
        "new_job_id": new_job_id,
        "failure_mode": failure_mode,
        "original_topic": original_topic,
        "new_topic": new_topic,
        "rewritten": new_topic != original_topic,
        "kb": kb_name,
    }


@app.get("/api/research/jobs")
async def api_research_jobs(
    limit: int = 50,
    status: str | None = None,
    compact: bool = False,
    before_id: int | None = None,
):
    """Return recent research jobs, newest first.

    Query params:
      - limit:      how many jobs to return (default 50, capped at 500).
      - status:     filter to a specific job status (e.g. "queued", "error").
      - compact:    drop the heavy ``content`` blob from each row — use this
                    for dashboards that poll every few seconds so the payload
                    doesn't balloon to megabytes. Also short-circuits the
                    article-slug lookup for bulk listings.
      - before_id:  cursor — return only jobs with ``id < before_id``.
                    Pair with ``limit`` for "load more" style paging.
    """
    capped = max(1, min(limit, 500))

    # The DynamoDB backend knows how to strip the content field
    # server-side via ProjectionExpression, which fits many more rows
    # per page. Pull extra rows when a cursor is set so we still have
    # enough after the cursor filter.
    fetch = capped * 2 if before_id is not None else capped
    jobs = await db.get_jobs(limit=fetch, compact=compact)

    if before_id is not None:
        jobs = [j for j in jobs if int(j.get("id", 0)) < before_id]
    jobs = jobs[:capped]

    slug_index = await asyncio.to_thread(_build_slug_index)
    rows = [_job_to_api_format(j, slug_index=slug_index) for j in jobs]

    if status:
        rows = [r for r in rows if r.get("status") == status]
    if compact:
        for r in rows:
            r.pop("content", None)
    return rows


# --- Auto-discovery routes --------------------------------------------------


@app.get("/api/auto-discovery/status")
async def api_auto_discovery_status(request: Request):
    """Per-KB auto-discovery status for the admin UI."""
    from app.auto_discovery import get_status_for_kb
    from app.config import AUTO_DISCOVERY_ENABLED, SERPER_CALLS_PER_JOB_ESTIMATE
    redis = request.app.state.redis
    results = []
    for kb in storage.list_kbs():
        try:
            results.append(await get_status_for_kb(kb, arq_pool=redis))
        except Exception as exc:
            results.append({"kb": kb, "error": str(exc)})
    return {
        "global_enabled": AUTO_DISCOVERY_ENABLED,
        "calls_per_job_estimate": SERPER_CALLS_PER_JOB_ESTIMATE,
        "kbs": results,
    }


@app.put("/api/auto-discovery/config/{kb}")
async def api_auto_discovery_config(kb: str, request: Request):
    """Upsert per-KB auto-discovery configuration."""
    if kb not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb}")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")

    # Validate optional fields
    allowed = {"enabled", "daily_budget", "max_per_hour", "strategy", "seed_topics", "llm_sample"}
    unknown = set(body.keys()) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown fields: {sorted(unknown)}")

    valid_strategies = (
        # Baseline KG / LLM / hybrid
        "kg_entities", "llm", "hybrid",
        # Self-mining strategies
        "all", "contradiction", "stale", "orphan_entity",
        "question", "blind_spot", "stub", "wikilink",
    )
    if "strategy" in body and body["strategy"] not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"strategy must be one of: {', '.join(valid_strategies)}",
        )
    for int_field in ("daily_budget", "max_per_hour", "llm_sample"):
        if int_field in body:
            try:
                body[int_field] = int(body[int_field])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail=f"{int_field} must be an integer",
                )
            if body[int_field] < 0:
                raise HTTPException(
                    status_code=400, detail=f"{int_field} must be non-negative",
                )
    if "seed_topics" in body and body["seed_topics"] is not None:
        if not isinstance(body["seed_topics"], list):
            raise HTTPException(status_code=400, detail="seed_topics must be a list of strings or null")
        body["seed_topics"] = [str(s) for s in body["seed_topics"]]

    cfg = await db.upsert_auto_discovery_config(kb, **body)
    return {"kb": kb, "config": cfg}


@app.post("/api/auto-discovery/run/{kb}")
async def api_auto_discovery_run(kb: str, request: Request):
    """Trigger a refill + enqueue cycle for a single KB via the worker.

    The discovery pipeline does heavy DynamoDB scans + LLM calls for the
    contradiction / stale / orphan / blind-spot strategies, so running
    it synchronously inside the web handler blocks the event loop for
    minutes. We hand it off to the arq queue instead and return
    immediately — the user can poll ``/api/auto-discovery/status`` or
    watch the last_run snapshot in Redis.
    """
    if kb not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb}")
    redis = request.app.state.redis
    await redis.enqueue_job(
        "auto_discovery_single_kb_task", kb,
        _queue_name=ARQ_QUEUE_NAME,
    )
    return {"status": "queued", "kb": kb}


# --- Per-KB settings -------------------------------------------------------


@app.get("/api/kb-settings/{kb}")
async def api_kb_settings_get(kb: str):
    """Return per-KB LLM provider/model + persona settings."""
    if kb not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb}")
    cfg = await db.get_kb_settings(kb)
    return {"kb": kb, "config": cfg}


@app.put("/api/kb-settings/{kb}")
async def api_kb_settings_put(kb: str, request: Request):
    """Upsert per-KB LLM and persona settings.

    Body fields (all optional, all nullable — empty string clears):
        synthesis_provider, synthesis_model,
        query_provider, query_model,
        persona
    """
    if kb not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb}")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")

    allowed = {
        "synthesis_provider", "synthesis_model",
        "query_provider", "query_model",
        "persona",
    }
    unknown = set(body.keys()) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown fields: {sorted(unknown)}")

    valid_providers = {"minimax", "bedrock", "kimi"}
    for prov_field in ("synthesis_provider", "query_provider"):
        val = body.get(prov_field)
        if val and val not in valid_providers:
            raise HTTPException(
                status_code=400,
                detail=f"{prov_field} must be one of: {sorted(valid_providers)}",
            )

    cfg = await db.upsert_kb_settings(kb, **body)
    return {"kb": kb, "config": cfg}


@app.get("/api/research/history")
async def api_research_history():
    """Full research history from SQLite: jobs + article updates."""
    jobs = await db.get_jobs(limit=100)
    updates = await db.get_article_updates(limit=100)
    return {
        "jobs": [_job_to_api_format(j) for j in jobs],
        "article_updates": updates,
    }


@app.get("/api/research/sources/{job_id}")
async def api_research_sources(job_id: int):
    """Get all search result sources for a specific research job."""
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    sources = await db.get_sources(job_id)
    return {"job_id": job_id, "status": job["status"], "sources": sources}


@app.get("/research/review/{job_id}", response_class=HTMLResponse)
async def view_source_review(request: Request, job_id: int):
    """NotebookLM-style source-review page.

    Renders all collected sources as cards grouped by tier, with toggle
    buttons to select/deselect each source. A "Synthesize selected"
    button at the bottom kicks off the synthesis pass via the existing
    POST /api/research/synthesize/{job_id} endpoint.
    """
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found")
    sources = await db.get_sources(job_id)
    return render(
        "source_review.html",
        job=job,
        sources=sources,
        kb_names=storage.list_kbs(),
    )


@app.put("/api/research/sources/{job_id}")
async def api_update_sources(job_id: int, request: Request):
    """Update source selection for a job awaiting review.

    Body: {"source_ids": [1, 2, 3], "selected": true}
    Or:   {"select_all": true/false}
    """
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "awaiting_review":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not awaiting review (status: {job['status']})",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if "select_all" in body:
        count = await db.select_all_sources(job_id, selected=bool(body["select_all"]))
        return {"job_id": job_id, "updated": count}

    source_ids = body.get("source_ids", [])
    selected = body.get("selected", True)
    if not isinstance(source_ids, list) or not source_ids:
        raise HTTPException(status_code=400, detail="source_ids must be a non-empty list")

    count = await db.update_source_selection(job_id, source_ids, selected=selected)
    return {"job_id": job_id, "updated": count}


@app.post("/api/research/synthesize/{job_id}")
async def api_synthesize(job_id: int, request: Request):
    """Trigger synthesis from selected sources after review.

    Body: {"kb": "personal"} (optional, defaults to "personal")
    """
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "awaiting_review":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not awaiting review (status: {job['status']})",
        )

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"

    redis = request.app.state.redis
    await redis.enqueue_job(
        "research_synthesize_task",
        job["topic"],
        job_id,
        kb_name,
        _queue_name=ARQ_QUEUE_NAME,
    )
    return {"job_id": job_id, "status": "queued_for_synthesis", "kb": kb_name}


@app.post("/api/quality/pass")
async def api_quality_pass(request: Request):
    """Run a quality improvement pass: find shallow articles, enrich, crosslink."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    max_articles = body.get("max_articles", 10) if isinstance(body, dict) else 10

    redis = request.app.state.redis
    await redis.enqueue_job(
        "quality_task", kb_name, max_articles,
        _queue_name="wikidelve",
    )
    return {"status": "queued", "kb": kb_name, "max_articles": max_articles}


@app.post("/api/quality/enrich/{kb_name}/{slug}")
async def api_enrich_article(request: Request, kb_name: str, slug: str):
    """Enrich a single article with Minimax."""
    redis = request.app.state.redis
    await redis.enqueue_job(
        "enrich_task", kb_name, slug,
        _queue_name="wikidelve",
    )
    return {"status": "queued", "kb": kb_name, "slug": slug, "action": "enrich"}


@app.post("/api/quality/crosslink/{kb_name}/{slug}")
async def api_crosslink_article(request: Request, kb_name: str, slug: str):
    """Add crosslinks to a single article."""
    redis = request.app.state.redis
    await redis.enqueue_job(
        "crosslink_task", kb_name, slug,
        _queue_name="wikidelve",
    )
    return {"status": "queued", "kb": kb_name, "slug": slug, "action": "crosslink"}


@app.get("/api/quality/shallow/{kb_name}")
async def api_shallow_articles(kb_name: str):
    """List shallow/low-quality articles that need improvement."""
    from app.quality import find_shallow_articles
    return await asyncio.to_thread(find_shallow_articles, kb_name)


@app.get("/api/quality/duplicates/{kb_name}")
async def api_duplicates(kb_name: str):
    """Find duplicate/overlapping articles."""
    from app.quality import find_duplicates
    return await asyncio.to_thread(find_duplicates, kb_name)


@app.get("/api/quality/broken-wikilinks/{kb_name}")
async def api_broken_wikilinks(kb_name: str):
    """Find [[Wikilinks]] that point at articles which don't exist yet."""
    from app.quality import find_broken_wikilinks
    return await asyncio.to_thread(find_broken_wikilinks, kb_name)


@app.post("/api/quality/broken-wikilinks/{kb_name}/research")
async def api_broken_wikilinks_research(request: Request, kb_name: str):
    """Queue research jobs for every missing wikilink target in ``kb_name``.

    Body (optional): ``{"targets": ["slug-a", "slug-b"]}`` — a subset of
    target slugs from the broken-wikilinks report. If omitted, every
    missing target is queued (the scanner's full list).
    """
    from app.quality import find_broken_wikilinks

    try:
        body = await request.json()
    except Exception:
        body = {}
    selected_slugs: set[str] | None = None
    if isinstance(body, dict) and isinstance(body.get("targets"), list):
        selected_slugs = {
            s for s in body["targets"] if isinstance(s, str) and s
        }

    report = await asyncio.to_thread(find_broken_wikilinks, kb_name)
    redis = request.app.state.redis

    queued: list[dict] = []
    skipped: list[dict] = []
    for target in report.get("by_target", []):
        if selected_slugs is not None and target["slug"] not in selected_slugs:
            continue
        topic = target["target"].strip()
        if len(topic) < 10:
            # Too short for meaningful research; pad with KB context so the
            # research pipeline has something to work with.
            topic = f"{topic} overview and reference for {kb_name}"
        # Cooldown check — don't re-research something already in-flight.
        existing = await db.check_cooldown(topic)
        if existing:
            skipped.append({"topic": topic, "existing_job_id": existing["id"]})
            continue
        job_id = await db.create_job(topic)
        await redis.enqueue_job(
            "research_task", topic, kb_name, job_id,
            _queue_name="wikidelve",
        )
        queued.append({"job_id": job_id, "topic": topic, "slug": target["slug"]})

    return {"kb": kb_name, "queued": queued, "skipped": skipped}


@app.post("/api/quality/factcheck/{kb_name}/{slug}")
async def api_fact_check(request: Request, kb_name: str, slug: str):
    """Fact-check an article: extract claims, verify against Serper."""
    redis = request.app.state.redis
    await redis.enqueue_job("fact_check_task", kb_name, slug, _queue_name="wikidelve")
    return {"status": "queued", "kb": kb_name, "slug": slug, "action": "fact_check"}


@app.post("/api/quality/freshness/{kb_name}/{slug}")
async def api_freshness_audit(request: Request, kb_name: str, slug: str):
    """Freshness audit: check for stale content and auto-update."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    auto_update = body.get("auto_update", True) if isinstance(body, dict) else True

    redis = request.app.state.redis
    await redis.enqueue_job(
        "freshness_task", kb_name, slug, auto_update,
        _queue_name="wikidelve",
    )
    return {"status": "queued", "kb": kb_name, "slug": slug, "action": "freshness_audit", "auto_update": auto_update}


@app.post("/api/quality/freshness-batch")
async def api_freshness_batch(request: Request):
    """Batch freshness audit across stalest articles in a KB."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    max_articles = body.get("max_articles", 10) if isinstance(body, dict) else 10
    auto_update = body.get("auto_update", True) if isinstance(body, dict) else True

    redis = request.app.state.redis
    await redis.enqueue_job(
        "freshness_batch_task", kb_name, max_articles, auto_update,
        _queue_name="wikidelve",
    )
    return {"status": "queued", "kb": kb_name, "max_articles": max_articles, "auto_update": auto_update}


@app.post("/api/github/check-releases")
async def api_github_releases(request: Request):
    """Check tracked repos for new releases and auto-research them."""
    redis = request.app.state.redis
    await redis.enqueue_job("github_releases_task", _queue_name="wikidelve")
    return {"status": "queued", "action": "check_releases"}


@app.post("/api/github/index-repos")
async def api_github_index(request: Request):
    """Index READMEs and changelogs from your own repos."""
    redis = request.app.state.redis
    await redis.enqueue_job("github_index_task", _queue_name="wikidelve")
    return {"status": "queued", "action": "index_repos"}


@app.get("/api/github/tracked")
async def api_github_tracked():
    """List all tracked GitHub repos."""
    from app.github_monitor import TRACKED_REPOS, OWN_REPOS
    return {"tracked": TRACKED_REPOS, "own": OWN_REPOS}


@app.get("/api/status")
async def api_grokmaxx_status():
    """GrokMaxxing status — are workers busy? How many jobs pending?"""
    job_stats = await db.get_job_stats()
    all_stats = {}
    total_articles = 0
    total_words = 0
    for kb_name in storage.list_kbs():
        articles = await asyncio.to_thread(get_articles, kb_name)
        words = sum(a["word_count"] for a in articles)
        all_stats[kb_name] = {"articles": len(articles), "words": words}
        total_articles += len(articles)
        total_words += words

    return {
        "grokmaxxing": job_stats.get("active", 0) > 0,
        "jobs": {
            "total": job_stats.get("total", 0),
            "complete": job_stats.get("complete", 0),
            "active": job_stats.get("active", 0),
            "queued": job_stats.get("queued", 0),
            "awaiting_review": job_stats.get("awaiting_review", 0),
            "errors": job_stats.get("errors", 0),
            "cancelled": job_stats.get("cancelled", 0),
            "words_generated": job_stats.get("total_words", 0),
            "added_to_wiki": job_stats.get("added_to_wiki", 0),
        },
        "wiki": {
            "total_articles": total_articles,
            "total_words": total_words,
            "kbs": all_stats,
        },
    }


@app.get("/api/quality/scores/{kb_name}")
async def api_quality_scores(kb_name: str):
    """Get quality scores for all articles in a KB, worst first."""
    from app.quality import score_all_articles
    scores = await asyncio.to_thread(score_all_articles, kb_name)
    avg = sum(s["score"] for s in scores) / max(len(scores), 1)
    return {
        "kb": kb_name,
        "average_score": round(avg, 1),
        "total_articles": len(scores),
        "distribution": {
            "excellent (80-100)": sum(1 for s in scores if s["score"] >= 80),
            "good (60-79)": sum(1 for s in scores if 60 <= s["score"] < 80),
            "fair (40-59)": sum(1 for s in scores if 40 <= s["score"] < 60),
            "poor (20-39)": sum(1 for s in scores if 20 <= s["score"] < 40),
            "bad (0-19)": sum(1 for s in scores if s["score"] < 20),
        },
        "worst_10": scores[:10],
        "best_10": scores[-10:][::-1],
    }


@app.get("/api/quality/score/{kb_name}/{slug}")
async def api_quality_score_single(kb_name: str, slug: str):
    """Get quality score for a single article with full breakdown."""
    from app.quality import score_article_quality
    return await asyncio.to_thread(score_article_quality, kb_name, slug)


@app.get("/api/quality/wikilinks/{kb_name}")
async def api_check_wikilinks(kb_name: str):
    """Check all articles for broken wikilinks and suggest fixes."""
    from app.quality import check_wikilinks
    return await asyncio.to_thread(check_wikilinks, kb_name)


@app.post("/api/quality/wikilinks/{kb_name}")
async def api_fix_wikilinks(kb_name: str, request: Request):
    """Fix broken wikilinks across all articles in a KB.

    Body: {"slug": "optional-single-article", "auto_remove": true}
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    slug = body.get("slug") if isinstance(body, dict) else None
    auto_remove = body.get("auto_remove", True) if isinstance(body, dict) else True

    from app.quality import fix_wikilinks
    return await asyncio.to_thread(fix_wikilinks, kb_name, slug, auto_remove)


# --- Article version history + diff ----------------------------------------


@app.get("/api/articles/{kb_name}/{slug}/versions")
async def api_article_versions(kb_name: str, slug: str):
    """Return historical snapshots of an article (newest first).

    Snapshots are written automatically by ``update_article`` just before
    the file is overwritten. Each row contains a full content blob — the
    diff endpoint computes per-pair diffs on demand.
    """
    if kb_name not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb_name}")
    versions = await db.get_article_versions(kb_name, slug, limit=50)
    # Strip the heavy ``full_content`` field from the list view — clients
    # fetch individual versions on demand.
    summary = [
        {
            "id": v["id"],
            "kb": v["kb"],
            "article_slug": v["article_slug"],
            "change_type": v.get("change_type"),
            "job_id": v.get("job_id"),
            "content_hash": v.get("content_hash"),
            "created_at": v.get("created_at"),
            "note": v.get("note"),
            "word_count": len((v.get("full_content") or "").split()),
        }
        for v in versions
    ]
    return {"kb": kb_name, "slug": slug, "versions": summary}


@app.get("/api/articles/{kb_name}/{slug}/versions/{version_id}")
async def api_article_version(kb_name: str, slug: str, version_id: int):
    """Return the full snapshot of a single historical version."""
    if kb_name not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb_name}")
    version = await db.get_article_version_by_id(version_id)
    if not version or version["kb"] != kb_name or version["article_slug"] != slug:
        raise HTTPException(status_code=404, detail="Version not found")
    return version


@app.get("/api/articles/{kb_name}/{slug}/versions/{version_id}/diff")
async def api_article_version_diff(kb_name: str, slug: str, version_id: int):
    """Unified diff between a historical snapshot and the current article.

    Returns a list of diff lines suitable for ``<pre>`` rendering. The
    diff is computed on the fly using ``difflib.unified_diff`` — no
    pre-computed diffs are stored anywhere.
    """
    import difflib

    if kb_name not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb_name}")

    version = await db.get_article_version_by_id(version_id)
    if not version or version["kb"] != kb_name or version["article_slug"] != slug:
        raise HTTPException(status_code=404, detail="Version not found")

    # Read the *current* file from disk for the "after" side of the diff
    current_article = get_article(kb_name, slug)
    if not current_article:
        raise HTTPException(status_code=404, detail="Current article not found")

    before = (version.get("full_content") or "").splitlines(keepends=False)
    # Re-read raw markdown without the wikilink rewrite so the diff is
    # apples-to-apples with the snapshot (which is the raw file body).
    raw_current = read_article_text(kb_name, slug) or current_article.get("raw_markdown", "")
    after = raw_current.splitlines(keepends=False)

    diff_lines = list(
        difflib.unified_diff(
            before,
            after,
            fromfile=f"{slug}@{version.get('created_at', 'snapshot')}",
            tofile=f"{slug}@current",
            lineterm="",
            n=3,
        )
    )

    # Compact stats so the UI can show "+12 / -3" without parsing the diff
    additions = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

    return {
        "kb": kb_name,
        "slug": slug,
        "version_id": version_id,
        "snapshot_at": version.get("created_at"),
        "additions": additions,
        "deletions": deletions,
        "diff": diff_lines,
    }


# --- "You might also research" suggestions ---------------------------------


@app.get("/api/articles/{kb_name}/{slug}/suggestions")
async def api_article_suggestions(kb_name: str, slug: str):
    """Surface related research topics for an article.

    Combines three signal sources:
      1. Hybrid search using the article title as a query (FTS+vector+graph)
      2. Knowledge graph neighbours from get_related_by_graph
      3. Pending topic_candidate rows whose source_ref points at this slug

    Results are deduped by slug. Limit 12.
    """
    if kb_name not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"Unknown KB: {kb_name}")
    article = get_article(kb_name, slug)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    title = article.get("title") or slug.replace("-", " ")

    suggestions: list[dict] = []
    seen: set[str] = {slug}

    # 1) Hybrid search using the title — surfaces semantically similar articles
    try:
        from app.hybrid_search import hybrid_search
        hits = await hybrid_search(title, kb_name=kb_name, limit=10)
        for h in hits:
            h_slug = h.get("slug")
            if not h_slug or h_slug in seen:
                continue
            seen.add(h_slug)
            suggestions.append({
                "kind": "related_article",
                "slug": h_slug,
                "kb": h.get("kb"),
                "title": h.get("title") or h_slug.replace("-", " ").title(),
                "url": f"/wiki/{h.get('kb')}/{h_slug}",
                "score": h.get("rrf_score", 0.0),
            })
    except Exception as exc:
        logger.warning("hybrid_search for suggestions failed: %s", exc)

    # 2) Knowledge graph neighbours
    try:
        from app.knowledge_graph import get_related_by_graph
        neighbours = await get_related_by_graph(slug, kb_name=kb_name, depth=2)
        for n in neighbours[:10]:
            n_slug = n.get("slug")
            if not n_slug or n_slug in seen:
                continue
            seen.add(n_slug)
            suggestions.append({
                "kind": "graph_neighbour",
                "slug": n_slug,
                "kb": n.get("kb"),
                "title": n_slug.replace("-", " ").title(),
                "url": f"/wiki/{n.get('kb')}/{n_slug}",
                "score": float(n.get("score", 0)),
                "hop": n.get("hop"),
            })
    except Exception as exc:
        logger.warning("graph neighbours for suggestions failed: %s", exc)

    # 3) Pending topic candidates linked to this article via source_ref
    try:
        pending = await db.get_pending_candidates(kb_name, 20)
        for cand in pending:
            ref = cand.get("source_ref") or ""
            if slug not in ref:
                continue
            suggestions.append({
                "kind": "pending_topic",
                "topic": cand.get("topic"),
                "source": cand.get("source"),
                "score": float(cand.get("score") or 0),
            })
    except Exception as exc:
        logger.warning("pending candidates for suggestions failed: %s", exc)

    return {
        "kb": kb_name,
        "slug": slug,
        "title": title,
        "suggestions": suggestions[:12],
    }


@app.get("/api/articles/{kb_name}/{slug}/related")
async def api_related_articles(kb_name: str, slug: str):
    """Find articles related to this one based on shared tags and wikilinks."""
    article = get_article(kb_name, slug)
    if not article:
        raise HTTPException(status_code=404)
    return await asyncio.to_thread(_compute_related_articles, kb_name, slug, article)


def _compute_related_articles(kb_name: str, slug: str, article: dict) -> dict:
    """Sync helper for api_related_articles.

    Runs on a worker thread so the blocking S3 reads don't starve the event
    loop. Tag overlap and outgoing wikilinks are cheap, so they drive the
    back-link probe: only candidates with a tag overlap OR an incoming
    wikilink get the expensive ``read_article_text`` back-link check.
    """
    import re as _re
    from app.wiki import _wikilink_to_slug

    all_articles = get_articles(kb_name)
    article_tags = set(article.get("tags", []))

    outgoing: set[str] = set()
    body = article.get("raw_markdown", "")
    for match in _re.findall(r'\[\[([^\]]+)\]\]', body):
        outgoing.add(_wikilink_to_slug(match))

    related: list[dict] = []
    for a in all_articles:
        if a["slug"] == slug:
            continue
        score = 0
        reasons: list[str] = []

        shared_tags = article_tags & set(a.get("tags", []))
        if shared_tags:
            score += len(shared_tags) * 3
            reasons.append(f"tags: {', '.join(shared_tags)}")

        if a["slug"] in outgoing:
            score += 10
            reasons.append("linked from this article")

        if score > 0:
            related.append({
                "slug": a["slug"],
                "title": a["title"],
                "score": score,
                "reasons": reasons,
                "tags": a.get("tags", [])[:5],
            })

    # Back-link probe only for the top tag/outgoing matches. Unbounded
    # scans across the whole KB were the single biggest blocker on the
    # hot article page.
    related.sort(key=lambda x: x["score"], reverse=True)
    for entry in related[:25]:
        other_text = read_article_text(kb_name, entry["slug"])
        if other_text and slug in other_text.lower():
            entry["score"] += 5
            entry["reasons"].append("links back to this article")

    related.sort(key=lambda x: x["score"], reverse=True)
    return {"article": slug, "related": related[:15]}


@app.get("/api/github/releases")
async def api_github_latest_releases():
    """Get latest releases for all tracked repos."""
    from app.github_monitor import check_all_releases
    return await check_all_releases()


@app.post("/api/media/youtube")
async def api_youtube_ingest(request: Request):
    """Ingest a YouTube video: extract transcript → synthesize → add to wiki.

    Body: {"url": "https://youtube.com/watch?v=..."} or {"urls": [...]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    urls = []
    if isinstance(body, dict):
        if body.get("url"):
            urls = [body["url"]]
        elif body.get("urls"):
            urls = body["urls"][:10]  # max 10 at once

    if not urls:
        raise HTTPException(status_code=400, detail="url or urls required")

    redis = request.app.state.redis
    jobs = []
    for url in urls:
        job_id = await db.create_job(f"YouTube: {url[:80]}")
        await redis.enqueue_job("youtube_task", url, job_id, _queue_name="wikidelve")
        jobs.append({"job_id": job_id, "url": url, "status": "queued"})

    return {"jobs": jobs, "count": len(jobs)}


@app.post("/api/ingest/document")
async def api_ingest_document(request: Request):
    """Ingest a PDF/ePub from URL: download → extract text → synthesize → wiki.

    Body: {"url": "https://example.com/book.pdf"} or {"urls": [...]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    urls = []
    if isinstance(body, dict):
        if body.get("url"):
            urls = [body["url"]]
        elif body.get("urls"):
            urls = body["urls"][:20]

    if not urls:
        raise HTTPException(status_code=400, detail="url or urls required")

    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"

    redis = request.app.state.redis
    jobs = []
    for url in urls:
        await redis.enqueue_job("ingest_document_task", url, kb_name, _queue_name="wikidelve")
        jobs.append({"url": url, "status": "queued"})

    return {"jobs": jobs, "count": len(jobs)}


@app.post("/api/ingest/directory")
async def api_ingest_directory(request: Request):
    """Crawl an open directory for PDFs/ePubs and ingest them.

    Body: {"url": "https://example.com/books/", "max_files": 10}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    url = body.get("url") if isinstance(body, dict) else None
    if not url:
        raise HTTPException(status_code=400, detail="url required")

    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    max_files = body.get("max_files", 10) if isinstance(body, dict) else 10

    redis = request.app.state.redis
    await redis.enqueue_job("ingest_directory_task", url, kb_name, min(max_files, 50), _queue_name="wikidelve")
    return {"status": "queued", "url": url, "max_files": max_files}


# --- Palace routes -----------------------------------------------------------

@app.get("/palace", response_class=HTMLResponse)
async def palace_page(request: Request):
    """Interactive palace view navigation."""
    resp = render("palace.html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.get("/api/palace/map")
async def api_palace_map(kb: str | None = None):
    """Get the full palace map or a single wing."""
    from app.palace import generate_palace_map
    return await generate_palace_map(kb)


@app.get("/api/palace/hall/{kb}/{hall}")
async def api_palace_hall(kb: str, hall: str):
    """List articles in a specific hall."""
    classifications, articles = await asyncio.gather(
        db.get_classifications_by_hall(kb, hall),
        asyncio.to_thread(get_articles, kb),
    )
    article_map = {a["slug"]: a for a in articles}
    results = []
    for c in classifications:
        art = article_map.get(c["slug"], {})
        results.append({
            "slug": c["slug"],
            "kb": c["kb"],
            "hall": c["hall"],
            "confidence": c["confidence"],
            "title": art.get("title", c["slug"]),
            "summary": art.get("summary", ""),
            "word_count": art.get("word_count", 0),
            "tags": art.get("tags", []),
        })
    return {"hall": hall, "kb": kb, "articles": results, "count": len(results)}


@app.get("/api/palace/room/{kb}/{room_name}")
async def api_palace_room(kb: str, room_name: str):
    """List articles in a specific room."""
    rooms = await db.get_rooms(kb)
    room = next((r for r in rooms if r["name"] == room_name), None)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    # Parallelize: members, articles, and classifications in flight together.
    members, articles, classifications = await asyncio.gather(
        db.get_room_members(room["id"]),
        asyncio.to_thread(get_articles, kb),
        db.get_classifications(kb),
    )
    article_map = {a["slug"]: a for a in articles}
    slug_to_hall = {c["slug"]: c["hall"] for c in classifications}
    results = []
    for m in members:
        art = article_map.get(m["slug"], {})
        results.append({
            "slug": m["slug"],
            "kb": m["kb"],
            "relevance": m["relevance"],
            "hall": slug_to_hall.get(m["slug"]),
            "title": art.get("title", m["slug"]),
            "summary": art.get("summary", ""),
            "word_count": art.get("word_count", 0),
        })
    return {"room": room_name, "kb": kb, "articles": results, "count": len(results)}


@app.get("/api/palace/article/{kb}/{slug}")
async def api_palace_article(kb: str, slug: str):
    """Get palace context for a single article."""
    classification = await db.get_article_classification(slug, kb)
    rooms = await db.get_article_rooms(slug, kb)
    return {
        "slug": slug,
        "kb": kb,
        "hall": classification["hall"] if classification else None,
        "confidence": classification["confidence"] if classification else None,
        "rooms": [{"name": r["name"], "relevance": r["relevance"]} for r in rooms],
    }


@app.post("/api/palace/classify")
async def api_palace_classify(request: Request):
    """Trigger palace classification for all articles in a KB."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    redis = request.app.state.redis
    await redis.enqueue_job("palace_classify_task", kb_name, _queue_name=ARQ_QUEUE_NAME)
    return {"status": "queued", "kb": kb_name}


@app.post("/api/palace/cluster")
async def api_palace_cluster(request: Request):
    """Trigger room clustering from knowledge graph data."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    redis = request.app.state.redis
    await redis.enqueue_job("palace_cluster_task", kb_name, _queue_name=ARQ_QUEUE_NAME)
    return {"status": "queued", "kb": kb_name}


@app.get("/gm", include_in_schema=False)
async def _gm():
    s = await db.get_job_stats()
    return {"status": "we're so back", "mode": "max overdrive", "words": s.get("total_words", 0)}


@app.get("/api/stats")
async def api_stats():
    stats: dict[str, dict] = {}
    total_articles = 0
    total_words = 0
    for kb_name in storage.list_kbs():
        articles = await asyncio.to_thread(get_articles, kb_name)
        words = sum(a["word_count"] for a in articles)
        stats[kb_name] = {"articles": len(articles), "words": words}
        total_articles += len(articles)
        total_words += words

    jobs = await db.get_jobs(limit=1000)
    return {
        "kbs": stats,
        "total_articles": total_articles,
        "total_words": total_words,
        "research_jobs": len(jobs),
    }


# --- Chat sessions -----------------------------------------------------------

@app.get("/api/chat/sessions")
async def api_chat_sessions():
    """List recent chat sessions."""
    return await db.get_chat_sessions(limit=30)


@app.get("/api/chat/sessions/{session_id}")
async def api_chat_messages(session_id: str):
    """Get messages for a chat session."""
    return await db.get_chat_messages(session_id)


@app.post("/api/chat/sessions/{session_id}/messages")
async def api_add_chat_message(session_id: str, request: Request):
    """Add a message to a chat session."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    role = body.get("role", "user")
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content required")
    return await db.add_chat_message(session_id, role, content)


@app.post("/api/chat/sessions/{session_id}/ask")
async def api_chat_ask(session_id: str, request: Request):
    """Streaming RAG chat endpoint.

    Body: ``{question: str, kb?: str | null, use_tools?: bool = false}``

    Streams Server-Sent Events with ``event:`` types:
        - ``status``: lifecycle markers (retrieving, thinking, ...)
        - ``delta``: incremental token chunks from the LLM
        - ``sources``: the retrieved passages used to ground the answer
        - ``gap``: chat-gap topics queued for auto-discovery
        - ``done``: terminal event

    Persists the user message + final assistant message to ``chat_messages``.
    Parses ``[GAP] <topic>`` sentinels from the response and writes them
    to ``topic_candidates`` with ``source='chat_gap'``.
    """
    from app.chat import (
        retrieve_context, get_recent_history, build_chat_prompt,
        parse_gap_topics, CHAT_TOOLS, run_tool,
    )
    from app.llm import llm_chat_stream, llm_chat_tools
    from app.sse import sse_response

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="question is too short")
    # Upper bound prevents context-window stuffing / DoS via huge prompts.
    MAX_QUESTION_LEN = 4096
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"question exceeds {MAX_QUESTION_LEN} character limit",
        )

    kb = body.get("kb")
    if kb is not None and (not isinstance(kb, str) or not kb.strip()):
        raise HTTPException(status_code=400, detail="kb must be a non-empty string or null")
    use_tools = bool(body.get("use_tools", False))

    redis = request.app.state.redis

    async def stream():
        # Persist the user turn first so the session is auto-created and
        # subsequent /sessions/{id} reads see it.
        try:
            await db.add_chat_message(session_id, "user", question)
        except Exception as exc:
            logger.warning("Failed to persist user message: %s", exc)

        yield {"event": "status", "data": "retrieving"}

        passages = await retrieve_context(question, kb=kb)
        history = await get_recent_history(session_id)

        # Retrieval audit log: forensic trail for which chunks the chat
        # pulled for a given question, and how many (if any) tripped
        # the injection detector. Lives in the chat_events table next
        # to the frontend observability events.
        flagged = sum(1 for p in passages if p.get("injection_flags"))
        try:
            await db.log_chat_event(
                event="retrieve",
                session_id=session_id,
                user_input=question[:500],
                result=(
                    f"passages={len(passages)} flagged={flagged} "
                    f"kb={kb or 'all'}"
                ),
            )
        except Exception as exc:
            logger.debug("retrieval audit log failed: %s", exc)

        yield {
            "event": "sources",
            "data": [
                {
                    "slug": p["slug"],
                    "kb": p["kb"],
                    "title": p.get("title"),
                    "url": p.get("url"),
                    "score": p.get("score"),
                }
                for p in passages
            ],
        }

        system_msg, user_msg = build_chat_prompt(question, passages, history)

        # Optional tool-use loop. v1: tools run non-streaming, only the
        # final text answer streams.
        if use_tools:
            yield {"event": "status", "data": "thinking_with_tools"}
            tool_messages: list[dict] = []
            for round_idx in range(3):
                augmented_user = user_msg
                if tool_messages:
                    augmented_user = (
                        user_msg
                        + "\n\n## Tool results so far\n"
                        + "\n".join(
                            f"- {m['name']}({m.get('input')}) → {m.get('result')}"
                            for m in tool_messages
                        )
                    )
                try:
                    result = await llm_chat_tools(
                        system_msg, augmented_user, CHAT_TOOLS,
                        max_tokens=2000, temperature=0.2,
                    )
                except Exception as exc:
                    yield {"event": "error", "data": str(exc)}
                    yield {"event": "done", "data": "error"}
                    return

                if result.get("type") == "tool_use":
                    tool_name = result.get("name", "")
                    tool_input = result.get("input") or {}
                    yield {
                        "event": "tool_use",
                        "data": {"name": tool_name, "input": tool_input},
                    }
                    tool_result = await run_tool(tool_name, tool_input, redis=redis)
                    tool_messages.append(
                        {"name": tool_name, "input": tool_input, "result": tool_result}
                    )
                    continue

                # Final text from the tool loop — done; fall through to streaming
                # the final answer using the regular path.
                user_msg = augmented_user
                break

        yield {"event": "status", "data": "thinking"}

        buffer_parts: list[str] = []
        try:
            async for chunk in llm_chat_stream(
                system_msg, user_msg, max_tokens=2000, temperature=0.2,
            ):
                buffer_parts.append(chunk)
                yield {"event": "delta", "data": chunk}
        except Exception as exc:
            logger.exception("Chat stream failed")
            yield {"event": "error", "data": str(exc)}
            yield {"event": "done", "data": "error"}
            return

        full_answer = "".join(buffer_parts)

        # Persist the assistant message
        try:
            await db.add_chat_message(session_id, "assistant", full_answer)
        except Exception as exc:
            logger.warning("Failed to persist assistant message: %s", exc)

        # Parse [GAP] sentinels and queue them for auto-discovery
        gaps = parse_gap_topics(full_answer)
        gap_kb = kb or "personal"
        for gap_topic in gaps:
            try:
                inserted = await db.insert_topic_candidate(
                    gap_kb, gap_topic, "chat_gap", session_id, 1.0,
                )
                if inserted:
                    yield {
                        "event": "gap",
                        "data": {"kb": gap_kb, "topic": gap_topic},
                    }
            except Exception as exc:
                logger.warning("Failed to insert chat_gap candidate: %s", exc)

        yield {"event": "done", "data": "complete"}

    return sse_response(stream(), request=request)


@app.delete("/api/chat/sessions/{session_id}")
async def api_delete_chat_session(session_id: str):
    """Delete a chat session."""
    await db.delete_chat_session(session_id)
    return {"status": "deleted", "session_id": session_id}


# --- Chat observability ------------------------------------------------------

@app.post("/api/chat/event")
async def api_log_chat_event(request: Request):
    """Log a chat interaction event from the frontend."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    event = body.get("event", "")
    if not event:
        raise HTTPException(status_code=400, detail="event required")

    log_chat_interaction(
        event=event,
        session_id=body.get("session_id"),
        user_input=body.get("user_input"),
        command=body.get("command"),
        action=body.get("action"),
    )
    await db.log_chat_event(
        event=event,
        session_id=body.get("session_id"),
        user_input=body.get("user_input"),
        command=body.get("command"),
        action=body.get("action"),
        result=body.get("result"),
        error=body.get("error"),
    )
    return {"status": "logged"}


@app.get("/api/chat/analytics")
async def api_chat_analytics():
    """Chat analytics dashboard data."""
    return await db.get_chat_analytics()


@app.get("/api/chat/events")
async def api_chat_events(limit: int = 100, event: str | None = None):
    """List recent chat events."""
    return await db.get_chat_events(limit=limit, event_type=event)


# --- Title refinement -------------------------------------------------------

@app.post("/api/articles/refine-titles")
async def api_refine_titles(request: Request):
    """Batch-refine article titles using content headings.

    Reads each article's markdown, extracts the clean title from the
    first # heading, and updates the frontmatter if it differs.
    Body: {"kb": "personal", "dry_run": true}
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    kb_name = body.get("kb", "personal") if isinstance(body, dict) else "personal"
    dry_run = body.get("dry_run", False) if isinstance(body, dict) else False

    if kb_name not in storage.list_kbs():
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_name}")

    changes, total = await asyncio.to_thread(_refine_titles_sync, kb_name, dry_run)

    return {
        "kb": kb_name,
        "dry_run": dry_run,
        "total": total,
        "updated": len(changes),
        "changes": changes[:50],
    }


def _refine_titles_sync(kb_name: str, dry_run: bool) -> tuple[list[dict], int]:
    """Sync worker for api_refine_titles — iterates and rewrites on a thread."""
    from app.wiki import (
        extract_title,
        parse_frontmatter,
        _serialize_frontmatter,
        write_article_text,
    )

    changes: list[dict] = []
    total = 0
    for slug, text in storage.iter_articles(kb_name, subdir="wiki"):
        total += 1
        meta, body_text = parse_frontmatter(text)
        old_title = meta.get("title", slug)
        new_title = extract_title(body_text, str(old_title))
        if new_title and new_title != str(old_title) and len(new_title) > 3:
            changes.append({
                "slug": slug,
                "old_title": str(old_title),
                "new_title": new_title,
            })
            if not dry_run:
                meta["title"] = new_title
                fm_str = _serialize_frontmatter(meta)
                write_article_text(kb_name, slug, fm_str + "\n\n" + body_text)
    return changes, total


# --- KB management routes ---------------------------------------------------

@app.get("/api/kbs")
async def api_list_kbs():
    """List all knowledge bases with article counts."""
    result = []
    for name in storage.list_kbs():
        articles = await asyncio.to_thread(get_articles, name)
        result.append({
            "name": name,
            "articles": len(articles),
            "words": sum(a["word_count"] for a in articles),
        })
    return result


@app.post("/api/kbs")
async def api_create_kb(request: Request):
    """Create a new knowledge base.

    Body: {"name": "my-new-kb"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    name = body.get("name", "").strip() if isinstance(body, dict) else ""
    if not name or len(name) < 2:
        raise HTTPException(status_code=400, detail="name is required (min 2 chars)")

    from app.config import register_kb
    try:
        kb_path = register_kb(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Rebuild search index to pick up new KB
    try:
        await build_search_index()
    except Exception:
        pass

    safe_name = kb_path.name
    return {"name": safe_name, "path": str(kb_path), "status": "created"}


# --- Storage backend introspection ------------------------------------------

@app.get("/api/storage/status")
async def api_storage_status():
    """Report the active storage / db / vector backends and discovered KBs."""
    import os as _os
    return {
        "storage_backend": storage.backend_name(),
        "db_backend": _os.getenv("DB_BACKEND", "sqlite"),
        "vector_backend": _os.getenv("VECTOR_BACKEND", "sqlite"),
        "s3_bucket": _os.getenv("S3_BUCKET") or None,
        "s3_vectors_bucket": _os.getenv("S3_VECTORS_BUCKET") or None,
        "dynamodb_table": _os.getenv("DYNAMODB_TABLE") or None,
        "kbs": storage.list_kbs(),
    }


# --- Helpers ----------------------------------------------------------------

def _build_slug_index() -> dict[str, str]:
    """Map every known article slug to its KB. Used once per jobs request."""
    all_slugs: dict[str, str] = {}
    for kb_name in storage.list_kbs():
        for article in get_articles(kb_name):
            all_slugs[article["slug"]] = kb_name
    return all_slugs


def _job_to_api_format(
    job: dict,
    slug_index: dict[str, str] | None = None,
    resolve_slug: bool = True,
) -> dict:
    """Convert a DB job row to the API response format.

    ``slug_index`` lets callers pass a pre-built slug → kb map so a list
    of 200 jobs doesn't scan every KB 200 times. ``resolve_slug=False``
    skips the lookup entirely for hot paths (e.g. the admin dashboard)
    where the caller doesn't need the resolved article link.
    """
    wiki_slug: str | None = job.get("wiki_slug") or None
    wiki_kb: str | None = job.get("wiki_kb") or None
    # Fall back to fuzzy reconstruction only for legacy rows that were
    # written before we started persisting the resolved slug (pre-fix
    # jobs set added_to_wiki=1 without recording which slug won).
    if resolve_slug and job.get("added_to_wiki") and not wiki_slug:
        wiki_slug, wiki_kb = _find_article_slug(job["topic"], slug_index=slug_index)

    return {
        "id": job["id"],
        "topic": job["topic"],
        "status": job["status"],
        "created": job["created_at"],
        "completed": job.get("completed_at"),
        "sources": job.get("sources_count", 0),
        "word_count": job.get("word_count", 0),
        "error": job.get("error"),
        "added_to_wiki": bool(job.get("added_to_wiki")),
        "content": job.get("content"),
        "wiki_slug": wiki_slug,
        "wiki_kb": wiki_kb or "personal",
    }


def _find_article_slug(
    topic: str,
    slug_index: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Find the best matching article slug for a research topic.

    Callers that format many jobs in a single request should pass
    ``slug_index`` from ``_build_slug_index()`` so the full-KB scan only
    happens once per request.
    """
    # Strip the synthetic prefixes that local/YouTube pipelines stick
    # on the topic field — otherwise "local:my auth flow" derives the
    # bogus candidate "local_my-auth-flow" which never matches what
    # create_article actually wrote (slug comes from the content's
    # first heading, not this prefixed string).
    clean_topic = topic
    for prefix in ("local:", "youtube:", "YouTube:"):
        if clean_topic.lower().startswith(prefix.lower()):
            clean_topic = clean_topic[len(prefix):].strip()
            break

    safe = "".join(c if c.isalnum() or c in " -" else "_" for c in clean_topic)[:80]
    candidate = safe.lower().replace(" ", "-").strip("-")
    if not candidate:
        return None, None

    all_slugs = slug_index if slug_index is not None else _build_slug_index()

    if candidate in all_slugs:
        return candidate, all_slugs[candidate]

    parts = candidate.split("-")
    for length in range(len(parts) - 1, 0, -1):
        prefix = "-".join(parts[:length])
        if prefix in all_slugs:
            return prefix, all_slugs[prefix]
        for slug in all_slugs:
            if slug.startswith(prefix):
                return slug, all_slugs[slug]

    # Fallback: fuzzy match via wiki module. Skip when we're operating
    # off a shared slug_index — callers in that mode are listing many
    # jobs and fuzzy-matching every miss would reintroduce the quadratic
    # behaviour the index is meant to prevent.
    if slug_index is None:
        from app.wiki import find_related_article
        for kb_name in storage.list_kbs():
            match = find_related_article(kb_name, topic)
            if match:
                return match["slug"], kb_name

    return candidate, None


def _jobs_to_template_format(jobs: list[dict]) -> list[dict]:
    """Convert DB job rows to the format expected by Jinja templates."""
    slug_index = _build_slug_index()
    return [_job_to_api_format(j, slug_index=slug_index) for j in jobs]
