"""In-process metrics registry + LLM token usage tracker.

Keeps it simple: counters/gauges/histograms live in a single thread-safe
registry and a parallel token-usage tracker buckets LLM calls by day so
the admin dashboard and /metrics endpoint can both read from one source.

No external Prometheus client dep — we emit the text format ourselves.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable


_LOCK = threading.Lock()

# name -> {label_tuple: float}.  label_tuple is a sorted tuple of (k, v) pairs.
_counters: dict[str, dict[tuple, float]] = defaultdict(dict)
_gauges: dict[str, dict[tuple, float]] = defaultdict(dict)
# name -> {label_tuple: [sum, count, [bucket_counts]]}
_histograms: dict[str, dict[tuple, list]] = defaultdict(dict)

_HELP: dict[str, str] = {}
_TYPES: dict[str, str] = {}

# Latency histogram buckets (seconds). Covers the whole range we care
# about — sub-ms health checks up to multi-minute research jobs.
DEFAULT_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0, 30.0, 60.0, 300.0,
)


def _labels_key(labels: dict | None) -> tuple:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _register(name: str, help_text: str, metric_type: str) -> None:
    _HELP.setdefault(name, help_text)
    _TYPES.setdefault(name, metric_type)


def inc_counter(name: str, value: float = 1.0, labels: dict | None = None, *, help: str = "") -> None:
    key = _labels_key(labels)
    with _LOCK:
        _register(name, help, "counter")
        _counters[name][key] = _counters[name].get(key, 0.0) + value


def set_gauge(name: str, value: float, labels: dict | None = None, *, help: str = "") -> None:
    key = _labels_key(labels)
    with _LOCK:
        _register(name, help, "gauge")
        _gauges[name][key] = value


def observe_histogram(
    name: str,
    value: float,
    labels: dict | None = None,
    *,
    help: str = "",
    buckets: Iterable[float] = DEFAULT_BUCKETS,
) -> None:
    key = _labels_key(labels)
    buckets = list(buckets)
    with _LOCK:
        _register(name, help, "histogram")
        entry = _histograms[name].get(key)
        if entry is None:
            entry = [0.0, 0, [0] * len(buckets), buckets]
            _histograms[name][key] = entry
        entry[0] += value
        entry[1] += 1
        for i, b in enumerate(buckets):
            if value <= b:
                entry[2][i] += 1


# ---------------------------------------------------------------------------
# LLM token usage tracking
# ---------------------------------------------------------------------------

# Best-effort pricing per 1M tokens (USD). Override via env if needed.
# Zero means "don't display a cost figure" (we still show counts).
_PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, kind) -> (input_per_mtok, output_per_mtok)
    ("minimax", "chat"): (0.20, 1.10),       # MiniMax-Text-01 ballpark
    ("minimax", "embed"): (0.01, 0.0),
    ("bedrock", "chat"): (3.0, 15.0),        # Claude Sonnet ballpark
    ("bedrock", "embed"): (0.10, 0.0),
}


# day_str -> {(provider, model, kind) -> {"calls": n, "input": n, "output": n}}
_usage_by_day: dict[str, dict[tuple, dict]] = defaultdict(dict)
USAGE_RETENTION_DAYS = 14


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _prune_old_days(today: str) -> None:
    cutoff_epoch = time.time() - (USAGE_RETENTION_DAYS * 86400)
    for day in list(_usage_by_day):
        try:
            day_epoch = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            _usage_by_day.pop(day, None)
            continue
        if day_epoch < cutoff_epoch:
            _usage_by_day.pop(day, None)


def record_llm_call(
    *,
    provider: str,
    model: str,
    kind: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record a single LLM invocation. Safe to call from any thread.

    In-memory 14-day rolling window + Prometheus counters update
    synchronously; the all-time DB total is scheduled as a fire-and-forget
    task so request handlers don't block on a SQLite write.
    """
    key = (provider or "unknown", model or "unknown", kind or "chat")
    day = _today()
    with _LOCK:
        _prune_old_days(day)
        bucket = _usage_by_day[day].setdefault(
            key, {"calls": 0, "input": 0, "output": 0},
        )
        bucket["calls"] += 1
        bucket["input"] += max(0, int(input_tokens or 0))
        bucket["output"] += max(0, int(output_tokens or 0))

    # Mirror as Prometheus counters for /metrics scrapers.
    labels = {"provider": key[0], "model": key[1], "kind": key[2]}
    inc_counter("kb_llm_calls_total", 1.0, labels, help="LLM API calls by provider/model/kind")
    if input_tokens:
        inc_counter("kb_llm_input_tokens_total", float(input_tokens), labels, help="Total input tokens")
    if output_tokens:
        inc_counter("kb_llm_output_tokens_total", float(output_tokens), labels, help="Total output tokens")

    _schedule_persist_total(key[0], key[1], key[2], int(input_tokens or 0), int(output_tokens or 0))


def _schedule_persist_total(
    provider: str, model: str, kind: str,
    input_tokens: int, output_tokens: int,
) -> None:
    """Fire-and-forget the all-time DB upsert.

    Called from inside synchronous and async contexts alike — if a running
    loop exists we schedule a task; otherwise we create a fresh loop just
    long enough to run the coro. Any failure is swallowed (logged in db).
    """
    import asyncio

    async def _do():
        try:
            from app import db
            await db.record_llm_usage_total(
                provider, model, kind,
                input_tokens=input_tokens, output_tokens=output_tokens,
            )
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop.create_task(_do())
    else:
        try:
            asyncio.run(_do())
        except Exception:
            pass


async def get_all_time_totals() -> dict:
    """Aggregate the lifetime rows from the DB into dashboard shape."""
    from app import db
    try:
        rows = await db.get_llm_usage_totals()
    except Exception:
        return {"rows": [], "totals": {"calls": 0, "input": 0, "output": 0, "cost_usd": 0.0}}

    total_calls = 0
    total_input = 0
    total_output = 0
    total_cost = 0.0
    out_rows = []
    for r in rows:
        calls = int(r.get("calls", 0) or 0)
        in_tok = int(r.get("input_tokens", 0) or 0)
        out_tok = int(r.get("output_tokens", 0) or 0)
        cost = _estimate_cost(r.get("provider", ""), r.get("kind", ""), in_tok, out_tok)
        total_calls += calls
        total_input += in_tok
        total_output += out_tok
        total_cost += cost
        out_rows.append({
            "provider": r.get("provider", ""),
            "model": r.get("model", ""),
            "kind": r.get("kind", ""),
            "calls": calls,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(cost, 4),
            "first_seen": r.get("first_seen", ""),
            "last_seen": r.get("last_seen", ""),
        })

    return {
        "rows": out_rows,
        "totals": {
            "calls": total_calls,
            "input": total_input,
            "output": total_output,
            "cost_usd": round(total_cost, 4),
        },
    }


def _estimate_cost(provider: str, kind: str, input_tokens: int, output_tokens: int) -> float:
    rates = _PRICING.get((provider, kind))
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate


def get_usage_summary() -> dict:
    """Return a JSON-friendly snapshot for the admin dashboard."""
    today = _today()
    with _LOCK:
        _prune_old_days(today)
        days = sorted(_usage_by_day.keys(), reverse=True)
        per_day = []
        totals_week = {"calls": 0, "input": 0, "output": 0, "cost_usd": 0.0}
        for day in days[:7]:
            entries = _usage_by_day.get(day, {})
            day_calls = sum(e["calls"] for e in entries.values())
            day_in = sum(e["input"] for e in entries.values())
            day_out = sum(e["output"] for e in entries.values())
            day_cost = sum(
                _estimate_cost(p, k, e["input"], e["output"])
                for (p, _m, k), e in entries.items()
            )
            per_day.append({
                "date": day,
                "calls": day_calls,
                "input_tokens": day_in,
                "output_tokens": day_out,
                "cost_usd": round(day_cost, 4),
            })
            totals_week["calls"] += day_calls
            totals_week["input"] += day_in
            totals_week["output"] += day_out
            totals_week["cost_usd"] += day_cost

        today_entries = _usage_by_day.get(today, {})
        today_breakdown = [
            {
                "provider": p,
                "model": m,
                "kind": k,
                "calls": e["calls"],
                "input_tokens": e["input"],
                "output_tokens": e["output"],
                "cost_usd": round(_estimate_cost(p, k, e["input"], e["output"]), 4),
            }
            for (p, m, k), e in sorted(today_entries.items())
        ]

    return {
        "today": _today(),
        "per_day": per_day,
        "totals_week": {
            **totals_week,
            "cost_usd": round(totals_week["cost_usd"], 4),
        },
        "today_breakdown": today_breakdown,
    }


# ---------------------------------------------------------------------------
# Prometheus text rendering
# ---------------------------------------------------------------------------

def _format_labels(label_tuple: tuple, extra: dict | None = None) -> str:
    pairs = list(label_tuple)
    if extra:
        pairs.extend(sorted(extra.items()))
    if not pairs:
        return ""
    rendered = ",".join(f'{k}="{_escape_label(str(v))}"' for k, v in pairs)
    return "{" + rendered + "}"


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def prometheus_text() -> str:
    """Serialize the registry as Prometheus text exposition format."""
    out: list[str] = []
    with _LOCK:
        names = sorted(set(_counters) | set(_gauges) | set(_histograms))
        for name in names:
            help_text = _HELP.get(name, "")
            metric_type = _TYPES.get(name, "untyped")
            if help_text:
                out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} {metric_type}")

            if name in _counters:
                for labels, value in _counters[name].items():
                    out.append(f"{name}{_format_labels(labels)} {value}")
            elif name in _gauges:
                for labels, value in _gauges[name].items():
                    out.append(f"{name}{_format_labels(labels)} {value}")
            elif name in _histograms:
                for labels, entry in _histograms[name].items():
                    total, count, bucket_counts, buckets = entry
                    # bucket_counts are already cumulative (each observation
                    # bumps every bucket it satisfies), so emit them directly.
                    for b, bc in zip(buckets, bucket_counts):
                        out.append(
                            f"{name}_bucket{_format_labels(labels, {'le': str(b)})} {bc}"
                        )
                    out.append(
                        f"{name}_bucket{_format_labels(labels, {'le': '+Inf'})} {count}"
                    )
                    out.append(f"{name}_sum{_format_labels(labels)} {total}")
                    out.append(f"{name}_count{_format_labels(labels)} {count}")
    out.append("")
    return "\n".join(out)


def reset_all() -> None:
    """Test-only helper: clear every registered metric."""
    with _LOCK:
        _counters.clear()
        _gauges.clear()
        _histograms.clear()
        _HELP.clear()
        _TYPES.clear()
        _usage_by_day.clear()
