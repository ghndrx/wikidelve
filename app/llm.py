"""
Shared LLM abstraction layer.

Supports two providers:
  - minimax  (default) — Minimax API with bearer token auth
  - bedrock  — AWS Bedrock with IAM credentials or instance roles

Set LLM_PROVIDER=bedrock in .env to switch. Bedrock auth uses the standard
boto3 credential chain: env vars, ~/.aws/credentials, IAM instance role, etc.
"""

import json
import logging
import re
import unicodedata
from typing import AsyncIterator, Optional

import httpx

from app.config import (
    LLM_PROVIDER,
    # Minimax
    MINIMAX_API_KEY,
    MINIMAX_BASE,
    MINIMAX_MODEL,
    MINIMAX_TIMEOUT,
    # Kimi / Moonshot
    KIMI_API_KEY,
    KIMI_BASE,
    KIMI_MODEL,
    # Bedrock
    BEDROCK_API_KEY,
    BEDROCK_REGION,
    BEDROCK_MODEL,
    BEDROCK_EMBED_MODEL,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
)

logger = logging.getLogger("kb-service.llm")

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _strip_non_latin(text: str) -> str:
    """Remove CJK, Arabic, Cyrillic, and other non-Latin script characters.

    Preserves ASCII, Latin Extended, accented characters, and common symbols.
    """
    _BLOCKED_PREFIXES = ("CJK", "ARABIC", "CYRILLIC", "DEVANAGARI", "HANGUL",
                         "HIRAGANA", "KATAKANA", "THAI", "TIBETAN", "ETHIOPIC")
    def _ok(c):
        if ord(c) < 128:
            return True
        name = unicodedata.name(c, "")
        return not any(name.startswith(p) for p in _BLOCKED_PREFIXES)
    return "".join(c for c in text if _ok(c))


def _resolve_provider_model(
    provider: Optional[str], model: Optional[str],
) -> tuple[str, str]:
    """Resolve the active (provider, model) pair for a single call.

    Defaults fall back to the global LLM_PROVIDER env var and the matching
    provider's default model. Raises ValueError on unknown provider strings.
    """
    active_provider = (provider or LLM_PROVIDER or "minimax").lower()
    if active_provider not in ("minimax", "bedrock", "kimi"):
        raise ValueError(
            f"Unknown LLM provider: {active_provider!r}. "
            f"Supported: 'minimax', 'bedrock', 'kimi'."
        )
    if active_provider == "bedrock":
        active_model = model or BEDROCK_MODEL
    elif active_provider == "kimi":
        active_model = model or KIMI_MODEL
    else:
        active_model = model or MINIMAX_MODEL
    return active_provider, active_model


async def llm_chat(
    system_msg: str,
    user_msg: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    *,
    client: Optional[httpx.AsyncClient] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """Send a chat completion and return the content string.

    If *client* is provided it is reused (caller manages its lifetime),
    otherwise a fresh one is created per call.
    Optional *model* / *provider* kwargs override the global defaults for
    this single call (used by per-KB synthesis settings and chat retrieval).
    All output is sanitized to remove non-Latin script artifacts.
    """
    active_provider, active_model = _resolve_provider_model(provider, model)
    if active_provider == "bedrock":
        result = await _bedrock_chat(
            system_msg, user_msg, max_tokens, temperature, model=active_model,
        )
    elif active_provider == "kimi":
        result = await _kimi_chat(
            system_msg, user_msg, max_tokens, temperature,
            client=client, model=active_model,
        )
    else:
        result = await _minimax_chat(
            system_msg, user_msg, max_tokens, temperature,
            client=client, model=active_model,
        )

    return _strip_non_latin(result)


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding backend is in an open circuit state."""


class _EmbeddingCircuit:
    """Minimal circuit breaker for the embedding backend.

    After ``failure_threshold`` consecutive failures, the circuit opens for
    ``cooldown_seconds`` and every call short-circuits with
    ``EmbeddingUnavailable`` so request handlers fall back to FTS-only
    instead of spending the provider's full timeout on every hit.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        import time
        if (time.monotonic() - self._opened_at) >= self.cooldown_seconds:
            # Half-open: allow one probe. Caller records the outcome.
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            import time
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None


_embedding_circuit = _EmbeddingCircuit()


async def llm_embed(texts: list[str], embed_type: str = "db") -> list[list[float]]:
    """Embed one or more texts. Returns a list of vectors (same order).

    Wrapped in a circuit breaker so a prolonged provider outage degrades
    callers to a fast ``EmbeddingUnavailable`` instead of timing out on
    every request.
    """
    if not _embedding_circuit.allow():
        raise EmbeddingUnavailable("embedding backend temporarily unavailable")

    try:
        if LLM_PROVIDER == "bedrock":
            vectors = await _bedrock_embed(texts)
        else:
            vectors = await _minimax_embed(texts, embed_type)
    except Exception:
        _embedding_circuit.record_failure()
        raise

    if not vectors or (texts and len(vectors) != len(texts)):
        _embedding_circuit.record_failure()
        raise EmbeddingUnavailable(
            f"embedding backend returned {len(vectors or [])} vectors for {len(texts)} texts"
        )

    _embedding_circuit.record_success()
    return vectors


# ---------------------------------------------------------------------------
# Minimax implementation
# ---------------------------------------------------------------------------

async def _minimax_chat(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    model: Optional[str] = None,
) -> str:
    if not MINIMAX_API_KEY:
        raise ValueError("MINIMAX_API_KEY not set")

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": model or MINIMAX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    async def _do(c: httpx.AsyncClient) -> str:
        resp = await c.post(
            f"{MINIMAX_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=MINIMAX_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        _record_minimax_usage(data, model or MINIMAX_MODEL, kind="chat")
        return _extract_minimax_content(data)

    if client is not None:
        return await _do(client)

    # Shared pool path — reuses keep-alives across every LLM call.
    from app.http_client import get_http_client
    return await _do(get_http_client())


async def _kimi_chat(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    model: Optional[str] = None,
) -> str:
    """Kimi / Moonshot AI chat — OpenAI-compatible API with 256K context."""
    if not KIMI_API_KEY:
        raise ValueError("KIMI_API_KEY not set")

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": model or KIMI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json",
    }

    async def _do(c: httpx.AsyncClient) -> str:
        resp = await c.post(
            f"{KIMI_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        _record_kimi_usage(data, model or KIMI_MODEL, kind="chat")
        return _extract_minimax_content(data)  # Same OpenAI format

    if client is not None:
        return await _do(client)

    from app.http_client import get_http_client
    return await _do(get_http_client())


def _extract_minimax_content(data: dict) -> str:
    choices = data.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            f"API response has no choices. Keys: {list(data.keys())}, "
            f"base_resp: {data.get('base_resp', 'N/A')}"
        )

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError(f"First choice is not a dict: {type(first)}")

    content = ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
    elif isinstance(first.get("text"), str):
        content = first["text"]
    elif isinstance(first.get("delta"), dict):
        content = first["delta"].get("content", "")

    if not content:
        logger.warning("Empty content from Minimax. Choice: %s", json.dumps(first, default=str)[:500])
        raise ValueError(
            f"API returned empty content. Finish reason: "
            f"{first.get('finish_reason', 'unknown')}"
        )

    return _THINK_RE.sub("", content).strip()


def _record_llm_usage(
    *, provider: str, model: str, kind: str,
    input_tokens: int = 0, output_tokens: int = 0,
) -> None:
    """Forward a single LLM call to the metrics tracker, swallowing errors.

    Instrumentation must never break the caller, so any failure here
    (missing attrs, metrics module reload, etc.) is logged and ignored.
    """
    try:
        from app import metrics
        metrics.record_llm_call(
            provider=provider, model=model, kind=kind,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
    except Exception as exc:
        logger.debug("metrics.record_llm_call failed: %s", exc)


def _record_minimax_usage(data: dict, model: str, *, kind: str) -> None:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return
    in_toks = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out_toks = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    _record_llm_usage(
        provider="minimax", model=model, kind=kind,
        input_tokens=in_toks, output_tokens=out_toks,
    )


def _record_kimi_usage(data: dict, model: str, *, kind: str) -> None:
    """Record Kimi/Moonshot usage — same OpenAI format as Minimax."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return
    in_toks = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out_toks = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    _record_llm_usage(
        provider="kimi", model=model, kind=kind,
        input_tokens=in_toks, output_tokens=out_toks,
    )


def _record_bedrock_usage(data: dict, model: str, *, kind: str) -> None:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return
    in_toks = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
    out_toks = int(usage.get("outputTokens") or usage.get("output_tokens") or 0)
    _record_llm_usage(
        provider="bedrock", model=model, kind=kind,
        input_tokens=in_toks, output_tokens=out_toks,
    )


async def _minimax_embed(texts: list[str], embed_type: str) -> list[list[float]]:
    if not MINIMAX_API_KEY:
        raise ValueError("MINIMAX_API_KEY not set")
    if not texts:
        return []

    MAX_CHUNK_CHARS = 8000
    truncated = [t[:MAX_CHUNK_CHARS] for t in texts]

    from app.http_client import get_http_client
    client = get_http_client()
    resp = await client.post(
        f"{MINIMAX_BASE}/embeddings",
        headers={
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "embo-01",
            "texts": truncated,
            "type": embed_type,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()

    vectors = data.get("vectors") or []
    if len(vectors) != len(texts):
        logger.warning("Expected %d embeddings, got %d", len(texts), len(vectors))
    # Minimax embed endpoint doesn't surface token counts reliably;
    # approximate using the 4-chars-per-token rule of thumb so the dashboard
    # still captures traffic volume.
    approx_input = sum(len(t) for t in truncated) // 4
    _record_llm_usage(
        provider="minimax",
        model="embo-01",
        kind="embed",
        input_tokens=approx_input,
        output_tokens=0,
    )
    return vectors


# ---------------------------------------------------------------------------
# AWS Bedrock implementation
# ---------------------------------------------------------------------------

def _get_bedrock_client():
    """Create a boto3 Bedrock Runtime client.

    Uses the standard boto3 credential chain:
      1. Explicit keys (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
      2. AWS_PROFILE / ~/.aws/credentials
      3. IAM instance role / ECS task role / EKS IRSA
    """
    import boto3

    kwargs = {"region_name": BEDROCK_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = AWS_SESSION_TOKEN

    return boto3.client("bedrock-runtime", **kwargs)


def _bedrock_base_url() -> str:
    """Bedrock Runtime base URL for API key (Bearer token) auth."""
    return f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com"


async def _bedrock_chat(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> str:
    """Call Bedrock's converse API (works across model families).

    Supports two auth modes:
      - BEDROCK_API_KEY: Bearer token via direct HTTP (no boto3)
      - boto3: SigV4 via credential chain
    """
    if BEDROCK_API_KEY:
        return await _bedrock_chat_bearer(
            system_msg, user_msg, max_tokens, temperature, model=model,
        )

    import asyncio
    client = _get_bedrock_client()

    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    system = [{"text": system_msg}] if system_msg else []

    inference_config = {"maxTokens": max_tokens, "temperature": temperature}
    active_model = model or BEDROCK_MODEL

    def _invoke():
        resp = client.converse(
            modelId=active_model,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
        )
        _record_bedrock_usage(resp, active_model, kind="chat")
        output = resp.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])
        parts = []
        for block in content_blocks:
            if "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)

    content = await asyncio.get_event_loop().run_in_executor(None, _invoke)

    if not content:
        raise ValueError("Bedrock returned empty content")

    return _THINK_RE.sub("", content).strip()


async def _bedrock_chat_bearer(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> str:
    """Call Bedrock converse via Bearer token (API key) auth."""
    active_model = model or BEDROCK_MODEL
    url = f"{_bedrock_base_url()}/model/{active_model}/converse"
    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    system = [{"text": system_msg}] if system_msg else []

    payload = {
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        payload["system"] = system

    from app.http_client import get_http_client
    client = get_http_client()
    resp = await client.post(
        url,
        headers={
            "Authorization": f"Bearer {BEDROCK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    _record_bedrock_usage(data, active_model, kind="chat")

    output = data.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])
    parts = [block["text"] for block in content_blocks if "text" in block]
    content = "\n".join(parts)

    if not content:
        raise ValueError("Bedrock returned empty content")

    return _THINK_RE.sub("", content).strip()


async def _bedrock_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using Bedrock (Titan Embeddings or Cohere).

    Supports Bearer token (API key) and boto3 auth.
    """
    if BEDROCK_API_KEY:
        return await _bedrock_embed_bearer(texts)

    import asyncio
    client = _get_bedrock_client()
    model = BEDROCK_EMBED_MODEL

    MAX_CHUNK_CHARS = 8000

    def _embed_single(text: str) -> list[float]:
        truncated = text[:MAX_CHUNK_CHARS]

        if "cohere" in model.lower():
            body = json.dumps({
                "texts": [truncated],
                "input_type": "search_document",
            })
            resp = client.invoke_model(modelId=model, body=body)
            result = json.loads(resp["body"].read())
            return result["embeddings"][0]
        else:
            body = json.dumps({
                "inputText": truncated,
            })
            resp = client.invoke_model(modelId=model, body=body)
            result = json.loads(resp["body"].read())
            return result["embedding"]

    def _embed_all():
        return [_embed_single(t) for t in texts]

    vectors = await asyncio.get_event_loop().run_in_executor(None, _embed_all)
    approx_input = sum(len(t) for t in texts) // 4
    _record_llm_usage(
        provider="bedrock", model=model, kind="embed",
        input_tokens=approx_input, output_tokens=0,
    )
    return vectors


async def _bedrock_embed_bearer(texts: list[str]) -> list[list[float]]:
    """Embed texts using Bedrock via Bearer token (API key) auth."""
    model = BEDROCK_EMBED_MODEL
    MAX_CHUNK_CHARS = 8000
    results = []

    async with httpx.AsyncClient(timeout=120) as client:
        for text in texts:
            truncated = text[:MAX_CHUNK_CHARS]

            if "cohere" in model.lower():
                payload = {"texts": [truncated], "input_type": "search_document"}
            else:
                payload = {"inputText": truncated}

            url = f"{_bedrock_base_url()}/model/{model}/invoke"
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {BEDROCK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            if "cohere" in model.lower():
                results.append(data["embeddings"][0])
            else:
                results.append(data["embedding"])

    approx_input = sum(len(t) for t in texts) // 4
    _record_llm_usage(
        provider="bedrock", model=model, kind="embed",
        input_tokens=approx_input, output_tokens=0,
    )
    return results


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------


async def llm_chat_stream(
    system_msg: str,
    user_msg: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    *,
    client: Optional[httpx.AsyncClient] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream a chat completion as an async iterator of text deltas.

    Yields strings (token chunks). Each chunk is passed through
    ``_strip_non_latin`` per-chunk; a chunk that ends mid-multi-byte may
    drop a single character at the boundary, which is acceptable for the
    streaming use case (the alternative is buffering the entire stream).
    """
    active_provider, active_model = _resolve_provider_model(provider, model)
    if active_provider == "bedrock":
        gen = _bedrock_chat_stream(
            system_msg, user_msg, max_tokens, temperature, model=active_model,
        )
    elif active_provider == "kimi":
        # Kimi streaming uses same SSE format as Minimax (OpenAI-compatible)
        gen = _kimi_chat_stream(
            system_msg, user_msg, max_tokens, temperature,
            client=client, model=active_model,
        )
    else:
        gen = _minimax_chat_stream(
            system_msg, user_msg, max_tokens, temperature,
            client=client, model=active_model,
        )

    async for chunk in gen:
        if not chunk:
            continue
        yield _strip_non_latin(chunk)


async def _minimax_chat_stream(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    if not MINIMAX_API_KEY:
        raise ValueError("MINIMAX_API_KEY not set")

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": model or MINIMAX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    url = f"{MINIMAX_BASE}/chat/completions"

    async def _do(c: httpx.AsyncClient):
        last_obj = None
        async with c.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                last_obj = obj
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
        # Final SSE frame usually carries usage totals.
        if last_obj:
            _record_minimax_usage(last_obj, model or MINIMAX_MODEL, kind="chat")

    if client is not None:
        async for chunk in _do(client):
            yield chunk
        return

    from app.http_client import get_http_client
    async for chunk in _do(get_http_client()):
        yield chunk


async def _kimi_chat_stream(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Kimi/Moonshot streaming — same SSE format as OpenAI/Minimax."""
    if not KIMI_API_KEY:
        raise ValueError("KIMI_API_KEY not set")

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": model or KIMI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    url = f"{KIMI_BASE}/chat/completions"

    async def _do(c: httpx.AsyncClient):
        last_obj = None
        async with c.stream("POST", url, headers=headers, json=payload, timeout=300) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                last_obj = obj
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
        if last_obj:
            _record_kimi_usage(last_obj, model or KIMI_MODEL, kind="chat")

    if client is not None:
        async for chunk in _do(client):
            yield chunk
        return

    from app.http_client import get_http_client
    async for chunk in _do(get_http_client()):
        yield chunk


async def _bedrock_chat_stream(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream Bedrock converse output via boto3 ``converse_stream``.

    Bridges sync iteration over the boto3 event stream into an async
    generator using ``run_in_executor`` + ``asyncio.Queue``.
    """
    if BEDROCK_API_KEY:
        async for chunk in _bedrock_chat_stream_bearer(
            system_msg, user_msg, max_tokens, temperature, model=model,
        ):
            yield chunk
        return

    import asyncio
    import threading

    active_model = model or BEDROCK_MODEL
    bedrock_client = _get_bedrock_client()
    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    system = [{"text": system_msg}] if system_msg else []
    inference_config = {"maxTokens": max_tokens, "temperature": temperature}

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _producer():
        try:
            resp = bedrock_client.converse_stream(
                modelId=active_model,
                messages=messages,
                system=system,
                inferenceConfig=inference_config,
            )
            for event in resp.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta and delta["text"]:
                        loop.call_soon_threadsafe(queue.put_nowait, delta["text"])
                elif "metadata" in event:
                    meta = event["metadata"]
                    if isinstance(meta, dict) and "usage" in meta:
                        _record_bedrock_usage(meta, active_model, kind="chat")
                elif "messageStop" in event:
                    break
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


async def _bedrock_chat_stream_bearer(
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream Bedrock converse via Bearer token (API key) auth.

    Uses the Bedrock Runtime ``/converse-stream`` HTTP endpoint, which
    returns SSE-style events.
    """
    active_model = model or BEDROCK_MODEL
    url = f"{_bedrock_base_url()}/model/{active_model}/converse-stream"
    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    system = [{"text": system_msg}] if system_msg else []
    payload = {
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {BEDROCK_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.amazon.eventstream",
            },
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if "contentBlockDelta" in obj:
                    delta = obj["contentBlockDelta"].get("delta", {})
                    text = delta.get("text")
                    if text:
                        yield text
                elif "metadata" in obj:
                    meta = obj["metadata"]
                    if isinstance(meta, dict) and "usage" in meta:
                        _record_bedrock_usage(meta, active_model, kind="chat")
                elif "messageStop" in obj:
                    return


# ---------------------------------------------------------------------------
# Tool-calling abstraction
# ---------------------------------------------------------------------------


_MINIMAX_TOOL_SYSTEM = """You have access to these tools:
{tools_json}

If you need to use a tool, respond with ONLY a single JSON object on one line:
{{"type": "tool_use", "name": "<tool_name>", "input": {{"<arg>": "<value>"}}}}

If you can answer directly, respond with normal text instead. Do not mix
tool-call JSON and prose in the same response."""


def _parse_minimax_tool_response(text: str) -> Optional[dict]:
    """Try to parse a Minimax response as a tool-use JSON object.

    Returns the parsed dict if it looks like a tool call, otherwise None.
    Accepts JSON either bare or wrapped in a ```json ... ``` fence.
    """
    if not text:
        return None
    stripped = text.strip()

    # Strip markdown code fence if present
    if stripped.startswith("```"):
        # Drop first line, drop trailing fence
        parts = stripped.split("\n", 1)
        if len(parts) == 2:
            inner = parts[1]
            if inner.endswith("```"):
                inner = inner[: -3]
            stripped = inner.strip()

    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "tool_use":
        return None
    if not isinstance(obj.get("name"), str):
        return None
    if not isinstance(obj.get("input"), dict):
        return None
    return {"type": "tool_use", "name": obj["name"], "input": obj["input"]}


async def llm_chat_tools(
    system_msg: str,
    user_msg: str,
    tools: list[dict],
    *,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict:
    """Call the LLM with optional tool-calling support.

    Returns one of:
        {"type": "text", "content": str}
        {"type": "tool_use", "name": str, "input": dict}

    *tools* is a list of dicts with keys ``name``, ``description`` and
    ``input_schema`` (JSON Schema). Both providers receive the same shape;
    Bedrock uses native ``toolConfig``, Minimax falls back to a JSON-prompt
    shim. ``_strip_non_latin`` is applied to text responses only — tool
    inputs are returned as-is to avoid corrupting JSON.
    """
    active_provider, active_model = _resolve_provider_model(provider, model)
    if active_provider == "bedrock":
        return await _bedrock_chat_tools(
            system_msg, user_msg, tools, max_tokens, temperature, model=active_model,
        )
    return await _minimax_chat_tools(
        system_msg, user_msg, tools, max_tokens, temperature, model=active_model,
    )


async def _minimax_chat_tools(
    system_msg: str,
    user_msg: str,
    tools: list[dict],
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> dict:
    """Minimax tool-calling via prompt shim (no native tool support)."""
    tools_json = json.dumps(tools, indent=2)
    shim_system = _MINIMAX_TOOL_SYSTEM.format(tools_json=tools_json)
    full_system = f"{system_msg}\n\n{shim_system}" if system_msg else shim_system

    raw = await _minimax_chat(
        full_system, user_msg, max_tokens, temperature, model=model,
    )
    parsed = _parse_minimax_tool_response(raw)
    if parsed is not None:
        return parsed
    return {"type": "text", "content": _strip_non_latin(raw)}


async def _bedrock_chat_tools(
    system_msg: str,
    user_msg: str,
    tools: list[dict],
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> dict:
    """Bedrock tool-calling via native ``toolConfig`` on converse API."""
    if BEDROCK_API_KEY:
        return await _bedrock_chat_tools_bearer(
            system_msg, user_msg, tools, max_tokens, temperature, model=model,
        )

    import asyncio

    active_model = model or BEDROCK_MODEL
    bedrock_client = _get_bedrock_client()
    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    system = [{"text": system_msg}] if system_msg else []
    inference_config = {"maxTokens": max_tokens, "temperature": temperature}

    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "inputSchema": {"json": t["input_schema"]},
                }
            }
            for t in tools
        ]
    }

    def _invoke():
        return bedrock_client.converse(
            modelId=active_model,
            messages=messages,
            system=system,
            inferenceConfig=inference_config,
            toolConfig=tool_config,
        )

    resp = await asyncio.get_event_loop().run_in_executor(None, _invoke)
    _record_bedrock_usage(resp, active_model, kind="chat")
    return _bedrock_extract_tool_or_text(resp)


async def _bedrock_chat_tools_bearer(
    system_msg: str,
    user_msg: str,
    tools: list[dict],
    max_tokens: int,
    temperature: float,
    *,
    model: Optional[str] = None,
) -> dict:
    active_model = model or BEDROCK_MODEL
    url = f"{_bedrock_base_url()}/model/{active_model}/converse"
    messages = [{"role": "user", "content": [{"text": user_msg}]}]
    system = [{"text": system_msg}] if system_msg else []

    payload = {
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "inputSchema": {"json": t["input_schema"]},
                    }
                }
                for t in tools
            ]
        },
    }
    if system:
        payload["system"] = system

    from app.http_client import get_http_client
    client = get_http_client()
    resp = await client.post(
        url,
        headers={
            "Authorization": f"Bearer {BEDROCK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    _record_bedrock_usage(data, active_model, kind="chat")

    return _bedrock_extract_tool_or_text(data)


def _bedrock_extract_tool_or_text(resp: dict) -> dict:
    """Walk a Bedrock converse response and return our unified shape."""
    output = resp.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    text_parts: list[str] = []
    for block in content_blocks:
        if "toolUse" in block:
            tool = block["toolUse"]
            return {
                "type": "tool_use",
                "name": tool.get("name", ""),
                "input": tool.get("input") or {},
            }
        if "text" in block:
            text_parts.append(block["text"])

    text = "\n".join(text_parts).strip()
    return {"type": "text", "content": _strip_non_latin(text)}
