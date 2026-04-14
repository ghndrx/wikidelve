"""Unit tests for app/llm.py retry classification.

Pure-function tests for ``_is_retryable`` and the backoff wrapper.
The actual provider calls aren't exercised here — that's the
integration tests' job. We verify that:

  - Transient HTTP / network errors classify as retryable
  - Permanent HTTP errors (4xx auth / validation) do NOT retry
  - Botocore throttle codes classify as retryable when present
  - Retry-After header is parsed when the server sends one
  - The backoff wrapper actually retries N times then raises
"""

import asyncio

import httpx
import pytest

from app import llm


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------

class TestIsRetryableHTTP:
    def _status_error(self, status: int, retry_after: str | None = None):
        req = httpx.Request("POST", "https://x/y")
        headers = {"retry-after": retry_after} if retry_after else {}
        resp = httpx.Response(status, headers=headers, request=req)
        return httpx.HTTPStatusError("err", request=req, response=resp)

    @pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504, 529])
    def test_retryable_status_codes(self, status):
        retry, hint = llm._is_retryable(self._status_error(status))
        assert retry is True
        assert hint is None  # no Retry-After header in these cases

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_permanent_status_codes_not_retryable(self, status):
        retry, _ = llm._is_retryable(self._status_error(status))
        assert retry is False

    def test_retry_after_seconds_parsed(self):
        retry, hint = llm._is_retryable(self._status_error(429, retry_after="12"))
        assert retry is True
        assert hint == 12.0

    def test_retry_after_garbage_falls_back_to_no_hint(self):
        retry, hint = llm._is_retryable(self._status_error(429, retry_after="banana"))
        assert retry is True
        assert hint is None


class TestIsRetryableNetwork:
    def test_timeout_is_retryable(self):
        retry, _ = llm._is_retryable(httpx.ReadTimeout("slow"))
        assert retry is True

    def test_connect_error_is_retryable(self):
        retry, _ = llm._is_retryable(httpx.ConnectError("dns?"))
        assert retry is True

    def test_remote_protocol_error_is_retryable(self):
        retry, _ = llm._is_retryable(httpx.RemoteProtocolError("trunc"))
        assert retry is True

    def test_random_exception_not_retryable(self):
        retry, _ = llm._is_retryable(RuntimeError("we made a mistake"))
        assert retry is False


class TestIsRetryableBoto:
    def test_throttling_exception_retryable(self):
        # Construct the shape botocore.exceptions.ClientError uses
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            pytest.skip("botocore not installed in this environment")

        err = ClientError(
            error_response={
                "Error": {"Code": "ThrottlingException", "Message": "slow down"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            operation_name="Converse",
        )
        retry, _ = llm._is_retryable(err)
        assert retry is True

    def test_validation_exception_not_retryable(self):
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            pytest.skip("botocore not installed")

        err = ClientError(
            error_response={
                "Error": {"Code": "ValidationException", "Message": "bad"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            operation_name="Converse",
        )
        retry, _ = llm._is_retryable(err)
        assert retry is False


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------

class TestWithRetry:
    @pytest.mark.asyncio
    async def test_returns_value_on_first_success(self):
        async def factory():
            return "ok"

        result = await llm._with_retry(factory, label="test")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_until_success(self, monkeypatch):
        # Force tiny delays so the test stays fast.
        monkeypatch.setattr(llm, "LLM_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(llm, "LLM_RETRY_MAX_DELAY", 0.01)
        monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 3)

        attempts = {"n": 0}

        async def factory():
            attempts["n"] += 1
            if attempts["n"] < 3:
                req = httpx.Request("POST", "https://x")
                resp = httpx.Response(503, request=req)
                raise httpx.HTTPStatusError("flake", request=req, response=resp)
            return "finally"

        result = await llm._with_retry(factory, label="test")
        assert result == "finally"
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(llm, "LLM_RETRY_MAX_DELAY", 0.01)
        monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 2)

        attempts = {"n": 0}

        async def factory():
            attempts["n"] += 1
            req = httpx.Request("POST", "https://x")
            resp = httpx.Response(503, request=req)
            raise httpx.HTTPStatusError("perma", request=req, response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            await llm._with_retry(factory, label="test")
        # 1 initial + 2 retries = 3 attempts
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self, monkeypatch):
        monkeypatch.setattr(llm, "LLM_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(llm, "LLM_MAX_RETRIES", 5)

        attempts = {"n": 0}

        async def factory():
            attempts["n"] += 1
            raise ValueError("not network")

        with pytest.raises(ValueError):
            await llm._with_retry(factory, label="test")
        assert attempts["n"] == 1  # didn't retry the non-retryable error
