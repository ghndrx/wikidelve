"""Unit tests for the composite rate-limit key in app.main."""

from types import SimpleNamespace

import pytest

from app.main import _rate_limit_key


def _fake_request(
    client_ip: str = "10.0.0.1",
    path_params: dict | None = None,
    query_params: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal Request-shaped object for the key function."""
    return SimpleNamespace(
        client=SimpleNamespace(host=client_ip),
        path_params=path_params or {},
        query_params=query_params or {},
        headers={},
    )


class TestRateLimitKey:
    def test_path_kb_wins(self):
        req = _fake_request(path_params={"kb": "personal"})
        assert _rate_limit_key(req).endswith("|personal")

    def test_path_kb_name_wins(self):
        req = _fake_request(path_params={"kb_name": "work"})
        assert _rate_limit_key(req).endswith("|work")

    def test_query_kb_used_when_path_empty(self):
        req = _fake_request(query_params={"kb": "docs"})
        assert _rate_limit_key(req).endswith("|docs")

    def test_fallback_bucket_when_no_kb(self):
        req = _fake_request()
        assert _rate_limit_key(req).endswith("|_")

    def test_includes_client_ip(self):
        req = _fake_request(client_ip="10.9.8.7", path_params={"kb": "personal"})
        assert _rate_limit_key(req).startswith("10.9.8.7|")

    def test_different_kbs_give_different_keys(self):
        a = _rate_limit_key(_fake_request(path_params={"kb": "one"}))
        b = _rate_limit_key(_fake_request(path_params={"kb": "two"}))
        assert a != b

    def test_different_ips_give_different_keys(self):
        a = _rate_limit_key(_fake_request(client_ip="10.0.0.1", path_params={"kb": "x"}))
        b = _rate_limit_key(_fake_request(client_ip="10.0.0.2", path_params={"kb": "x"}))
        assert a != b
