"""Verify the per-request metrics middleware records histogram + counter
observations into app.metrics, and that /metrics renders Prometheus text."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import asyncio
    from contextlib import asynccontextmanager
    from app.main import app
    from app import db, metrics

    @asynccontextmanager
    async def _test_lifespan(app_):
        app_.state.redis = None
        await db.init_db()
        yield

    app.router.lifespan_context = _test_lifespan
    metrics.reset_all()

    with TestClient(app) as c:
        yield c

    metrics.reset_all()


class TestMetricsMiddleware:
    def test_health_endpoint_increments_counter(self, client):
        from app import metrics
        metrics.reset_all()
        client.get("/health")
        text = metrics.prometheus_text()
        assert "kb_http_requests_total" in text

    def test_histogram_observed(self, client):
        from app import metrics
        metrics.reset_all()
        client.get("/health")
        text = metrics.prometheus_text()
        assert "kb_http_request_duration_seconds" in text
        assert "_bucket" in text

    def test_route_template_used_not_actual_path(self, client):
        from app import metrics
        metrics.reset_all()
        client.get("/health")
        text = metrics.prometheus_text()
        assert 'path="/health"' in text

    def test_metrics_endpoint_returns_prometheus_text(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text
        assert "# TYPE" in body
        assert "kb_http_requests_total" in body

    def test_usage_summary_endpoint(self, client):
        resp = client.get("/api/usage/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "today" in data
        assert "totals_week" in data
