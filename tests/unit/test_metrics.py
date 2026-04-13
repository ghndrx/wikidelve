"""Unit tests for app.metrics — in-process registry + token usage tracker."""

import pytest

from app import metrics


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset_all()
    yield
    metrics.reset_all()


class TestCounters:
    def test_inc_counter_accumulates(self):
        metrics.inc_counter("foo_total", 1, {"a": "1"}, help="doc")
        metrics.inc_counter("foo_total", 2, {"a": "1"})
        text = metrics.prometheus_text()
        assert 'foo_total{a="1"} 3.0' in text

    def test_counter_labels_are_independent(self):
        metrics.inc_counter("foo_total", 1, {"a": "1"})
        metrics.inc_counter("foo_total", 5, {"a": "2"})
        text = metrics.prometheus_text()
        assert 'foo_total{a="1"} 1.0' in text
        assert 'foo_total{a="2"} 5.0' in text

    def test_help_and_type_rendered(self):
        metrics.inc_counter("bar_total", 1, help="useful help")
        text = metrics.prometheus_text()
        assert "# HELP bar_total useful help" in text
        assert "# TYPE bar_total counter" in text


class TestGauges:
    def test_gauge_replaces_value(self):
        metrics.set_gauge("jobs_running", 3, help="h")
        metrics.set_gauge("jobs_running", 5)
        text = metrics.prometheus_text()
        assert "jobs_running 5" in text
        assert "jobs_running 3" not in text


class TestHistograms:
    def test_bucket_counts_and_sum(self):
        metrics.observe_histogram(
            "dur_seconds", 0.05, {"path": "/x"},
            help="duration", buckets=(0.01, 0.1, 1.0),
        )
        metrics.observe_histogram(
            "dur_seconds", 0.5, {"path": "/x"},
            buckets=(0.01, 0.1, 1.0),
        )
        text = metrics.prometheus_text()
        assert 'dur_seconds_bucket{path="/x",le="0.01"} 0' in text
        assert 'dur_seconds_bucket{path="/x",le="0.1"} 1' in text
        assert 'dur_seconds_bucket{path="/x",le="1.0"} 2' in text
        assert 'dur_seconds_bucket{path="/x",le="+Inf"} 2' in text
        assert 'dur_seconds_count{path="/x"} 2' in text
        assert 'dur_seconds_sum{path="/x"} 0.55' in text


class TestLabelEscaping:
    def test_escapes_quotes_and_backslashes(self):
        metrics.inc_counter("x_total", 1, {"label": 'has "quote" and \\'}, help="h")
        text = metrics.prometheus_text()
        assert 'label="has \\"quote\\" and \\\\"' in text


class TestUsageTracking:
    def test_single_call_shows_in_summary(self):
        metrics.record_llm_call(
            provider="minimax", model="text-01", kind="chat",
            input_tokens=100, output_tokens=50,
        )
        summary = metrics.get_usage_summary()
        assert summary["totals_week"]["calls"] == 1
        assert summary["totals_week"]["input"] == 100
        assert summary["totals_week"]["output"] == 50
        assert summary["totals_week"]["cost_usd"] >= 0.0
        today_rows = summary["today_breakdown"]
        assert len(today_rows) == 1
        assert today_rows[0]["provider"] == "minimax"
        assert today_rows[0]["model"] == "text-01"
        assert today_rows[0]["kind"] == "chat"

    def test_multiple_calls_aggregate_per_key(self):
        for _ in range(3):
            metrics.record_llm_call(
                provider="bedrock", model="sonnet", kind="chat",
                input_tokens=10, output_tokens=5,
            )
        summary = metrics.get_usage_summary()
        row = summary["today_breakdown"][0]
        assert row["calls"] == 3
        assert row["input_tokens"] == 30
        assert row["output_tokens"] == 15

    def test_counters_mirrored_into_registry(self):
        metrics.record_llm_call(
            provider="minimax", model="text-01", kind="chat",
            input_tokens=42, output_tokens=9,
        )
        text = metrics.prometheus_text()
        assert "kb_llm_calls_total" in text
        assert "kb_llm_input_tokens_total" in text
        assert "kb_llm_output_tokens_total" in text

    def test_zero_tokens_still_counts_call(self):
        metrics.record_llm_call(
            provider="bedrock", model="titan-embed", kind="embed",
            input_tokens=0, output_tokens=0,
        )
        summary = metrics.get_usage_summary()
        assert summary["totals_week"]["calls"] == 1

    def test_cost_is_zero_for_unknown_pricing(self):
        metrics.record_llm_call(
            provider="fictional", model="foo", kind="chat",
            input_tokens=1000, output_tokens=1000,
        )
        summary = metrics.get_usage_summary()
        assert summary["totals_week"]["cost_usd"] == 0.0
