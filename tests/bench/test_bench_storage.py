"""Latency benchmarks for storage operations across local + moto-S3.

Run with `make bench` — produces `.benchmarks/latest.json` and
`docs/benchmarks/latency.md`. Numbers are relative (moto runs in-process),
so the absolute times are unrealistically low. The RATIOS between
backends are what matter for regression detection.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def seeded_storage(bench_storage):
    """Seed bench-kb with N articles and return (storage, backend, N)."""
    inst, backend = bench_storage
    n = 100
    for i in range(n):
        inst.write_text("bench-kb", f"wiki/article-{i:04d}.md",
                        f"---\ntitle: Article {i}\ntags: [bench]\n---\n\n"
                        f"# Article {i}\n\n{'Lorem ipsum. ' * 50}\n")
    return inst, backend, n


@pytest.mark.bench
class TestIterArticles:
    def test_iter_articles_100(self, seeded_storage, benchmark):
        inst, backend, n = seeded_storage

        def run():
            return list(inst.iter_articles("bench-kb"))

        result = benchmark(run)
        assert len(result) == n


@pytest.mark.bench
class TestReadTextSingle:
    def test_read_single_cold(self, seeded_storage, benchmark):
        inst, backend, n = seeded_storage

        def run():
            return inst.read_text("bench-kb", "wiki/article-0050.md")

        result = benchmark(run)
        assert result is not None
        assert "Article 50" in result


@pytest.mark.bench
class TestReadTextBatch:
    def test_read_50_articles(self, seeded_storage, benchmark):
        inst, backend, n = seeded_storage

        def run():
            results = []
            for i in range(50):
                results.append(inst.read_text("bench-kb", f"wiki/article-{i:04d}.md"))
            return results

        result = benchmark(run)
        assert len(result) == 50
        assert all(r is not None for r in result)


@pytest.mark.bench
class TestListSlugs:
    def test_list_slugs_100(self, seeded_storage, benchmark):
        inst, backend, n = seeded_storage

        def run():
            return inst.list_slugs("bench-kb")

        result = benchmark(run)
        assert len(result) == n
