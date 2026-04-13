# WikiDelve dev tasks.
#
# Default Python comes from the project venv. Override with PY=... if
# you run outside .venv. Every target sets DB_BACKEND=sqlite +
# STORAGE_BACKEND=local so stray production env vars can't leak in.

PY ?= .venv/bin/python
PYTEST ?= $(PY) -m pytest
TEST_ENV := DB_BACKEND=sqlite STORAGE_BACKEND=local

.PHONY: help test test-heavy coverage coverage-unit coverage-parity \
        coverage-aws coverage-html coverage-bump bench bench-aws \
        bench-check lint install-dev clean-coverage clean-bench

help:
	@echo "WikiDelve dev targets:"
	@echo ""
	@echo "  make install-dev        Install dev/test dependencies"
	@echo ""
	@echo "  make test               Fast unit tests (default pre-commit path)"
	@echo "  make test-heavy         Include @pytest.mark.heavy"
	@echo ""
	@echo "  make coverage           Full suite + coverage report"
	@echo "  make coverage-unit      tests/unit only"
	@echo "  make coverage-parity    tests/parity (local + moto AWS, offline)"
	@echo "  make coverage-aws       tests/parity + real AWS tier (requires WIKIDELVE_AWS_TEST_BUCKET/TABLE)"
	@echo "  make coverage-html      Open HTML report in a browser"
	@echo "  make coverage-bump      Ratchet fail_under upward if coverage improved"
	@echo ""
	@echo "  make bench              Run latency benchmark suite + render report"
	@echo "  make bench-aws          Run latency benchmarks against real AWS backends"
	@echo "  make bench-check        Fail if bench ratios regressed > 2x"
	@echo ""
	@echo "  make clean-coverage     Remove .coverage/htmlcov/coverage.xml"
	@echo "  make clean-bench        Remove .benchmarks/"

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

# ----- Unit tests ---------------------------------------------------------

test:
	$(TEST_ENV) $(PYTEST) tests/unit -x -q -m "not heavy" --no-header

test-heavy:
	$(TEST_ENV) $(PYTEST) tests/unit -x -q --no-header

# ----- Coverage -----------------------------------------------------------

coverage:
	$(TEST_ENV) $(PYTEST) tests/unit tests/parity \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		--cov-report=html \
		-m "not heavy and not bench"

coverage-unit:
	$(TEST_ENV) $(PYTEST) tests/unit \
		--cov=app \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		-m "not heavy"

coverage-parity:
	$(TEST_ENV) $(PYTEST) tests/parity \
		--cov=app \
		--cov-report=term-missing \
		-v

# Real-AWS parity tier. Refuses to run unless all three guard env vars
# are set AND the test resources differ from production (guard lives
# in tests/parity/conftest.py).
coverage-aws:
	@if [ -z "$$WIKIDELVE_AWS_TEST_BUCKET" ] || [ -z "$$WIKIDELVE_AWS_TEST_TABLE" ]; then \
		echo "ERROR: set WIKIDELVE_AWS_TEST_BUCKET + WIKIDELVE_AWS_TEST_TABLE"; \
		echo "  source them from wikidelve-infra terraform outputs:"; \
		echo "    cd ../wikidelve-infra && terraform output -raw test_env_exports"; \
		exit 1; \
	fi
	INTEGRATION_AWS=1 $(PYTEST) tests/parity \
		--cov=app \
		--cov-report=term-missing \
		-v \
		-k "real"

coverage-html: coverage
	@echo "Open htmlcov/index.html in a browser"
	@command -v xdg-open >/dev/null && xdg-open htmlcov/index.html || true

coverage-bump:
	$(PY) scripts/ratchet_coverage.py

# ----- Latency bench ------------------------------------------------------

bench:
	$(TEST_ENV) $(PYTEST) tests/bench \
		--benchmark-only \
		--benchmark-json=.benchmarks/latest.json \
		-m bench
	$(PY) scripts/render_bench_report.py

# Bench against real AWS — produces absolute latency numbers, not
# relative moto ratios. Same safety guard as coverage-aws.
bench-aws:
	@if [ -z "$$WIKIDELVE_AWS_TEST_BUCKET" ] || [ -z "$$WIKIDELVE_AWS_TEST_TABLE" ]; then \
		echo "ERROR: set WIKIDELVE_AWS_TEST_BUCKET + WIKIDELVE_AWS_TEST_TABLE"; \
		exit 1; \
	fi
	INTEGRATION_AWS=1 $(PYTEST) tests/bench \
		--benchmark-only \
		--benchmark-json=.benchmarks/aws.json \
		-k "real" \
		-m bench
	$(PY) scripts/render_bench_report.py --source .benchmarks/aws.json \
		--output docs/benchmarks/latency-aws.md

bench-check:
	$(PY) scripts/render_bench_report.py --check

# ----- Cleanup ------------------------------------------------------------

clean-coverage:
	rm -rf .coverage coverage.xml htmlcov

clean-bench:
	rm -rf .benchmarks
