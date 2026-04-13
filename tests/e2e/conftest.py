"""Playwright E2E test configuration."""

import pytest


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": True}
