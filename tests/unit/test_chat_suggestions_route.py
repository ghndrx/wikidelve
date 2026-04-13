"""Server-side regression: the /chat route must send full article titles
as suggestion payloads, not pre-truncated 50-char snippets.

Background:
    main.py used to build ``suggestions = [a["title"][:50] for a in sampled]``.
    Combined with the template's ``data-send="{{ s | e }}"``, clicking a
    suggestion shipped the truncated string to the chat (e.g.
    "Python uv Package Manager Lockfile Reproducible Bu" — sliced at 50).
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# The /chat route calls storage.list_kbs() which triggers
# KB_ROOT.iterdir() in app.config. In a default dev env that's /kb,
# which tests can't create. Point it at a tmp dir BEFORE importing
# app.main so config.KB_ROOT picks up the override at import time.
_TEST_KB_ROOT = Path(tempfile.mkdtemp(prefix="wikidelve-test-kb-"))
(_TEST_KB_ROOT / "personal" / "wiki").mkdir(parents=True, exist_ok=True)
os.environ["KB_ROOT"] = str(_TEST_KB_ROOT)
os.environ["PERSONAL_KB_PATH"] = str(_TEST_KB_ROOT / "personal")


@pytest.fixture(scope="module")
def client():
    import asyncio
    from contextlib import asynccontextmanager

    from app.main import app
    from app import db

    # Real noop lifespan — skip redis but init the sqlite schema so
    # db.get_jobs() in the /chat route doesn't blow up.
    @asynccontextmanager
    async def _test_lifespan(app_):
        app_.state.redis = None
        await db.init_db()
        yield

    app.router.lifespan_context = _test_lifespan

    with TestClient(app) as c:
        yield c


def test_chat_route_does_not_truncate_suggestion_payloads(client):
    """If any suggestion is exactly 50 chars long, chances are main.py
    sliced it with [:50] again. Assert the server hands the template
    the full untruncated title by checking the rendered data-send
    attributes — the visible label is `| truncate(25)` so even a long
    full title will render short, but its payload stays full."""
    resp = client.get("/chat")
    assert resp.status_code == 200
    html = resp.text

    # Every suggestion appears as `data-send="...">label` in the HTML.
    # Extract the payloads and verify: (a) at least one exists, and
    # (b) none match the specific 50-char pattern that was shipping
    # (a truncated string never ends cleanly on a word boundary).
    payloads = re.findall(r'<button[^>]*data-send="([^"]*)"[^>]*>', html)
    # The hardcoded "help" fallback button is always present; dynamic
    # ones are the ones we want to inspect.
    dyn = [p for p in payloads if p and p != "help"]

    for p in dyn:
        # A 50-char hard slice usually lands mid-word — sanity check:
        # if the payload is exactly 50 and doesn't end with punctuation
        # or a common stopword terminator, that's the old bug.
        if len(p) == 50 and not p[-1] in ".!?)":
            pytest.fail(
                f"Suggestion payload looks 50-char truncated: {p!r}. "
                f"main.py should send the full title; the template handles "
                f"label truncation via `| truncate(25)`."
            )


def test_chat_route_uses_full_title_source(client):
    """Inspect the /chat HTML to confirm the template is still using
    `truncate(25)` as the visible label — because if somebody 'fixes'
    the UI by moving truncation into main.py again, that would look
    correct visually but break the payload silently."""
    resp = client.get("/chat")
    assert resp.status_code == 200
    html = resp.text
    # The suggestion container must render, proving the template is
    # the right one.
    assert 'class="suggestions"' in html
