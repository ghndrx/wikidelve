"""Regression tests for the /chat welcome suggestion buttons.

The buttons originally used ``onclick="send(this.title)"``. When the CSP
inline-handler shim got stricter, it silently dropped ``this.title``
(resolving it as ``window.this.title`` -> undefined) and the buttons
became no-ops — clicking "yt-dlp Subtitle..." did absolutely nothing.

This test asserts the markup + script that replaces that pattern:
  1. The suggestions are rendered as ``<button data-send="...">`` so
     the delegation handler has an explicit payload.
  2. ``search.html`` includes a delegated click handler keyed on
     ``.suggestion[data-send]`` that calls ``send(payload)``.
  3. No ``onclick="...this.title..."`` sneaks back in.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


SEARCH_HTML = Path(__file__).resolve().parents[2] / "templates" / "search.html"


@pytest.fixture(scope="module")
def html() -> str:
    return SEARCH_HTML.read_text(encoding="utf-8")


def test_suggestions_use_data_send_attribute(html: str):
    assert 'data-send="{{ s | e }}"' in html, (
        "Suggestion button must carry the text to send in a data-send "
        "attribute so the delegation handler can pick it up. "
        "Don't switch back to onclick='send(this.title)' — the CSP shim "
        "silently drops `this.*` references."
    )


def test_delegation_handler_wired(html: str):
    assert ".suggestion[data-send]" in html, (
        "search.html must contain a delegated click handler that matches "
        "'.suggestion[data-send]' and calls send(...)."
    )
    # And it must actually call send() inside the handler block.
    match = re.search(
        r"\.suggestion\[data-send\].*?send\(\s*text\s*\)",
        html,
        flags=re.DOTALL,
    )
    assert match, "Delegation handler must call send(text) with the data-send payload."


def test_no_this_dot_property_in_inline_handlers(html: str):
    offenders = re.findall(r'on[a-z]+="[^"]*\bthis\.[a-zA-Z_]', html)
    assert not offenders, (
        "Inline handler references `this.property` — the CSP shim can't "
        "resolve that. Use a data-* attribute + delegated listener instead."
    )


def test_help_suggestion_is_wired(html: str):
    """The hardcoded 'What can you do?' button also needs a data-send."""
    assert 'data-send="help"' in html


def test_suggestion_button_has_type_button(html: str):
    """Without type='button', a <button> inside a form defaults to submit
    and refreshes the page on click. The welcome box is inside the chat
    container — if that ever gets wrapped in a form, the default submit
    behavior would kill the click. Belt + braces."""
    # At least one suggestion in the loop must be type=button.
    assert 'type="button" class="suggestion"' in html, (
        "Suggestion buttons should have type='button' so they never "
        "accidentally submit a form."
    )


def test_data_send_is_unfiltered_template_variable(html: str):
    """Regression: ``main.py`` used to pre-truncate suggestions to 50
    chars via ``a['title'][:50]`` so ``data-send`` shipped the
    truncated string. The visible label already uses ``| truncate(25)``
    so the payload must be the raw ``{{ s }}`` — anything shorter
    means the backend is pre-truncating again."""
    assert 'data-send="{{ s | e }}"' in html, (
        "Suggestion button must carry the unfiltered {{ s | e }} as its "
        "data-send payload. If the backend is also truncating, clicking "
        "a suggestion sends a snipped title (bug from 2026-04)."
    )
    assert '| truncate(25)' in html, (
        "The visible label, not the payload, is the place for truncate()."
    )
