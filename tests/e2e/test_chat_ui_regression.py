"""Playwright regression tests for /chat UI bugs that have shipped
to users in the past. Each test pins one of the bugs we fixed so the
same regression can't slip through again.

Run against the live instance at localhost:8888.
"""

import re

import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8888"


class TestKeydownDoesNotSendOnEveryKey:
    """The shim-parser bug: onkeydown="if(Enter)send()" was stripped to
    just send(), so every single keystroke fired a send. Assert typing
    a non-Enter key into the input does NOT produce a chat message."""

    def test_typing_characters_produces_no_messages(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        inp = page.locator("#input")
        expect(inp).to_be_visible(timeout=10_000)

        inp.click()
        inp.type("hello world", delay=20)

        # Give any misfired listeners a full tick to run.
        page.wait_for_timeout(400)

        # Any rendered .msg element means a send() fired. Should be zero.
        assert page.locator(".msg").count() == 0, (
            "Typing into the chat input triggered a send() — the CSP "
            "shim keydown regression is back."
        )

        # The input still holds the typed text, proving keystrokes weren't
        # consumed by an errant preventDefault.
        assert inp.input_value() == "hello world"

    def test_enter_key_submits_once(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        inp = page.locator("#input")
        expect(inp).to_be_visible(timeout=10_000)

        inp.click()
        inp.type("test query", delay=10)
        inp.press("Enter")

        # A user message bubble should appear.
        page.wait_for_selector(".msg.user", timeout=5000)
        assert page.locator(".msg.user").count() == 1


class TestSuggestionButtons:
    """The `send(this.title)` bug: clicking a welcome-screen suggestion
    pill silently did nothing because the shim couldn't resolve
    `this.title`. Assert clicking a suggestion actually populates/sends."""

    def test_help_suggestion_sends(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        help_btn = page.locator('.suggestion[data-send="help"]')
        expect(help_btn).to_be_visible(timeout=10_000)

        help_btn.click()

        # `send('help')` should show the user bubble AND the AI help
        # response (the help command is local, not a real research run).
        page.wait_for_selector(".msg.user", timeout=5000)
        user_msgs = page.locator(".msg.user .msg-bubble")
        assert user_msgs.count() >= 1
        # The user bubble should contain 'help'.
        assert "help" in user_msgs.first.inner_text().lower()

    def test_dynamic_suggestion_click_sends_full_title(self, page: Page):
        """The bug: the button's visible text was truncated (`truncate(25)`)
        but `send(this.title)` was supposed to send the full title.
        Verify the first dynamic suggestion sends its full data-send
        payload, not the truncated label."""
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        # First non-"help" suggestion.
        sugg = page.locator('.suggestion[data-send]:not([data-send="help"])').first
        if sugg.count() == 0:
            pytest.skip("No dynamic suggestions available — KB has no long-title articles")
        expect(sugg).to_be_visible(timeout=5000)

        full_title = sugg.get_attribute("data-send") or ""
        assert len(full_title) >= 10, "suggestion payload too short to be meaningful"

        sugg.click()
        page.wait_for_selector(".msg.user", timeout=5000)

        # The user bubble text should start with the full title, not the
        # 25-char truncated label.
        user_text = page.locator(".msg.user .msg-bubble").first.inner_text()
        assert full_title[:20] in user_text, (
            f"Suggestion click sent truncated label instead of full payload. "
            f"Expected to find {full_title[:20]!r} in message {user_text!r}."
        )


class TestChatSessionListLayout:
    """The chat session list had the delete × on its own line below the
    title. Assert it's inline with the title now."""

    def test_delete_x_is_inline_with_title(self, page: Page):
        page.goto(f"{BASE_URL}/chat", wait_until="domcontentloaded")
        items = page.locator(".chat-sess-item")
        if items.count() == 0:
            pytest.skip("No chat sessions to verify layout against")
        first = items.first
        link = first.locator(".chat-sess-link")
        delete = first.locator(".chat-sess-del")

        link_box = link.bounding_box()
        del_box = delete.bounding_box()
        assert link_box and del_box, "link/delete bounding boxes unavailable"

        # Delete × should sit on the same horizontal row as the title
        # (vertical centers within 8px of each other) and to the right.
        link_center = link_box["y"] + link_box["height"] / 2
        del_center = del_box["y"] + del_box["height"] / 2
        assert abs(link_center - del_center) < 8, (
            f"Delete × is on a different row than the session title: "
            f"link_center={link_center}, del_center={del_center}"
        )
        assert del_box["x"] > link_box["x"], "Delete × should be to the right of the title"
