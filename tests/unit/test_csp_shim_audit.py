"""Static audit of every template to guarantee no inline handler contains
a pattern the CSP inline-handler shim cannot safely parse.

The shim silently drops `if (...)` guards, `return`, `=>`, assignments,
`{ ... }` blocks, etc. — when that happens, a handler like
``onkeydown="if(event.key==='Enter'){send();}"`` degrades into
``send()`` firing on *every* keydown. This regression shipped on
2026-04-12 as the "every keystroke sends a chat" bug.

This test asserts no template in the repo reintroduces that class of
handler. Add to the allowlist only with a matching unit or E2E test
that proves the handler behaves correctly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"

# Patterns the CSP shim refuses to rewire. Must stay in sync with
# UNSAFE_PATTERNS in templates/base.html.
_UNSAFE_PATTERNS = [
    re.compile(r"\bif\s*\("),
    re.compile(r"\belse\b"),
    re.compile(r"\bfor\s*\("),
    re.compile(r"\bwhile\s*\("),
    re.compile(r"\breturn\b"),
    re.compile(r"=>"),
    re.compile(r"\{[^}]"),
    re.compile(r"[^=!<>]=[^=]"),
    re.compile(r"\bnew\s+"),
    re.compile(r"\bawait\b"),
    re.compile(r"\bthis\."),
]

# Inline handler attribute regex: on* on a real element (not a meta tag).
_HANDLER_RE = re.compile(r'on([a-z]+)\s*=\s*"([^"]*)"')

_EVENT_NAMES = {
    "click", "change", "input", "submit", "keydown", "keyup",
    "mouseover", "mouseout", "focus", "blur", "load", "error",
}


def _strip_js_strings(src: str) -> str:
    """Blank out quoted literals so assignment checks don't trip on
    things like `'/api/foo?kb=personal'`."""
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c in ("'", '"'):
            quote = c
            out.append(quote + quote)
            i += 1
            while i < n:
                d = src[i]
                if d == "\\" and i + 1 < n:
                    i += 2
                    continue
                if d == quote:
                    i += 1
                    break
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


_JINJA_INTERP = re.compile(r"\{\{.*?\}\}|\{%.*?%\}")


def _strip_jinja(src: str) -> str:
    """Replace Jinja substitutions with a neutral placeholder — the
    browser sees the rendered value, not the braces, so the audit
    should ignore them too."""
    return _JINJA_INTERP.sub("__JINJA__", src)


def _is_unsafe(handler: str) -> str | None:
    stripped = _strip_js_strings(_strip_jinja(handler))
    for pat in _UNSAFE_PATTERNS:
        if pat.search(stripped):
            return pat.pattern
    return None


_SCRIPT_BLOCK_RE = re.compile(
    r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE,
)
_STYLE_BLOCK_RE = re.compile(
    r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE,
)


def _blank(match: re.Match) -> str:
    # Preserve line count so reported line numbers stay accurate.
    return "\n" * match.group(0).count("\n")


def _strip_non_markup_blocks(text: str) -> str:
    text = _SCRIPT_BLOCK_RE.sub(_blank, text)
    text = _STYLE_BLOCK_RE.sub(_blank, text)
    return text


@pytest.mark.parametrize("template_path", sorted(TEMPLATES_DIR.glob("*.html")))
def test_no_unsafe_inline_handlers(template_path: Path):
    raw = template_path.read_text(encoding="utf-8")
    # Only scan real element attributes — not JS strings inside
    # <script> blocks or CSS inside <style> blocks.
    text = _strip_non_markup_blocks(raw)
    offenders: list[str] = []
    for m in _HANDLER_RE.finditer(text):
        event_name = m.group(1)
        if event_name not in _EVENT_NAMES:
            continue
        handler = m.group(2)
        reason = _is_unsafe(handler)
        if reason is not None:
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(
                f"  {template_path.name}:{line}  on{event_name}=...  [matched {reason}]\n"
                f"    {handler[:140]}"
            )
    assert not offenders, (
        "Inline event handlers contain patterns the CSP shim cannot safely "
        "parse. Rewrite these as addEventListener in a <script nonce> block "
        "(see templates/search.html for the pattern).\n\n"
        + "\n".join(offenders)
    )


def test_shim_unsafe_patterns_are_comprehensive():
    """Sanity-check that the canonical known-bad patterns are caught."""
    known_bad = [
        "if(event.key==='Enter'){send();}",
        "return send()",
        "x = 1",
        "this.style.opacity='1'",
        "(function(){send();})()",
        "new Date()",
        "await fetch('/x')",
        "() => send()",
        # Regression: onclick="send(this.title)" passed as safe but broke
        # at runtime because the shim resolved `this.title` as
        # window.this.title -> undefined. Caused the suggestion buttons
        # on /chat to silently do nothing.
        "send(this.title)",
        "fn(this.dataset.id)",
    ]
    for snippet in known_bad:
        assert _is_unsafe(snippet), f"expected unsafe: {snippet!r}"


def test_shim_accepts_safe_patterns():
    """Plain function calls — with literals, this, numbers, nulls,
    and dotted names — must still be considered safe."""
    safe = [
        "send()",
        "fn('hello')",
        "fn(\"a\", 'b')",
        "fn(this)",
        "fn(null)",
        "fn(true, false)",
        "fn(1, -2, 3.5)",
        "ns.fn('x')",
        "event.preventDefault(); send()",
        "send('/api/articles?kb=personal')",
        "send('/api/search?q=a&limit=5')",
    ]
    for snippet in safe:
        assert not _is_unsafe(snippet), f"expected safe: {snippet!r}"
