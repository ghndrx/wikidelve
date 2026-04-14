"""
PDF rendering for the Documents feature.

Pipeline: markdown source → HTML (with bundled CSS) → PDF bytes.

Renderer choice for MVP: ``xhtml2pdf`` (pure-Python, no system deps,
ships with the existing pip install). It produces serviceable PDFs
for one-pagers, briefs, PRDs — anything text-forward. For pixel-
perfect typography we'd swap to ``weasyprint`` later (better
fidelity but pulls in cairo / pango system libs, ~100MB image
bloat). The renderer interface is just ``markdown_text → bytes`` so
the swap is one file change.

Citation expansion: ``[ref:kb/slug]`` markers in the source resolve
to numbered footnotes + a References section appended to the doc.
The agent inserts these naturally during drafting (per
DOC_CHAT_AGENT_PROMPT).
"""

import io
import logging
import re
from typing import Optional

import markdown as _md

logger = logging.getLogger("kb-service.document_renderer")


# Default stylesheet — restrained, paper-friendly, prints well to A4.
# Lives inline so the renderer is self-contained and themes can
# override later by passing extra_css.
_DEFAULT_CSS = """
@page {
    size: A4;
    margin: 2cm 2cm 2.5cm 2cm;
    @bottom-center { content: counter(page) " / " counter(pages); }
}
body {
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 {
    font-size: 22pt;
    margin: 0 0 4pt 0;
    border-bottom: 1.5pt solid #1a1a1a;
    padding-bottom: 4pt;
}
h2 {
    font-size: 15pt;
    margin: 18pt 0 6pt 0;
    color: #2a2a2a;
}
h3 {
    font-size: 12pt;
    margin: 14pt 0 4pt 0;
    color: #3a3a3a;
}
p { margin: 0 0 8pt 0; }
ul, ol { margin: 0 0 10pt 18pt; }
li { margin-bottom: 2pt; }
code {
    font-family: "Courier", monospace;
    font-size: 10pt;
    background: #f4f4f4;
    padding: 1pt 3pt;
}
pre {
    font-family: "Courier", monospace;
    font-size: 9pt;
    background: #f4f4f4;
    padding: 8pt;
    border-left: 2pt solid #888;
    white-space: pre-wrap;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0;
}
th, td {
    border: 0.5pt solid #888;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
}
th { background: #ececec; font-weight: bold; }
blockquote {
    margin: 8pt 12pt;
    padding-left: 10pt;
    border-left: 2pt solid #888;
    color: #4a4a4a;
    font-style: italic;
}
.doc-meta {
    color: #6a6a6a;
    font-size: 9pt;
    margin-bottom: 16pt;
}
.references {
    margin-top: 24pt;
    padding-top: 8pt;
    border-top: 0.5pt solid #888;
    font-size: 9pt;
}
.references h2 {
    font-size: 11pt;
    margin: 0 0 6pt 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
sup.cite {
    font-size: 7pt;
    color: #06c;
}
"""


_REF_RE = re.compile(r"\[ref:([^\]]+)\]")


def _expand_citations(markdown_text: str) -> tuple[str, list[dict]]:
    """Replace ``[ref:kb/slug]`` markers with numbered superscripts.

    Returns the rewritten markdown plus an ordered list of unique
    references in the order they first appear. Same ref reused later
    in the doc reuses its number — no duplicates in the bibliography.
    """
    seen: dict[str, int] = {}
    refs: list[dict] = []

    def _sub(match: re.Match) -> str:
        target = match.group(1).strip()
        if target not in seen:
            seen[target] = len(seen) + 1
            kb, _, slug = target.partition("/")
            refs.append({
                "n": seen[target],
                "kb": kb or "personal",
                "slug": slug or target,
                "raw": target,
            })
        return f'<sup class="cite">[{seen[target]}]</sup>'

    rewritten = _REF_RE.sub(_sub, markdown_text)
    return rewritten, refs


def _references_html(refs: list[dict]) -> str:
    if not refs:
        return ""
    lines = ['<div class="references"><h2>References</h2><ol>']
    for r in refs:
        lines.append(
            f'<li>[{r["n"]}] '
            f'<a href="/wiki/{r["kb"]}/{r["slug"]}">{r["kb"]}/{r["slug"]}</a>'
            f'</li>'
        )
    lines.append("</ol></div>")
    return "".join(lines)


def render_html(
    markdown_text: str,
    *,
    title: Optional[str] = None,
    extra_css: Optional[str] = None,
    meta_line: Optional[str] = None,
) -> str:
    """Markdown source → standalone HTML doc string. Useful for
    previews and as the input to the PDF stage."""
    body, refs = _expand_citations(markdown_text)
    html_body = _md.markdown(
        body,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
    )
    refs_html = _references_html(refs)
    css = _DEFAULT_CSS + ("\n" + extra_css if extra_css else "")

    title_html = f"<h1>{title}</h1>" if title else ""
    meta_html = f'<div class="doc-meta">{meta_line}</div>' if meta_line else ""

    return (
        "<!doctype html>"
        "<html><head>"
        '<meta charset="utf-8"/>'
        f"<title>{title or 'Document'}</title>"
        f"<style>{css}</style>"
        "</head><body>"
        f"{title_html}{meta_html}{html_body}{refs_html}"
        "</body></html>"
    )


def render_pdf(
    markdown_text: str,
    *,
    title: Optional[str] = None,
    extra_css: Optional[str] = None,
    meta_line: Optional[str] = None,
) -> bytes:
    """Markdown source → PDF bytes. Caller persists the bytes via
    documents.commit_version(..., rendered=...).

    Raises ``RuntimeError`` if the underlying renderer fails — the
    document drafting flow catches that and surfaces it on the job
    so the user knows their commit didn't produce a viewable PDF.
    """
    # Lazy import so test envs without xhtml2pdf can still import
    # this module for render_html (HTML-only path).
    try:
        from xhtml2pdf import pisa
    except ImportError as exc:
        raise RuntimeError(
            "xhtml2pdf is not installed — add it to requirements"
        ) from exc

    html = render_html(
        markdown_text, title=title, extra_css=extra_css, meta_line=meta_line,
    )
    buf = io.BytesIO()
    result = pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError(
            f"PDF render failed: {result.err} error(s) reported by xhtml2pdf"
        )
    pdf_bytes = buf.getvalue()
    if not pdf_bytes:
        raise RuntimeError("PDF render produced 0 bytes")
    logger.info(
        "Rendered PDF: %d bytes from %d-char markdown",
        len(pdf_bytes), len(markdown_text),
    )
    return pdf_bytes
