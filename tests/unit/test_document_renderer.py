"""Unit tests for app/document_renderer.py.

Covers:
  - HTML rendering shape (title / meta / body / refs section)
  - Citation expansion: numbered footnotes + dedup of repeated refs
  - PDF rendering produces non-empty bytes starting with %PDF
  - Documents commit path renders + persists binary alongside markdown
"""

import re

import pytest

from app import document_renderer as dr


# ---------------------------------------------------------------------------
# Citation expansion
# ---------------------------------------------------------------------------

class TestCitationExpansion:
    def test_no_refs_means_empty_list(self):
        body, refs = dr._expand_citations("Plain text, no refs.")
        assert body == "Plain text, no refs."
        assert refs == []

    def test_single_ref_numbered(self):
        body, refs = dr._expand_citations("Per the docs[ref:personal/foo].")
        assert "<sup" in body and "[1]" in body
        assert refs == [{"n": 1, "kb": "personal", "slug": "foo", "raw": "personal/foo"}]

    def test_repeated_ref_dedups(self):
        body, refs = dr._expand_citations(
            "First[ref:personal/foo] then again[ref:personal/foo]."
        )
        # Same ref should reuse number 1 — no [2] for the duplicate.
        assert body.count("[1]") == 2
        assert "[2]" not in body
        assert len(refs) == 1

    def test_multiple_refs_in_order(self):
        body, refs = dr._expand_citations(
            "[ref:personal/a] then [ref:personal/b] then [ref:personal/a]"
        )
        nums = re.findall(r"\[(\d+)\]", body)
        assert nums == ["1", "2", "1"]
        assert [r["slug"] for r in refs] == ["a", "b"]

    def test_kb_default_when_missing(self):
        body, refs = dr._expand_citations("[ref:loose-slug]")
        assert refs[0]["kb"] == "personal"
        assert refs[0]["slug"] == "loose-slug"


# ---------------------------------------------------------------------------
# HTML rendering shape
# ---------------------------------------------------------------------------

class TestRenderHTML:
    def test_basic_html_doc_structure(self):
        html = dr.render_html(
            "## Hello\n\nworld",
            title="My Doc",
        )
        assert html.startswith("<!doctype html>")
        assert "<title>My Doc</title>" in html
        assert "<h1>My Doc</h1>" in html
        assert "<h2>Hello</h2>" in html
        assert "world" in html

    def test_meta_line_renders(self):
        html = dr.render_html(
            "body", title="X", meta_line="2026-04-14 · v3",
        )
        assert "2026-04-14 · v3" in html
        assert 'class="doc-meta"' in html

    def test_references_section_appended(self):
        html = dr.render_html(
            "Claim[ref:personal/foo].", title="X",
        )
        assert 'class="references"' in html
        assert "personal/foo" in html

    def test_no_refs_no_section(self):
        html = dr.render_html("plain", title="X")
        assert "References" not in html

    def test_extra_css_appended(self):
        html = dr.render_html(
            "x", title="X", extra_css="body { color: hotpink; }",
        )
        assert "hotpink" in html


# ---------------------------------------------------------------------------
# PDF rendering — exercises xhtml2pdf if installed, else skips
# ---------------------------------------------------------------------------

class TestRenderPDF:
    def test_renders_basic_doc(self):
        try:
            pdf = dr.render_pdf(
                "## Section\n\nSome text.",
                title="Smoke Test",
            )
        except RuntimeError as exc:
            if "not installed" in str(exc):
                pytest.skip("xhtml2pdf not installed in this env")
            raise
        # Real PDFs start with the magic %PDF marker.
        assert pdf[:4] == b"%PDF"
        # Sanity: > 1 KB even for a tiny doc
        assert len(pdf) > 1024

    def test_renders_with_table(self):
        try:
            pdf = dr.render_pdf(
                "| col | col |\n|---|---|\n| a | b |\n",
                title="Table Test",
            )
        except RuntimeError as exc:
            if "not installed" in str(exc):
                pytest.skip("xhtml2pdf not installed")
            raise
        assert pdf[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Documents commit-path integration: render runs at commit time
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_kb_for_documents(tmp_path, monkeypatch):
    import app.config as _config
    import app.storage as _storage
    kb_root = tmp_path / "kb"
    kb_dir = kb_root / "personal"
    (kb_dir / "wiki").mkdir(parents=True)
    prev = dict(_config.KB_DIRS)
    _config.KB_DIRS.clear()
    _config.KB_DIRS["personal"] = kb_dir
    monkeypatch.setattr(_config, "KB_ROOT", kb_root)
    _storage.set_storage(_storage._build_default())
    yield "personal"
    _config.KB_DIRS.clear()
    _config.KB_DIRS.update(prev)
    _storage.set_storage(_storage._build_default())


class TestCommitPendingRendersPDF:
    def test_pending_commit_writes_pdf_blob(self, tmp_kb_for_documents):
        try:
            from xhtml2pdf import pisa  # noqa
        except ImportError:
            pytest.skip("xhtml2pdf not installed")

        from app import documents

        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Render Test", "render brief")
        documents.write_pending_draft(
            kb, slug, "## Hello\n\nbody text", "first draft",
        )
        entry = documents.commit_pending_draft(kb, slug)
        assert entry["v"] == 1

        pdf = documents.get_rendered(kb, slug, version=1)
        assert pdf is not None
        assert pdf[:4] == b"%PDF"

    def test_render_failure_does_not_block_commit(
        self, tmp_kb_for_documents, monkeypatch,
    ):
        # Force the renderer to raise — commit should still land the
        # markdown as v+1, just without a binary. The user can
        # re-render later.
        from app import documents

        def _boom(*_a, **_kw):
            raise RuntimeError("renderer is sad")

        monkeypatch.setattr(
            "app.document_renderer.render_pdf", _boom,
        )

        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Boom", "brief")
        documents.write_pending_draft(kb, slug, "body", "draft")
        entry = documents.commit_pending_draft(kb, slug)
        assert entry["v"] == 1
        # Markdown landed
        assert documents.get_markdown(kb, slug, 1) == "body"
        # Binary did not
        assert documents.get_rendered(kb, slug, 1) is None
