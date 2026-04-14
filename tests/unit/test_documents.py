"""Unit tests for app/documents.py.

Coverage focus by risk:
  1. _validate_manifest — autonomy_mode + doc_type + seed shape
  2. create_document → commit_version → list / get round-trip
  3. Pending-draft lifecycle (write / discard / commit)
  4. add_pinned_fact dedup
  5. Version trim past MAX_VERSIONS
"""

import json
from pathlib import Path

import pytest

from app import documents


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


# ---------------------------------------------------------------------------
# _validate_manifest
# ---------------------------------------------------------------------------

class TestValidateManifest:
    def _base(self, **overrides):
        m = {"title": "Test", "doc_type": "pdf"}
        m.update(overrides)
        return m

    def test_minimal_passes(self):
        out = documents._validate_manifest(self._base())
        assert out["slug"] == "test"
        assert out["doc_type"] == "pdf"
        assert out["autonomy_mode"] == documents.DEFAULT_AUTONOMY
        assert out["document_version"] == 1

    def test_default_autonomy_is_propose(self):
        out = documents._validate_manifest(self._base())
        assert out["autonomy_mode"] == "propose"

    def test_unknown_doc_type_rejected(self):
        with pytest.raises(ValueError, match="doc_type"):
            documents._validate_manifest(self._base(doc_type="zip"))

    def test_unknown_autonomy_rejected(self):
        with pytest.raises(ValueError, match="autonomy_mode"):
            documents._validate_manifest(self._base(autonomy_mode="rogue"))

    def test_seed_articles_must_have_kb_and_slug(self):
        with pytest.raises(ValueError, match="seed_article"):
            documents._validate_manifest(
                self._base(seed_articles=[{"slug": "x"}]),
            )

    def test_pinned_facts_truncated(self):
        out = documents._validate_manifest(
            self._base(pinned_facts=["x" * 1000]),
        )
        assert len(out["pinned_facts"][0]) == 500


# ---------------------------------------------------------------------------
# create_document → version lifecycle
# ---------------------------------------------------------------------------

class TestVersionLifecycle:
    def test_create_then_commit_v1(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(
            kb, "First Doc", "A test brief",
            seed_articles=[{"kb": "personal", "slug": "some-article"}],
        )
        manifest = documents.get_manifest(kb, slug)
        assert manifest["current_version"] == 0
        assert manifest["seed_articles"][0]["slug"] == "some-article"

        # Commit v1
        entry = documents.commit_version(
            kb, slug, "# Hello\n\nFirst draft", None,
            trigger="initial draft",
        )
        assert entry["v"] == 1
        assert documents.get_markdown(kb, slug) == "# Hello\n\nFirst draft"
        assert documents.get_manifest(kb, slug)["current_version"] == 1

    def test_commit_increments_version(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Inc", "brief")
        documents.commit_version(kb, slug, "v1 text", None, trigger="a")
        documents.commit_version(kb, slug, "v2 text", None, trigger="b")
        documents.commit_version(kb, slug, "v3 text", None, trigger="c")
        m = documents.get_manifest(kb, slug)
        assert m["current_version"] == 3
        assert documents.get_markdown(kb, slug) == "v3 text"
        assert documents.get_markdown(kb, slug, version=1) == "v1 text"
        assert documents.get_markdown(kb, slug, version=2) == "v2 text"

    def test_commit_rejects_oversize_markdown(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Big", "brief")
        big = "x" * (documents.MAX_MARKDOWN_BYTES + 1)
        with pytest.raises(ValueError, match="markdown"):
            documents.commit_version(kb, slug, big, None, trigger="overflow")

    def test_commit_rejects_oversize_rendered(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "BigPdf", "brief")
        big = b"\x00" * (documents.MAX_RENDERED_BYTES + 1)
        with pytest.raises(ValueError, match="rendered"):
            documents.commit_version(kb, slug, "ok", big, trigger="overflow")

    def test_commit_unknown_doc_raises(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        with pytest.raises(ValueError, match="not found"):
            documents.commit_version(kb, "ghost", "x", None, trigger="?")


# ---------------------------------------------------------------------------
# Pending draft lifecycle (propose mode)
# ---------------------------------------------------------------------------

class TestPendingDraft:
    def test_write_then_get(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "P", "brief")
        documents.write_pending_draft(kb, slug, "draft body", "tightened tone")
        pending = documents.get_pending_draft(kb, slug)
        assert pending["markdown"] == "draft body"
        assert pending["summary"] == "tightened tone"
        assert "proposed_at" in pending

    def test_discard_returns_true_when_present(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "P", "brief")
        documents.write_pending_draft(kb, slug, "x", "y")
        assert documents.discard_pending_draft(kb, slug) is True
        assert documents.get_pending_draft(kb, slug) is None

    def test_discard_returns_false_when_absent(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "P", "brief")
        assert documents.discard_pending_draft(kb, slug) is False

    def test_commit_pending_promotes_to_v_plus_1(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "P", "brief")
        # Establish a v1 first so promotion goes to v2
        documents.commit_version(kb, slug, "v1", None, trigger="init")
        documents.write_pending_draft(kb, slug, "v2 from agent", "trim")
        entry = documents.commit_pending_draft(kb, slug)
        assert entry is not None
        assert entry["v"] == 2
        assert documents.get_markdown(kb, slug) == "v2 from agent"
        # Pending was cleared
        assert documents.get_pending_draft(kb, slug) is None

    def test_commit_pending_with_no_draft_returns_none(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "P", "brief")
        assert documents.commit_pending_draft(kb, slug) is None

    def test_oversize_pending_rejected(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "P", "brief")
        with pytest.raises(ValueError, match="too large"):
            documents.write_pending_draft(
                kb, slug,
                "x" * (documents.MAX_MARKDOWN_BYTES + 1),
                "overflow",
            )


# ---------------------------------------------------------------------------
# Pinned facts
# ---------------------------------------------------------------------------

class TestPinnedFacts:
    def test_add_first_fact(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Doc", "brief")
        ok = documents.add_pinned_fact(kb, slug, "pricing is $99/mo")
        assert ok is True
        assert "pricing is $99/mo" in documents.get_manifest(kb, slug)["pinned_facts"]

    def test_dedup_same_fact(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Doc", "brief")
        documents.add_pinned_fact(kb, slug, "fact A")
        documents.add_pinned_fact(kb, slug, "fact A")
        facts = documents.get_manifest(kb, slug)["pinned_facts"]
        assert facts.count("fact A") == 1

    def test_empty_fact_rejected(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Doc", "brief")
        assert documents.add_pinned_fact(kb, slug, "   ") is False

    def test_unknown_doc_returns_false(self, tmp_kb_for_documents):
        assert documents.add_pinned_fact(tmp_kb_for_documents, "ghost", "fact") is False


# ---------------------------------------------------------------------------
# History log
# ---------------------------------------------------------------------------

class TestHistory:
    def test_create_appends_event(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Doc", "brief")
        history = documents.get_history(kb, slug)
        assert len(history) >= 1
        assert history[0]["type"] == "create"

    def test_chat_turn_appended(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Doc", "brief")
        documents.append_chat_turn(kb, slug, "user", "make it shorter")
        documents.append_chat_turn(
            kb, slug, "agent", "I tightened §2",
            metadata={"diff_summary": "removed 3 paragraphs"},
        )
        history = documents.get_history(kb, slug)
        types = [e["type"] for e in history]
        assert types.count("turn") == 2

    def test_invalid_role_rejected(self, tmp_kb_for_documents):
        kb = tmp_kb_for_documents
        slug = documents.create_document(kb, "Doc", "brief")
        with pytest.raises(ValueError, match="role"):
            documents.append_chat_turn(kb, slug, "robot", "msg")
