"""Unit tests for app/scaffolds.py.

Coverage focus by risk:
  1. _safe_rel_path — file-system escape (highest risk; bug here =
     write outside the scaffold tree)
  2. _validate_manifest — input shape contract for agent output
  3. create_scaffold — size caps, file caps, entrypoint enforcement
  4. Index lifecycle (touch / drop / self-heal)
"""

import json
from pathlib import Path

import pytest

from app import scaffolds


# ---------------------------------------------------------------------------
# _safe_rel_path
# ---------------------------------------------------------------------------

class TestSafeRelPath:
    def test_simple_path_passes(self):
        assert scaffolds._safe_rel_path("index.html") == "index.html"

    def test_nested_path_passes(self):
        assert scaffolds._safe_rel_path("css/styles.css") == "css/styles.css"

    def test_strips_leading_slash(self):
        # The function strips leading slashes via .strip('/'), then
        # the absolute-path guard checks the original (unstripped)
        # value — which means a leading slash is rejected before
        # normalisation. That's the safer of the two readings.
        with pytest.raises(ValueError, match="unsafe"):
            scaffolds._safe_rel_path("/etc/passwd")

    def test_backslash_normalized(self):
        assert scaffolds._safe_rel_path("css\\styles.css") == "css/styles.css"

    def test_dotdot_rejected_at_start(self):
        with pytest.raises(ValueError, match="path escape"):
            scaffolds._safe_rel_path("../etc/passwd")

    def test_dotdot_rejected_in_middle(self):
        with pytest.raises(ValueError, match="path escape"):
            scaffolds._safe_rel_path("a/../b")

    def test_dotdot_rejected_via_backslash(self):
        with pytest.raises(ValueError, match="path escape"):
            scaffolds._safe_rel_path("a\\..\\b")

    def test_colon_rejected(self):
        # Windows-drive style escape attempt
        with pytest.raises(ValueError, match="unsafe"):
            scaffolds._safe_rel_path("C:/Windows/System32")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            scaffolds._safe_rel_path("")

    def test_single_dot_passes(self):
        # A single '.' segment is harmless — pathlib normalises it
        # away. We allow it rather than reject every funky-looking
        # path that's actually safe.
        assert scaffolds._safe_rel_path("./index.html") == "index.html"


# ---------------------------------------------------------------------------
# _validate_manifest
# ---------------------------------------------------------------------------

class TestValidateManifest:
    def _base(self, **overrides):
        m = {
            "title": "Test Scaffold",
            "scaffold_type": "landing-page",
            "framework": "vanilla",
            "preview_type": "static",
            "entrypoint": "index.html",
        }
        m.update(overrides)
        return m

    def test_minimal_passes(self):
        out = scaffolds._validate_manifest(self._base())
        assert out["slug"] == "test-scaffold"
        assert out["scaffold_version"] == 1
        assert out["framework"] == "vanilla"

    def test_unknown_type_falls_back_to_other(self):
        out = scaffolds._validate_manifest(self._base(scaffold_type="not-a-thing"))
        assert out["scaffold_type"] == "other"

    def test_unknown_framework_rejected(self):
        with pytest.raises(ValueError, match="framework"):
            scaffolds._validate_manifest(self._base(framework="cobol"))

    def test_unknown_preview_type_rejected(self):
        with pytest.raises(ValueError, match="preview_type"):
            scaffolds._validate_manifest(self._base(preview_type="hologram"))

    def test_missing_entrypoint_rejected(self):
        m = self._base()
        del m["entrypoint"]
        with pytest.raises(ValueError, match="entrypoint"):
            scaffolds._validate_manifest(m)

    def test_unsafe_entrypoint_rejected(self):
        with pytest.raises(ValueError, match="path escape"):
            scaffolds._validate_manifest(self._base(entrypoint="../escape.html"))

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="dict"):
            scaffolds._validate_manifest("not a dict")  # type: ignore

    def test_description_truncated(self):
        out = scaffolds._validate_manifest(self._base(description="x" * 5000))
        assert len(out["description"]) == 2000


# ---------------------------------------------------------------------------
# create_scaffold — uses real storage, isolated tmp KB
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_kb_for_scaffolds(tmp_path, monkeypatch):
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


class TestCreateScaffold:
    def _files(self, *paths):
        return [{"path": p, "content": f"<!-- {p} -->\n"} for p in paths]

    def _manifest(self, **overrides):
        m = {
            "title": "Test",
            "scaffold_type": "landing-page",
            "framework": "vanilla",
            "preview_type": "static",
            "entrypoint": "index.html",
        }
        m.update(overrides)
        return m

    def test_basic_create(self, tmp_kb_for_scaffolds):
        kb = tmp_kb_for_scaffolds
        slug = scaffolds.create_scaffold(
            kb, self._manifest(),
            self._files("index.html", "styles.css"),
        )
        assert slug == "test"
        # Manifest persisted
        manifest = scaffolds.get_manifest(kb, slug)
        assert manifest["entrypoint"] == "index.html"
        assert set(manifest["files"]) == {"index.html", "styles.css"}
        # Files persisted
        assert scaffolds.get_file(kb, slug, "index.html") == "<!-- index.html -->\n"

    def test_no_files_rejected(self, tmp_kb_for_scaffolds):
        with pytest.raises(ValueError, match="at least one file"):
            scaffolds.create_scaffold(
                tmp_kb_for_scaffolds, self._manifest(), [],
            )

    def test_too_many_files_rejected(self, tmp_kb_for_scaffolds):
        files = [{"path": f"f{i}.txt", "content": "x"} for i in range(scaffolds.MAX_FILES + 1)]
        # Entrypoint must exist or earlier guards trigger first; use f0
        with pytest.raises(ValueError, match=r"max is \d+"):
            scaffolds.create_scaffold(
                tmp_kb_for_scaffolds,
                self._manifest(entrypoint="f0.txt"),
                files,
            )

    def test_oversized_file_rejected(self, tmp_kb_for_scaffolds):
        big = "x" * (scaffolds.MAX_FILE_BYTES + 1)
        with pytest.raises(ValueError, match="max per file"):
            scaffolds.create_scaffold(
                tmp_kb_for_scaffolds, self._manifest(),
                [{"path": "index.html", "content": big}],
            )

    def test_entrypoint_must_exist_in_files(self, tmp_kb_for_scaffolds):
        with pytest.raises(ValueError, match="entrypoint"):
            scaffolds.create_scaffold(
                tmp_kb_for_scaffolds,
                self._manifest(entrypoint="missing.html"),
                self._files("index.html"),
            )

    def test_path_escape_rejected_via_files(self, tmp_kb_for_scaffolds):
        # Even if validate_manifest passed because entrypoint is OK,
        # individual file paths must also pass _safe_rel_path.
        with pytest.raises(ValueError, match="path escape"):
            scaffolds.create_scaffold(
                tmp_kb_for_scaffolds, self._manifest(),
                [
                    {"path": "index.html", "content": "<!doctype html>"},
                    {"path": "../escape.txt", "content": "boom"},
                ],
            )

    def test_index_updated_on_create(self, tmp_kb_for_scaffolds):
        kb = tmp_kb_for_scaffolds
        scaffolds.create_scaffold(kb, self._manifest(title="One"), self._files("index.html"))
        scaffolds.create_scaffold(kb, self._manifest(title="Two"), self._files("index.html"))
        listing = scaffolds.list_scaffolds(kb)
        slugs = {s["slug"] for s in listing}
        assert {"one", "two"} <= slugs


class TestDeleteScaffold:
    def test_delete_clears_files_and_index(self, tmp_kb_for_scaffolds):
        kb = tmp_kb_for_scaffolds
        slug = scaffolds.create_scaffold(
            kb, {"title": "Doomed", "scaffold_type": "landing-page",
                 "framework": "vanilla", "preview_type": "static",
                 "entrypoint": "index.html"},
            [{"path": "index.html", "content": "<!doctype html>"}],
        )
        scaffolds.delete_scaffold(kb, slug)
        assert scaffolds.get_manifest(kb, slug) is None
        assert all(s["slug"] != slug for s in scaffolds.list_scaffolds(kb))


class TestListScaffoldsSelfHealing:
    def test_dropped_manifest_is_filtered(self, tmp_kb_for_scaffolds):
        from app import storage as _storage
        kb = tmp_kb_for_scaffolds
        slug = scaffolds.create_scaffold(
            kb, {"title": "Ghost", "scaffold_type": "landing-page",
                 "framework": "vanilla", "preview_type": "static",
                 "entrypoint": "index.html"},
            [{"path": "index.html", "content": "x"}],
        )
        # Manually delete the manifest behind list_scaffolds' back —
        # the index entry stays but the manifest read returns None.
        _storage.delete(kb, f"scaffolds/{slug}/manifest.json")
        listing = scaffolds.list_scaffolds(kb)
        assert all(s["slug"] != slug for s in listing)
