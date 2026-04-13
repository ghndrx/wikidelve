"""Unit tests for app.storage — LocalStorage + S3Storage + module helpers."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.storage import LocalStorage, S3Storage, get_storage, set_storage


# ===========================================================================
# LocalStorage
# ===========================================================================


@pytest.fixture
def local_storage(tmp_path, monkeypatch):
    kb_dirs = {"alpha": tmp_path / "alpha", "beta": tmp_path / "beta"}
    for p in kb_dirs.values():
        (p / "wiki").mkdir(parents=True)
        (p / "raw").mkdir(parents=True)
    monkeypatch.setattr("app.storage.KB_DIRS", kb_dirs)
    return LocalStorage()


class TestLocalStorage:
    def test_write_then_read(self, local_storage):
        local_storage.write_text("alpha", "wiki/demo.md", "body text")
        assert local_storage.read_text("alpha", "wiki/demo.md") == "body text"

    def test_read_missing_returns_none(self, local_storage):
        assert local_storage.read_text("alpha", "wiki/missing.md") is None

    def test_read_oserror_returns_none(self, local_storage, monkeypatch, caplog):
        """Coverage: lines 76-78 — OSError during read_text."""
        local_storage.write_text("alpha", "wiki/demo.md", "body")
        p = local_storage._path("alpha", "wiki/demo.md")
        # Make the file unreadable by patching Path.read_text to raise
        with patch.object(Path, "read_text", side_effect=OSError("disk error")):
            with caplog.at_level(logging.WARNING):
                result = local_storage.read_text("alpha", "wiki/demo.md")
        assert result is None
        assert "disk error" in caplog.text

    def test_exists(self, local_storage):
        assert local_storage.exists("alpha", "wiki/missing.md") is False
        local_storage.write_text("alpha", "wiki/demo.md", "hello")
        assert local_storage.exists("alpha", "wiki/demo.md") is True

    def test_delete(self, local_storage):
        local_storage.write_text("alpha", "wiki/demo.md", "hello")
        assert local_storage.delete("alpha", "wiki/demo.md") is True
        assert local_storage.exists("alpha", "wiki/demo.md") is False
        assert local_storage.delete("alpha", "wiki/demo.md") is False

    def test_delete_oserror_returns_false(self, local_storage, caplog):
        """Coverage: lines 92-94 — OSError during delete."""
        local_storage.write_text("alpha", "wiki/demo.md", "hello")
        with patch.object(Path, "unlink", side_effect=OSError("perm denied")):
            with caplog.at_level(logging.WARNING):
                result = local_storage.delete("alpha", "wiki/demo.md")
        assert result is False
        assert "perm denied" in caplog.text

    def test_list_slugs_skips_underscored(self, local_storage):
        local_storage.write_text("alpha", "wiki/one.md", "a")
        local_storage.write_text("alpha", "wiki/two.md", "b")
        local_storage.write_text("alpha", "wiki/_hidden.md", "x")
        slugs = local_storage.list_slugs("alpha")
        assert slugs == ["one", "two"]

    def test_list_slugs_empty_kb(self, local_storage):
        assert local_storage.list_slugs("alpha") == []

    def test_list_slugs_nonexistent_subdir(self, local_storage):
        assert local_storage.list_slugs("alpha", subdir="nope") == []

    def test_iter_articles(self, local_storage):
        local_storage.write_text("alpha", "wiki/one.md", "alpha body")
        local_storage.write_text("alpha", "wiki/two.md", "beta body")
        local_storage.write_text("alpha", "wiki/_skip.md", "nope")
        results = dict(local_storage.iter_articles("alpha"))
        assert set(results.keys()) == {"one", "two"}
        assert results["one"] == "alpha body"

    def test_iter_articles_empty(self, local_storage):
        assert list(local_storage.iter_articles("alpha")) == []

    def test_iter_articles_nonexistent_subdir(self, local_storage):
        """Coverage: line 110 — missing subdir returns empty iter."""
        assert list(local_storage.iter_articles("alpha", subdir="nope")) == []

    def test_iter_articles_oserror_skips_file(self, local_storage, caplog):
        """Coverage: lines 117-118 — OSError reading one article skips it."""
        local_storage.write_text("alpha", "wiki/good.md", "ok content")
        local_storage.write_text("alpha", "wiki/bad.md", "will fail")
        original_read = Path.read_text
        bad_path = local_storage._path("alpha", "wiki/bad.md")

        def patched_read(self, *a, **kw):
            if self == bad_path:
                raise OSError("io error")
            return original_read(self, *a, **kw)

        with patch.object(Path, "read_text", patched_read):
            with caplog.at_level(logging.WARNING):
                results = dict(local_storage.iter_articles("alpha"))
        assert "good" in results
        assert "bad" not in results
        assert "io error" in caplog.text

    def test_init_kb_creates_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.storage.KB_DIRS", {"fresh": tmp_path / "fresh"})
        LocalStorage().init_kb("fresh")
        assert (tmp_path / "fresh" / "wiki").is_dir()
        assert (tmp_path / "fresh" / "raw").is_dir()

    def test_list_kbs(self, local_storage):
        assert local_storage.list_kbs() == ["alpha", "beta"]

    def test_write_creates_parent_dirs(self, local_storage):
        local_storage.write_text("alpha", "nested/dir/file.md", "ok")
        assert local_storage.read_text("alpha", "nested/dir/file.md") == "ok"

    def test_path_traversal_backslash(self, local_storage):
        with pytest.raises(ValueError, match="Invalid rel_path"):
            local_storage.read_text("alpha", "wiki\\..\\secret.md")

    def test_path_traversal_absolute(self, local_storage):
        with pytest.raises(ValueError, match="Invalid rel_path"):
            local_storage.read_text("alpha", "/etc/passwd")

    def test_path_traversal_dotdot(self, local_storage):
        with pytest.raises(ValueError, match="Invalid rel_path segment"):
            local_storage.read_text("alpha", "../outside.md")

    def test_root_fallback_to_kb_root(self, tmp_path, monkeypatch):
        """Coverage: line 56 — KB not in KB_DIRS falls back to KB_ROOT / kb."""
        monkeypatch.setattr("app.storage.KB_DIRS", {})
        monkeypatch.setattr("app.storage.KB_ROOT", tmp_path)
        ls = LocalStorage()
        root = ls._root("unknown_kb")
        assert root == tmp_path / "unknown_kb"


# ===========================================================================
# S3Storage
# ===========================================================================


class TestS3Storage:
    """Test S3Storage with fully mocked boto3 client."""

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        # NoSuchKey exception class
        client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        return client

    @pytest.fixture
    def s3(self, mock_client):
        storage = S3Storage(bucket="test-bucket", prefix="pfx")
        with patch.object(type(storage), "_client", new_callable=PropertyMock, return_value=mock_client):
            yield storage, mock_client

    @pytest.fixture
    def s3_no_prefix(self, mock_client):
        storage = S3Storage(bucket="test-bucket", prefix="")
        with patch.object(type(storage), "_client", new_callable=PropertyMock, return_value=mock_client):
            yield storage, mock_client

    def test_init_requires_bucket(self):
        """Coverage: line 147."""
        with pytest.raises(ValueError, match="bucket name"):
            S3Storage(bucket="", prefix="")

    def test_key_with_prefix(self, s3):
        storage, _ = s3
        assert storage._key("mykb", "wiki/test.md") == "pfx/mykb/wiki/test.md"

    def test_key_without_prefix(self, s3_no_prefix):
        storage, _ = s3_no_prefix
        assert storage._key("mykb", "wiki/test.md") == "mykb/wiki/test.md"

    def test_key_rejects_traversal(self, s3):
        storage, _ = s3
        with pytest.raises(ValueError):
            storage._key("mykb", "../escape.md")
        with pytest.raises(ValueError):
            storage._key("mykb", "/absolute.md")
        with pytest.raises(ValueError):
            storage._key("mykb", "wiki\\bad.md")

    def test_kb_prefix(self, s3):
        storage, _ = s3
        assert storage._kb_prefix("mykb", "wiki") == "pfx/mykb/wiki/"
        assert storage._kb_prefix("mykb") == "pfx/mykb/"

    def test_kb_prefix_no_prefix(self, s3_no_prefix):
        storage, _ = s3_no_prefix
        assert storage._kb_prefix("mykb", "wiki") == "mykb/wiki/"

    def test_read_text_success(self, s3):
        storage, client = s3
        body = MagicMock()
        body.read.return_value = b"hello world"
        client.get_object.return_value = {"Body": body}
        result = storage.read_text("mykb", "wiki/test.md")
        assert result == "hello world"
        client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="pfx/mykb/wiki/test.md"
        )

    def test_read_text_no_such_key(self, s3):
        """Coverage: line 182-183."""
        storage, client = s3
        client.get_object.side_effect = client.exceptions.NoSuchKey("missing")
        assert storage.read_text("mykb", "wiki/missing.md") is None

    def test_read_text_other_exception(self, s3, caplog):
        """Coverage: lines 184-190."""
        storage, client = s3
        client.get_object.side_effect = RuntimeError("network error")
        with caplog.at_level(logging.WARNING):
            result = storage.read_text("mykb", "wiki/broken.md")
        assert result is None
        assert "network error" in caplog.text

    def test_write_text(self, s3):
        storage, client = s3
        storage.write_text("mykb", "wiki/test.md", "content here")
        client.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="pfx/mykb/wiki/test.md",
            Body=b"content here",
            ContentType="text/markdown; charset=utf-8",
        )

    def test_delete_existing(self, s3):
        """Coverage: lines 207-214."""
        storage, client = s3
        client.head_object.return_value = {}
        client.delete_object.return_value = {}
        assert storage.delete("mykb", "wiki/test.md") is True
        client.delete_object.assert_called_once()

    def test_delete_missing(self, s3):
        """Coverage: lines 210-211."""
        storage, client = s3
        client.head_object.side_effect = Exception("404")
        assert storage.delete("mykb", "wiki/missing.md") is False
        client.delete_object.assert_not_called()

    def test_delete_error_on_delete_call(self, s3, caplog):
        """Coverage: lines 215-217."""
        storage, client = s3
        client.head_object.return_value = {}
        client.delete_object.side_effect = RuntimeError("s3 delete failed")
        with caplog.at_level(logging.WARNING):
            result = storage.delete("mykb", "wiki/test.md")
        assert result is False

    def test_exists_true(self, s3):
        storage, client = s3
        client.head_object.return_value = {}
        assert storage.exists("mykb", "wiki/test.md") is True

    def test_exists_false(self, s3):
        storage, client = s3
        client.head_object.side_effect = Exception("404")
        assert storage.exists("mykb", "wiki/missing.md") is False

    def test_list_slugs(self, s3):
        """Coverage: lines 227-241."""
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "pfx/mykb/wiki/alpha.md"},
                    {"Key": "pfx/mykb/wiki/beta.md"},
                    {"Key": "pfx/mykb/wiki/_hidden.md"},
                    {"Key": "pfx/mykb/wiki/nested/deep.md"},
                    {"Key": "pfx/mykb/wiki/readme.txt"},
                ]
            }
        ]
        slugs = storage.list_slugs("mykb")
        assert slugs == ["alpha", "beta"]

    def test_list_slugs_empty(self, s3):
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]
        assert storage.list_slugs("mykb") == []

    def test_iter_articles(self, s3):
        """Coverage: lines 243-272."""
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "pfx/mykb/wiki/alpha.md"},
                    {"Key": "pfx/mykb/wiki/beta.md"},
                    {"Key": "pfx/mykb/wiki/_skip.md"},
                ]
            }
        ]
        body_a = MagicMock()
        body_a.read.return_value = b"alpha content"
        body_b = MagicMock()
        body_b.read.return_value = b"beta content"
        client.get_object.side_effect = [
            {"Body": body_a},
            {"Body": body_b},
        ]
        results = dict(storage.iter_articles("mykb"))
        assert results == {"alpha": "alpha content", "beta": "beta content"}

    def test_iter_articles_empty(self, s3):
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]
        assert list(storage.iter_articles("mykb")) == []

    def test_iter_articles_fetch_error_skips(self, s3):
        """Coverage: lines 262-264 — fetch error skips that article."""
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "pfx/mykb/wiki/good.md"}]}
        ]
        client.get_object.side_effect = RuntimeError("read fail")
        results = list(storage.iter_articles("mykb"))
        assert results == []

    def test_init_kb_noop(self, s3):
        """Coverage: lines 274-277."""
        storage, _ = s3
        assert storage.init_kb("mykb") is None

    def test_list_kbs(self, s3):
        """Coverage: lines 279-297."""
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "pfx/personal/"},
                    {"Prefix": "pfx/work/"},
                    {"Prefix": "pfx/_reserved/"},
                    {"Prefix": "pfx/tfstate/"},
                    {"Prefix": ""},
                ]
            }
        ]
        kbs = storage.list_kbs()
        assert kbs == ["personal", "work"]

    def test_list_kbs_empty(self, s3):
        storage, client = s3
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]
        assert storage.list_kbs() == []


# ===========================================================================
# Module-level helpers
# ===========================================================================


class TestModuleHelpers:
    def test_set_and_get_storage(self, monkeypatch):
        """Coverage: lines 329-331, 348."""
        import app.storage as mod
        prev = mod._default
        try:
            mock = MagicMock()
            set_storage(mock)
            assert get_storage() is mock
        finally:
            mod._default = prev

    def test_build_default_local(self, monkeypatch):
        """Coverage: line 332 — default builds LocalStorage."""
        import app.storage as mod
        monkeypatch.setattr(mod, "_BACKEND", "local")
        prev = mod._default
        mod._default = None
        try:
            s = get_storage()
            assert isinstance(s, LocalStorage)
        finally:
            mod._default = prev

    def test_build_default_s3_missing_bucket(self, monkeypatch):
        """Coverage: lines 329-331 — s3 backend without bucket raises."""
        import app.storage as mod
        monkeypatch.setattr(mod, "_BACKEND", "s3")
        monkeypatch.setattr(mod, "_S3_BUCKET", "")
        prev = mod._default
        mod._default = None
        try:
            with pytest.raises(RuntimeError, match="S3_BUCKET"):
                get_storage()
        finally:
            mod._default = prev

    def test_convenience_passthroughs(self, monkeypatch):
        """Coverage: lines 367, 371 — module-level functions delegate."""
        import app.storage as mod
        mock = MagicMock()
        mock.exists.return_value = True
        mock.list_slugs.return_value = ["a", "b"]
        prev = mod._default
        try:
            set_storage(mock)
            assert mod.exists("kb", "wiki/a.md") is True
            assert mod.list_slugs("kb") == ["a", "b"]
            mock.exists.assert_called_once_with("kb", "wiki/a.md")
            mock.list_slugs.assert_called_once_with("kb", "wiki", ".md")
        finally:
            mod._default = prev

    def test_backend_name(self, monkeypatch):
        import app.storage as mod
        prev = mod._default
        try:
            set_storage(LocalStorage())
            assert mod.backend_name() == "local"
        finally:
            mod._default = prev


# ===========================================================================
# _s3_client (coverage: lines 302-314)
# ===========================================================================


class TestS3Client:
    def test_s3_client_with_explicit_keys(self, monkeypatch):
        """Coverage: lines 302-314."""
        import app.storage as mod
        mod._s3_client.cache_clear()
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET")
        try:
            import boto3
            from botocore.config import Config
            mock_client = MagicMock()
            with patch("boto3.client", return_value=mock_client) as mock_call:
                result = mod._s3_client()
                assert result is mock_client
                call_kwargs = mock_call.call_args
                assert call_kwargs[1]["region_name"] == "eu-west-1"
                assert call_kwargs[1]["aws_access_key_id"] == "AKID"
                assert call_kwargs[1]["aws_secret_access_key"] == "SECRET"
        except ImportError:
            pytest.skip("boto3 not installed")
        finally:
            mod._s3_client.cache_clear()
