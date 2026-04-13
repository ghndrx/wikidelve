"""Unit tests for app/s3_sync.py — S3 backing store."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app import s3_sync


class TestS3SyncDisabled:
    """Tests when S3 is not configured."""

    def test_not_enabled_by_default(self):
        with patch.object(s3_sync, "_enabled", False):
            assert s3_sync.is_enabled() is False

    def test_upload_noop_when_disabled(self):
        with patch.object(s3_sync, "_enabled", False):
            # Should not raise
            s3_sync.upload_file("/some/path")

    def test_delete_noop_when_disabled(self):
        with patch.object(s3_sync, "_enabled", False):
            s3_sync.delete_file("/some/path")

    def test_pull_returns_zero_when_disabled(self):
        with patch.object(s3_sync, "_enabled", False):
            assert s3_sync.pull_from_s3() == 0

    def test_push_returns_zero_when_disabled(self):
        with patch.object(s3_sync, "_enabled", False):
            assert s3_sync.push_to_s3() == 0

    def test_sync_db_noop_when_disabled(self):
        with patch.object(s3_sync, "_enabled", False):
            s3_sync.sync_db()


class TestS3KeyGeneration:
    def test_kb_path(self):
        key = s3_sync._s3_key("/kb/personal/wiki/test.md")
        assert key == "personal/wiki/test.md"

    def test_with_prefix(self):
        with patch.object(s3_sync, "S3_PREFIX", "backups"):
            key = s3_sync._s3_key("/kb/personal/wiki/test.md")
            assert key == "backups/personal/wiki/test.md"

    def test_db_path(self):
        key = s3_sync._s3_key("/kb/wikidelve.db")
        assert key == "wikidelve.db"

    def test_research_path(self):
        key = s3_sync._s3_key("/kb/research/output.md")
        assert key == "research/output.md"

    def test_no_kb_in_path(self):
        key = s3_sync._s3_key("/tmp/somefile.txt")
        assert key == "somefile.txt"


class TestS3Upload:
    """Tests with mocked boto3 client."""

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_upload_calls_s3(self, tmp_path):
        test_file = tmp_path / "test.md"
        test_file.write_text("content")

        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            s3_sync.upload_file(test_file)

        mock_client.upload_file.assert_called_once()
        args = mock_client.upload_file.call_args[0]
        assert args[0] == str(test_file)
        assert args[1] == "test-bucket"

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    def test_upload_missing_file_noop(self, tmp_path):
        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            s3_sync.upload_file(tmp_path / "nonexistent.md")
        mock_client.upload_file.assert_not_called()

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_delete_calls_s3(self):
        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            s3_sync.delete_file("/kb/personal/wiki/old.md")

        mock_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="personal/wiki/old.md",
        )

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_upload_handles_s3_error(self, tmp_path):
        test_file = tmp_path / "test.md"
        test_file.write_text("content")

        mock_client = MagicMock()
        mock_client.upload_file.side_effect = Exception("S3 error")
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            # Should not raise
            s3_sync.upload_file(test_file)


class TestS3Push:
    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_push_uploads_all_files(self, tmp_path):
        # Create some files
        (tmp_path / "wiki").mkdir()
        (tmp_path / "wiki" / "article.md").write_text("content")
        (tmp_path / "research.md").write_text("research")

        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.push_to_s3(tmp_path)

        assert count == 2
        assert mock_client.upload_file.call_count == 2

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    def test_push_skips_pycache(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "module.pyc").write_text("bytecode")
        (tmp_path / "article.md").write_text("content")

        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.push_to_s3(tmp_path)

        assert count == 1  # only article.md, not pycache

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    def test_push_skips_tmp_files(self, tmp_path):
        (tmp_path / "article.md").write_text("content")
        (tmp_path / "scratch.tmp").write_text("temp data")

        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.push_to_s3(tmp_path)

        assert count == 1  # only article.md

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    def test_push_handles_upload_error(self, tmp_path):
        (tmp_path / "article.md").write_text("content")
        (tmp_path / "good.md").write_text("ok")

        mock_client = MagicMock()
        mock_client.upload_file.side_effect = [Exception("S3 error"), None]
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.push_to_s3(tmp_path)

        # Both attempted; the exception is caught inside upload_file so
        # push_to_s3 counts both (it tries upload_file which swallows the error)
        assert count == 2


class TestS3Delete:
    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_delete_handles_error(self):
        mock_client = MagicMock()
        mock_client.delete_object.side_effect = Exception("S3 error")
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            # Should not raise
            s3_sync.delete_file("/kb/personal/wiki/old.md")


class TestS3Pull:
    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_pull_downloads_missing_files(self, tmp_path):
        from datetime import datetime, timezone

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "personal/wiki/article.md",
                        "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    },
                ],
            },
        ]
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.pull_from_s3(tmp_path)

        assert count == 1
        mock_client.download_file.assert_called_once()

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_pull_skips_newer_local_files(self, tmp_path):
        from datetime import datetime, timezone

        # Create a local file that is newer than S3 version
        local_file = tmp_path / "personal" / "wiki" / "article.md"
        local_file.parent.mkdir(parents=True)
        local_file.write_text("local content")

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "personal/wiki/article.md",
                        "LastModified": datetime(2020, 1, 1, tzinfo=timezone.utc),
                    },
                ],
            },
        ]
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.pull_from_s3(tmp_path)

        assert count == 0
        mock_client.download_file.assert_not_called()

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "backups")
    def test_pull_with_prefix(self, tmp_path):
        from datetime import datetime, timezone

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "backups/personal/wiki/article.md",
                        "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    },
                ],
            },
        ]
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.pull_from_s3(tmp_path)

        assert count == 1

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_pull_handles_s3_error(self, tmp_path):
        mock_client = MagicMock()
        mock_client.get_paginator.side_effect = Exception("Connection failed")
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.pull_from_s3(tmp_path)
        assert count == 0

    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_pull_skips_empty_relative_key(self, tmp_path):
        from datetime import datetime, timezone

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "", "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc)}]},
        ]
        with patch.object(s3_sync, "_get_client", return_value=mock_client):
            count = s3_sync.pull_from_s3(tmp_path)
        assert count == 0


class TestSyncDb:
    @patch.object(s3_sync, "_enabled", True)
    @patch.object(s3_sync, "S3_BUCKET", "test-bucket")
    @patch.object(s3_sync, "S3_PREFIX", "")
    def test_sync_db_uploads_db_file(self, tmp_path):
        db_path = tmp_path / "wikidelve.db"
        db_path.write_text("sqlite data")

        mock_client = MagicMock()
        with patch.object(s3_sync, "_get_client", return_value=mock_client), \
             patch("app.config.DB_PATH", db_path):
            s3_sync.sync_db()
        mock_client.upload_file.assert_called_once()


class TestS3AsyncOps:
    @patch.object(s3_sync, "_enabled", False)
    def test_upload_async_noop_when_disabled(self):
        # Should not raise
        s3_sync.upload_file_async("/some/path")

    @patch.object(s3_sync, "_enabled", False)
    def test_delete_async_noop_when_disabled(self):
        s3_sync.delete_file_async("/some/path")
