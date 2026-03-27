# -*- coding: utf-8 -*-
"""
Tests for FileScanner component
"""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import time

from ragindexer.FileScanner import (
    FileScanner,
    FileFormat,
    ChangeType,
    ScanResult,
    FileChange,
)


class TestFileScanner:
    """Test cases for FileScanner"""

    @pytest.fixture
    def temp_scan_dir(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files in subdirectories
            (tmppath / "subdir1").mkdir()
            (tmppath / "subdir1" / "file1.txt").write_text("Test content 1")
            (tmppath / "subdir1" / "file2.pdf").write_bytes(b"PDF content")

            (tmppath / "subdir2").mkdir()
            (tmppath / "subdir2" / "document.docx").write_bytes(b"DOCX content")
            (tmppath / "README.md").write_text("# Readme")

            # File that should be ignored
            (tmppath / "ignored.exe").write_bytes(b"Executable")

            yield tmppath

    def test_scan_detects_supported_files(self, temp_scan_dir):
        """Test that scan detects all supported file formats"""
        scanner = FileScanner(temp_scan_dir)
        result = scanner.scan()

        assert result.total_files == 4
        assert len(result.files) == 4

        # Check that all expected files are found
        rel_paths = set(result.files.keys())
        assert "subdir1/file1.txt" in rel_paths
        assert "subdir1/file2.pdf" in rel_paths
        assert "subdir2/document.docx" in rel_paths
        assert "README.md" in rel_paths

    def test_scan_ignores_unsupported_files(self, temp_scan_dir):
        """Test that scan ignores files with unsupported extensions"""
        scanner = FileScanner(temp_scan_dir)
        result = scanner.scan()

        # .exe file should be ignored
        assert "ignored.exe" not in result.files

    def test_file_info_metadata(self, temp_scan_dir):
        """Test that FileInfo contains correct metadata"""
        scanner = FileScanner(temp_scan_dir)
        result = scanner.scan()

        file_info = result.files["subdir1/file1.txt"]
        assert file_info.format == FileFormat.TXT
        assert file_info.file_size == len("Test content 1")
        assert file_info.relative_path == "subdir1/file1.txt"
        assert file_info.absolute_path == temp_scan_dir / "subdir1/file1.txt"
        assert isinstance(file_info.modified_time, datetime)
        assert isinstance(file_info.file_hash, str)
        assert len(file_info.file_hash) == 64  # SHA256 hex string

    def test_file_hash_consistency(self, temp_scan_dir):
        """Test that hash is consistent for same file"""
        scanner = FileScanner(temp_scan_dir)
        result1 = scanner.scan()
        result2 = scanner.scan()

        file_hash1 = result1.files["subdir1/file1.txt"].file_hash
        file_hash2 = result2.files["subdir1/file1.txt"].file_hash

        assert file_hash1 == file_hash2

    def test_detect_changes_added_files(self, temp_scan_dir):
        """Test detection of newly added files"""
        scanner = FileScanner(temp_scan_dir)

        # Initial scan
        scan1 = scanner.scan()

        # Add a new file
        time.sleep(0.01)  # Ensure different timestamp
        (temp_scan_dir / "newfile.md").write_text("New content")

        # Second scan
        scan2 = scanner.scan()

        # Detect changes
        changes = scanner.detect_changes(scan1, scan2)

        assert "newfile.md" in changes
        assert changes["newfile.md"].change_type == ChangeType.ADDED
        assert changes["newfile.md"].file_info is not None

    def test_detect_changes_modified_files(self, temp_scan_dir):
        """Test detection of modified files"""
        scanner = FileScanner(temp_scan_dir)

        # Initial scan
        scan1 = scanner.scan()
        original_hash = scan1.files["subdir1/file1.txt"].file_hash

        # Modify file
        time.sleep(0.01)
        (temp_scan_dir / "subdir1" / "file1.txt").write_text("Modified content!")

        # Second scan
        scan2 = scanner.scan()

        # Detect changes
        changes = scanner.detect_changes(scan1, scan2)

        assert "subdir1/file1.txt" in changes
        assert changes["subdir1/file1.txt"].change_type == ChangeType.MODIFIED
        assert changes["subdir1/file1.txt"].previous_hash == original_hash
        assert changes["subdir1/file1.txt"].file_info.file_hash != original_hash

    def test_detect_changes_deleted_files(self, temp_scan_dir):
        """Test detection of deleted files"""
        scanner = FileScanner(temp_scan_dir)

        # Initial scan
        scan1 = scanner.scan()

        # Delete file
        (temp_scan_dir / "subdir1" / "file1.txt").unlink()

        # Second scan
        scan2 = scanner.scan()

        # Detect changes
        changes = scanner.detect_changes(scan1, scan2)

        assert "subdir1/file1.txt" in changes
        assert changes["subdir1/file1.txt"].change_type == ChangeType.DELETED
        assert changes["subdir1/file1.txt"].file_info is None

    def test_get_changed_files_excludes_unchanged(self, temp_scan_dir):
        """Test that get_changed_files excludes unchanged files"""
        scanner = FileScanner(temp_scan_dir)

        scan1 = scanner.scan()
        scan2 = scanner.scan()

        # All files are the same, so should have no changes
        changed = scanner.get_changed_files(scan1, scan2)

        assert len(changed) == 0

    def test_total_size_calculation(self, temp_scan_dir):
        """Test that total_size_bytes is calculated correctly"""
        scanner = FileScanner(temp_scan_dir)
        result = scanner.scan()

        expected_size = (
            len("Test content 1") + len(b"PDF content") + len(b"DOCX content") + len("# Readme")
        )

        assert result.total_size_bytes == expected_size

    def test_invalid_scan_root(self):
        """Test that invalid scan root raises error"""
        with pytest.raises(ValueError, match="does not exist"):
            FileScanner("/nonexistent/path")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "file.txt"
            file_path.write_text("test")

            with pytest.raises(ValueError, match="not a directory"):
                FileScanner(file_path)

    def test_empty_directory(self):
        """Test scanning empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = FileScanner(tmpdir)
            result = scanner.scan()

            assert result.total_files == 0
            assert result.total_size_bytes == 0
            assert len(result.files) == 0

    def test_directory_with_only_unsupported_files(self):
        """Test directory containing only unsupported files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir).joinpath("file.exe").write_bytes(b"content")
            Path(tmpdir).joinpath("file.bin").write_bytes(b"content")

            scanner = FileScanner(tmpdir)
            result = scanner.scan()

            assert result.total_files == 0
