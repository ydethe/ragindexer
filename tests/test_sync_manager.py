# -*- coding: utf-8 -*-
"""
Tests for SyncManager component
"""

import pytest
from pathlib import Path
import tempfile
import time

from ragindexer.SyncManager import (
    SyncManager,
    SyncStatus,
    FileSyncResult,
    SyncOperationResult,
)
from ragindexer.FileScanner import ChangeType


class TestSyncManager:
    """Test cases for SyncManager"""

    @pytest.fixture
    def temp_docs_dir(self):
        """Create a temporary directory with test documents"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create subdirectory
            (tmppath / "docs").mkdir()

            # Create test files
            (tmppath / "docs" / "test1.txt").write_text(
                "This is test document 1.\nIt has multiple lines.\nFor testing purposes."
            )
            (tmppath / "docs" / "test2.md").write_text(
                "# Test Document 2\n\nThis is a markdown document.\n\nWith multiple paragraphs."
            )

            yield tmppath

    @pytest.fixture
    def sync_manager(self, temp_docs_dir):
        """Create a SyncManager instance with in-memory database"""
        return SyncManager(
            scan_root=temp_docs_dir / "docs",
            persistence_path=None,  # In-memory for tests
            chunk_size=256,
            overlap_size=25,
        )

    def test_sync_manager_initialization(self, sync_manager):
        """Test SyncManager initialization"""
        assert sync_manager.scan_root.exists()
        assert sync_manager.file_scanner is not None
        assert sync_manager.document_parser is not None
        assert sync_manager.chunking_service is not None
        assert sync_manager.embedding_service is not None
        assert sync_manager.vector_db is not None
        assert sync_manager.last_scan_result is None

    def test_full_sync_creates_embeddings(self, sync_manager):
        """Test that full_sync processes all files and creates embeddings"""
        result = sync_manager.full_sync()

        assert result.overall_status == SyncStatus.COMPLETED
        assert result.total_files_added == 2
        assert result.total_files_modified == 0
        assert result.total_files_deleted == 0
        assert result.total_errors == 0
        assert result.total_chunks_created > 0
        assert len(result.file_results) == 2

        # Check file results
        for rel_path, file_result in result.file_results.items():
            assert file_result.status == SyncStatus.COMPLETED
            assert file_result.change_type == ChangeType.ADDED
            assert file_result.chunks_count > 0

    def test_full_sync_updates_last_scan_result(self, sync_manager):
        """Test that full_sync updates the last scan result"""
        assert sync_manager.last_scan_result is None

        sync_manager.full_sync()

        assert sync_manager.last_scan_result is not None
        assert len(sync_manager.last_scan_result.files) == 2

    def test_incremental_sync_without_changes(self, sync_manager):
        """Test incremental_sync when no files have changed"""
        # Do initial full sync
        sync_manager.full_sync()

        # Run incremental sync without changes
        result = sync_manager.incremental_sync()

        assert result.overall_status == SyncStatus.COMPLETED
        assert result.total_files_processed == 0
        assert result.total_files_added == 0
        assert result.total_files_modified == 0
        assert result.total_files_deleted == 0
        assert result.total_errors == 0

    def test_incremental_sync_with_new_file(self, sync_manager, temp_docs_dir):
        """Test incremental_sync detects and processes new files"""
        # Do initial full sync
        sync_manager.full_sync()

        # Add a new file
        time.sleep(0.01)  # Ensure different timestamp
        (temp_docs_dir / "docs" / "test3.txt").write_text("New test document.\nAdded later.")

        # Run incremental sync
        result = sync_manager.incremental_sync()

        assert result.overall_status == SyncStatus.COMPLETED
        assert result.total_files_processed == 1
        assert result.total_files_added == 1
        assert result.total_files_modified == 0
        assert result.total_files_deleted == 0
        assert result.total_errors == 0

    def test_incremental_sync_with_modified_file(self, sync_manager, temp_docs_dir):
        """Test incremental_sync detects and processes modified files"""
        # Do initial full sync
        sync_manager.full_sync()

        # Modify a file
        time.sleep(0.01)  # Ensure different timestamp
        (temp_docs_dir / "docs" / "test1.txt").write_text(
            "Modified content.\nCompletely different.\nWith new text."
        )

        # Run incremental sync
        result = sync_manager.incremental_sync()

        assert result.overall_status == SyncStatus.COMPLETED
        assert result.total_files_processed == 1
        assert result.total_files_added == 0
        assert result.total_files_modified == 1
        assert result.total_files_deleted == 0
        assert result.total_errors == 0

    def test_incremental_sync_with_deleted_file(self, sync_manager, temp_docs_dir):
        """Test incremental_sync detects and processes deleted files"""
        # Do initial full sync
        sync_manager.full_sync()

        # Delete a file
        (temp_docs_dir / "docs" / "test1.txt").unlink()

        # Run incremental sync
        result = sync_manager.incremental_sync()

        assert result.overall_status == SyncStatus.COMPLETED
        assert result.total_files_processed == 1
        assert result.total_files_added == 0
        assert result.total_files_modified == 0
        assert result.total_files_deleted == 1
        assert result.total_errors == 0

    def test_incremental_sync_before_full_sync_raises_error(self, sync_manager):
        """Test that incremental_sync raises error if full_sync not done first"""
        with pytest.raises(ValueError, match="No previous scan result"):
            sync_manager.incremental_sync()

    def test_get_last_scan_result(self, sync_manager):
        """Test get_last_scan_result method"""
        assert sync_manager.get_last_scan_result() is None

        sync_manager.full_sync()

        scan_result = sync_manager.get_last_scan_result()
        assert scan_result is not None
        assert len(scan_result.files) == 2

    def test_get_statistics(self, sync_manager):
        """Test get_statistics method"""
        sync_manager.full_sync()

        stats = sync_manager.get_statistics()
        assert "collection_name" in stats
        assert "point_count" in stats
        assert stats["point_count"] > 0

    def test_full_sync_with_empty_directory(self):
        """Test full_sync with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "empty_docs").mkdir()

            manager = SyncManager(
                scan_root=tmppath / "empty_docs",
                persistence_path=None,
            )

            result = manager.full_sync()

            assert result.overall_status == SyncStatus.COMPLETED
            assert result.total_files_processed == 0
            assert result.total_files_added == 0
            assert result.total_chunks_created == 0

    def test_sync_result_contains_file_details(self, sync_manager):
        """Test that sync results contain detailed per-file information"""
        result = sync_manager.full_sync()

        assert len(result.file_results) == 2

        for rel_path, file_result in result.file_results.items():
            assert isinstance(file_result, FileSyncResult)
            assert file_result.relative_path == rel_path
            assert file_result.status == SyncStatus.COMPLETED
            assert file_result.chunks_count > 0
            assert file_result.duration_seconds > 0
            assert file_result.synced_at is not None

    def test_multiple_sync_cycles(self, sync_manager, temp_docs_dir):
        """Test multiple sync cycles with various changes"""
        # First full sync
        result1 = sync_manager.full_sync()
        assert result1.total_files_added == 2

        # Add file
        time.sleep(0.01)
        (temp_docs_dir / "docs" / "test3.txt").write_text("Third document.")

        # Incremental sync
        result2 = sync_manager.incremental_sync()
        assert result2.total_files_added == 1
        assert result2.total_files_processed == 1

        # Modify file
        time.sleep(0.01)
        (temp_docs_dir / "docs" / "test1.txt").write_text("Modified again.")

        # Incremental sync
        result3 = sync_manager.incremental_sync()
        assert result3.total_files_modified == 1
        assert result3.total_files_processed == 1

        # Delete file
        (temp_docs_dir / "docs" / "test2.md").unlink()

        # Incremental sync
        result4 = sync_manager.incremental_sync()
        assert result4.total_files_deleted == 1
        assert result4.total_files_processed == 1

    def test_sync_preserves_metadata(self, sync_manager):
        """Test that sync preserves document metadata through the pipeline"""
        sync_manager.full_sync()

        stats = sync_manager.get_statistics()
        # Verify embeddings were created (point_count > 0)
        assert stats["point_count"] > 0
