# -*- coding: utf-8 -*-
"""
Sync Manager Component

Orchestrates the complete indexing pipeline and manages file change synchronization.
Coordinates FileScanner, DocumentParser, ChunkingService, EmbeddingService, and VectorDatabaseService.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from ragindexer.FileScanner import FileScanner, ScanResult, FileChange, ChangeType
from ragindexer.DocumentParser import DocumentParser
from ragindexer.ChunkingService import ChunkingService
from ragindexer.EmbeddingService import EmbeddingService
from ragindexer.VectorDatabaseService import VectorDatabaseService

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Status of a sync operation"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some files succeeded, some failed


class FileSyncResult(BaseModel):
    """
    Result of syncing a single file.

    Attributes:
        relative_path: Path of the synced file
        status: Success/failure status
        change_type: Type of change (added, modified, deleted)
        chunks_count: Number of chunks created (for new/modified)
        error: Error message if failed
        duration_seconds: Time taken for sync
        synced_at: Timestamp of sync
    """

    relative_path: str
    status: SyncStatus
    change_type: ChangeType
    chunks_count: int = 0
    error: Optional[str] = None
    duration_seconds: float = 0.0
    synced_at: datetime = Field(default_factory=datetime.now)


class SyncOperationResult(BaseModel):
    """
    Result of a complete sync operation across multiple files.

    Attributes:
        scan_root: Root folder scanned
        total_files_processed: Number of files processed
        total_files_added: Number of files added to index
        total_files_modified: Number of files modified
        total_files_deleted: Number of files deleted
        total_chunks_created: Total chunks created/updated
        total_errors: Number of failed files
        file_results: Detailed results per file
        overall_status: Overall sync status
        duration_seconds: Total time taken
        synced_at: Timestamp of sync
    """

    scan_root: Path
    total_files_processed: int = 0
    total_files_added: int = 0
    total_files_modified: int = 0
    total_files_deleted: int = 0
    total_chunks_created: int = 0
    total_errors: int = 0
    file_results: Dict[str, FileSyncResult] = Field(default_factory=dict)
    overall_status: SyncStatus
    duration_seconds: float = 0.0
    synced_at: datetime = Field(default_factory=datetime.now)


class SyncManager:
    """
    Orchestrates the complete indexing pipeline and synchronizes file changes.

    Coordinates all components:
    - FileScanner: Detects file changes
    - DocumentParser: Extracts text content
    - ChunkingService: Splits text into chunks
    - EmbeddingService: Generates vector embeddings
    - VectorDatabaseService: Stores and indexes embeddings

    Manages:
    - Full initial indexing
    - Incremental updates on file changes
    - Deletion of embeddings for removed files
    - Error tracking and logging
    """

    def __init__(
        self,
        scan_root: Path | str,
        persistence_path: Optional[Path] = None,
        chunk_size: int = 512,
        overlap_size: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        logger_instance: Optional[logging.Logger] = None,
    ):
        """
        Initialize the SyncManager.

        Args:
            scan_root: Root directory to scan for documents
            persistence_path: Path for vector database persistence (None = in-memory)
            chunk_size: Target chunk size in tokens (default 512)
            overlap_size: Overlap size between chunks in tokens (default 50)
            embedding_model: HuggingFace model ID for embeddings
            logger_instance: Logger to use (defaults to module logger)

        Raises:
            ValueError: If scan_root does not exist or is not a directory
        """
        self.scan_root = Path(scan_root)
        self.logger = logger_instance or logger

        # Initialize all components
        self.file_scanner = FileScanner(self.scan_root, logger_instance=self.logger)
        self.document_parser = DocumentParser(logger_instance=self.logger)
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            logger_instance=self.logger,
        )
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            logger_instance=self.logger,
        )
        self.vector_db = VectorDatabaseService(
            vector_size=self.embedding_service.embedding_dim,
            persistence_path=persistence_path,
            logger_instance=self.logger,
        )

        # Track the last scan result for change detection
        self.last_scan_result: Optional[ScanResult] = None

        self.logger.info(
            f"SyncManager initialized: scan_root={self.scan_root}, "
            f"embedding_model={embedding_model}"
        )

    def full_sync(self) -> SyncOperationResult:
        """
        Perform a complete initial indexing of all documents in the scan root.

        This will:
        1. Scan the folder for all documents
        2. Parse each document
        3. Chunk the text
        4. Generate embeddings
        5. Store in vector database

        Returns:
            SyncOperationResult with detailed status per file

        Note:
            Use incremental_sync() for subsequent runs to only process changes.
        """
        self.logger.info(f"Starting full sync of {self.scan_root}")
        start_time = datetime.now()

        # Scan for all files
        current_scan = self.file_scanner.scan()

        # Process all files as "added"
        result = SyncOperationResult(
            scan_root=self.scan_root,
            overall_status=SyncStatus.IN_PROGRESS,
        )

        for rel_path, file_info in current_scan.files.items():
            file_start_time = datetime.now()
            try:
                chunks_count = self._process_file_for_indexing(file_info)

                file_result = FileSyncResult(
                    relative_path=rel_path,
                    status=SyncStatus.COMPLETED,
                    change_type=ChangeType.ADDED,
                    chunks_count=chunks_count,
                    duration_seconds=(datetime.now() - file_start_time).total_seconds(),
                )
                result.file_results[rel_path] = file_result
                result.total_files_added += 1
                result.total_chunks_created += chunks_count

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Failed to process {rel_path}: {error_msg}")

                file_result = FileSyncResult(
                    relative_path=rel_path,
                    status=SyncStatus.FAILED,
                    change_type=ChangeType.ADDED,
                    error=error_msg,
                    duration_seconds=(datetime.now() - file_start_time).total_seconds(),
                )
                result.file_results[rel_path] = file_result
                result.total_errors += 1

        # Update tracking
        self.last_scan_result = current_scan
        result.total_files_processed = len(current_scan.files)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Determine overall status
        if result.total_errors == 0:
            result.overall_status = SyncStatus.COMPLETED
        elif result.total_errors < result.total_files_processed:
            result.overall_status = SyncStatus.PARTIAL
        else:
            result.overall_status = SyncStatus.FAILED

        self.logger.info(
            f"Full sync completed: {result.total_files_added} files indexed, "
            f"{result.total_errors} errors, {result.duration_seconds:.2f}s"
        )

        return result

    def incremental_sync(self) -> SyncOperationResult:
        """
        Perform an incremental sync, processing only changed files.

        This will:
        1. Scan the folder for current files
        2. Compare with last scan
        3. Process added/modified files through the pipeline
        4. Delete embeddings for removed files

        Returns:
            SyncOperationResult with detailed status per file

        Raises:
            ValueError: If last_scan_result is not set (must call full_sync first)
        """
        if self.last_scan_result is None:
            raise ValueError("No previous scan result. Call full_sync() before incremental_sync()")

        self.logger.info(f"Starting incremental sync of {self.scan_root}")
        start_time = datetime.now()

        # Scan for current files
        current_scan = self.file_scanner.scan()

        # Detect changes
        changes = self.file_scanner.get_changed_files(self.last_scan_result, current_scan)

        result = SyncOperationResult(
            scan_root=self.scan_root,
            overall_status=SyncStatus.IN_PROGRESS,
        )

        # Process changes
        for rel_path, change in changes.items():
            file_start_time = datetime.now()

            if change.change_type == ChangeType.DELETED:
                try:
                    self._process_file_deletion(rel_path)

                    file_result = FileSyncResult(
                        relative_path=rel_path,
                        status=SyncStatus.COMPLETED,
                        change_type=ChangeType.DELETED,
                        duration_seconds=(datetime.now() - file_start_time).total_seconds(),
                    )
                    result.file_results[rel_path] = file_result
                    result.total_files_deleted += 1

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Failed to delete embeddings for {rel_path}: {error_msg}")

                    file_result = FileSyncResult(
                        relative_path=rel_path,
                        status=SyncStatus.FAILED,
                        change_type=ChangeType.DELETED,
                        error=error_msg,
                        duration_seconds=(datetime.now() - file_start_time).total_seconds(),
                    )
                    result.file_results[rel_path] = file_result
                    result.total_errors += 1

            elif change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
                try:
                    # For modified files, delete old embeddings first
                    if change.change_type == ChangeType.MODIFIED:
                        self._process_file_deletion(rel_path)

                    chunks_count = self._process_file_for_indexing(change.file_info)

                    file_result = FileSyncResult(
                        relative_path=rel_path,
                        status=SyncStatus.COMPLETED,
                        change_type=change.change_type,
                        chunks_count=chunks_count,
                        duration_seconds=(datetime.now() - file_start_time).total_seconds(),
                    )
                    result.file_results[rel_path] = file_result

                    if change.change_type == ChangeType.ADDED:
                        result.total_files_added += 1
                    else:
                        result.total_files_modified += 1

                    result.total_chunks_created += chunks_count

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(f"Failed to process {rel_path}: {error_msg}")

                    file_result = FileSyncResult(
                        relative_path=rel_path,
                        status=SyncStatus.FAILED,
                        change_type=change.change_type,
                        error=error_msg,
                        duration_seconds=(datetime.now() - file_start_time).total_seconds(),
                    )
                    result.file_results[rel_path] = file_result
                    result.total_errors += 1

        # Update tracking
        self.last_scan_result = current_scan
        result.total_files_processed = len(changes)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Determine overall status
        if result.total_errors == 0:
            result.overall_status = SyncStatus.COMPLETED
        elif result.total_errors < result.total_files_processed:
            result.overall_status = SyncStatus.PARTIAL
        else:
            result.overall_status = SyncStatus.FAILED

        self.logger.info(
            f"Incremental sync completed: {result.total_files_added} added, "
            f"{result.total_files_modified} modified, {result.total_files_deleted} deleted, "
            f"{result.total_errors} errors, {result.duration_seconds:.2f}s"
        )

        return result

    def _process_file_for_indexing(self, file_info) -> int:
        """
        Process a single file through the complete pipeline.

        Pipeline:
        1. Parse document (extract text)
        2. Chunk text into semantic units
        3. Generate embeddings
        4. Store in vector database

        Args:
            file_info: FileInfo object from FileScanner

        Returns:
            Number of chunks created for the file

        Raises:
            Exception: Any errors during processing
        """
        # Step 1: Parse
        parsed_doc = self.document_parser.parse(file_info)

        # Step 2: Chunk
        chunking_result = self.chunking_service.chunk(parsed_doc)

        # Step 3: Embed
        embedding_result = self.embedding_service.embed_chunks(chunking_result.chunks)

        # Step 4: Store
        self.vector_db.add_embeddings(embedding_result.embedded_chunks)

        return len(chunking_result.chunks)

    def _process_file_deletion(self, relative_path: str) -> None:
        """
        Delete all embeddings associated with a deleted file.

        Args:
            relative_path: Relative path of the deleted file

        Raises:
            Exception: If deletion fails
        """
        self.vector_db.delete_document(relative_path)

    def get_last_scan_result(self) -> Optional[ScanResult]:
        """
        Get the most recent scan result.

        Returns:
            Last ScanResult if available, None otherwise
        """
        return self.last_scan_result

    def get_statistics(self) -> Dict[str, object]:
        """
        Get current database statistics.

        Returns:
            Dictionary with database stats (number of embeddings, etc.)
        """
        return self.vector_db.get_statistics()
