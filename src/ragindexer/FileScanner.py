# -*- coding: utf-8 -*-
"""
File Scanner Component

Recursively scans a folder to identify all documents to be indexed.
Provides file metadata and change detection between scans.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
import hashlib
from enum import Enum
import logging

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class FileFormat(str, Enum):
    """Supported document formats"""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    MARKDOWN = "md"


class FileInfo(BaseModel):
    """
    Information about a file detected during scan.

    Attributes:
        relative_path: Path relative to scan root (e.g., "subdir/file.pdf")
        absolute_path: Full absolute path to the file
        format: Document format (pdf, docx, txt, md)
        file_size: File size in bytes
        modified_time: Last modification timestamp
        file_hash: SHA256 hash of file content (first 8KB for performance)
        detected_at: Timestamp when file was detected during scan
    """

    relative_path: str
    absolute_path: Path
    format: FileFormat
    file_size: int
    modified_time: datetime
    file_hash: str
    detected_at: datetime = Field(default_factory=datetime.now)

    @field_validator("file_size")
    @classmethod
    def validate_file_size(cls, v):
        if v < 0:
            raise ValueError("file_size must be non-negative")
        return v


class ChangeType(str, Enum):
    """Types of file changes detected"""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class FileChange(BaseModel):
    """
    Represents a detected change in a file between scans.

    Attributes:
        file_info: Information about the current file (None if deleted)
        change_type: Type of change detected
        previous_hash: Hash from previous scan (None if newly added)
        previous_modified_time: Modified time from previous scan (None if newly added)
    """

    file_info: Optional[FileInfo] = None
    change_type: ChangeType
    previous_hash: Optional[str] = None
    previous_modified_time: Optional[datetime] = None


class ScanResult(BaseModel):
    """
    Result of a folder scan operation.

    Attributes:
        scan_root: Root folder that was scanned
        files: Dictionary of detected files (key: relative_path)
        scan_time: When the scan was performed
        total_files: Total number of files found
        total_size_bytes: Total size of all files
    """

    scan_root: Path
    files: Dict[str, FileInfo] = Field(default_factory=dict)
    scan_time: datetime = Field(default_factory=datetime.now)

    @property
    def total_files(self) -> int:
        """Count of files detected"""
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        """Total size of all detected files in bytes"""
        return sum(f.file_size for f in self.files.values())


class FileScanner:
    """
    Recursively scans a folder to identify documents to index.

    Supports detection of:
    - New files
    - Modified files (via hash and timestamp)
    - Deleted files (when compared to previous scan)

    Only files with supported extensions (.pdf, .docx, .doc, .txt, .md) are included.
    """

    SUPPORTED_FORMATS = {fmt.value for fmt in FileFormat}
    HASH_CHUNK_SIZE = 8192  # Read first 8KB for hash (performance)

    def __init__(self, scan_root: Path | str, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the FileScanner.

        Args:
            scan_root: Root directory to scan
            logger_instance: Logger to use (defaults to module logger)

        Raises:
            ValueError: If scan_root does not exist or is not a directory
        """
        self.scan_root = Path(scan_root)
        self.logger = logger_instance or logger

        if not self.scan_root.exists():
            raise ValueError(f"Scan root does not exist: {self.scan_root}")
        if not self.scan_root.is_dir():
            raise ValueError(f"Scan root is not a directory: {self.scan_root}")

    def scan(self) -> ScanResult:
        """
        Recursively scan the root folder and return all detected files.

        Returns:
            ScanResult containing all detected files with metadata
        """
        result = ScanResult(scan_root=self.scan_root)

        self.logger.info(f"Starting file scan from: {self.scan_root}")

        for file_path in self.scan_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Check if file format is supported
            suffix = file_path.suffix.lstrip(".").lower()
            if suffix not in self.SUPPORTED_FORMATS:
                continue

            try:
                file_info = self._create_file_info(file_path)
                result.files[file_info.relative_path] = file_info
                self.logger.debug(f"Detected: {file_info.relative_path}")
            except Exception as e:
                self.logger.warning(f"Failed to scan file {file_path}: {e}")

        self.logger.info(
            f"Scan completed: {result.total_files} files, "
            f"{result.total_size_bytes / 1024 / 1024:.2f} MB"
        )

        return result

    def _create_file_info(self, file_path: Path) -> FileInfo:
        """
        Create FileInfo object for a given file.

        Args:
            file_path: Absolute path to the file

        Returns:
            FileInfo object with metadata
        """
        # Normalize path separators to forward slashes for cross-platform consistency
        relative_path = str(file_path.relative_to(self.scan_root)).replace("\\", "/")
        suffix = file_path.suffix.lstrip(".").lower()

        stat = file_path.stat()
        file_hash = self._compute_file_hash(file_path)

        return FileInfo(
            relative_path=relative_path,
            absolute_path=file_path,
            format=FileFormat(suffix),
            file_size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            file_hash=file_hash,
        )

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file (first HASH_CHUNK_SIZE bytes for performance).

        Args:
            file_path: Path to file

        Returns:
            Hex string of hash
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(self.HASH_CHUNK_SIZE)
                hasher.update(chunk)
        except Exception as e:
            self.logger.warning(f"Could not compute hash for {file_path}: {e}")

        return hasher.hexdigest()

    def detect_changes(
        self, previous_scan: ScanResult, current_scan: ScanResult
    ) -> Dict[str, FileChange]:
        """
        Compare two scan results to detect added, modified, and deleted files.

        Args:
            previous_scan: ScanResult from previous scan
            current_scan: ScanResult from current scan

        Returns:
            Dictionary of FileChange objects (key: relative_path)
        """
        changes = {}

        # Detect added and modified files
        for rel_path, current_file in current_scan.files.items():
            if rel_path not in previous_scan.files:
                changes[rel_path] = FileChange(
                    file_info=current_file,
                    change_type=ChangeType.ADDED,
                )
                self.logger.info(f"Detected new file: {rel_path}")
            else:
                previous_file = previous_scan.files[rel_path]

                if (
                    current_file.file_hash != previous_file.file_hash
                    or current_file.modified_time != previous_file.modified_time
                ):
                    changes[rel_path] = FileChange(
                        file_info=current_file,
                        change_type=ChangeType.MODIFIED,
                        previous_hash=previous_file.file_hash,
                        previous_modified_time=previous_file.modified_time,
                    )
                    self.logger.info(f"Detected modified file: {rel_path}")
                else:
                    changes[rel_path] = FileChange(
                        file_info=current_file,
                        change_type=ChangeType.UNCHANGED,
                    )

        # Detect deleted files
        for rel_path, previous_file in previous_scan.files.items():
            if rel_path not in current_scan.files:
                changes[rel_path] = FileChange(
                    change_type=ChangeType.DELETED,
                    previous_hash=previous_file.file_hash,
                )
                self.logger.info(f"Detected deleted file: {rel_path}")

        return changes

    def get_changed_files(
        self, previous_scan: ScanResult, current_scan: ScanResult
    ) -> Dict[str, FileChange]:
        """
        Get only files that have changed (exclude UNCHANGED).

        Args:
            previous_scan: ScanResult from previous scan
            current_scan: ScanResult from current scan

        Returns:
            Dictionary of FileChange objects for added/modified/deleted files
        """
        all_changes = self.detect_changes(previous_scan, current_scan)
        return {
            path: change
            for path, change in all_changes.items()
            if change.change_type != ChangeType.UNCHANGED
        }
