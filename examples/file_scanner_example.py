# -*- coding: utf-8 -*-
"""
Example usage of the FileScanner component

This example demonstrates:
1. Basic file scanning
2. Change detection between scans
3. Working with the metadata
"""

import logging
from pathlib import Path
from datetime import datetime

from ragindexer.FileScanner import FileScanner, ChangeType


def setup_logging():
    """Configure logging for the example"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def example_basic_scan():
    """Example 1: Basic folder scan"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Folder Scan")
    print("=" * 60)

    # Create scanner for a directory
    scan_root = Path("./documents")  # Your documents folder
    scanner = FileScanner(scan_root)

    # Perform initial scan
    result = scanner.scan()

    print(f"\nScanned folder: {result.scan_root}")
    print(f"Total files found: {result.total_files}")
    print(f"Total size: {result.total_size_bytes / 1024 / 1024:.2f} MB")
    print(f"Scan time: {result.scan_time}")

    # List all detected files
    print("\nDetected files:")
    for rel_path, file_info in result.files.items():
        size_kb = file_info.file_size / 1024
        print(f"  - {rel_path:50} " f"({size_kb:8.1f} KB) " f"[{file_info.format.value.upper()}]")


def example_change_detection():
    """Example 2: Detect changes between scans"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Change Detection")
    print("=" * 60)

    scan_root = Path("./documents")
    scanner = FileScanner(scan_root)

    # Perform initial scan
    print("\n[Step 1] Performing initial scan...")
    scan1 = scanner.scan()
    print(f"Initial scan: {scan1.total_files} files")

    # Simulate some time passing and documents being modified
    print("\n[Step 2] Waiting... (in real scenario, user adds/modifies/deletes files)")
    print("(You would modify files here in your actual workflow)")

    # Perform second scan
    print("\n[Step 3] Performing second scan...")
    scan2 = scanner.scan()
    print(f"Second scan: {scan2.total_files} files")

    # Detect changes
    print("\n[Step 4] Detecting changes...")
    changes = scanner.detect_changes(scan1, scan2)

    # Group changes by type
    added = {p: c for p, c in changes.items() if c.change_type == ChangeType.ADDED}
    modified = {p: c for p, c in changes.items() if c.change_type == ChangeType.MODIFIED}
    deleted = {p: c for p, c in changes.items() if c.change_type == ChangeType.DELETED}
    unchanged = {p: c for p, c in changes.items() if c.change_type == ChangeType.UNCHANGED}

    print(f"\nChange summary:")
    print(f"  Added:     {len(added)} file(s)")
    print(f"  Modified:  {len(modified)} file(s)")
    print(f"  Deleted:   {len(deleted)} file(s)")
    print(f"  Unchanged: {len(unchanged)} file(s)")

    # Show details of changes
    if added:
        print("\nAdded files:")
        for path in added:
            print(f"  + {path}")

    if modified:
        print("\nModified files:")
        for path, change in modified.items():
            print(f"  ~ {path}")
            print(f"      Previous hash: {change.previous_hash[:16]}...")
            print(f"      New hash:      {change.file_info.file_hash[:16]}...")

    if deleted:
        print("\nDeleted files:")
        for path in deleted:
            print(f"  - {path}")


def example_metadata_inspection():
    """Example 3: Inspect file metadata in detail"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Detailed Metadata Inspection")
    print("=" * 60)

    scan_root = Path("./documents")
    scanner = FileScanner(scan_root)
    result = scanner.scan()

    if result.total_files == 0:
        print("No files found in scan root")
        return

    # Show detailed info for each file
    print(f"\nDetailed metadata for {result.total_files} file(s):\n")

    for i, (rel_path, file_info) in enumerate(result.files.items(), 1):
        print(f"{i}. {rel_path}")
        print(f"   Format:        {file_info.format.value}")
        print(
            f"   Size:          {file_info.file_size:,} bytes ({file_info.file_size / 1024:.1f} KB)"
        )
        print(f"   Modified:      {file_info.modified_time}")
        print(f"   SHA256 Hash:   {file_info.file_hash}")
        print(f"   Detected at:   {file_info.detected_at}")
        print()


def example_incremental_update():
    """Example 4: Incremental update workflow (only re-index changed files)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Incremental Update Workflow")
    print("=" * 60)

    scan_root = Path("./documents")
    scanner = FileScanner(scan_root)

    # Initial scan
    print("\n[Phase 1] Initial indexing...")
    scan1 = scanner.scan()
    print(f"Initial index created for {scan1.total_files} files")

    # In a real scenario, files would be modified here
    print("\n[Phase 2] Waiting for file changes...")
    print("(In production, you might wait for filesystem events)")

    # Second scan
    print("\n[Phase 3] Detecting changes for re-indexing...")
    scan2 = scanner.scan()

    # Get only changed files
    changed = scanner.get_changed_files(scan1, scan2)

    if not changed:
        print("No changes detected - skipping re-indexing")
    else:
        print(f"Found {len(changed)} file(s) to process:")

        # Show which files need to be re-indexed
        to_reindex = {
            p: c
            for p, c in changed.items()
            if c.change_type in (ChangeType.ADDED, ChangeType.MODIFIED)
        }
        to_delete = {p: c for p, c in changed.items() if c.change_type == ChangeType.DELETED}

        if to_reindex:
            print(f"\n  Re-index {len(to_reindex)} file(s):")
            for path in to_reindex:
                print(f"    → {path}")

        if to_delete:
            print(f"\n  Remove {len(to_delete)} file(s) from index:")
            for path in to_delete:
                print(f"    ✗ {path}")


if __name__ == "__main__":
    setup_logging()

    # Run examples
    try:
        example_basic_scan()
        example_change_detection()
        example_metadata_inspection()
        example_incremental_update()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("To run these examples, create a './documents' folder with some files")
        print("Supported formats: .pdf, .docx, .doc, .txt, .md")
