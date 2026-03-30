# FileScanner Component Documentation

## Overview

The **FileScanner** component is responsible for recursively scanning folders to identify and catalog documents for indexing. It provides:

- **Recursive folder scanning** with support for multiple document formats
- **File metadata extraction** (size, timestamps, hash)
- **Change detection** between scans (added, modified, deleted files)
- **Cross-platform path handling** (Windows, Linux, macOS)
- **Performance optimization** with partial file hashing

## Supported Formats

The FileScanner supports the following document formats:

- `.pdf` - PDF documents
- `.docx` - Microsoft Word (OOXML format)
- `.doc` - Microsoft Word (legacy format)
- `.txt` - Plain text files
- `.md` - Markdown files

## Architecture

### Data Models (Pydantic)

#### FileFormat
An enum representing supported document formats.

```python
class FileFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    MARKDOWN = "md"
```

#### FileInfo
Represents metadata about a single detected file.

**Fields:**
- `relative_path` (str): Path relative to scan root (e.g., `"subdir/file.pdf"`)
- `absolute_path` (Path): Full absolute path to the file
- `format` (FileFormat): Document format
- `file_size` (int): File size in bytes
- `modified_time` (datetime): Last modification timestamp
- `file_hash` (str): SHA256 hash of file content (first 8KB for performance)
- `detected_at` (datetime): When the file was detected

#### ChangeType
Enum representing types of detected changes.

```python
class ChangeType(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
```

#### FileChange
Represents a detected change between two scans.

**Fields:**
- `file_info` (Optional[FileInfo]): Current file info (None if deleted)
- `change_type` (ChangeType): Type of change
- `previous_hash` (Optional[str]): Hash from previous scan
- `previous_modified_time` (Optional[datetime]): Modified time from previous scan

#### ScanResult
Complete result of a folder scan operation.

**Fields:**
- `scan_root` (Path): Root folder that was scanned
- `files` (Dict[str, FileInfo]): Dictionary of detected files
- `scan_time` (datetime): When the scan was performed
- `total_files` (property): Count of detected files
- `total_size_bytes` (property): Total size in bytes

## Usage

### Basic Scanning

```python
from pathlib import Path
from ragindexer.FileScanner import FileScanner

# Create a scanner
scan_root = Path("./documents")
scanner = FileScanner(scan_root)

# Perform a scan
result = scanner.scan()

# Access results
print(f"Found {result.total_files} files")
print(f"Total size: {result.total_size_bytes / 1024 / 1024:.2f} MB")

# Iterate through detected files
for rel_path, file_info in result.files.items():
    print(f"{rel_path}: {file_info.format.value}")
```

### Change Detection

```python
# Perform initial scan
scan1 = scanner.scan()

# ... later, after files have been modified ...

# Perform second scan
scan2 = scanner.scan()

# Detect changes
changes = scanner.detect_changes(scan1, scan2)

# Iterate through changes
for rel_path, change in changes.items():
    if change.change_type == ChangeType.ADDED:
        print(f"New file: {rel_path}")
    elif change.change_type == ChangeType.MODIFIED:
        print(f"Modified file: {rel_path}")
    elif change.change_type == ChangeType.DELETED:
        print(f"Deleted file: {rel_path}")
```

### Incremental Updates (Only Changed Files)

```python
# Get only changed files (excluding UNCHANGED)
changed_files = scanner.get_changed_files(scan1, scan2)

# Separate by change type
added = {p: c for p, c in changed_files.items()
         if c.change_type == ChangeType.ADDED}
modified = {p: c for p, c in changed_files.items()
            if c.change_type == ChangeType.MODIFIED}
deleted = {p: c for p, c in changed_files.items()
           if c.change_type == ChangeType.DELETED}

# Re-index only changed files
for path in list(added.keys()) + list(modified.keys()):
    print(f"Re-indexing: {path}")

# Remove deleted files from index
for path in deleted:
    print(f"Removing: {path}")
```

## Performance Considerations

### File Hash Calculation

The FileScanner uses SHA256 hashing for change detection. For performance reasons, it only reads the **first 8KB** of each file:

```python
HASH_CHUNK_SIZE = 8192  # 8KB
```

This is sufficient for detecting file modifications and provides a good balance between:
- **Detection accuracy**: Most real file modifications will affect the first 8KB
- **Performance**: Hash computation is very fast, even for large files
- **Memory usage**: Only 8KB of data is loaded into memory per file

### Scan Optimization

- Only files with supported extensions are processed
- Directories are not hashed (only files)
- Recursive scanning uses Python's `Path.rglob()` for efficiency

## Error Handling

### Invalid Scan Root

```python
from ragindexer.FileScanner import FileScanner

try:
    scanner = FileScanner("/nonexistent/path")
except ValueError as e:
    print(f"Error: {e}")  # "Scan root does not exist..."
```

### File Read Errors

If a file cannot be read during scanning:
- A warning is logged
- The file is skipped
- Scanning continues with other files

### Hash Computation Errors

If a file's hash cannot be computed:
- A warning is logged
- The scan continues (file is still added to results with hash = empty string)

## Logging

The FileScanner uses Python's standard logging module:

```python
import logging

logger = logging.getLogger("ragindexer.FileScanner")
logger.setLevel(logging.DEBUG)

# Configure handler
handler = logging.StreamHandler()
logger.addHandler(handler)
```

## Cross-Platform Compatibility

The FileScanner automatically normalizes path separators to forward slashes (`/`) for consistent behavior across Windows, Linux, and macOS:

```python
# On Windows: "subdir\file.txt" → "subdir/file.txt"
# On Linux:   "subdir/file.txt" → "subdir/file.txt"
# On macOS:   "subdir/file.txt" → "subdir/file.txt"
```

## Integration with Other Components

The FileScanner outputs are consumed by:

1. **Document Parser** - Receives list of files to parse
2. **Sync Manager** - Uses change detection results to determine what to reindex
3. **Vector Database** - Uses relative_path as unique identifier for documents

## Example Workflow

```python
from pathlib import Path
from ragindexer.FileScanner import FileScanner, ChangeType

# 1. Initialize scanner
scanner = FileScanner(Path("./documents"))

# 2. Initial indexing
print("Starting initial index...")
scan1 = scanner.scan()

for rel_path, file_info in scan1.files.items():
    # Send to Document Parser
    parse_document(file_info)

# 3. Later, periodically check for changes
while True:
    # Wait for some time or filesystem events
    time.sleep(60)

    # Scan again
    scan2 = scanner.scan()

    # Detect changes
    changes = scanner.get_changed_files(scan1, scan2)

    if changes:
        print(f"Found {len(changes)} changed file(s)")

        for rel_path, change in changes.items():
            if change.change_type == ChangeType.DELETED:
                remove_from_index(rel_path)
            else:  # ADDED or MODIFIED
                parse_document(change.file_info)

    # Update baseline
    scan1 = scan2
```

## Testing

The FileScanner includes comprehensive tests:

```bash
# Run all FileScanner tests
python -m pytest tests/test_file_scanner.py -v

# Run with coverage
python -m pytest tests/test_file_scanner.py --cov=ragindexer.FileScanner

# Run specific test
python -m pytest tests/test_file_scanner.py::TestFileScanner::test_scan_detects_supported_files -v
```

Current test coverage: **95%**

## Troubleshooting

### "Scan root does not exist" error
- Ensure the path exists and is a valid directory
- Check file permissions

### "No files found"
- Verify files have supported extensions (.pdf, .docx, .doc, .txt, .md)
- Ensure files are readable (check permissions)

### Hash computation warnings
- Files may be locked by another process
- Check file permissions
- Try scanning again later

## Future Enhancements

- Real-time folder monitoring using `watchdog` library
- Batch hashing for better performance
- Filtering options (ignore patterns, max file size)
- Parallel scanning for very large directories
- Incremental hash updates only for modified timestamps
