# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ragindexer** is a RAG (Retrieval Augmented Generation) indexer for document scanning and semantic search. It integrates:
- **File scanning** for recursive folder indexing
- **Document parsing** (PDF with OCR, DOCX, TXT, Markdown)
- **Text chunking** with semantic overlap
- **Vector embeddings** for semantic search
- **Vector database** for storing and querying embeddings
- **MCP Server** for model context protocol integration

The project ingests documents, indexes them with embeddings, and provides query capabilities.

## Development Commands

### Setup
```bash
# Install base dependencies
uv sync

# Install all dev tools (linting, testing, docs)
uv sync --extra dev,test,doc
```

### Testing
```bash
# Run all tests using .venv
./.venv/Scripts/python.exe -m pytest

# Run specific test file
./.venv/Scripts/python.exe -m pytest tests/test_file_scanner.py -v
./.venv/Scripts/python.exe -m pytest tests/test_document_parser.py -v

# Run tests matching pattern
./.venv/Scripts/python.exe -m pytest tests/ -k "pattern"

# Run with coverage report
./.venv/Scripts/python.exe -m pytest --cov=ragindexer --cov-report=html
```

### Code Quality
```bash
# Format code with black
./.venv/Scripts/python.exe -m black src/ tests/

# Lint with ruff
./.venv/Scripts/python.exe -m ruff check src/ tests/

# Fix auto-fixable ruff issues
./.venv/Scripts/python.exe -m ruff check --fix src/ tests/
```

### Documentation
```bash
# Generate HTML docs, coverage reports, and badges
bash generate_doc.sh
```

## Architecture & Components

### Component 1: FileScanner (✅ Implemented)
**File**: `src/ragindexer/FileScanner.py`

Recursively scans folders to identify documents with:
- Support for .pdf, .docx, .doc, .txt, .md formats
- File metadata extraction (size, timestamps, SHA256 hash)
- Change detection (added, modified, deleted files)
- Cross-platform path normalization

**Pydantic Models**:
- `FileInfo`: File metadata (relative_path, format, size, hash, timestamps)
- `ScanResult`: Scan results with file collection
- `FileChange`: Detected changes between scans
- `ChangeType` & `FileFormat`: Enums for types

**Usage**:
```python
from ragindexer.FileScanner import FileScanner

scanner = FileScanner("/path/to/docs")
result = scanner.scan()  # Returns ScanResult
changes = scanner.detect_changes(scan1, scan2)
```

**Tests**: 12/12 passing, 95% coverage. See `tests/test_file_scanner.py`

### Component 2: DocumentParser (✅ Implemented)
**File**: `src/ragindexer/DocumentParser.py`

Extracts textual content from detected documents with:
- PDF text extraction + metadata (title, author, page count)
- DOCX parsing (paragraphs, tables, core properties)
- TXT & Markdown support with multi-encoding handling
- Character count and metadata preservation

**Pydantic Models**:
- `ParsedDocument`: Extracted content with metadata and character count
- `DocumentMetadata`: Title, author, page count, extraction timestamp
- Integrates with `FileInfo` from FileScanner

**Usage**:
```python
from ragindexer import DocumentParser, FileScanner

scanner = FileScanner("/path/to/docs")
scan_result = scanner.scan()

parser = DocumentParser()
for file_info in scan_result.files.values():
    parsed_doc = parser.parse(file_info)
    # parsed_doc.content is ready for chunking
    # parsed_doc.metadata contains extracted metadata
```

**Tests**: 11/11 passing, 63% coverage. See `tests/test_document_parser.py`

### Component 3: Chunking Service (✅ Implemented)
**File**: `src/ragindexer/ChunkingService.py`

Splits text into semantic chunks with overlap for continuity.

### Component 4: Embedding Service (✅ Implemented)
**File**: `src/ragindexer/EmbeddingService.py`

Generates vector embeddings using sentence-transformers (CPU-optimized).

### Component 5: Vector Database Service (✅ Implemented)
**File**: `src/ragindexer/VectorDatabaseService.py`

Stores and queries embeddings using Qdrant.

### Component 6: Sync Manager (✅ Implemented)
**File**: `src/ragindexer/SyncManager.py`

Orchestrates the complete indexing pipeline.

### Component 8: Settings (✅ Implemented)
**File**: `src/ragindexer/Settings.py`

Centralizes configuration from environment variables and .env files. See `docs/Settings.md` for detailed documentation.

### Component 7: MCP Server (⏳ Not Yet Implemented)
See `Architecture.md` for specifications:
- MCP Server - Expose search API for Claude Code integration

## Core Modules
- **FileScanner.py**: Recursive folder scanning with metadata and change detection
- **DocumentParser.py**: Text extraction from multiple document formats
- **Settings.py**: Configuration and initialization with logging
- **__init__.py**: Logger and module exports

## Dependencies & Tools

**Core dependencies**:
- `pydantic` & `pydantic-settings` - Data validation
- `rich` - Enhanced logging and progress bars
- `PyPDF2` - PDF text extraction
- `python-docx` - DOCX parsing

**Development tools**:
- `black` - Code formatting (line length: 100)
- `ruff` - Linting (line length: 100)
- `pytest` with plugins: asyncio, cov, html, instafail, mock, picked, sugar, xdist
- `pdoc3` - API documentation
- `pre-commit` - Git hooks

## Environment Setup

Create `.env` file (use `sample.env` as template) with required variables:
- Logging level
- Database URLs (if implementing DB components)
- API keys (if needed)

## Environment Variable

All Python commands **must use the `.venv` virtual environment**:
```bash
./.venv/Scripts/python.exe -m <command>
```

On Windows bash, use forward slashes with the `.venv` path:
```bash
./.venv/Scripts/python.exe
```

## Test Report & Coverage

After running pytest:
- HTML test report: `htmldoc/tests/report.html`
- Coverage HTML: `htmldoc/coverage/index.html`
- Coverage XML: `htmldoc/coverage/coverage.xml`

## Documentation Files

- `Specification.md` - User requirements and constraints
- `Architecture.md` - Component design and data flows
- `docs/FileScanner.md` - FileScanner detailed documentation
- `docs/DocumentParser.md` - DocumentParser detailed documentation
- `examples/file_scanner_example.py` - FileScanner usage examples
- `examples/document_parser_example.py` - DocumentParser usage examples
