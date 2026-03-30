# ragindexer

**RAG (Retrieval Augmented Generation) Indexer** - An open-source, self-hosted document indexing and semantic search system.

ragindexer scans a folder of documents, extracts their content, generates semantic embeddings, and exposes a vector search API via the Model Context Protocol (MCP).

## Features

✅ **Multi-format document support**
- PDF (text extraction + OCR ready)
- DOCX/DOC (with table extraction)
- TXT (multi-encoding)
- Markdown (structure preservation)

✅ **Intelligent indexing pipeline**
- Recursive folder scanning with change detection
- Automatic text extraction with metadata
- Semantic text chunking with overlap
- Local vector embeddings (no GPU required)
- Fast semantic search

✅ **Self-hosted & open-source**
- No cloud dependencies
- Linux, Windows, macOS compatible
- Single-user design
- All components are standard, open-source packages

✅ **MCP Server integration**
- Query your documents via Claude Code/Codex
- Semantic search with context
- Document listing and filtering

## Quick Start

### Prerequisites
- Python 3.12+
- Linux (recommended) or Windows/macOS

### Setup

```bash
# Clone repository
git clone <repo-url>
cd ragindexer

# Install dependencies (using uv)
uv sync --all-groups

# Copy environment template
cp sample.env .env
```

### Basic Usage

```python
from ragindexer import FileScanner, DocumentParser

# Scan documents
scanner = FileScanner("/path/to/your/documents")
scan_result = scanner.scan()
print(f"Found {scan_result.total_files} documents")

# Extract content
parser = DocumentParser()
for file_info in scan_result.files.values():
    parsed_doc = parser.parse(file_info)
    print(f"{parsed_doc.metadata.source_file}: {parsed_doc.character_count} chars")
    # Content ready for chunking and embedding
```

See [examples/](examples/) for complete usage examples.

## Architecture

ragindexer is built as a modular pipeline:

```
[Documents Folder]
       ↓
[Component 1: File Scanner] ✅
    Detects all documents, extracts metadata
       ↓
[Component 2: Document Parser] ✅
    Extracts text from each document
       ↓
[Component 3: Chunking Service]
    Splits text into semantic chunks
       ↓
[Component 4: Embedding Service]
    Generates semantic vectors
       ↓
[Component 5: Vector Database]
    Stores embeddings for search
       ↓
[Component 6: Sync Manager]
    Orchestrates updates and deletions
       ↓
[Component 7: MCP Server]
    Exposes search API to Claude Code
       ↓
[Claude Code / Codex]
    Uses RAG context in conversations
```

### Implementation Status

| Component | Status | Coverage | Tests |
|-----------|--------|----------|-------|
| 1. File Scanner | ✅ Done | 95% | 12/12 |
| 2. Document Parser | ✅ Done | 63% | 11/11 |
| 3. Chunking Service | 🚧 Planned | - | - |
| 4. Embedding Service | 🚧 Planned | - | - |
| 5. Vector Database | 🚧 Planned | - | - |
| 6. Sync Manager | 🚧 Planned | - | - |
| 7. MCP Server | 🚧 Planned | - | - |
| 8. Configuration | 🚧 Planned | - | - |

See [Architecture.md](Architecture.md) and [Specification.md](Specification.md) for detailed component specifications.

## Testing

Run the test suite:

```bash
# All tests
./.venv/Scripts/python.exe -m pytest

# Specific test file
./.venv/Scripts/python.exe -m pytest tests/test_document_parser.py -v

# With coverage report
./.venv/Scripts/python.exe -m pytest --cov=ragindexer --cov-report=html

# View coverage
open htmldoc/coverage/index.html  # macOS
xdg-open htmldoc/coverage/index.html  # Linux
start htmldoc\coverage\index.html  # Windows
```

**Current test status**: 23 tests, 100% pass rate, 72% coverage

## Documentation

- **[Specification.md](Specification.md)** - User requirements and constraints
- **[Architecture.md](Architecture.md)** - System design and component specifications
- **[CLAUDE.md](CLAUDE.md)** - Development guidance for Claude Code

### Component Documentation
- **[docs/FileScanner.md](docs/FileScanner.md)** - File scanning details
- **[docs/DocumentParser.md](docs/DocumentParser.md)** - Document parsing details

### Examples
- **[examples/file_scanner_example.py](examples/file_scanner_example.py)** - FileScanner usage
- **[examples/document_parser_example.py](examples/document_parser_example.py)** - DocumentParser usage

## Dependencies

### Core
- `pydantic>=2.12.5` - Data validation
- `pydantic-settings>=2.13.1` - Configuration
- `rich>=14.3.3` - Enhanced logging
- `PyPDF2>=3.0.1` - PDF text extraction
- `python-docx>=1.0.0` - DOCX parsing

### Development
- `pytest` with plugins - Testing framework
- `black` - Code formatting
- `ruff` - Linting
- `pdoc3` - API documentation

## Environment Variables

Create a `.env` file (see `sample.env` for template):

```env
# Logging
LOG_LEVEL=INFO

# Document scanning
SCAN_ROOT=/path/to/documents
SUPPORTED_FORMATS=pdf,docx,txt,md

# Vector embeddings (when implemented)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector database (when implemented)
VECTOR_DB_TYPE=chroma
VECTOR_DB_PATH=./data/vector_db

# MCP Server (when implemented)
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8000
```

## Development

### Code Quality

```bash
# Format code
./.venv/Scripts/python.exe -m black src/ tests/

# Lint
./.venv/Scripts/python.exe -m ruff check src/ tests/

# Fix linting issues
./.venv/Scripts/python.exe -m ruff check --fix src/ tests/
```

### Generate Documentation

```bash
bash generate_doc.sh
```

This generates:
- API documentation in `htmldoc/api/`
- Test reports in `htmldoc/tests/`
- Coverage reports in `htmldoc/coverage/`

## Project Structure

```
ragindexer/
├── src/ragindexer/
│   ├── FileScanner.py          # ✅ Component 1: File scanning
│   ├── DocumentParser.py        # ✅ Component 2: Document parsing
│   ├── Settings.py              # Configuration management
│   └── __init__.py              # Module exports
├── tests/
│   ├── test_file_scanner.py    # 12 tests, 95% coverage
│   ├── test_document_parser.py  # 11 tests, 63% coverage
│   └── coverage.conf            # Coverage configuration
├── docs/
│   ├── FileScanner.md           # FileScanner documentation
│   └── DocumentParser.md        # DocumentParser documentation
├── examples/
│   ├── file_scanner_example.py
│   └── document_parser_example.py
├── Architecture.md              # Component design
├── Specification.md             # User requirements
├── CLAUDE.md                    # Development guidance
├── pyproject.toml               # Project configuration
├── sample.env                   # Environment template
└── README.md                    # This file
```

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `pytest tests/`
2. Code is formatted: `black src/ tests/`
3. Code is linted: `ruff check src/ tests/`
4. New components include tests
5. Documentation is updated

## License

To be defined.

## Roadmap

### Phase 1 (Current) ✅
- [x] Component 1: File Scanner
- [x] Component 2: Document Parser

### Phase 2 (Next)
- [ ] Component 3: Chunking Service
- [ ] Component 4: Embedding Service
- [ ] Component 5: Vector Database

### Phase 3
- [ ] Component 6: Sync Manager
- [ ] Component 7: MCP Server
- [ ] Full integration testing

### Phase 4
- [ ] Performance optimization
- [ ] OCR support for scanned PDFs
- [ ] Advanced filtering and search
- [ ] CLI interface

## Support

For issues, questions, or contributions:
- Check [Architecture.md](Architecture.md) for design details
- Review test files for usage examples
- See component documentation in `docs/` folder

---

Built with Python, Pydantic, and open-source libraries. No GPU required.
