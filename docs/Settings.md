# Settings - Component 8: Configuration & Settings

## Overview

The **Settings** component centralizes all configuration parameters for the ragindexer application. It provides a unified interface for managing:
- File scanning parameters
- Embedding model configuration
- Text chunking settings
- Vector database parameters
- MCP server configuration
- Logging level

The Settings class uses **Pydantic** for configuration validation and **pydantic-settings** to load from environment variables and `.env` files.

## Features

- **Environment-based configuration** via `.env` files or environment variables
- **Type validation** for all parameters
- **Sensible defaults** for quick setup
- **Flexible persistence** mode (in-memory or persistent storage)
- **Extra fields support** for future extensibility
- **Rich logging** with version information

## Configuration Parameters

### Logging Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `LOGLEVEL` | str | `"info"` | Logging level: debug, info, warning, error, critical |

**Validation**: Must be one of the valid logging levels.

### File Scanning Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SCAN_ROOT` | str | `"./documents"` | Root directory to scan for documents |

**Usage with FileScanner**:
```python
from ragindexer import FileScanner, settings

scanner = FileScanner(settings.get_scan_root())
result = scanner.scan()
```

### Embedding Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EMBEDDING_MODEL` | str | `"all-MiniLM-L6-v2"` | Sentence Transformers model name for embeddings |

**Supported Models** (no GPU required):
- `all-MiniLM-L6-v2` (384 dimensions, 22M parameters) - **Recommended for production**
- `all-mpnet-base-v2` (768 dimensions, 110M parameters) - Higher quality, slower
- `sentence-transformers/all-MiniLM-L6-v2` - Explicit model path
- See [Sentence Transformers](https://www.sbert.net/docs/pretrained_models/) for more options

**Usage with EmbeddingService**:
```python
from ragindexer import EmbeddingService, settings

embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)
```

### Chunking Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `CHUNK_SIZE` | int | `512` | Target chunk size in tokens (~0.75 words per token) |
| `OVERLAP_SIZE` | int | `50` | Overlap between consecutive chunks for semantic continuity |

**Validation**:
- `CHUNK_SIZE` must be > 0
- `OVERLAP_SIZE` must be >= 0 and typically < `CHUNK_SIZE`

**Usage with ChunkingService**:
```python
from ragindexer import ChunkingService, settings

chunking_service = ChunkingService(
    chunk_size=settings.CHUNK_SIZE,
    overlap_size=settings.OVERLAP_SIZE
)
```

**Recommended Values**:
- Small documents: `CHUNK_SIZE=256`, `OVERLAP_SIZE=32`
- Standard documents: `CHUNK_SIZE=512`, `OVERLAP_SIZE=50` (default)
- Large documents: `CHUNK_SIZE=1024`, `OVERLAP_SIZE=100`

### Vector Database Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `QDRANT_URL` | str | `"http://localhost:6333"` | URL for Qdrant vector database |
| `QDRANT_PERSISTENCE_PATH` | str | `"./data/qdrant"` | Path for persistent storage (or "none" for in-memory) |

**QDRANT_URL Examples**:
- Local development: `http://localhost:6333`
- Docker Compose: `http://qdrant:6333`
- Remote server: `http://qdrant.example.com:6333`

**QDRANT_PERSISTENCE_PATH Options**:
- **Persistent (production)**: `"./data/qdrant"` - Stores embeddings to disk
- **In-memory (development)**: `"none"` - Stores only in RAM (lost on restart)

**Usage with VectorDatabaseService**:
```python
from ragindexer import VectorDatabaseService, settings

vector_db = VectorDatabaseService(
    qdrant_url=settings.QDRANT_URL,
    persistence_path=settings.get_qdrant_persistence_path()
)
```

### MCP Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MCP_HOST` | str | `"localhost"` | Host for MCP server |
| `MCP_PORT` | int | `5000` | Port for MCP server (1-65535) |

**Validation**: `MCP_PORT` must be between 1 and 65535.

**Common Configurations**:
- **Development**: `MCP_HOST=localhost`, `MCP_PORT=5000`
- **Docker**: `MCP_HOST=0.0.0.0`, `MCP_PORT=5000`
- **Production**: `MCP_HOST=0.0.0.0`, `MCP_PORT=8000`

## Usage

### Default Configuration

```python
from ragindexer import settings

# Access settings directly
print(settings.LOGLEVEL)              # "info"
print(settings.SCAN_ROOT)             # "./documents"
print(settings.EMBEDDING_MODEL)       # "all-MiniLM-L6-v2"
print(settings.CHUNK_SIZE)            # 512
print(settings.MCP_PORT)              # 5000
```

### Loading from `.env` File

Create a `.env` file in your project root:

```bash
LOGLEVEL=debug
SCAN_ROOT=/path/to/documents
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=1024
OVERLAP_SIZE=100
QDRANT_URL=http://qdrant:6333
QDRANT_PERSISTENCE_PATH=./data/qdrant
MCP_HOST=0.0.0.0
MCP_PORT=8000
```

The settings will automatically load from this file:

```python
from ragindexer import settings

# All values are now loaded from .env
print(settings.LOGLEVEL)              # "debug"
print(settings.CHUNK_SIZE)            # 1024
```

### Using Environment Variables

```bash
export LOGLEVEL=warning
export CHUNK_SIZE=256
python your_script.py
```

```python
from ragindexer import settings

print(settings.LOGLEVEL)              # "warning"
print(settings.CHUNK_SIZE)            # 256
```

### Environment Variable Priority

Settings are loaded in this order (highest priority first):
1. Environment variables
2. `.env` file (specified by `RAGINDEXER_ENV_FILE` or default `.env`)
3. Default values in Settings class

### Programmatic Configuration (Testing)

```python
from ragindexer import Settings

# Create settings with custom values (no .env file)
settings = Settings(
    LOGLEVEL="debug",
    SCAN_ROOT="/custom/path",
    CHUNK_SIZE=1024,
    _env_file=None  # Skip .env loading
)

print(settings.CHUNK_SIZE)            # 1024
```

## Helper Methods

### `get_qdrant_persistence_path()`

Returns the Qdrant persistence path as a `Path` object, or `None` if persistence is disabled.

```python
from ragindexer import settings

persistence_path = settings.get_qdrant_persistence_path()
if persistence_path:
    print(f"Using persistent storage at: {persistence_path}")
else:
    print("Using in-memory storage")
```

### `get_scan_root()`

Returns the scan root directory as a `Path` object.

```python
from ragindexer import settings
from pathlib import Path

scan_root = settings.get_scan_root()
assert isinstance(scan_root, Path)
```

## Integration with Other Components

### Full Pipeline Example

```python
from pathlib import Path
from ragindexer import (
    settings,
    FileScanner,
    DocumentParser,
    ChunkingService,
    EmbeddingService,
    VectorDatabaseService,
)

# Initialize all components using settings
scanner = FileScanner(settings.get_scan_root())
parser = DocumentParser()
chunking_service = ChunkingService(
    chunk_size=settings.CHUNK_SIZE,
    overlap_size=settings.OVERLAP_SIZE
)
embedding_service = EmbeddingService(
    model_name=settings.EMBEDDING_MODEL
)
vector_db = VectorDatabaseService(
    qdrant_url=settings.QDRANT_URL,
    persistence_path=settings.get_qdrant_persistence_path()
)

# Run pipeline
scan_result = scanner.scan()
for file_info in scan_result.files.values():
    parsed_doc = parser.parse(file_info)
    chunks = chunking_service.chunk(parsed_doc)
    embeddings = embedding_service.embed_chunks(chunks.chunks)
    vector_db.add_embeddings(embeddings.embedded_chunks)

print(f"Indexed {vector_db.get_collection_stats().points_count} embeddings")
```

## Validation

The Settings class includes validators for all important fields:

| Field | Validation Rule | Error Example |
|-------|-----------------|---------------|
| `LOGLEVEL` | Must be one of: debug, info, warning, error, critical | `ValueError: LOGLEVEL must be one of [...]` |
| `CHUNK_SIZE` | Must be > 0 | `ValueError: CHUNK_SIZE must be positive` |
| `OVERLAP_SIZE` | Must be >= 0 | `ValueError: OVERLAP_SIZE must be non-negative` |
| `MCP_PORT` | Must be between 1-65535 | `ValueError: MCP_PORT must be between 1 and 65535` |

## Testing

The Settings component has 14 tests covering:
- Default values
- Environment variable override
- Persistence path handling
- Configuration file loading
- Field validation
- Extra fields support

Run tests with:
```bash
./.venv/Scripts/python.exe -m pytest tests/test_settings.py -v
```

## Advanced: Custom Environment File

You can specify a custom environment file path via the `RAGINDEXER_ENV_FILE` environment variable:

```bash
export RAGINDEXER_ENV_FILE=/etc/ragindexer/production.env
python your_script.py
```

## Logging Configuration

Once loaded, Settings configures the root logger with the specified level:

```python
import logging
from ragindexer import settings

logger = logging.getLogger("ragindexer")
# Logger level is automatically set to settings.LOGLEVEL
```

The logger uses **RichHandler** for enhanced formatting and tracebacks.

## Environment File Examples

### Development Setup
```bash
# .env.development
LOGLEVEL=debug
SCAN_ROOT=./test_documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
OVERLAP_SIZE=50
QDRANT_URL=http://localhost:6333
QDRANT_PERSISTENCE_PATH=none
MCP_HOST=localhost
MCP_PORT=5000
```

### Production Setup
```bash
# .env.production
LOGLEVEL=info
SCAN_ROOT=/var/lib/ragindexer/documents
EMBEDDING_MODEL=all-mpnet-base-v2
CHUNK_SIZE=1024
OVERLAP_SIZE=100
QDRANT_URL=http://qdrant-server:6333
QDRANT_PERSISTENCE_PATH=/var/lib/ragindexer/data/qdrant
MCP_HOST=0.0.0.0
MCP_PORT=8000
```

### Docker Setup
```bash
# .env.docker
LOGLEVEL=info
SCAN_ROOT=/data/documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
OVERLAP_SIZE=50
QDRANT_URL=http://qdrant:6333
QDRANT_PERSISTENCE_PATH=/data/qdrant
MCP_HOST=0.0.0.0
MCP_PORT=5000
```
