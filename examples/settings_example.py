# -*- coding: utf-8 -*-
"""
Example: Using the Settings component

This example demonstrates various ways to configure ragindexer using the
Settings component and how to integrate it with other components.
"""

from ragindexer import Settings
from pathlib import Path

settings=Settings()

print("=" * 70)
print("Settings Component - Configuration Examples")
print("=" * 70)

# Example 1: Access default settings
print("\n1. DEFAULT SETTINGS")
print("-" * 70)
print(f"Logging Level:           {settings.LOGLEVEL}")
print(f"Scan Root:               {settings.SCAN_ROOT}")
print(f"Embedding Model:         {settings.EMBEDDING_MODEL}")
print(f"Chunk Size:              {settings.CHUNK_SIZE} tokens")
print(f"Overlap Size:            {settings.OVERLAP_SIZE} tokens")
print(f"Qdrant URL:              {settings.QDRANT_URL}")
print(f"Qdrant Persistence:      {settings.QDRANT_PERSISTENCE_PATH}")
print(f"MCP Host:Port:           {settings.MCP_HOST}:{settings.MCP_PORT}")

# Example 2: Use helper methods
print("\n2. HELPER METHODS")
print("-" * 70)
scan_root = settings.get_scan_root()
print(f"Scan Root (as Path):     {scan_root}")
print(f"Scan Root type:          {type(scan_root).__name__}")

persistence = settings.get_qdrant_persistence_path()
if persistence:
    print(f"Persistence Path:        {persistence}")
else:
    print(f"Persistence:             In-memory mode")

# Example 3: Create custom settings for testing
print("\n3. CUSTOM SETTINGS (Testing)")
print("-" * 70)
custom_settings = Settings(
    LOGLEVEL="debug",
    SCAN_ROOT="/custom/documents",
    CHUNK_SIZE=256,
    OVERLAP_SIZE=32,
    _env_file=None,  # Don't load from .env
)
print(f"Custom LOGLEVEL:         {custom_settings.LOGLEVEL}")
print(f"Custom SCAN_ROOT:        {custom_settings.SCAN_ROOT}")
print(f"Custom CHUNK_SIZE:       {custom_settings.CHUNK_SIZE}")
print(f"Custom OVERLAP_SIZE:     {custom_settings.OVERLAP_SIZE}")

# Example 4: Integration with other components
print("\n4. INTEGRATION WITH COMPONENTS")
print("-" * 70)

try:
    from ragindexer import (
        FileScanner,
        DocumentParser,
        ChunkingService,
        EmbeddingService,
    )

    # Initialize components with settings
    scanner = FileScanner(settings.get_scan_root())
    parser = DocumentParser()
    chunking_service = ChunkingService(
        chunk_size=settings.CHUNK_SIZE, overlap_size=settings.OVERLAP_SIZE
    )
    embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL)

    print(f"[OK] FileScanner initialized for: {settings.get_scan_root()}")
    print(f"[OK] ChunkingService with chunk_size={settings.CHUNK_SIZE}")
    print(f"[OK] EmbeddingService with model={settings.EMBEDDING_MODEL}")

except Exception as e:
    print(f"Note: Component import/initialization skipped ({type(e).__name__})")

# Example 5: Configuration scenarios
print("\n5. RECOMMENDED CONFIGURATIONS")
print("-" * 70)

configs = {
    "Development": {
        "LOGLEVEL": "debug",
        "CHUNK_SIZE": 512,
        "OVERLAP_SIZE": 50,
        "QDRANT_PERSISTENCE_PATH": "none",
    },
    "Testing": {
        "LOGLEVEL": "info",
        "CHUNK_SIZE": 256,
        "OVERLAP_SIZE": 32,
        "QDRANT_PERSISTENCE_PATH": "none",
    },
    "Production": {
        "LOGLEVEL": "warning",
        "CHUNK_SIZE": 1024,
        "OVERLAP_SIZE": 100,
        "QDRANT_PERSISTENCE_PATH": "./data/qdrant",
    },
}

for scenario, config in configs.items():
    print(f"\n  {scenario}:")
    for key, value in config.items():
        print(f"    {key}: {value}")

# Example 6: Environment variable override
print("\n6. ENVIRONMENT VARIABLE OVERRIDE")
print("-" * 70)
import os

# Show how to set environment variables
print("To override settings via environment variables, use:")
print("  export LOGLEVEL=warning")
print("  export CHUNK_SIZE=1024")
print("  export QDRANT_URL=http://qdrant-prod:6333")
print("")
print(f"Current LOGLEVEL: {os.getenv('LOGLEVEL', '(not set)')}")
print(f"Current CHUNK_SIZE: {os.getenv('CHUNK_SIZE', '(not set)')}")

# Example 7: .env file configuration
print("\n7. .ENV FILE CONFIGURATION")
print("-" * 70)
print("Create a .env file in your project root with:")
print("""
LOGLEVEL=info
SCAN_ROOT=./documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
OVERLAP_SIZE=50
QDRANT_URL=http://localhost:6333
QDRANT_PERSISTENCE_PATH=./data/qdrant
MCP_HOST=localhost
MCP_PORT=5000
""")

# Example 8: Validation
print("\n8. SETTINGS VALIDATION")
print("-" * 70)
print("Settings validates all parameters:")
print("  [OK] LOGLEVEL must be: debug, info, warning, error, critical")
print("  [OK] CHUNK_SIZE must be > 0")
print("  [OK] OVERLAP_SIZE must be >= 0")
print("  [OK] MCP_PORT must be between 1-65535")

print("\nTrying invalid settings:")
try:
    invalid_settings = Settings(CHUNK_SIZE=-1, _env_file=None)
except Exception as e:
    print(f"  [OK] Caught validation error: {type(e).__name__}")

print("\n" + "=" * 70)
print("For more information, see docs/Settings.md")
print("=" * 70)
