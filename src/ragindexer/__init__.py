# -*- coding: utf-8 -*-
"""

.. include:: ../../README.md

# Testing

## Run the tests

To run tests, just run:

    python -m pytest

## Test reports

[See test report](../tests/report.html)

[See coverage](../coverage/index.html)

"""

import logging

from ragindexer.FileScanner import FileScanner, FileInfo, FileFormat, ScanResult, FileChange
from ragindexer.DocumentParser import (
    DocumentParser,
    ParsedDocument,
    DocumentMetadata,
)
from ragindexer.ChunkingService import (
    ChunkingService,
    TextChunk,
    ChunkMetadata,
    ChunkingResult,
)
from ragindexer.EmbeddingService import (
    EmbeddingService,
    EmbeddedChunk,
    EmbeddingResult,
)
from ragindexer.VectorDatabaseService import (
    VectorDatabaseService,
    StoredEmbedding,
    SearchResult,
    VectorDatabaseResult,
)
from ragindexer.SyncManager import (
    SyncManager,
    SyncStatus,
    FileSyncResult,
    SyncOperationResult,
)
from ragindexer.Settings import Settings
from ragindexer.Orchestrator import (
    SyncEventHandler,
    PipelineOrchestrator,
)

# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger(__name__)

__all__ = [
    "FileScanner",
    "FileInfo",
    "FileFormat",
    "ScanResult",
    "FileChange",
    "DocumentParser",
    "ParsedDocument",
    "DocumentMetadata",
    "ChunkingService",
    "TextChunk",
    "ChunkMetadata",
    "ChunkingResult",
    "EmbeddingService",
    "EmbeddedChunk",
    "EmbeddingResult",
    "VectorDatabaseService",
    "StoredEmbedding",
    "SearchResult",
    "VectorDatabaseResult",
    "SyncManager",
    "SyncStatus",
    "FileSyncResult",
    "SyncOperationResult",
    "Settings",
    "settings",
    "SyncEventHandler",
    "PipelineOrchestrator",
    "logger",
]
