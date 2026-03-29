from pathlib import Path
import logging
import os

from rich.logging import RichHandler

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration for ragindexer application.

    Loads settings from:
    1. Environment variables
    2. .env file (specified by RAGINDEXER_ENV_FILE or default ".env")
    3. Default values

    All settings are validated by Pydantic.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # Application configuration
    ENV_FILE: str = Field(
        default=".env", description="Path to .env file for loading environment variables"
    )

    # Logging configuration
    LOGLEVEL: str = Field(
        default="info", description="Logging level (debug, info, warning, error, critical)"
    )

    # File scanning configuration
    SCAN_ROOT: str = Field(
        default="./documents", description="Root directory to scan for documents"
    )

    # Embedding model configuration
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformers model for embeddings (local, no GPU required)",
    )

    # Chunking configuration
    CHUNK_SIZE: int = Field(
        default=512, description="Target chunk size in tokens (approximate: 1 token ≈ 0.75 words)"
    )
    OVERLAP_SIZE: int = Field(
        default=50, description="Overlap size between chunks for semantic continuity"
    )

    # Pipeline orchestrator configuration
    DEBOUNCE_DELAY: float = Field(
        default=2.0, description="Seconds to wait before syncing on filesystem changes"
    )

    # Vector database configuration
    QDRANT_URL: str = Field(
        default="http://localhost:6333", description="URL for Qdrant vector database"
    )
    QDRANT_PERSISTENCE_PATH: str = Field(
        default="./data/qdrant",
        description="Path for persistent storage of Qdrant data (None for in-memory)",
    )

    # MCP Server configuration
    MCP_HOST: str = Field(default="localhost", description="Host for MCP server")
    MCP_PORT: int = Field(default=5000, description="Port for MCP server")

    @field_validator("LOGLEVEL")
    @classmethod
    def validate_loglevel(cls, v: str) -> str:
        """Validate that LOGLEVEL is a valid logging level"""
        valid_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in valid_levels:
            raise ValueError(f"LOGLEVEL must be one of {valid_levels}, got '{v}'")
        return v.lower()

    @field_validator("CHUNK_SIZE")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate that CHUNK_SIZE is positive"""
        if v <= 0:
            raise ValueError(f"CHUNK_SIZE must be positive, got {v}")
        return v

    @field_validator("OVERLAP_SIZE")
    @classmethod
    def validate_overlap_size(cls, v: int) -> int:
        """Validate that OVERLAP_SIZE is non-negative"""
        if v < 0:
            raise ValueError(f"OVERLAP_SIZE must be non-negative, got {v}")
        return v

    @field_validator("DEBOUNCE_DELAY")
    @classmethod
    def validate_debounce_delay(cls, v: float) -> float:
        """Validate that DEBOUNCE_DELAY is positive"""
        if v <= 0:
            raise ValueError(f"DEBOUNCE_DELAY must be positive, got {v}")
        return v

    @field_validator("MCP_PORT")
    @classmethod
    def validate_mcp_port(cls, v: int) -> int:
        """Validate that MCP_PORT is in valid range (1-65535)"""
        if v <= 0 or v > 65535:
            raise ValueError(f"MCP_PORT must be between 1 and 65535, got {v}")
        return v

    def __init__(self, **data):
        # Extract _env_file from data if provided, otherwise use environment or default
        _env_file = data.pop("_env_file", None)
        if _env_file is None:
            # Use ENV_FILE from environment, or fall back to default
            _env_file = os.environ.get("RAGINDEXER_ENV_FILE", ".env")

        super().__init__(_env_file=_env_file, _env_file_encoding="utf-8", **data)
        from importlib.metadata import version

        ragindexer_version = version("ragindexer")
        logger = logging.getLogger(__name__)
        logger.addHandler(RichHandler(rich_tracebacks=False))
        logger.setLevel(self.LOGLEVEL.upper())
        logger.info(f"RagIndexer version: '{ragindexer_version}'")
        logger.info(f"Loaded environment: '{_env_file}'")

    def get_qdrant_persistence_path(self) -> Path | None:
        """
        Get the Qdrant persistence path as a Path object.

        Returns None if persistence is disabled (e.g., for in-memory mode).
        """
        if self.QDRANT_PERSISTENCE_PATH and self.QDRANT_PERSISTENCE_PATH.lower() != "none":
            return Path(self.QDRANT_PERSISTENCE_PATH)
        return None

    def get_scan_root(self) -> Path:
        """Get the scan root directory as a Path object."""
        return Path(self.SCAN_ROOT)

