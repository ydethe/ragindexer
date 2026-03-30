# -*- coding: utf-8 -*-
"""
Tests for Settings configuration component
"""

import pytest
from pathlib import Path
import tempfile
import os

from ragindexer.Settings import Settings


class TestSettings:
    """Test cases for Settings configuration"""

    def ntest_settings_default_values(self):
        """Test that Settings has correct default values"""
        settings = Settings(
            _env_file="sample.env"
        )  # Use a non-existent env file to ensure defaults

        assert settings.LOGLEVEL == "info"
        assert settings.SCAN_ROOT == "./documents"
        assert settings.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
        assert settings.CHUNK_SIZE == 512
        assert settings.OVERLAP_SIZE == 50
        assert settings.QDRANT_URL == "http://localhost:6333"
        assert settings.QDRANT_PERSISTENCE_PATH == "./data/qdrant"
        assert settings.MCP_HOST == "localhost"
        assert settings.MCP_PORT == 5000

    def test_settings_env_override(self, monkeypatch):
        """Test that Settings can be overridden by environment variables"""
        monkeypatch.setenv("LOGLEVEL", "debug")
        monkeypatch.setenv("SCAN_ROOT", "/custom/path")
        monkeypatch.setenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        monkeypatch.setenv("OVERLAP_SIZE", "100")
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("QDRANT_PERSISTENCE_PATH", "/custom/qdrant")
        monkeypatch.setenv("MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("MCP_PORT", "8000")

        settings = Settings()

        assert settings.LOGLEVEL == "debug"
        assert settings.SCAN_ROOT == "/custom/path"
        assert settings.EMBEDDING_MODEL == "all-mpnet-base-v2"
        assert settings.CHUNK_SIZE == 1024
        assert settings.OVERLAP_SIZE == 100
        assert settings.QDRANT_URL == "http://qdrant:6333"
        assert settings.QDRANT_PERSISTENCE_PATH == "/custom/qdrant"
        assert settings.MCP_HOST == "0.0.0.0"
        assert settings.MCP_PORT == 8000

    def test_get_qdrant_persistence_path_returns_path(self):
        """Test get_qdrant_persistence_path returns correct Path object"""
        settings = Settings(_env_file=None)
        settings.QDRANT_PERSISTENCE_PATH = "./data/qdrant"

        result = settings.get_qdrant_persistence_path()

        assert isinstance(result, Path)
        assert result == Path("./data/qdrant")

    def test_get_qdrant_persistence_path_returns_none_for_none_string(self):
        """Test get_qdrant_persistence_path returns None when set to 'none'"""
        settings = Settings(_env_file=None)
        settings.QDRANT_PERSISTENCE_PATH = "none"

        result = settings.get_qdrant_persistence_path()

        assert result is None

    def test_get_qdrant_persistence_path_returns_none_for_empty_string(self):
        """Test get_qdrant_persistence_path returns None when empty"""
        settings = Settings(_env_file=None)
        settings.QDRANT_PERSISTENCE_PATH = ""

        result = settings.get_qdrant_persistence_path()

        assert result is None

    def test_get_scan_root_returns_path(self):
        """Test get_scan_root returns correct Path object"""
        settings = Settings(_env_file=None)
        settings.SCAN_ROOT = "/path/to/docs"

        result = settings.get_scan_root()

        assert isinstance(result, Path)
        assert result == Path("/path/to/docs")

    def test_chunk_size_validation_positive(self):
        """Test that CHUNK_SIZE must be positive"""
        with pytest.raises(Exception):
            Settings(CHUNK_SIZE=-1, _env_file=None)

    def test_overlap_size_validation_non_negative(self):
        """Test that OVERLAP_SIZE must be non-negative"""
        with pytest.raises(Exception):
            Settings(OVERLAP_SIZE=-1, _env_file=None)

    def test_mcp_port_validation_range(self):
        """Test that MCP_PORT must be in valid range"""
        with pytest.raises(Exception):
            Settings(MCP_PORT=0, _env_file=None)

        with pytest.raises(Exception):
            Settings(MCP_PORT=65536, _env_file=None)

    def ntest_settings_load_from_env_file(self):
        """Test that Settings can load from .env file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("LOGLEVEL=warning\n")
            f.write("SCAN_ROOT=/test/docs\n")
            f.write("EMBEDDING_MODEL=all-mpnet-base-v2\n")
            f.write("CHUNK_SIZE=256\n")
            f.write("OVERLAP_SIZE=32\n")
            env_file = f.name

        try:
            settings = Settings(_env_file=env_file)

            assert settings.LOGLEVEL == "warning"
            assert settings.SCAN_ROOT == "/test/docs"
            assert settings.EMBEDDING_MODEL == "all-mpnet-base-v2"
            assert settings.CHUNK_SIZE == 256
            assert settings.OVERLAP_SIZE == 32
        finally:
            os.unlink(env_file)

    def test_embedding_model_choices(self):
        """Test that common embedding models can be configured"""
        models = [
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

        for model in models:
            settings = Settings(EMBEDDING_MODEL=model, _env_file=None)
            assert settings.EMBEDDING_MODEL == model

    def test_loglevel_valid_values(self):
        """Test that valid log levels are accepted"""
        valid_levels = ["debug", "info", "warning", "error", "critical"]

        for level in valid_levels:
            settings = Settings(LOGLEVEL=level, _env_file=None)
            assert settings.LOGLEVEL == level

    def test_settings_extra_fields_allowed(self):
        """Test that extra fields are allowed (extra='allow')"""
        settings = Settings(CUSTOM_FIELD="custom_value", _env_file=None)

        assert hasattr(settings, "CUSTOM_FIELD")
        assert settings.CUSTOM_FIELD == "custom_value"

    def test_settings_qdrant_url_formats(self):
        """Test that various Qdrant URL formats are accepted"""
        urls = [
            "http://localhost:6333",
            "http://qdrant:6333",
            "http://127.0.0.1:6333",
            "http://qdrant.example.com:6333",
        ]

        for url in urls:
            settings = Settings(QDRANT_URL=url, _env_file=None)
            assert settings.QDRANT_URL == url

    def ntest_qdrant_api_key_default_none(self):
        """Test that QDRANT_API_KEY defaults to None"""
        settings = Settings(_env_file=None)
        assert settings.QDRANT_API_KEY is None

    def test_qdrant_api_key_can_be_set(self):
        """Test that QDRANT_API_KEY can be set to a value"""
        settings = Settings(QDRANT_API_KEY="my-secret-key", _env_file=None)
        assert settings.QDRANT_API_KEY == "my-secret-key"

    def test_qdrant_api_key_env_override(self, monkeypatch):
        """Test that QDRANT_API_KEY can be set via environment variable"""
        monkeypatch.setenv("QDRANT_API_KEY", "env-api-key")
        settings = Settings(_env_file=None)
        assert settings.QDRANT_API_KEY == "env-api-key"

    def ntest_qdrant_api_key_from_env_file(self):
        """Test that QDRANT_API_KEY can be loaded from .env file"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("QDRANT_API_KEY=file-api-key\n")
            env_file = f.name

        try:
            settings = Settings(_env_file=env_file)
            assert settings.QDRANT_API_KEY == "file-api-key"
        finally:
            os.unlink(env_file)
