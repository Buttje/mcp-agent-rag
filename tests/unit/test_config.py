"""Tests for configuration management."""

import json
from pathlib import Path

import pytest

from mcp_agent_rag.config import Config


def test_config_default_path():
    """Test default config path."""
    path = Config.get_default_config_path()
    assert ".mcp-agent-rag" in path
    assert "config.json" in path


def test_config_default_data_dir():
    """Test default data directory."""
    data_dir = Config.get_default_data_dir()
    assert ".mcp-agent-rag" in str(data_dir)


def test_config_load_or_create(test_config):
    """Test config creation with defaults."""
    assert "embedding_model" in test_config.data
    assert "databases" in test_config.data
    assert test_config.get("embedding_model") == "nomic-embed-text"


def test_config_save_and_load(temp_dir):
    """Test saving and loading config."""
    config_path = temp_dir / "config.json"

    # Create and save
    config1 = Config(str(config_path))
    config1.set("test_key", "test_value")
    config1.save()

    assert config_path.exists()

    # Load and verify
    config2 = Config(str(config_path))
    assert config2.get("test_key") == "test_value"


def test_config_add_database(test_config):
    """Test adding database."""
    test_config.add_database("testdb", "/path/to/db", "Test database", 10)

    assert test_config.database_exists("testdb")
    db_info = test_config.get_database("testdb")
    assert db_info["path"] == "/path/to/db"
    assert db_info["description"] == "Test database"
    assert db_info["doc_count"] == 10


def test_config_update_database(test_config):
    """Test updating database."""
    test_config.add_database("testdb", "/path/to/db")
    test_config.update_database("testdb", doc_count=5, description="Updated")

    db_info = test_config.get_database("testdb")
    assert db_info["doc_count"] == 5
    assert db_info["description"] == "Updated"


def test_config_list_databases(test_config):
    """Test listing databases."""
    test_config.add_database("db1", "/path/1")
    test_config.add_database("db2", "/path/2")

    databases = test_config.list_databases()
    assert len(databases) == 2
    assert "db1" in databases
    assert "db2" in databases


def test_config_get_database_path(test_config):
    """Test getting database path."""
    test_config.add_database("testdb", "/path/to/db")
    path = test_config.get_database_path("testdb")
    assert path == Path("/path/to/db")


def test_config_database_not_exists(test_config):
    """Test checking non-existent database."""
    assert not test_config.database_exists("nonexistent")
    assert test_config.get_database("nonexistent") is None
    assert test_config.get_database_path("nonexistent") is None
