"""Tests for database manager."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.database import DatabaseManager


@pytest.fixture
def db_manager(test_config):
    """Create database manager."""
    with patch("mcp_agent_rag.database.OllamaEmbedder"):
        return DatabaseManager(test_config)


def test_create_database(db_manager, test_config):
    """Test creating database."""
    success = db_manager.create_database("testdb", "Test description")
    assert success is True
    assert test_config.database_exists("testdb")

    db_info = test_config.get_database("testdb")
    assert db_info["description"] == "Test description"


def test_create_duplicate_database(db_manager, test_config):
    """Test creating duplicate database."""
    db_manager.create_database("testdb")
    success = db_manager.create_database("testdb")
    assert success is False


def test_list_databases(db_manager, test_config):
    """Test listing databases."""
    db_manager.create_database("db1", "First")
    db_manager.create_database("db2", "Second")

    databases = db_manager.list_databases()
    assert len(databases) == 2
    assert "db1" in databases
    assert "db2" in databases


def test_load_database(db_manager, test_config, temp_dir):
    """Test loading database."""
    # Create database first
    db_path = temp_dir / "testdb"
    db_path.mkdir(parents=True)
    test_config.add_database("testdb", str(db_path))
    test_config.save()

    # Load it
    db = db_manager.load_database("testdb")
    assert db is not None
    assert "testdb" in db_manager.databases


def test_load_nonexistent_database(db_manager):
    """Test loading non-existent database."""
    db = db_manager.load_database("nonexistent")
    assert db is None


def test_load_multiple_databases(db_manager, test_config, temp_dir):
    """Test loading multiple databases."""
    # Create databases
    for name in ["db1", "db2"]:
        db_path = temp_dir / name
        db_path.mkdir(parents=True)
        test_config.add_database(name, str(db_path))
    test_config.save()

    # Load them
    loaded = db_manager.load_multiple_databases(["db1", "db2"])
    assert len(loaded) == 2
    assert "db1" in loaded
    assert "db2" in loaded
