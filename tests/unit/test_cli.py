"""Tests for CLI."""

import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.cli import handle_database_command, handle_server_command


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    manager = Mock()
    manager.create_database.return_value = True
    manager.add_documents.return_value = {"processed": 5, "skipped": 1, "failed": 0}
    manager.list_databases.return_value = {
        "db1": {"description": "First", "doc_count": 10, "last_updated": "2024-01-01"},
        "db2": {"description": "Second", "doc_count": 20, "last_updated": "2024-01-02"},
    }
    return manager


def test_handle_database_create(test_config, mock_db_manager):
    """Test handling database create command."""
    args = Mock()
    args.db_command = "create"
    args.name = "newdb"
    args.description = "Test DB"

    with patch("mcp_agent_rag.cli.DatabaseManager", return_value=mock_db_manager):
        handle_database_command(args, test_config, Mock())

    mock_db_manager.create_database.assert_called_once_with("newdb", "Test DB")


def test_handle_database_add(test_config, mock_db_manager):
    """Test handling database add command."""
    args = Mock()
    args.db_command = "add"
    args.database = "testdb"
    args.path = "/path/to/docs"
    args.url = None
    args.glob = None
    args.recursive = False
    args.skip_existing = False

    with patch("mcp_agent_rag.cli.DatabaseManager", return_value=mock_db_manager):
        handle_database_command(args, test_config, Mock())

    mock_db_manager.add_documents.assert_called_once()


def test_handle_database_list(test_config, mock_db_manager, capsys):
    """Test handling database list command."""
    args = Mock()
    args.db_command = "list"

    with patch("mcp_agent_rag.cli.DatabaseManager", return_value=mock_db_manager):
        handle_database_command(args, test_config, Mock())

    captured = capsys.readouterr()
    assert "db1" in captured.out
    assert "db2" in captured.out


def test_handle_database_list_empty(test_config, capsys):
    """Test listing empty databases."""
    args = Mock()
    args.db_command = "list"

    mock_manager = Mock()
    mock_manager.list_databases.return_value = {}

    with patch("mcp_agent_rag.cli.DatabaseManager", return_value=mock_manager):
        handle_database_command(args, test_config, Mock())

    captured = capsys.readouterr()
    assert "No databases found" in captured.out


def test_handle_server_start_stdio(test_config):
    """Test handling server start command with stdio."""
    # Create test database
    test_config.add_database("testdb", "/tmp/testdb")
    test_config.save()

    args = Mock()
    args.server_command = "start"
    args.active_databases = "testdb"
    args.transport = "stdio"

    mock_server = Mock()

    with patch("mcp_agent_rag.cli.MCPServer", return_value=mock_server):
        handle_server_command(args, test_config, Mock())

    mock_server.run_stdio.assert_called_once()


def test_handle_server_start_nonexistent_database(test_config, capsys):
    """Test starting server with non-existent database."""
    args = Mock()
    args.server_command = "start"
    args.active_databases = "nonexistent"
    args.transport = "stdio"

    with pytest.raises(SystemExit):
        handle_server_command(args, test_config, Mock())


def test_handle_database_add_no_path_or_url(test_config, mock_db_manager):
    """Test add command without path or URL."""
    args = Mock()
    args.db_command = "add"
    args.database = "testdb"
    args.path = None
    args.url = None

    with pytest.raises(SystemExit):
        with patch("mcp_agent_rag.cli.DatabaseManager", return_value=mock_db_manager):
            handle_database_command(args, test_config, Mock())
