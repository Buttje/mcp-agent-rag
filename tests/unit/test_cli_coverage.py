"""Integration tests for CLI to improve coverage."""

import sys
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag import cli


def test_main_no_args(capsys):
    """Test CLI with no arguments."""
    with patch("sys.argv", ["mcp-rag"]):
        with pytest.raises(SystemExit):
            cli.main()


def test_main_database_create_success(test_config, temp_dir, monkeypatch):
    """Test database create via CLI."""
    config_path = temp_dir / "config.json"
    test_config.config_path = config_path
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "create",
        "--name", "testdb",
        "--description", "Test DB"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            mock_manager = Mock()
            mock_manager.create_database.return_value = True
            MockDBManager.return_value = mock_manager

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config
                test_config.get_database_path = Mock(return_value="/tmp/testdb")

                cli.main()

def test_main_database_create_failure(test_config, temp_dir):
    """Test database create failure via CLI."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "create",
        "--name", "testdb"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            mock_manager = Mock()
            mock_manager.create_database.return_value = False
            MockDBManager.return_value = mock_manager

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config

                with pytest.raises(SystemExit):
                    cli.main()


def test_main_database_add_success(test_config, temp_dir, sample_text_file):
    """Test database add via CLI."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "add",
        "--database", "testdb",
        "--path", str(sample_text_file)
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            mock_manager = Mock()
            mock_manager.add_documents.return_value = {
                "processed": 1, "skipped": 0, "failed": 0
            }
            MockDBManager.return_value = mock_manager

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config

                cli.main()


def test_main_database_add_with_url(test_config, temp_dir):
    """Test database add with URL via CLI."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "add",
        "--database", "testdb",
        "--url", "http://example.com/doc.html"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            mock_manager = Mock()
            mock_manager.add_documents.return_value = {
                "processed": 1, "skipped": 0, "failed": 0
            }
            MockDBManager.return_value = mock_manager

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config

                cli.main()


def test_main_database_list_empty(test_config, temp_dir, capsys):
    """Test database list with no databases."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "list"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            mock_manager = Mock()
            mock_manager.list_databases.return_value = {}
            MockDBManager.return_value = mock_manager

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config

                cli.main()

                captured = capsys.readouterr()
                assert "No databases found" in captured.out


def test_main_database_list_with_data(test_config, temp_dir, capsys):
    """Test database list with databases."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "list"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            mock_manager = Mock()
            mock_manager.list_databases.return_value = {
                "db1": {"description": "First", "doc_count": 10, "last_updated": "2024-01-01"}
            }
            MockDBManager.return_value = mock_manager

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config

                cli.main()

                captured = capsys.readouterr()
                assert "db1" in captured.out


def test_main_server_start(test_config, temp_dir):
    """Test server start via CLI."""
    config_path = temp_dir / "config.json"
    test_config.add_database("db1", "/tmp/db1")
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "server", "start",
        "--active-databases", "db1",
        "--transport", "stdio"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.MCPServer") as MockServer:
            mock_server = Mock()
            mock_server.run_stdio = Mock(side_effect=KeyboardInterrupt)
            MockServer.return_value = mock_server

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                mock_config = test_config
                mock_config.database_exists = Mock(return_value=True)
                MockConfig.return_value = mock_config

                try:
                    cli.main()
                except KeyboardInterrupt:
                    pass


def test_main_server_start_nonexistent_db(test_config, temp_dir):
    """Test server start with non-existent database."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "server", "start",
        "--active-databases", "nonexistent"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.Config") as MockConfig:
            mock_config = test_config
            mock_config.database_exists = Mock(return_value=False)
            MockConfig.return_value = mock_config

            with pytest.raises(SystemExit):
                cli.main()


def test_main_error_handling(test_config, temp_dir):
    """Test error handling in main."""
    config_path = temp_dir / "config.json"
    test_config.save()

    args = [
        "mcp-rag",
        "--config", str(config_path),
        "database", "create",
        "--name", "testdb"
    ]

    with patch("sys.argv", args):
        with patch("mcp_agent_rag.cli.DatabaseManager") as MockDBManager:
            MockDBManager.side_effect = Exception("Test error")

            with patch("mcp_agent_rag.cli.Config") as MockConfig:
                MockConfig.return_value = test_config

                with pytest.raises(SystemExit):
                    cli.main()
