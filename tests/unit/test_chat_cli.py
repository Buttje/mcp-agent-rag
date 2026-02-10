"""Tests for chat_cli module."""

import json
import sys
from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.chat_cli import MCPClient, create_mcp_tool_query_data, start_mcp_server, main


def test_mcp_client_call_tool():
    """Test MCP client tool calling."""
    # Create a mock process
    mock_process = Mock()
    mock_process.stdin = Mock()
    mock_process.stdout = Mock()
    mock_process.poll.return_value = None

    # Mock the response
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"context": "test context", "citations": []},
    }
    mock_process.stdout.readline.return_value = (
        (json.dumps(response) + "\n").encode()
    )

    client = MCPClient(mock_process, verbose=False)

    result = client.call_tool("query-get_data", {"prompt": "test"})

    assert result["context"] == "test context"
    assert result["citations"] == []


def test_mcp_client_call_tool_verbose(capsys):
    """Test MCP client tool calling with verbose mode."""
    # Create a mock process
    mock_process = Mock()
    mock_process.stdin = Mock()
    mock_process.stdout = Mock()
    mock_process.poll.return_value = None

    # Mock the response
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"context": "test context", "citations": [{"source": "test.txt"}]},
    }
    mock_process.stdout.readline.return_value = (
        (json.dumps(response) + "\n").encode()
    )

    client = MCPClient(mock_process, verbose=True)

    result = client.call_tool("query-get_data", {"prompt": "test"})

    assert result["context"] == "test context"
    assert result["citations"] == [{"source": "test.txt"}]

    # Check verbose output
    captured = capsys.readouterr()
    assert "🔧 [MCP Tool Call]" in captured.out
    assert "Tool: query-get_data" in captured.out
    assert "✅ Result:" in captured.out


def test_mcp_client_error_response():
    """Test MCP client handles error responses."""
    mock_process = Mock()
    mock_process.stdin = Mock()
    mock_process.stdout = Mock()
    mock_process.poll.return_value = None

    # Mock an error response
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32600, "message": "Invalid request"},
    }
    mock_process.stdout.readline.return_value = (
        (json.dumps(response) + "\n").encode()
    )

    client = MCPClient(mock_process)

    with pytest.raises(RuntimeError, match="MCP error: Invalid request"):
        client.call_tool("query-get_data", {"prompt": "test"})


def test_mcp_client_close():
    """Test MCP client closes process properly."""
    mock_process = Mock()
    mock_process.poll.return_value = None

    client = MCPClient(mock_process)
    client.close()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()


def test_create_mcp_tool_query_data():
    """Test creating MCP query tool."""
    mock_client = Mock()
    mock_client.call_tool.return_value = {
        "context": "Test context",
        "citations": [{"source": "test.txt", "chunk": 0}],
    }

    tool = create_mcp_tool_query_data(mock_client, verbose=False)
    result = tool("test prompt")

    assert "Test context" in result
    assert "test.txt" in result
    mock_client.call_tool.assert_called_once_with(
        "query-get_data", {"prompt": "test prompt", "max_results": 5}
    )


def test_create_mcp_tool_query_data_verbose(capsys):
    """Test creating MCP query tool with verbose mode."""
    mock_client = Mock()
    mock_client.call_tool.return_value = {
        "context": "Test context",
        "citations": [{"source": "test.txt", "chunk": 0}],
    }

    tool = create_mcp_tool_query_data(mock_client, verbose=True)
    result = tool("test prompt", max_results=3)

    assert "Test context" in result
    assert "test.txt" in result

    # Check verbose output
    captured = capsys.readouterr()
    assert "💭 [Agent Decision: Using query_data tool]" in captured.out
    assert "Query: test prompt" in captured.out
    assert "Max results: 3" in captured.out

    mock_client.call_tool.assert_called_once_with(
        "query-get_data", {"prompt": "test prompt", "max_results": 3}
    )


def test_mcp_client_no_truncation():
    """Test that MCP client handles long responses without truncation."""
    mock_process = Mock()
    mock_process.stdin = Mock()
    mock_process.stdout = Mock()
    mock_process.poll.return_value = None

    # Create a response with long text that would be prone to truncation
    # Repeat text multiple times to simulate a long response
    base_text = "Graufell ist ein mächtischer Mann, der einst ein Bauer aus Helmark war. "
    repeat_count = 10
    long_text = base_text * repeat_count
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"context": long_text, "citations": []},
    }
    mock_process.stdout.readline.return_value = (
        (json.dumps(response) + "\n").encode()
    )

    client = MCPClient(mock_process)
    result = client.call_tool("query-get_data", {"prompt": "test"})

    # Verify the full text is returned without truncation
    assert result["context"] == long_text
    assert result["context"].startswith("Graufell ist ein mächtischer Mann")


def test_mcp_client_utf8_encoding():
    """Test that MCP client handles UTF-8 characters correctly."""
    mock_process = Mock()
    mock_process.stdin = Mock()
    mock_process.stdout = Mock()
    mock_process.poll.return_value = None

    # Create a response with German text containing umlauts
    text_with_umlauts = "Graufell ist ein mächtischer Mann mit Fähigkeiten über Äther."
    response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"context": text_with_umlauts, "citations": []},
    }
    # Use the same JSON serialization as production code
    mock_process.stdout.readline.return_value = (
        (json.dumps(response) + "\n").encode('utf-8')
    )

    client = MCPClient(mock_process)
    result = client.call_tool("query-get_data", {"prompt": "test"})

    # Verify UTF-8 characters are preserved
    assert result["context"] == text_with_umlauts
    assert "mächtischer" in result["context"]
    assert "Fähigkeiten" in result["context"]
    assert "Äther" in result["context"]


def test_chat_cli_main_no_databases(test_config, capsys):
    """Test chat CLI main with no databases."""
    test_args = ["mcp-rag-cli"]

    with patch.object(sys, "argv", test_args):
        with patch("mcp_agent_rag.chat_cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config.get.side_effect = lambda key, default=None: {
                "log_level": "INFO",
            }.get(key, default)
            mock_config_class.return_value = mock_config

            with patch("mcp_agent_rag.chat_cli.Config.get_default_data_dir") as mock_dir:
                mock_dir.return_value = test_config.config_path.parent

                with patch("mcp_agent_rag.chat_cli.setup_logger"):
                    with patch("mcp_agent_rag.chat_cli.DatabaseManager") as mock_db_manager_class:
                        mock_db_manager = Mock()
                        mock_db_manager.list_databases.return_value = {}
                        mock_db_manager_class.return_value = mock_db_manager

                        with pytest.raises(SystemExit) as exc_info:
                            main()

                        assert exc_info.value.code == 1
                        captured = capsys.readouterr()
                        assert "No databases found" in captured.out


def test_chat_cli_main_with_custom_log_file(test_config):
    """Test chat CLI main with custom log file argument."""
    test_args = ["mcp-rag-cli", "--log", "/tmp/custom-chat.log"]

    with patch.object(sys, "argv", test_args):
        with patch("mcp_agent_rag.chat_cli.setup_logger") as mock_setup_logger:
            with patch("mcp_agent_rag.chat_cli.DatabaseManager") as MockDBManager:
                mock_db_manager = Mock()
                mock_db_manager.list_databases.return_value = {}
                MockDBManager.return_value = mock_db_manager

                with pytest.raises(SystemExit):
                    main()

                # Verify setup_logger was called with the custom log file
                mock_setup_logger.assert_called_once()
                call_args = mock_setup_logger.call_args
                assert call_args[1]["log_file"] == "/tmp/custom-chat.log"


def test_chat_cli_main_with_default_log_file(test_config):
    """Test chat CLI main with default log file."""
    test_args = ["mcp-rag-cli"]

    with patch.object(sys, "argv", test_args):
        with patch("mcp_agent_rag.chat_cli.setup_logger") as mock_setup_logger:
            with patch("mcp_agent_rag.chat_cli.DatabaseManager") as MockDBManager:
                mock_db_manager = Mock()
                mock_db_manager.list_databases.return_value = {}
                MockDBManager.return_value = mock_db_manager

                with pytest.raises(SystemExit):
                    main()

                # Verify setup_logger was called with the default log file
                mock_setup_logger.assert_called_once()
                call_args = mock_setup_logger.call_args
                assert "mcp-rag-cli.log" in call_args[1]["log_file"]


def test_chat_cli_main_with_verbose_flag(test_config, capsys):
    """Test chat CLI main with --verbose flag."""
    test_args = ["mcp-rag-cli", "--verbose"]

    with patch.object(sys, "argv", test_args):
        with patch("mcp_agent_rag.chat_cli.setup_logger"):
            with patch("mcp_agent_rag.chat_cli.DatabaseManager") as MockDBManager:
                # Mock database manager to return a test database
                mock_db_manager = Mock()
                test_db_info = {"test_db": {"doc_count": 5, "description": "Test DB"}}
                mock_db_manager.list_databases.return_value = test_db_info
                MockDBManager.return_value = mock_db_manager

                # Mock start_mcp_server to avoid starting actual server
                with patch("mcp_agent_rag.chat_cli.start_mcp_server") as mock_start_server:
                    mock_process = Mock()
                    mock_start_server.return_value = mock_process

                    # Mock Agent to avoid needing Ollama
                    with patch("mcp_agent_rag.chat_cli.Agent") as MockAgent:
                        # Simulate user selecting database and then quitting
                        with patch("builtins.input", side_effect=["1", "quit"]):
                            main()

                            # Verify MCPClient was created with verbose=True
                            # by checking the output contains verbose mode message
                            captured = capsys.readouterr()
                            assert "🔍 Verbose mode enabled" in captured.out

                            # Verify Agent was created with debug_mode and show_tool_calls
                            MockAgent.assert_called_once()
                            call_kwargs = MockAgent.call_args[1]
                            assert call_kwargs.get("debug_mode") is True
                            assert call_kwargs.get("show_tool_calls") is True
