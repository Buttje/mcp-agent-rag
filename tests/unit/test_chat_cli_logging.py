"""Tests for debug logging in chat_cli."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mcp_agent_rag.chat_cli import MCPClient
from mcp_agent_rag.utils.debug_logger import setup_debug_logger


class TestMCPClientLogging:
    """Test debug logging in MCPClient."""

    def test_call_tool_logs_request(self):
        """Test that tool calls are logged to debug logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Setup debug logger (needed for side effect)
            _ = setup_debug_logger(enabled=True, log_dir=log_dir)
            
            # Create mock process
            mock_process = Mock(spec=subprocess.Popen)
            mock_process.stdin = MagicMock()
            mock_process.stdout = MagicMock()
            
            # Mock the response
            response_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": "Test context",
                    "citations": []
                }
            }
            response_line = json.dumps(response_data) + "\n"
            mock_process.stdout.readline.return_value.decode.return_value = response_line
            
            # Create client
            client = MCPClient(mock_process, verbose=False)
            
            # Call tool
            result = client.call_tool("test-tool", {"arg": "value"})
            
            # Verify log file was created and contains expected content
            log_files = list(log_dir.glob("debug_*.log"))
            assert len(log_files) == 1
            
            log_content = log_files[0].read_text()
            
            # Check request was logged
            assert "mcp.client" in log_content
            assert "Calling MCP tool 'test-tool'" in log_content
            assert '"arg": "value"' in log_content
            
    def test_call_tool_logs_response(self):
        """Test that tool responses are logged to debug logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Setup debug logger (needed for side effect)
            _ = setup_debug_logger(enabled=True, log_dir=log_dir)
            
            # Create mock process
            mock_process = Mock(spec=subprocess.Popen)
            mock_process.stdin = MagicMock()
            mock_process.stdout = MagicMock()
            
            # Mock the response with context
            test_context = "This is a test context with some data"
            response_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "context": test_context,
                    "citations": [{"source": "test.pdf", "chunk": 1}],
                    "average_confidence": 0.95,
                    "databases_searched": ["testdb"]
                }
            }
            response_line = json.dumps(response_data) + "\n"
            mock_process.stdout.readline.return_value.decode.return_value = response_line
            
            # Create client
            client = MCPClient(mock_process, verbose=False)
            
            # Call tool
            result = client.call_tool("test-tool", {"arg": "value"})
            
            # Verify log contains response
            log_files = list(log_dir.glob("debug_*.log"))
            log_content = log_files[0].read_text()
            
            # Check response was logged
            assert "Received response from MCP tool 'test-tool'" in log_content
            assert "context_preview" in log_content
            assert "context_length" in log_content
            assert "citations" in log_content
            assert "average_confidence" in log_content
            
    def test_call_tool_without_debug_logger(self):
        """Test that tool calls work without debug logger enabled."""
        # No debug logger setup
        
        # Create mock process
        mock_process = Mock(spec=subprocess.Popen)
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        
        # Mock simple response
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"text": "Test response"}
        }
        response_line = json.dumps(response_data) + "\n"
        mock_process.stdout.readline.return_value.decode.return_value = response_line
        
        # Create client
        client = MCPClient(mock_process, verbose=False)
        
        # Call tool - should work without error
        result = client.call_tool("test-tool", {"arg": "value"})
        
        assert result == {"text": "Test response"}
        
    def test_call_tool_logs_error(self):
        """Test that tool errors are properly handled with debug logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Setup debug logger (needed for side effect)
            _ = setup_debug_logger(enabled=True, log_dir=log_dir)
            
            # Create mock process
            mock_process = Mock(spec=subprocess.Popen)
            mock_process.stdin = MagicMock()
            mock_process.stdout = MagicMock()
            
            # Mock error response
            error_response = {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32000, "message": "Test error"}
            }
            response_line = json.dumps(error_response) + "\n"
            mock_process.stdout.readline.return_value.decode.return_value = response_line
            
            # Create client
            client = MCPClient(mock_process, verbose=False)
            
            # Call tool - should raise error
            with pytest.raises(RuntimeError, match="MCP error: Test error"):
                client.call_tool("test-tool", {"arg": "value"})
            
            # Verify request was still logged
            log_files = list(log_dir.glob("debug_*.log"))
            log_content = log_files[0].read_text()
            
            assert "Calling MCP tool" in log_content


class TestMainFunctionLogging:
    """Test debug logging in main function."""
    
    @patch('mcp_agent_rag.chat_cli.setup_debug_logger')
    @patch('mcp_agent_rag.chat_cli.Config')
    @patch('mcp_agent_rag.chat_cli.DatabaseManager')
    @patch('builtins.input', side_effect=['quit'])  # Exit immediately
    @patch('mcp_agent_rag.chat_cli.start_mcp_server')
    @patch('mcp_agent_rag.chat_cli.Agent')
    def test_main_enables_debug_logger_with_flag(
        self, mock_agent, mock_start_server, mock_input, 
        mock_db_manager, mock_config, mock_setup_debug
    ):
        """Test that main() sets up debug logger when --debug flag is used."""
        from mcp_agent_rag.chat_cli import main
        
        # Mock config
        mock_config_instance = Mock()
        mock_config_instance.get.return_value = "INFO"
        mock_config_instance.get_default_data_dir.return_value = Path("/tmp/test")
        mock_config.return_value = mock_config_instance
        mock_config.get_default_data_dir.return_value = Path("/tmp/test")
        
        # Mock database manager with databases
        mock_db_instance = Mock()
        mock_db_instance.list_databases.return_value = {
            "testdb": {"doc_count": 1, "description": "Test"}
        }
        mock_db_manager.return_value = mock_db_instance
        
        # Mock MCP server
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_start_server.return_value = mock_process
        
        # Mock agent
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        # Run main with --debug
        with patch('sys.argv', ['chat_cli.py', '--debug']):
            try:
                main()
            except SystemExit:
                pass  # Expected on quit
        
        # Verify debug logger was set up
        mock_setup_debug.assert_called_once_with(enabled=True)
        
    @patch('mcp_agent_rag.chat_cli.setup_debug_logger')
    @patch('mcp_agent_rag.chat_cli.Config')
    @patch('mcp_agent_rag.chat_cli.DatabaseManager')
    @patch('builtins.input', side_effect=['quit'])
    @patch('mcp_agent_rag.chat_cli.start_mcp_server')
    @patch('mcp_agent_rag.chat_cli.Agent')
    def test_main_no_debug_logger_without_flag(
        self, mock_agent, mock_start_server, mock_input,
        mock_db_manager, mock_config, mock_setup_debug
    ):
        """Test that main() does not set up debug logger without --debug flag."""
        from mcp_agent_rag.chat_cli import main
        
        # Mock config
        mock_config_instance = Mock()
        mock_config_instance.get.return_value = "INFO"
        mock_config_instance.get_default_data_dir.return_value = Path("/tmp/test")
        mock_config.return_value = mock_config_instance
        mock_config.get_default_data_dir.return_value = Path("/tmp/test")
        
        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.list_databases.return_value = {
            "testdb": {"doc_count": 1, "description": "Test"}
        }
        mock_db_manager.return_value = mock_db_instance
        
        # Mock MCP server
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_start_server.return_value = mock_process
        
        # Mock agent
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        
        # Run main without --debug
        with patch('sys.argv', ['chat_cli.py']):
            try:
                main()
            except SystemExit:
                pass
        
        # Verify debug logger was not set up
        mock_setup_debug.assert_not_called()


class TestUserPromptAndResponseLogging:
    """Test logging of user prompts and agent responses."""
    
    def test_user_prompt_logged(self):
        """Test that user prompts are logged to debug logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Setup debug logger
            debug_logger = setup_debug_logger(enabled=True, log_dir=log_dir)
            
            # Log user prompt
            debug_logger.log_user_prompt("What is the capital of France?")
            
            # Verify log
            log_files = list(log_dir.glob("debug_*.log"))
            log_content = log_files[0].read_text()
            
            assert "User prompt" in log_content
            assert "What is the capital of France?" in log_content
            
    def test_agent_response_logged(self):
        """Test that agent responses are logged to debug logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            
            # Setup debug logger
            debug_logger = setup_debug_logger(enabled=True, log_dir=log_dir)
            
            # Log agent response
            debug_logger.log(
                "mcp.agent",
                "Agent response:",
                {"response": "The capital of France is Paris."}
            )
            
            # Verify log
            log_files = list(log_dir.glob("debug_*.log"))
            log_content = log_files[0].read_text()
            
            assert "mcp.agent" in log_content
            assert "Agent response" in log_content
            assert "capital of France is Paris" in log_content
