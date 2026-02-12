#!/usr/bin/env python3
"""Integration test for debug logging in mcp-rag-cli.

This script tests that debug logging works correctly by simulating
the initialization and tool call scenarios.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_agent_rag.chat_cli import MCPClient
from mcp_agent_rag.utils.debug_logger import setup_debug_logger, get_debug_logger


def test_debug_logging_integration():
    """Test that debug logging works end-to-end."""
    
    print("=" * 70)
    print("Debug Logging Integration Test")
    print("=" * 70)
    print()
    
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        print(f"Using temporary log directory: {log_dir}")
        
        # Setup debug logger
        print("Setting up debug logger...")
        debug_logger = setup_debug_logger(enabled=True, log_dir=log_dir)
        print(f"✓ Debug logger initialized: {debug_logger.enabled}")
        print()
        
        # Verify logger is accessible
        retrieved_logger = get_debug_logger()
        assert retrieved_logger is debug_logger, "Failed to retrieve debug logger"
        print("✓ Debug logger accessible via get_debug_logger()")
        print()
        
        # Test logging user prompt
        print("Testing user prompt logging...")
        test_prompt = "What is the capital of France?"
        debug_logger.log_user_prompt(test_prompt)
        print(f"✓ Logged user prompt: {test_prompt}")
        print()
        
        # Test logging agent response
        print("Testing agent response logging...")
        test_response = "The capital of France is Paris."
        debug_logger.log("mcp.agent", "Agent response:", {"response": test_response})
        print(f"✓ Logged agent response: {test_response}")
        print()
        
        # Test MCP client logging
        print("Testing MCP client tool call logging...")
        
        # Create mock process
        mock_process = Mock()
        mock_process.stdin = Mock()
        mock_process.stdout = Mock()
        
        # Mock response
        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "context": "Paris is the capital of France.",
                "citations": [{"source": "geography.pdf", "chunk": 1}],
                "average_confidence": 0.95
            }
        }
        response_line = json.dumps(response_data) + "\n"
        mock_process.stdout.readline.return_value.decode.return_value = response_line
        
        # Create client
        client = MCPClient(mock_process, verbose=False)
        
        # Call tool (this should log both request and response)
        print("Calling MCP tool...")
        result = client.call_tool("query-get_data", {"prompt": "capital of France"})
        print(f"✓ Tool call completed with {len(result.get('context', ''))} chars of context")
        print()
        
        # Verify log file was created
        log_files = list(log_dir.glob("debug_*.log"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
        log_file = log_files[0]
        print(f"✓ Log file created: {log_file.name}")
        print()
        
        # Read and verify log content
        print("Verifying log content...")
        log_content = log_file.read_text()
        
        checks = [
            ("Debug logging initialized", "Initialization message"),
            ("User prompt", "User prompt logging"),
            ("What is the capital of France", "User prompt content"),
            ("Agent response", "Agent response logging"),
            ("capital of France is Paris", "Agent response content"),
            ("mcp.client", "MCP client module"),
            ("Calling MCP tool", "Tool call request"),
            ("query-get_data", "Tool name"),
            ("Received response from MCP tool", "Tool call response"),
            ("context_preview", "Response content"),
        ]
        
        print()
        for text, description in checks:
            if text in log_content:
                print(f"  ✓ {description}: '{text}' found")
            else:
                print(f"  ✗ {description}: '{text}' NOT found")
                print(f"     Log preview: {log_content[:500]}")
                raise AssertionError(f"Missing expected text: {text}")
        
        print()
        print("=" * 70)
        print("✅ All tests passed! Debug logging is working correctly.")
        print("=" * 70)
        print()
        
        # Show log file preview
        print("Log file preview:")
        print("-" * 70)
        lines = log_content.split("\n")[:20]
        for line in lines:
            print(line)
        if len(log_content.split("\n")) > 20:
            print("... (truncated)")
        print("-" * 70)


if __name__ == "__main__":
    test_debug_logging_integration()
