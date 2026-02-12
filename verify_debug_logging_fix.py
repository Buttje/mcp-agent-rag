#!/usr/bin/env python3
"""Manual verification script for debug logging fix.

This script verifies that:
1. Only the MCP server creates debug logs
2. The chat CLI does not create debug logs
3. The server logs all required information
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Check that imports work correctly."""
    print("Checking imports...")
    
    # Chat CLI should NOT import debug logger functions
    chat_cli_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "chat_cli.py"
    content = chat_cli_source.read_text()
    
    # Verify debug logger is NOT imported
    assert "setup_debug_logger" not in content, "setup_debug_logger should not be imported in chat_cli"
    assert "get_debug_logger" not in content, "get_debug_logger should not be imported in chat_cli"
    print("✓ chat_cli.py does not import debug logger functions")
    
    # CLI should still import debug logger functions
    cli_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "cli.py"
    content = cli_source.read_text()
    
    # Verify debug logger IS imported
    assert "setup_debug_logger" in content, "setup_debug_logger should be imported in cli"
    assert "get_debug_logger" in content or "setup_debug_logger" in content, "debug logger functions should be imported in cli"
    print("✓ cli.py imports debug logger functions")

def check_logging_calls():
    """Check that logging calls are removed from chat_cli."""
    print("\nChecking logging calls...")
    
    chat_cli_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "chat_cli.py"
    content = chat_cli_source.read_text()
    
    # Verify no debug logger calls in chat_cli
    assert "debug_logger.log_user_prompt" not in content, "log_user_prompt should not be called in chat_cli"
    assert "debug_logger.log(" not in content, "debug_logger.log should not be called in chat_cli"
    print("✓ chat_cli.py does not call debug logger methods")
    
    # Verify debug logger calls still exist in server
    server_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "mcp" / "server.py"
    content = server_source.read_text()
    
    assert "debug_logger.log_json_rpc_request" in content, "log_json_rpc_request should be called in server"
    assert "debug_logger.log_json_rpc_response" in content, "log_json_rpc_response should be called in server"
    assert "debug_logger.log_final_response" in content, "log_final_response should be called in server"
    print("✓ server.py calls debug logger methods")
    
    # Verify debug logger calls in enhanced_rag
    rag_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "mcp" / "enhanced_rag.py"
    content = rag_source.read_text()
    
    assert "debug_logger.log_user_prompt" in content, "log_user_prompt should be called in enhanced_rag"
    assert "debug_logger.log_rag_query" in content, "log_rag_query should be called in enhanced_rag"
    assert "debug_logger.log_rag_results" in content, "log_rag_results should be called in enhanced_rag"
    assert "debug_logger.log_llm_request" in content, "log_llm_request should be called in enhanced_rag"
    assert "debug_logger.log_llm_response" in content, "log_llm_response should be called in enhanced_rag"
    assert "debug_logger.log_augmented_prompt" in content, "log_augmented_prompt should be called in enhanced_rag"
    print("✓ enhanced_rag.py calls debug logger methods")

def check_setup_calls():
    """Check that setup_debug_logger is only called in cli, not chat_cli."""
    print("\nChecking debug logger setup...")
    
    chat_cli_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "chat_cli.py"
    content = chat_cli_source.read_text()
    
    # Verify setup is NOT called in chat_cli
    assert "setup_debug_logger(enabled=True)" not in content, "setup_debug_logger should not be called in chat_cli"
    print("✓ chat_cli.py does not call setup_debug_logger")
    
    # Verify setup IS called in cli
    cli_source = Path(__file__).parent / "src" / "mcp_agent_rag" / "cli.py"
    content = cli_source.read_text()
    
    assert "setup_debug_logger(enabled=True)" in content, "setup_debug_logger should be called in cli"
    print("✓ cli.py calls setup_debug_logger when --debug is used")

def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Debug Logging Fix - Manual Verification")
    print("=" * 70)
    print()
    
    try:
        check_imports()
        check_logging_calls()
        check_setup_calls()
        
        print("\n" + "=" * 70)
        print("✓ All verification checks passed!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  - chat_cli.py no longer creates debug logs")
        print("  - Only the MCP server (cli.py) creates debug logs")
        print("  - All required logging is present in server code")
        print()
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Verification failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
