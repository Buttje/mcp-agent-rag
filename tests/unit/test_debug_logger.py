"""Tests for debug logger."""

import tempfile
from pathlib import Path

import pytest

from mcp_agent_rag.utils.debug_logger import DebugLogger, setup_debug_logger, get_debug_logger


def test_debug_logger_disabled():
    """Test debug logger when disabled."""
    logger = DebugLogger(enabled=False)
    
    # Should not create logger when disabled
    assert logger.enabled is False
    assert logger.logger is None
    
    # Log calls should be no-op
    logger.log("test", "message")  # Should not raise


def test_debug_logger_enabled():
    """Test debug logger when enabled."""
    import re
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        assert logger.enabled is True
        assert logger.logger is not None
        
        # Check log file was created
        log_files = list(log_dir.glob("debug_*.log"))
        assert len(log_files) == 1
        
        # Test logging
        logger.log("test_module", "Test message")
        
        # Verify log content
        log_content = log_files[0].read_text()
        assert "test_module" in log_content
        assert "Test message" in log_content
        
        # Verify timestamp format [YYYY-MM-DD HH:MM:SS]
        timestamp_pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]'
        assert re.search(timestamp_pattern, log_content), "Timestamp format not found in log"


def test_debug_logger_with_json_data():
    """Test debug logger with JSON data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        # Log with JSON data
        test_data = {"key": "value", "number": 42}
        logger.log("test", "Test with data", test_data)
        
        # Verify log content
        log_files = list(log_dir.glob("debug_*.log"))
        log_content = log_files[0].read_text()
        
        assert "Test with data" in log_content
        assert '"key": "value"' in log_content
        assert '"number": 42' in log_content


def test_debug_logger_json_rpc():
    """Test JSON-RPC logging methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        # Test request logging
        request = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        logger.log_json_rpc_request(request)
        
        # Test response logging
        response = {"jsonrpc": "2.0", "id": 1, "result": {}}
        logger.log_json_rpc_response(response)
        
        # Verify log content
        log_files = list(log_dir.glob("debug_*.log"))
        log_content = log_files[0].read_text()
        
        assert "mcp.server" in log_content
        assert "Received JSON-RPC request" in log_content
        assert "Sending JSON-RPC response" in log_content


def test_debug_logger_rag_operations():
    """Test RAG operation logging methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        # Test user prompt logging
        logger.log_user_prompt("What is the capital of France?")
        
        # Test RAG query logging
        logger.log_rag_query("mydb", "capital France", [0.1, 0.2, 0.3])
        
        # Test RAG results logging
        results = [
            {"source": "doc.pdf", "chunk_num": 1, "confidence": 0.95, "text": "Paris is the capital"}
        ]
        logger.log_rag_results("mydb", results, filtered_count=2)
        
        # Test final response logging
        citations = [{"source": "doc.pdf", "chunk": 1}]
        logger.log_final_response("Paris", citations)
        
        # Verify log content
        log_files = list(log_dir.glob("debug_*.log"))
        log_content = log_files[0].read_text()
        
        assert "User prompt" in log_content
        assert "capital of France" in log_content
        assert "Querying database" in log_content
        assert "mydb" in log_content
        assert "Retrieved" in log_content
        assert "Final response" in log_content


def test_setup_and_get_debug_logger():
    """Test global debug logger setup and retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        # Setup global logger
        logger = setup_debug_logger(enabled=True, log_dir=log_dir)
        assert logger is not None
        assert logger.enabled is True
        
        # Retrieve global logger
        retrieved = get_debug_logger()
        assert retrieved is logger
        
        # Test with disabled logger
        disabled_logger = setup_debug_logger(enabled=False)
        assert disabled_logger.enabled is False
        
        retrieved_disabled = get_debug_logger()
        assert retrieved_disabled is disabled_logger


def test_debug_logger_augmented_prompt():
    """Test augmented prompt logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        logger.log_augmented_prompt(
            prompt="What is the capital?",
            context="Paris is the capital of France."
        )
        
        log_files = list(log_dir.glob("debug_*.log"))
        log_content = log_files[0].read_text()
        
        assert "Augmented prompt" in log_content
        assert "original_prompt" in log_content
        assert "retrieved_context" in log_content


def test_debug_logger_thinking_step():
    """Test thinking step logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        logger.log_thinking_step("analyze", "Analyzing the user query")
        
        log_files = list(log_dir.glob("debug_*.log"))
        log_content = log_files[0].read_text()
        
        assert "agent.thinking" in log_content
        assert "Thinking step 'analyze'" in log_content
        assert "Analyzing the user query" in log_content


def test_debug_logger_llm_interactions():
    """Test LLM interaction logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = DebugLogger(enabled=True, log_dir=log_dir)
        
        # Test LLM request
        logger.log_llm_request("gpt-4", "Translate this", context="Some context")
        
        # Test LLM response
        logger.log_llm_response("gpt-4", "Translation result")
        
        log_files = list(log_dir.glob("debug_*.log"))
        log_content = log_files[0].read_text()
        
        assert "llm.request" in log_content
        assert "llm.response" in log_content
        assert "gpt-4" in log_content
