"""Tests for logging utilities."""

import logging
import sys
from pathlib import Path

import pytest

from mcp_agent_rag.utils import get_logger, setup_logger


def test_setup_logger_basic(temp_dir):
    """Test basic logger setup."""
    log_file = temp_dir / "test.log"
    logger = setup_logger("test", str(log_file), "INFO")

    assert logger.name == "test"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2  # Console and file


def test_setup_logger_level(temp_dir):
    """Test logger level configuration."""
    log_file = temp_dir / "test.log"
    logger = setup_logger("test", str(log_file), "DEBUG")

    assert logger.level == logging.DEBUG


def test_setup_logger_without_file():
    """Test logger without file handler."""
    logger = setup_logger("test", None, "INFO")

    assert len(logger.handlers) == 1  # Only console handler


def test_setup_logger_creates_directory(temp_dir):
    """Test that logger creates log directory."""
    log_dir = temp_dir / "logs"
    log_file = log_dir / "test.log"

    assert not log_dir.exists()

    logger = setup_logger("test", str(log_file), "INFO")

    assert log_dir.exists()
    logger.info("test message")
    assert log_file.exists()


def test_get_logger():
    """Test getting logger."""
    logger = get_logger("test")
    assert logger.name == "test"


def test_logger_writes_to_file(temp_dir):
    """Test that logger writes to file."""
    log_file = temp_dir / "test.log"
    logger = setup_logger("test", str(log_file), "INFO")

    logger.info("Test message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content


def test_logger_rotation(temp_dir):
    """Test log rotation."""
    log_file = temp_dir / "test.log"
    logger = setup_logger("test", str(log_file), "INFO", max_bytes=100, backup_count=2)

    # Write enough to trigger rotation
    for i in range(50):
        logger.info(f"Test message {i}" * 10)

    # Check that backup files are created
    log_dir = temp_dir
    log_files = list(log_dir.glob("test.log*"))
    assert len(log_files) > 1  # Original + backups


def test_console_handler_uses_stderr():
    """Test that console handler writes to stderr, not stdout.
    
    This is critical for MCP server operation where stdout is used
    for JSON-RPC communication and must not be polluted by log messages.
    """
    logger = setup_logger("test", None, "INFO")
    
    # Find the console handler
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            console_handler = handler
            break
    
    assert console_handler is not None, "Console handler not found"
    assert console_handler.stream == sys.stderr, "Console handler should write to stderr, not stdout"
