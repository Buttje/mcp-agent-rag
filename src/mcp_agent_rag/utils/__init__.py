"""Utilities for MCP-RAG."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from mcp_agent_rag.utils.debug_logger import (
    DebugLogger,
    get_debug_logger,
    log_if_debug,
    setup_debug_logger,
)
from mcp_agent_rag.utils.gpu_utils import (
    GPUInfo,
    check_pytorch_installed,
    detect_gpu,
    get_gpu_installation_instructions,
    recommend_pytorch_installation,
)


def setup_logger(
    name: str = "mcp-rag",
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler - use stderr to avoid interfering with stdout (used for JSON-RPC)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Also configure the mcp_agent_rag parent logger to ensure all module loggers
    # inherit the console and file handlers
    app_logger = logging.getLogger("mcp_agent_rag")
    app_logger.setLevel(getattr(logging, level.upper()))
    app_logger.handlers.clear()

    # Add same handlers to app logger
    app_logger.addHandler(console_handler)
    if log_file:
        app_logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "mcp-rag") -> logging.Logger:
    """Get or create logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
