"""Debug logger for detailed MCP-RAG operation logging.

This module provides a specialized debug logging system that creates timestamped
log files with detailed information about MCP server operations, RAG queries,
and agent thinking processes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class DebugLogger:
    """Debug logger for MCP-RAG operations.
    
    Creates timestamped debug log files in ~/.mcp-agent-rag/debug/ directory
    with detailed information about:
    - JSON-RPC messages received/sent
    - User prompts and augmented prompts
    - RAG database queries and results
    - Agent thinking steps
    - LLM interactions
    """

    def __init__(self, enabled: bool = False, log_dir: Optional[Path] = None):
        """Initialize debug logger.
        
        Args:
            enabled: Whether debug logging is enabled
            log_dir: Directory for debug logs (default: ~/.mcp-agent-rag/debug)
        """
        self.enabled = enabled
        
        if not enabled:
            self.logger = None
            return
        
        # Set up debug log directory
        if log_dir is None:
            from mcp_agent_rag.config import Config
            log_dir = Config.get_default_data_dir() / "debug"
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"debug_{timestamp}.log"
        
        # Set up logger
        self.logger = logging.getLogger(f"mcp-rag-debug-{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Don't propagate to parent loggers
        
        # Remove any existing handlers
        self.logger.handlers.clear()
        
        # File handler with detailed format
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Format: [timestamp] [module] message
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Log initialization
        self.log("debug_logger", f"Debug logging initialized: {log_file}")
    
    def log(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a debug message with optional data.
        
        Args:
            module: Name of the module logging the message
            message: Human-readable message
            data: Optional dictionary of data to log as JSON
        """
        if not self.enabled or not self.logger:
            return
        
        # Create logger for this specific module
        module_logger = logging.LoggerAdapter(self.logger, {'name': module})
        
        # Log the message
        if data:
            # Pretty-print JSON data
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            full_message = f"{message}\n{json_str}"
        else:
            full_message = message
        
        module_logger.debug(full_message)
    
    def log_json_rpc_request(self, request: Dict[str, Any]):
        """Log JSON-RPC request received from MCP host.
        
        Args:
            request: JSON-RPC request dictionary
        """
        self.log(
            "mcp.server",
            "Received JSON-RPC request from MCP host:",
            request
        )
    
    def log_json_rpc_response(self, response: Dict[str, Any]):
        """Log JSON-RPC response sent to MCP host.
        
        Args:
            response: JSON-RPC response dictionary
        """
        self.log(
            "mcp.server",
            "Sending JSON-RPC response to MCP host:",
            response
        )
    
    def log_user_prompt(self, prompt: str):
        """Log user prompt received by agent.
        
        Args:
            prompt: User's prompt/query
        """
        self.log(
            "mcp.agent",
            f"User prompt: {prompt}"
        )
    
    def log_augmented_prompt(self, prompt: str, context: str):
        """Log augmented prompt sent to LLM.
        
        Args:
            prompt: Original user prompt
            context: Retrieved context added to prompt
        """
        self.log(
            "mcp.agent",
            "Augmented prompt sent to LLM:",
            {
                "original_prompt": prompt,
                "retrieved_context": context
            }
        )
    
    def log_rag_query(self, database: str, query: str, embedding_preview: Optional[list] = None):
        """Log RAG database query.
        
        Args:
            database: Name of the database being queried
            query: Query text
            embedding_preview: Optional preview of query embedding (first 5 dimensions)
        """
        data = {
            "database": database,
            "query": query
        }
        if embedding_preview:
            data["embedding_preview"] = embedding_preview[:5]
        
        self.log(
            "rag.retrieval",
            f"Querying database '{database}':",
            data
        )
    
    def log_rag_results(self, database: str, results: list, filtered_count: int = 0):
        """Log RAG database results.
        
        Args:
            database: Name of the database
            results: List of result dictionaries
            filtered_count: Number of results filtered out by confidence threshold
        """
        # Simplify results for logging (avoid logging full text content)
        simplified_results = []
        for r in results:
            simplified_results.append({
                "source": r.get("source"),
                "chunk": r.get("chunk_num"),
                "confidence": r.get("confidence"),
                "text_preview": r.get("text", "")[:100] + "..." if r.get("text") else ""
            })
        
        data = {
            "database": database,
            "result_count": len(results),
            "filtered_count": filtered_count,
            "results": simplified_results
        }
        
        self.log(
            "rag.retrieval",
            f"Retrieved {len(results)} results from '{database}':",
            data
        )
    
    def log_thinking_step(self, step: str, description: str):
        """Log agent thinking step.
        
        Args:
            step: Step name/identifier
            description: Description of the thinking process
        """
        self.log(
            "agent.thinking",
            f"Thinking step '{step}': {description}"
        )
    
    def log_llm_request(self, model: str, prompt: str, context: Optional[str] = None):
        """Log request to LLM.
        
        Args:
            model: LLM model name
            prompt: Prompt sent to LLM
            context: Optional context added to prompt
        """
        data = {
            "model": model,
            "prompt": prompt
        }
        if context:
            data["context"] = context
        
        self.log(
            "llm.request",
            f"Sending request to LLM ({model}):",
            data
        )
    
    def log_llm_response(self, model: str, response: str):
        """Log response from LLM.
        
        Args:
            model: LLM model name
            response: Response from LLM
        """
        self.log(
            "llm.response",
            f"Received response from LLM ({model}):",
            {"response": response}
        )
    
    def log_final_response(self, response: str, citations: Optional[list] = None):
        """Log final response sent to user.
        
        Args:
            response: Final response text
            citations: Optional list of citations
        """
        data = {"response": response}
        if citations:
            data["citations"] = citations
        
        self.log(
            "mcp.agent",
            "Final response:",
            data
        )


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def setup_debug_logger(enabled: bool = False, log_dir: Optional[Path] = None) -> DebugLogger:
    """Set up global debug logger.
    
    Args:
        enabled: Whether debug logging is enabled
        log_dir: Directory for debug logs
    
    Returns:
        DebugLogger instance
    """
    global _debug_logger
    _debug_logger = DebugLogger(enabled=enabled, log_dir=log_dir)
    return _debug_logger


def get_debug_logger() -> Optional[DebugLogger]:
    """Get global debug logger instance.
    
    Returns:
        DebugLogger instance or None if not initialized
    """
    return _debug_logger
