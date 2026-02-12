"""MCP Server implementation."""

import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.mcp.enhanced_rag import AgenticRAG
from mcp_agent_rag.rag.ollama_utils import get_model_capabilities
from mcp_agent_rag.utils import get_debug_logger, get_logger

logger = get_logger(__name__)

# MCP Protocol Constants
MCP_PROTOCOL_VERSION_2025 = "2025-11-25"
MCP_PROTOCOL_VERSION_2024 = "2024-11-05"
# Default protocol version
MCP_PROTOCOL_VERSION = MCP_PROTOCOL_VERSION_2025

# MCP Tool names - single source of truth
MCP_TOOL_NAMES = ["getDatabases", "getInformationFor", "getInformationForDB"]


class MCPServer:
    """Model Context Protocol server."""

    def __init__(
        self,
        config: Config,
        active_databases: List[str],
        protocol_version: str = MCP_PROTOCOL_VERSION_2025,
    ):
        """Initialize MCP server.

        Args:
            config: Configuration instance
            active_databases: List of active database names
            protocol_version: MCP protocol version to use (2024-11-05 or 2025-11-25)
        """
        self.config = config
        self.active_databases = active_databases
        self.db_manager = DatabaseManager(config)
        self.agent = None
        self._single_db_agents = {}  # Cache for single-database agents
        
        # Set protocol version (validate it's one of the supported versions)
        if protocol_version not in [MCP_PROTOCOL_VERSION_2024, MCP_PROTOCOL_VERSION_2025]:
            raise ValueError(
                f"Unsupported protocol version: {protocol_version}. "
                f"Supported versions: {MCP_PROTOCOL_VERSION_2024}, {MCP_PROTOCOL_VERSION_2025}"
            )
        self.protocol_version = protocol_version

        # Load active databases
        self.loaded_databases = self.db_manager.load_multiple_databases(active_databases)
        if len(self.loaded_databases) != len(active_databases):
            missing = set(active_databases) - set(self.loaded_databases.keys())
            raise ValueError(f"Failed to load databases: {', '.join(missing)}")

        # Build combined prefix from all active database prefixes
        # Note: Duplicates are removed while preserving order
        prefixes = []
        seen = set()
        for db_name in active_databases:
            db_config = config.get_database(db_name)
            if db_config and db_config.get("prefix"):
                prefix = db_config["prefix"]
                if prefix not in seen:
                    prefixes.append(prefix)
                    seen.add(prefix)
        
        # Create combined prefix (e.g., "A1_B1_A2_" or "" if no prefixes)
        self.tool_prefix = "_".join(prefixes) + "_" if prefixes else ""
        
        logger.info(f"Initialized server with tool prefix: '{self.tool_prefix}'")

        # Initialize agentic RAG with LLM-based iterative retrieval
        # This creates internal database query tools that the LLM can call iteratively
        self.agent = AgenticRAG(config, self.loaded_databases)
        
        # Check if the generative model supports required capabilities (tools and thinking)
        generative_model = config.get("generative_model", "mistral:7b-instruct")
        ollama_host = config.get("ollama_host", "http://localhost:11434")
        
        capabilities, error = get_model_capabilities(generative_model, ollama_host)
        if error:
            logger.warning(f"Could not verify model capabilities for '{generative_model}': {error}")
            logger.warning("Server will continue but LLM-based agentic retrieval may not work correctly")
        else:
            # Check for tools capability (required for iterative retrieval)
            if "tools" not in capabilities:
                logger.warning(
                    f"Model '{generative_model}' does not support 'tools' capability. "
                    f"LLM-based agentic retrieval requires a model with tool calling support. "
                    f"Supported capabilities: {capabilities}"
                )
            else:
                logger.info(f"Model '{generative_model}' supports tools capability ✓")
            
            # Check for thinking capability (optional but recommended)
            if "thinking" in capabilities:
                logger.info(f"Model '{generative_model}' supports thinking capability ✓")
            else:
                logger.info(
                    f"Model '{generative_model}' does not support 'thinking' capability. "
                    f"Consider using a model with thinking support for better results."
                )

    def _initialize(self, params: Dict) -> Dict:
        """Handle initialize request - required by MCP specification.
        
        The initialize method is the first interaction between client and server.
        It negotiates protocol version and capabilities.
        
        Args:
            params: Dictionary containing:
                - protocolVersion: Client's supported protocol version
                - capabilities: Client's capabilities (optional)
                - clientInfo: Information about the client (optional)
        
        Returns:
            Dictionary with server's protocol version, capabilities, and info
        """
        client_version = params.get("protocolVersion", self.protocol_version)
        client_capabilities = params.get("capabilities", {})
        client_info = params.get("clientInfo", {})
        
        logger.info(f"Initialize request from client: {client_info.get('name', 'unknown')}")
        logger.info(f"Client protocol version: {client_version}")
        logger.info(f"Server protocol version: {self.protocol_version}")
        logger.info(f"Client capabilities: {client_capabilities}")
        
        # Server capabilities based on what we support
        server_capabilities = {
            "resources": {
                "subscribe": False,  # We don't support resource subscriptions yet
                "listChanged": False,
            },
            "tools": {
                "listChanged": False,  # Tool list is static
            },
            "prompts": {
                "listChanged": False,  # Prompts list is static (empty for now)
            },
            "logging": {},  # We support logging
        }
        
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": server_capabilities,
            "serverInfo": {
                "name": "mcp-agent-rag",
                "version": "1.0.0",
            },
            "instructions": f"MCP RAG Server with {len(self.active_databases)} active database(s): {', '.join(self.active_databases)}",
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response (or None for notifications)
        """
        # Debug logging: log incoming request
        debug_logger = get_debug_logger()
        if debug_logger:
            debug_logger.log_json_rpc_request(request)
        
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            # Handle initialize - required by MCP spec
            if method == "initialize":
                result = self._initialize(params)
            # Handle notifications - no response needed
            elif method and method.startswith("notifications/"):
                notification_type = method.split("/", 1)[1]
                logger.info(f"Received notification: {notification_type}")
                if notification_type == "initialized":
                    logger.info("Client initialized successfully")
                return None  # Notifications don't get responses
            # Check if method is a prefixed tool call (e.g., "Prefix_getInformationFor")
            # Route through _call_tool to ensure MCP-compliant response format
            elif (self.tool_prefix and method and method.startswith(self.tool_prefix) and
                  method[len(self.tool_prefix):] in MCP_TOOL_NAMES):
                # Route through _call_tool for MCP-compliant response
                result = self._call_tool({"name": method, "arguments": params})
            elif method == "getDatabases":
                result = self._get_databases(params)
            elif method == "getInformationFor":
                result = self._get_information_for(params)
            elif method == "getInformationForDB":
                result = self._get_information_for_db(params)
            elif method == "resources/list":
                result = self._list_resources(params)
            elif method == "resources/templates/list":
                result = self._list_resource_templates(params)
            elif method == "prompts/list":
                result = self._list_prompts(params)
            elif method == "logging/setLevel":
                result = self._set_log_level(params)
            elif method == "tools/list":
                result = self._list_tools(params)
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                return self._error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
            
            # Debug logging: log outgoing response
            if debug_logger:
                debug_logger.log_json_rpc_response(response)
            
            return response

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            error_response = self._error_response(
                request.get("id"), -32603, f"Internal error: {str(e)}"
            )
            
            # Debug logging: log error response
            if debug_logger:
                debug_logger.log_json_rpc_response(error_response)
            
            return error_response

    def _create_database(self, params: Dict) -> Dict:
        """Handle database-create."""
        name = params.get("name")
        if not name:
            raise ValueError("Missing required parameter: name")

        description = params.get("description", "")
        success = self.db_manager.create_database(name, description)

        if not success:
            raise ValueError(f"Database '{name}' already exists")

        db_path = self.config.get_database_path(name)
        return {
            "success": True,
            "message": f"Created database '{name}' at {db_path}",
            "name": name,
            "path": str(db_path),
        }

    def _add_documents(self, params: Dict) -> Dict:
        """Handle database-add."""
        database_name = params.get("database_name")
        if not database_name:
            raise ValueError("Missing required parameter: database_name")

        if not self.config.database_exists(database_name):
            raise ValueError(f"Database '{database_name}' does not exist")

        path = params.get("path")
        url = params.get("url")
        glob_pattern = params.get("glob")
        recursive = params.get("recursive", False)
        skip_existing = params.get("skip_existing", False)

        stats = self.db_manager.add_documents(
            database_name=database_name,
            path=path,
            url=url,
            glob_pattern=glob_pattern,
            recursive=recursive,
            skip_existing=skip_existing,
        )

        return {
            "success": True,
            "database": database_name,
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "failed": stats["failed"],
        }

    def _list_databases(self, params: Dict) -> Dict:
        """Handle database-list."""
        databases = self.db_manager.list_databases()
        result = []

        for name, info in databases.items():
            result.append({
                "name": name,
                "description": info.get("description", ""),
                "doc_count": info.get("doc_count", 0),
                "last_updated": info.get("last_updated", ""),
                "path": info.get("path", ""),
            })

        return {"databases": result}

    def _query_data(self, params: Dict) -> Dict:
        """Handle query-get_data.
        
        Uses LLM-based agentic RAG flow where the LLM iteratively queries
        databases using internal tools until it has enough information.
        """
        prompt = params.get("prompt")
        if not prompt:
            raise ValueError("Missing required parameter: prompt")

        max_results = params.get("max_results", 5)

        # Use LLM-based agentic RAG to get context
        # The LLM will create internal tool calls to query databases iteratively
        context = self.agent.get_context(prompt, max_results)

        result = {
            "prompt": prompt,
            "context": context["text"],
            "citations": context["citations"],
            "databases_searched": context["databases_searched"],
            "average_confidence": context.get("average_confidence", 0.0),
            "iterations": context.get("iterations", 0),
        }
        
        # Debug logging: log final response
        debug_logger = get_debug_logger()
        if debug_logger:
            debug_logger.log_final_response(
                response=context["text"],
                citations=context["citations"]
            )
        
        return result

    def _get_databases(self, params: Dict) -> Dict:
        """Handle getDatabases - returns list of activated databases.
        
        Returns:
            Dictionary containing list of active database names and their info
        """
        databases = []
        for db_name in self.active_databases:
            db_info = self.config.get_database(db_name)
            if db_info:
                databases.append({
                    "name": db_name,
                    "description": db_info.get("description", ""),
                    "doc_count": db_info.get("doc_count", 0),
                    "last_updated": db_info.get("last_updated", ""),
                    "path": db_info.get("path", ""),
                })
        
        return {
            "databases": databases,
            "count": len(databases),
        }

    def _get_information_for(self, params: Dict) -> Dict:
        """Handle getInformationFor - returns information from all activated databases.
        
        Uses LLM-based agentic RAG flow where the LLM iteratively queries all
        activated databases using internal tools until it has enough information.
        
        Args:
            params: Dictionary with 'prompt' parameter
            
        Returns:
            Dictionary with context, citations, databases searched, and confidence scores
        """
        prompt = params.get("prompt")
        if not prompt:
            raise ValueError("Missing required parameter: prompt")

        max_results = params.get("max_results", 5)

        # Use LLM-based agentic RAG to get context from all active databases
        # The LLM will create internal tool calls to query databases iteratively
        context = self.agent.get_context(prompt, max_results)

        result = {
            "prompt": prompt,
            "context": context["text"],
            "citations": context["citations"],
            "databases_searched": context["databases_searched"],
            "average_confidence": context.get("average_confidence", 0.0),
            "min_confidence_threshold": self.agent.min_confidence,
            "iterations": context.get("iterations", 0),
        }
        
        # Debug logging: log final response
        debug_logger = get_debug_logger()
        if debug_logger:
            debug_logger.log_final_response(
                response=context["text"],
                citations=context["citations"]
            )
        
        return result

    def _get_information_for_db(self, params: Dict) -> Dict:
        """Handle getInformationForDB - returns information from specific database.
        
        Uses LLM-based agentic RAG flow where the LLM iteratively queries the 
        specified database using internal tools until it has enough information.
        
        Args:
            params: Dictionary with 'prompt' and 'database_name' parameters
            
        Returns:
            Dictionary with context, citations, database searched, and confidence scores
        """
        prompt = params.get("prompt")
        database_name = params.get("database_name")
        
        if not prompt:
            raise ValueError("Missing required parameter: prompt")
        if not database_name:
            raise ValueError("Missing required parameter: database_name")
        
        # Check if database is in active databases
        if database_name not in self.active_databases:
            raise ValueError(
                f"Database '{database_name}' is not in active databases: "
                f"{', '.join(self.active_databases)}"
            )
        
        max_results = params.get("max_results", 5)
        
        # Get the specific database
        if database_name not in self.loaded_databases:
            raise ValueError(f"Database '{database_name}' is not loaded")
        
        # Get or create a cached single-database agent for efficiency
        # This avoids creating a new agent instance on every query
        if database_name not in self._single_db_agents:
            single_db = {database_name: self.loaded_databases[database_name]}
            self._single_db_agents[database_name] = AgenticRAG(self.config, single_db)
        
        single_db_agent = self._single_db_agents[database_name]
        
        # Use LLM-based agentic RAG to get context from the specific database
        # The LLM will create internal tool calls to query the database iteratively
        context = single_db_agent.get_context(prompt, max_results)
        
        result = {
            "prompt": prompt,
            "database": database_name,
            "context": context["text"],
            "citations": context["citations"],
            "average_confidence": context.get("average_confidence", 0.0),
            "min_confidence_threshold": single_db_agent.min_confidence,
            "iterations": context.get("iterations", 0),
        }
        
        # Debug logging: log final response
        debug_logger = get_debug_logger()
        if debug_logger:
            debug_logger.log_final_response(
                response=context["text"],
                citations=context["citations"]
            )
        
        return result

    def _list_resources(self, params: Dict) -> Dict:
        """Handle resources/list."""
        resources = []
        for name in self.active_databases:
            db_info = self.config.get_database(name)
            if db_info:
                resources.append({
                    "uri": f"database://{name}",
                    "name": name,
                    "description": db_info.get("description", ""),
                    "mimeType": "application/x-faiss-index",
                })
        return {"resources": resources}

    def _list_resource_templates(self, params: Dict) -> Dict:
        """Handle resources/templates/list.
        
        Returns resource templates that allow clients to discover
        parameterized resources according to MCP specification.
        
        Args:
            params: Optional parameters (e.g., cursor for pagination)
            
        Returns:
            Dictionary containing list of resource templates
        """
        # Define resource templates for dynamic resource discovery
        # These follow the MCP specification for resource templates
        resource_templates = []
        
        # Template for querying specific databases
        resource_templates.append({
            "uriTemplate": "database://{database_name}/query",
            "name": "Database Query Template",
            "description": "Template for querying a specific database by name",
            "mimeType": "application/json",
        })
        
        # Template for accessing database information
        resource_templates.append({
            "uriTemplate": "database://{database_name}/info",
            "name": "Database Info Template", 
            "description": "Template for accessing metadata and information about a specific database",
            "mimeType": "application/json",
        })
        
        return {"resourceTemplates": resource_templates}

    def _list_prompts(self, params: Dict) -> Dict:
        """Handle prompts/list.
        
        Returns list of available prompts according to MCP specification.
        Prompts are predefined, user-focused templates that servers expose to clients.
        
        Args:
            params: Optional parameters (e.g., cursor for pagination)
            
        Returns:
            Dictionary containing list of prompts (currently empty as this server
            doesn't define custom prompts yet, but this satisfies the MCP protocol)
        """
        # For now, return an empty list of prompts
        # In the future, this could be extended to include custom prompt templates
        # for common RAG operations
        return {"prompts": []}

    def _set_log_level(self, params: Dict) -> Dict:
        """Handle logging/setLevel.
        
        Allows clients to set the minimum severity level of log messages.
        Uses syslog RFC 5424 severity levels.
        
        Args:
            params: Dictionary containing 'level' parameter (one of: debug, info, 
                    notice, warning, error, critical, alert, emergency)
            
        Returns:
            Empty dictionary on success
            
        Raises:
            ValueError: If level is invalid
        """
        level = params.get("level")
        if not level:
            raise ValueError("Missing required parameter: level")
        
        # Valid log levels per MCP spec (syslog RFC 5424)
        valid_levels = {
            "debug", "info", "notice", "warning", 
            "error", "critical", "alert", "emergency"
        }
        
        if level not in valid_levels:
            raise ValueError(
                f"Invalid log level: {level}. "
                f"Must be one of: {', '.join(sorted(valid_levels))}"
            )
        
        # Map MCP log levels to Python logging levels
        level_map = {
            "debug": "DEBUG",
            "info": "INFO",
            "notice": "INFO",  # Python doesn't have NOTICE, use INFO
            "warning": "WARNING",
            "error": "ERROR",
            "critical": "CRITICAL",
            "alert": "CRITICAL",  # Python doesn't have ALERT, use CRITICAL
            "emergency": "CRITICAL",  # Python doesn't have EMERGENCY, use CRITICAL
        }
        
        python_level = level_map[level]
        
        # Update logger level
        logging.getLogger().setLevel(getattr(logging, python_level))
        logger.info(f"Log level set to: {level} (Python: {python_level})")
        
        # Return empty result to indicate success
        return {}


    def _list_tools(self, params: Dict) -> Dict:
        """Handle tools/list - dynamically generates tool descriptions based on active databases.
        
        Tool descriptions are optimized based on database metadata including
        prefix, description, and document types.
        """
        # Collect database information for tool descriptions
        db_descriptions = []
        db_names = []
        for db_name in self.active_databases:
            db_info = self.config.get_database(db_name)
            if db_info:
                desc = db_info.get("description", "")
                prefix = db_info.get("prefix", "")
                doc_count = db_info.get("doc_count", 0)
                
                # Format prefix display
                prefix_display = prefix if prefix else "no prefix"
                
                # Build a descriptive string
                if desc and doc_count:
                    db_descriptions.append(f"{db_name} ({prefix_display}): {desc} - {doc_count} documents")
                elif desc:
                    db_descriptions.append(f"{db_name} ({prefix_display}): {desc}")
                else:
                    db_descriptions.append(f"{db_name} ({prefix_display})")
                
                db_names.append(db_name)
        
        # Create detailed database list for descriptions
        db_list_text = "\n".join([f"  - {desc}" for desc in db_descriptions]) if db_descriptions else "  (no databases)"
        db_names_text = ", ".join(db_names) if db_names else "none"
        
        # Define base tools with dynamic descriptions
        base_tools = [
            {
                "name": "getDatabases",
                "description": (
                    f"Get list of activated databases in the MCP RAG server. "
                    f"Currently active: {db_names_text}. "
                    f"Returns database names, descriptions, document counts, and metadata."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "getInformationFor",
                "description": (
                    f"Returns information by scanning through all activated RAG databases "
                    f"using vector similarity search. Retrieves context from high-confidence matches "
                    f"(>85% confidence threshold). Results include citations with confidence scores. "
                    f"\n\nActive databases:\n{db_list_text}"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string", 
                            "description": "The query/prompt to search for in all databases"
                        },
                        "max_results": {
                            "type": "integer", 
                            "default": 5,
                            "description": "Maximum number of results per database (default: 5)"
                        },
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "getInformationForDB",
                "description": (
                    f"Returns information by scanning just the named database using vector similarity search. "
                    f"Retrieves context from high-confidence matches (>85% confidence threshold). "
                    f"Results include citations with confidence scores. "
                    f"\n\nAvailable databases:\n{db_list_text}"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The query/prompt to search for in the specified database"
                        },
                        "database_name": {
                            "type": "string",
                            "description": f"Name of the database to search in. Available: {db_names_text}"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of results (default: 5)"
                        },
                    },
                    "required": ["prompt", "database_name"],
                },
            },
        ]
        
        # Apply prefix to tool names if configured
        tools = []
        for tool in base_tools:
            tool_copy = tool.copy()
            tool_copy["name"] = f"{self.tool_prefix}{tool['name']}"
            tools.append(tool_copy)
        
        return {"tools": tools}

    def _call_tool(self, params: Dict) -> Dict:
        """Handle tools/call.
        
        Returns MCP-compliant response with content array format:
        {
            "content": [{"type": "text", "text": "<json_string>"}],
            "isError": false
        }
        
        All responses are JSON-formatted, including errors.
        """
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        # Strip prefix from tool name if present
        base_name = name
        if self.tool_prefix and name.startswith(self.tool_prefix):
            base_name = name[len(self.tool_prefix):]

        try:
            # Call the appropriate tool method
            if base_name == "getDatabases":
                result = self._get_databases(arguments)
            elif base_name == "getInformationFor":
                result = self._get_information_for(arguments)
            elif base_name == "getInformationForDB":
                result = self._get_information_for_db(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            # Wrap result in MCP-compliant format with JSON serialization
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ],
                "isError": False
            }
        except Exception as e:
            # Return error in MCP-compliant format as JSON
            error_result = {
                "error": str(e),
                "tool": name,
                "arguments": arguments,
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(error_result)
                    }
                ],
                "isError": True
            }

    def _error_response(self, request_id, code: int, message: str) -> Dict:
        """Create JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def run_stdio(self) -> None:
        """Run server with stdio transport."""
        logger.info(f"MCP server starting with stdio transport")
        logger.info(f"Active databases: {', '.join(self.active_databases)}")

        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.strip())
                    response = self.handle_request(request)
                    
                    # Only send response if it's not None (notifications don't get responses)
                    if response is not None:
                        print(json.dumps(response), flush=True)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = self._error_response(None, -32700, "Parse error")
                    print(json.dumps(error_response), flush=True)

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)

    def run_http(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Run server with HTTP transport.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        logger.info(f"MCP server starting with HTTP transport on {host}:{port}")
        logger.info(f"Active databases: {', '.join(self.active_databases)}")

        # Create request handler class with access to server instance
        server_instance = self

        class MCPHTTPHandler(BaseHTTPRequestHandler):
            """HTTP request handler for MCP server."""

            def do_POST(self):
                """Handle POST requests."""
                try:
                    # Read request body
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length)
                    
                    # Parse JSON-RPC request
                    request = json.loads(body.decode('utf-8'))
                    
                    # Handle request
                    response = server_instance.handle_request(request)
                    
                    # Only send response if not None (notifications don't get responses)
                    if response is not None:
                        # Send response
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                    else:
                        # For notifications, send 204 No Content
                        self.send_response(204)
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = server_instance._error_response(
                        None, -32700, "Parse error"
                    )
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(error_response).encode('utf-8'))
                    
                except Exception as e:
                    logger.error(f"Error handling request: {e}", exc_info=True)
                    error_response = server_instance._error_response(
                        None, -32603, f"Internal error: {str(e)}"
                    )
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(error_response).encode('utf-8'))

            def do_OPTIONS(self):
                """Handle OPTIONS requests for CORS."""
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def do_GET(self):
                """Handle GET requests."""
                # Health check endpoint
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    health = {
                        "status": "ok",
                        "active_databases": server_instance.active_databases,
                    }
                    self.wfile.write(json.dumps(health).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                """Override to use our logger."""
                logger.info(f"{self.address_string()} - {format % args}")

        # Create and start HTTP server
        try:
            httpd = HTTPServer((host, port), MCPHTTPHandler)
            logger.info(f"HTTP server listening on {host}:{port}")
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)

    def run_sse(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Run server with SSE (Server-Sent Events) transport.
        
        SSE is deprecated in favor of streamable HTTP but provided for backwards compatibility.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        logger.info(f"MCP server starting with SSE transport on {host}:{port}")
        logger.info(f"Active databases: {', '.join(self.active_databases)}")
        logger.warning("SSE transport is deprecated. Consider using HTTP transport instead.")

        server_instance = self

        class MCPSSEHandler(BaseHTTPRequestHandler):
            """SSE request handler for MCP server."""

            def do_POST(self):
                """Handle POST requests for sending messages."""
                try:
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length)
                    request = json.loads(body.decode('utf-8'))
                    
                    # Handle request
                    response = server_instance.handle_request(request)
                    
                    # Only send response if not None (notifications don't get responses)
                    if response is not None:
                        # Send response as HTTP 200 with JSON
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(json.dumps(response).encode('utf-8'))
                    else:
                        # For notifications, send 204 No Content
                        self.send_response(204)
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                    
                except Exception as e:
                    logger.error(f"Error handling request: {e}", exc_info=True)
                    error_response = server_instance._error_response(
                        None, -32603, f"Internal error: {str(e)}"
                    )
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(error_response).encode('utf-8'))

            def do_GET(self):
                """Handle GET requests for SSE stream."""
                # SSE endpoint
                if self.path == '/sse':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'keep-alive')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    # Send initial connection message
                    event_data = json.dumps({
                        "type": "connection",
                        "message": "Connected to MCP server",
                        "active_databases": server_instance.active_databases,
                    })
                    self.wfile.write(f"data: {event_data}\n\n".encode('utf-8'))
                    self.wfile.flush()
                    
                    # Keep connection alive
                    try:
                        keepalive_interval = 30  # seconds
                        while True:
                            time.sleep(keepalive_interval)
                            self.wfile.write(": keepalive\n\n".encode('utf-8'))
                            self.wfile.flush()
                    except Exception:
                        pass
                        
                # Health check endpoint
                elif self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    health = {
                        "status": "ok",
                        "active_databases": server_instance.active_databases,
                    }
                    self.wfile.write(json.dumps(health).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_OPTIONS(self):
                """Handle OPTIONS requests for CORS."""
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def log_message(self, format, *args):
                """Override to use our logger."""
                logger.info(f"{self.address_string()} - {format % args}")

        # Create and start SSE server
        try:
            httpd = HTTPServer((host, port), MCPSSEHandler)
            logger.info(f"SSE server listening on {host}:{port}")
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
