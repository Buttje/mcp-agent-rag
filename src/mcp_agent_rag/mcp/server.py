"""MCP Server implementation."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.mcp.agent import AgenticRAG
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class MCPServer:
    """Model Context Protocol server."""

    def __init__(self, config: Config, active_databases: List[str]):
        """Initialize MCP server.

        Args:
            config: Configuration instance
            active_databases: List of active database names
        """
        self.config = config
        self.active_databases = active_databases
        self.db_manager = DatabaseManager(config)
        self.agent = None

        # Load active databases
        self.loaded_databases = self.db_manager.load_multiple_databases(active_databases)
        if len(self.loaded_databases) != len(active_databases):
            missing = set(active_databases) - set(self.loaded_databases.keys())
            raise ValueError(f"Failed to load databases: {', '.join(missing)}")

        # Initialize agentic RAG
        self.agent = AgenticRAG(config, self.loaded_databases)

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
        client_version = params.get("protocolVersion", "2025-11-25")
        client_capabilities = params.get("capabilities", {})
        client_info = params.get("clientInfo", {})
        
        logger.info(f"Initialize request from client: {client_info.get('name', 'unknown')}")
        logger.info(f"Client protocol version: {client_version}")
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
                "listChanged": False,  # We don't support prompts yet
            },
            "logging": {},  # We support logging
        }
        
        return {
            "protocolVersion": "2025-11-25",  # The protocol version we support
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
            JSON-RPC response
        """
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            # Handle initialize - required by MCP spec
            if method == "initialize":
                result = self._initialize(params)
            # Handle initialized notification - no response needed
            elif method == "notifications/initialized":
                logger.info("Client initialized successfully")
                return None  # Notifications don't get responses
            elif method == "database/create":
                result = self._create_database(params)
            elif method == "database/add":
                result = self._add_documents(params)
            elif method == "database/list":
                result = self._list_databases(params)
            elif method == "query/get_data":
                result = self._query_data(params)
            elif method == "getDatabases":
                result = self._get_databases(params)
            elif method == "getInformationFor":
                result = self._get_information_for(params)
            elif method == "getInformationForDB":
                result = self._get_information_for_db(params)
            elif method == "resources/list":
                result = self._list_resources(params)
            elif method == "tools/list":
                result = self._list_tools(params)
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                return self._error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return self._error_response(
                request.get("id"), -32603, f"Internal error: {str(e)}"
            )

    def _create_database(self, params: Dict) -> Dict:
        """Handle database/create."""
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
        """Handle database/add."""
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
        """Handle database/list."""
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
        """Handle query/get_data."""
        prompt = params.get("prompt")
        if not prompt:
            raise ValueError("Missing required parameter: prompt")

        max_results = params.get("max_results", 5)

        # Use agentic RAG to get context
        context = self.agent.get_context(prompt, max_results)

        return {
            "prompt": prompt,
            "context": context["text"],
            "citations": context["citations"],
            "databases_searched": context["databases_searched"],
        }

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
        
        Args:
            params: Dictionary with 'prompt' parameter
            
        Returns:
            Dictionary with context, citations, and databases searched
        """
        prompt = params.get("prompt")
        if not prompt:
            raise ValueError("Missing required parameter: prompt")

        max_results = params.get("max_results", 5)

        # Use agentic RAG to get context from all active databases
        context = self.agent.get_context(prompt, max_results)

        return {
            "prompt": prompt,
            "context": context["text"],
            "citations": context["citations"],
            "databases_searched": context["databases_searched"],
        }

    def _get_information_for_db(self, params: Dict) -> Dict:
        """Handle getInformationForDB - returns information from specific database.
        
        Args:
            params: Dictionary with 'prompt' and 'database_name' parameters
            
        Returns:
            Dictionary with context, citations, and database searched
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
        
        db = self.loaded_databases[database_name]
        
        # Generate query embedding
        query_embedding = self.agent.embedder.embed_single(prompt)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return {
                "prompt": prompt,
                "database": database_name,
                "context": "",
                "citations": [],
            }
        
        # Search the specific database
        results = db.search(query_embedding, k=max_results)
        
        # Process results
        context_parts = []
        citations = []
        seen_sources = set()
        total_length = 0
        max_context_length = self.config.get("max_context_length", 4000)
        
        for distance, metadata in results:
            source = metadata.get("source", "")
            chunk_num = metadata.get("chunk_num", 0)
            source_key = f"{source}:{chunk_num}"
            
            # Skip duplicates
            if source_key in seen_sources:
                continue
            
            chunk_text = metadata.get("text", "")
            if not chunk_text:
                continue
            
            # Check if adding this would exceed limit
            if total_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            citations.append({
                "source": source,
                "chunk": chunk_num,
                "database": database_name,
            })
            seen_sources.add(source_key)
            total_length += len(chunk_text)
        
        # Compose final context
        context_text = "\n\n".join(context_parts)
        
        return {
            "prompt": prompt,
            "database": database_name,
            "context": context_text,
            "citations": citations,
        }

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

    def _list_tools(self, params: Dict) -> Dict:
        """Handle tools/list."""
        return {
            "tools": [
                {
                    "name": "database/create",
                    "description": "Create a new database with a unique name",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Database name"},
                            "description": {"type": "string", "description": "Database description"},
                        },
                        "required": ["name"],
                    },
                },
                {
                    "name": "database/add",
                    "description": "Add documents to an existing database",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "database_name": {"type": "string"},
                            "path": {"type": "string"},
                            "url": {"type": "string"},
                            "glob": {"type": "string"},
                            "recursive": {"type": "boolean"},
                            "skip_existing": {"type": "boolean"},
                        },
                        "required": ["database_name"],
                    },
                },
                {
                    "name": "database/list",
                    "description": "List all databases",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "query/get_data",
                    "description": "Retrieve context for a user's prompt from active databases",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "max_results": {"type": "integer", "default": 5},
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "getDatabases",
                    "description": "Get list of activated databases in the MCP RAG server",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
                {
                    "name": "getInformationFor",
                    "description": "Returns information by scanning through all activated databases",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string", 
                                "description": "The query/prompt to search for"
                            },
                            "max_results": {
                                "type": "integer", 
                                "default": 5,
                                "description": "Maximum number of results per database"
                            },
                        },
                        "required": ["prompt"],
                    },
                },
                {
                    "name": "getInformationForDB",
                    "description": "Returns information by scanning just the named database",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The query/prompt to search for"
                            },
                            "database_name": {
                                "type": "string",
                                "description": "Name of the database to search in"
                            },
                            "max_results": {
                                "type": "integer",
                                "default": 5,
                                "description": "Maximum number of results"
                            },
                        },
                        "required": ["prompt", "database_name"],
                    },
                },
            ]
        }

    def _call_tool(self, params: Dict) -> Dict:
        """Handle tools/call."""
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name == "database/create":
            return self._create_database(arguments)
        elif name == "database/add":
            return self._add_documents(arguments)
        elif name == "database/list":
            return self._list_databases(arguments)
        elif name == "query/get_data":
            return self._query_data(arguments)
        elif name == "getDatabases":
            return self._get_databases(arguments)
        elif name == "getInformationFor":
            return self._get_information_for(arguments)
        elif name == "getInformationForDB":
            return self._get_information_for_db(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

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
