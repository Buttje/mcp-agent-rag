"""MCP Server implementation."""

import json
import sys
from typing import Any, Dict, List, Optional

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

            if method == "database/create":
                result = self._create_database(params)
            elif method == "database/add":
                result = self._add_documents(params)
            elif method == "database/list":
                result = self._list_databases(params)
            elif method == "query/get_data":
                result = self._query_data(params)
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
                    print(json.dumps(response), flush=True)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = self._error_response(None, -32700, "Parse error")
                    print(json.dumps(error_response), flush=True)

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
