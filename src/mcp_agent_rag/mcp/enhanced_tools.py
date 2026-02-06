"""Enhanced MCP tools for better UX and workflow support."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class EnhancedMCPTools:
    """Enhanced MCP tools for workflow support.
    
    Implements additional tools:
    - ingest(path|url, db, options) -> job_id
    - status(job_id) -> status
    - search(query, db, top_k, filters) -> results
    - summarize(source_id) -> summary
    - delete(source_id|path|url) -> status
    """

    def __init__(self, config: Config, db_manager: DatabaseManager):
        """Initialize enhanced MCP tools.

        Args:
            config: Configuration instance
            db_manager: Database manager instance
        """
        self.config = config
        self.db_manager = db_manager
        self.jobs = {}  # Track async jobs
        self.job_counter = 0

    def ingest(self, params: Dict) -> Dict:
        """Ingest documents asynchronously.

        Args:
            params: Dictionary with:
                - database: Database name
                - path: File/directory path (optional)
                - url: URL to download (optional)
                - recursive: Recurse into directories (optional)
                - glob: Glob pattern (optional)
                - options: Additional options (optional)

        Returns:
            Dictionary with job_id
        """
        database = params.get("database")
        if not database:
            raise ValueError("Missing required parameter: database")

        if not self.config.database_exists(database):
            raise ValueError(f"Database '{database}' does not exist")

        path = params.get("path")
        url = params.get("url")
        recursive = params.get("recursive", False)
        glob_pattern = params.get("glob")
        options = params.get("options", {})

        # Create job ID
        self.job_counter += 1
        job_id = f"ingest_{self.job_counter}"

        # Store job info
        self.jobs[job_id] = {
            "id": job_id,
            "type": "ingest",
            "status": "running",
            "database": database,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        # For now, run synchronously (could be made async with threading)
        try:
            import time

            self.jobs[job_id]["started_at"] = time.time()

            stats = self.db_manager.add_documents(
                database_name=database,
                path=path,
                url=url,
                glob_pattern=glob_pattern,
                recursive=recursive,
                skip_existing=options.get("skip_existing", False),
            )

            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = time.time()
            self.jobs[job_id]["result"] = stats

        except Exception as e:
            logger.error(f"Ingest job {job_id} failed: {e}")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)

        return {
            "job_id": job_id,
            "status": self.jobs[job_id]["status"],
        }

    def status(self, params: Dict) -> Dict:
        """Get job status.

        Args:
            params: Dictionary with job_id

        Returns:
            Dictionary with job status
        """
        job_id = params.get("job_id")
        if not job_id:
            raise ValueError("Missing required parameter: job_id")

        if job_id not in self.jobs:
            raise ValueError(f"Job not found: {job_id}")

        job = self.jobs[job_id]

        result = {
            "job_id": job_id,
            "type": job["type"],
            "status": job["status"],
            "database": job.get("database"),
        }

        if job["started_at"]:
            result["started_at"] = job["started_at"]

        if job["completed_at"]:
            result["completed_at"] = job["completed_at"]
            result["duration_seconds"] = job["completed_at"] - job["started_at"]

        if job["result"]:
            result["result"] = job["result"]

        if job["error"]:
            result["error"] = job["error"]

        return result

    def search(self, params: Dict) -> Dict:
        """Search with advanced options and filters.

        Args:
            params: Dictionary with:
                - query: Search query
                - database: Database name (optional, searches all if not specified)
                - top_k: Number of results (default 10)
                - filters: Metadata filters (optional)

        Returns:
            Dictionary with search results
        """
        query = params.get("query")
        if not query:
            raise ValueError("Missing required parameter: query")

        database = params.get("database")
        top_k = params.get("top_k", 10)
        filters = params.get("filters", {})

        # Load database(s)
        if database:
            if not self.config.database_exists(database):
                raise ValueError(f"Database '{database}' does not exist")
            databases = {database: self.db_manager.load_database(database)}
        else:
            # Search all databases
            databases = self.db_manager.load_multiple_databases(
                list(self.config.list_databases().keys())
            )

        # Get embedder
        embedder = self.db_manager.embedder

        # Generate query embedding
        query_embedding = embedder.embed_single(query)
        if not query_embedding:
            raise ValueError("Failed to generate query embedding")

        # Search each database
        all_results = []
        for db_name, vector_db in databases.items():
            if vector_db:
                results = vector_db.search(query_embedding, k=top_k, filters=filters)
                for distance, metadata in results:
                    metadata["database"] = db_name
                    metadata["score"] = 1.0 / (1.0 + distance)  # Convert distance to score
                    all_results.append(metadata)

        # Sort by score and limit
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        all_results = all_results[:top_k]

        return {
            "query": query,
            "results": all_results,
            "count": len(all_results),
            "filters": filters,
        }

    def summarize(self, params: Dict) -> Dict:
        """Summarize a document or source.

        Args:
            params: Dictionary with source identifier

        Returns:
            Dictionary with summary
        """
        source = params.get("source")
        if not source:
            raise ValueError("Missing required parameter: source")

        # This would use the generative model to create a summary
        # For now, return a placeholder
        return {
            "source": source,
            "summary": "Summary functionality to be implemented with generative model",
        }

    def delete(self, params: Dict) -> Dict:
        """Delete documents from database.

        Args:
            params: Dictionary with:
                - database: Database name
                - source: Source identifier (file path or URL)

        Returns:
            Dictionary with deletion status
        """
        database = params.get("database")
        source = params.get("source")

        if not database:
            raise ValueError("Missing required parameter: database")
        if not source:
            raise ValueError("Missing required parameter: source")

        if not self.config.database_exists(database):
            raise ValueError(f"Database '{database}' does not exist")

        # This would implement deletion using manifest
        # For now, return a placeholder
        return {
            "database": database,
            "source": source,
            "status": "Delete functionality to be fully implemented with manifest",
        }

    def get_tool_definitions(self, prefix: str = "") -> List[Dict]:
        """Get MCP tool definitions for enhanced tools.

        Args:
            prefix: Tool name prefix

        Returns:
            List of tool definition dicts
        """
        tools = [
            {
                "name": f"{prefix}ingest",
                "description": "Ingest documents into a database (async)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Target database name",
                        },
                        "path": {
                            "type": "string",
                            "description": "File or directory path (optional)",
                        },
                        "url": {
                            "type": "string",
                            "description": "URL to download (optional)",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Recurse into directories",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Glob pattern for file filtering",
                        },
                        "options": {
                            "type": "object",
                            "description": "Additional options",
                        },
                    },
                    "required": ["database"],
                },
            },
            {
                "name": f"{prefix}status",
                "description": "Get status of an async job",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Job identifier",
                        },
                    },
                    "required": ["job_id"],
                },
            },
            {
                "name": f"{prefix}search",
                "description": "Search with advanced options and metadata filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "database": {
                            "type": "string",
                            "description": "Database to search (optional, searches all if not specified)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 10)",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Metadata filters (source, database, etc.)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": f"{prefix}summarize",
                "description": "Summarize a document or source",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source identifier",
                        },
                    },
                    "required": ["source"],
                },
            },
            {
                "name": f"{prefix}delete",
                "description": "Delete documents from database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database name",
                        },
                        "source": {
                            "type": "string",
                            "description": "Source identifier (file path or URL)",
                        },
                    },
                    "required": ["database", "source"],
                },
            },
        ]

        return tools
