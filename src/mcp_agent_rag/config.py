"""Configuration management for MCP-RAG."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class Config:
    """Manages MCP-RAG configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to config file. Defaults to ~/.mcp-agent-rag/config.json
        """
        if config_path is None:
            config_path = self.get_default_config_path()
        self.config_path = Path(config_path)
        self.data = self._load_or_create()

    @staticmethod
    def get_default_config_path() -> str:
        """Get default configuration path."""
        home = Path.home()
        return str(home / ".mcp-agent-rag" / "config.json")

    @staticmethod
    def get_default_data_dir() -> Path:
        """Get default data directory."""
        return Path.home() / ".mcp-agent-rag"

    def _load_or_create(self) -> Dict[str, Any]:
        """Load existing config or create default."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._create_default()

    def _create_default(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "embedding_model": "nomic-embed-text",
            "generative_model": "mistral:7b-instruct",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "databases": {},
            "ollama_host": "http://localhost:11434",
            "log_level": "INFO",
            "max_context_length": 4000,
        }

    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.data[key] = value

    def add_database(
        self, name: str, path: str, description: str = "", doc_count: int = 0
    ) -> None:
        """Add database to configuration.

        Args:
            name: Database name
            path: Path to database files
            description: Database description
            doc_count: Number of documents
        """
        if "databases" not in self.data:
            self.data["databases"] = {}

        self.data["databases"][name] = {
            "path": path,
            "description": description,
            "doc_count": doc_count,
            "last_updated": None,
        }

    def update_database(self, name: str, **kwargs) -> None:
        """Update database configuration.

        Args:
            name: Database name
            **kwargs: Fields to update
        """
        if name in self.data.get("databases", {}):
            self.data["databases"][name].update(kwargs)

    def get_database(self, name: str) -> Optional[Dict[str, Any]]:
        """Get database configuration.

        Args:
            name: Database name

        Returns:
            Database config or None if not found
        """
        return self.data.get("databases", {}).get(name)

    def list_databases(self) -> Dict[str, Dict[str, Any]]:
        """List all databases."""
        return self.data.get("databases", {})

    def database_exists(self, name: str) -> bool:
        """Check if database exists.

        Args:
            name: Database name

        Returns:
            True if database exists
        """
        return name in self.data.get("databases", {})

    def get_database_path(self, name: str) -> Optional[Path]:
        """Get path to database directory.

        Args:
            name: Database name

        Returns:
            Path to database directory or None
        """
        db_info = self.get_database(name)
        if db_info:
            return Path(db_info["path"])
        return None
