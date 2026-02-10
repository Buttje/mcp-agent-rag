"""Enhanced configuration with profiles and versioning."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class ConfigProfile:
    """Configuration profile for different use cases."""

    # Default profiles
    PROFILES = {
        "default": {
            "embedding_model": "nomic-embed-text",
            "generative_model": "mistral:7b-instruct",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "use_semantic_chunking": False,
            "use_hybrid_retrieval": False,
            "use_reranker": False,
            "bm25_alpha": 0.5,  # Weight for vector search in hybrid
            "reranker_type": "simple",  # simple, mmr, or chain
            "mmr_lambda": 0.5,
            "max_context_length": 4000,
            "ocr_enabled": True,
            "ocr_min_text_threshold": 100,
            "parallel_workers": 4,
            "ocr_workers": 2,
            "use_mmap": True,
            "log_level": "INFO",
        },
        "fast": {
            # Fast ingestion, lower quality
            "embedding_model": "nomic-embed-text",
            "generative_model": "mistral:7b-instruct",
            "chunk_size": 256,
            "chunk_overlap": 25,
            "use_semantic_chunking": False,
            "use_hybrid_retrieval": False,
            "use_reranker": False,
            "ocr_enabled": False,
            "parallel_workers": 8,
            "use_mmap": True,
            "log_level": "WARNING",
        },
        "quality": {
            # High quality retrieval, slower
            "embedding_model": "nomic-embed-text",
            "generative_model": "mistral:7b-instruct",
            "chunk_size": 512,
            "chunk_overlap": 100,
            "use_semantic_chunking": True,
            "use_hybrid_retrieval": True,
            "use_reranker": True,
            "bm25_alpha": 0.6,  # Favor vector search slightly
            "reranker_type": "chain",
            "mmr_lambda": 0.6,
            "max_context_length": 8000,
            "ocr_enabled": True,
            "ocr_min_text_threshold": 50,
            "parallel_workers": 4,
            "ocr_workers": 2,
            "use_mmap": True,
            "log_level": "INFO",
        },
        "balanced": {
            # Balanced performance and quality
            "embedding_model": "nomic-embed-text",
            "generative_model": "mistral:7b-instruct",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "use_semantic_chunking": True,
            "use_hybrid_retrieval": True,
            "use_reranker": True,
            "bm25_alpha": 0.5,
            "reranker_type": "simple",
            "mmr_lambda": 0.5,
            "max_context_length": 4000,
            "ocr_enabled": True,
            "ocr_min_text_threshold": 100,
            "parallel_workers": 6,
            "ocr_workers": 2,
            "use_mmap": True,
            "log_level": "INFO",
        },
    }

    @staticmethod
    def get_profile(name: str) -> Dict[str, Any]:
        """Get configuration profile by name.

        Args:
            name: Profile name

        Returns:
            Profile configuration dict
        """
        if name not in ConfigProfile.PROFILES:
            logger.warning(f"Unknown profile '{name}', using 'default'")
            name = "default"
        return ConfigProfile.PROFILES[name].copy()

    @staticmethod
    def list_profiles() -> list[str]:
        """List available profile names.

        Returns:
            List of profile names
        """
        return list(ConfigProfile.PROFILES.keys())


class EnhancedConfig:
    """Enhanced configuration with profiles and versioning.
    
    Adds:
    - Configuration profiles (fast, quality, balanced)
    - Schema versioning for migrations
    - Better organization and defaults
    """

    SCHEMA_VERSION = 2  # Increment for breaking changes

    def __init__(
        self,
        config_path: Optional[str] = None,
        profile: str = "default",
    ):
        """Initialize enhanced configuration.

        Args:
            config_path: Path to config file
            profile: Configuration profile to use
        """
        if config_path is None:
            config_path = self.get_default_config_path()
        self.config_path = Path(config_path)
        self.profile_name = profile
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
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check schema version
                schema_version = data.get("schema_version", 1)
                if schema_version < self.SCHEMA_VERSION:
                    logger.info(
                        f"Migrating config from version {schema_version} to {self.SCHEMA_VERSION}"
                    )
                    data = self._migrate_config(data, schema_version)
                
                return data
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return self._create_default()
        return self._create_default()

    def _create_default(self) -> Dict[str, Any]:
        """Create default configuration."""
        profile_config = ConfigProfile.get_profile(self.profile_name)
        
        return {
            "schema_version": self.SCHEMA_VERSION,
            "profile": self.profile_name,
            "ollama_host": "http://localhost:11434",
            "databases": {},
            # Agentic RAG inference probability thresholds
            "query_inference_threshold": 0.80,  # Inference threshold for generating RAG queries
            "iteration_confidence_threshold": 0.90,  # Threshold for accepting information completeness
            "final_augmentation_threshold": 0.80,  # Inference threshold for final prompt augmentation
            **profile_config,
        }

    def _migrate_config(self, data: Dict, from_version: int) -> Dict:
        """Migrate configuration to current schema version.

        Args:
            data: Old configuration data
            from_version: Old schema version

        Returns:
            Migrated configuration data
        """
        if from_version == 1 and self.SCHEMA_VERSION == 2:
            # Add new fields from default profile
            profile_config = ConfigProfile.get_profile("default")
            for key, value in profile_config.items():
                if key not in data:
                    data[key] = value
            
            data["schema_version"] = 2
            data["profile"] = data.get("profile", "default")
        
        # Ensure agentic RAG fields exist
        defaults = self._create_default()
        for key in ["query_inference_threshold", "iteration_confidence_threshold", "final_augmentation_threshold"]:
            if key not in data:
                data[key] = defaults[key]
            
            logger.info("Config migrated from v1 to v2")

        return data

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

    def switch_profile(self, profile_name: str) -> None:
        """Switch to a different configuration profile.

        Args:
            profile_name: Name of profile to switch to
        """
        profile_config = ConfigProfile.get_profile(profile_name)
        self.data.update(profile_config)
        self.data["profile"] = profile_name
        self.profile_name = profile_name
        logger.info(f"Switched to profile: {profile_name}")

    def add_database(
        self,
        name: str,
        path: str,
        description: str = "",
        doc_count: int = 0,
        prefix: str = "",
    ) -> None:
        """Add database to configuration."""
        if "databases" not in self.data:
            self.data["databases"] = {}

        self.data["databases"][name] = {
            "path": path,
            "description": description,
            "doc_count": doc_count,
            "last_updated": None,
            "prefix": prefix,
        }

    def update_database(self, name: str, **kwargs) -> None:
        """Update database configuration."""
        if name in self.data.get("databases", {}):
            self.data["databases"][name].update(kwargs)

    def get_database(self, name: str) -> Optional[Dict[str, Any]]:
        """Get database configuration."""
        return self.data.get("databases", {}).get(name)

    def list_databases(self) -> Dict[str, Dict[str, Any]]:
        """List all databases."""
        return self.data.get("databases", {})

    def database_exists(self, name: str) -> bool:
        """Check if database exists."""
        return name in self.data.get("databases", {})

    def get_database_path(self, name: str) -> Optional[Path]:
        """Get path to database directory."""
        db_info = self.get_database(name)
        if db_info:
            return Path(db_info["path"])
        return None

    def export_config(self) -> Dict:
        """Export configuration for backup/sharing.

        Returns:
            Configuration dict suitable for export
        """
        export = self.data.copy()
        # Remove sensitive or local-specific data if needed
        return export

    def import_config(self, config_data: Dict) -> None:
        """Import configuration from another source.

        Args:
            config_data: Configuration data to import
        """
        # Validate and merge
        if "schema_version" in config_data:
            schema_version = config_data["schema_version"]
            if schema_version > self.SCHEMA_VERSION:
                logger.warning(
                    f"Importing config from newer version ({schema_version}), "
                    "some settings may be lost"
                )
        
        self.data.update(config_data)
        self.data["schema_version"] = self.SCHEMA_VERSION
