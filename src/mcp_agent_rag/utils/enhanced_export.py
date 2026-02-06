"""Enhanced database export/import with compatibility metadata."""

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mcp_agent_rag.config import Config
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class EnhancedDatabaseExporter:
    """Export databases with full compatibility metadata.
    
    Export format includes:
    - Manifest with chunk hashes
    - Embeddings model ID
    - Chunking parameters
    - Schema version
    - Database metadata
    """

    EXPORT_VERSION = "2.0"  # Export format version

    def __init__(self, config: Config):
        """Initialize exporter.

        Args:
            config: Configuration instance
        """
        self.config = config

    def export_databases(
        self,
        database_names: List[str],
        output_path: Path,
        include_metadata: bool = True,
        include_cache: bool = False,
    ) -> bool:
        """Export databases to ZIP file with compatibility metadata.

        Args:
            database_names: List of database names to export
            output_path: Output ZIP file path
            include_metadata: Include full metadata
            include_cache: Include embedding cache

        Returns:
            True if successful
        """
        try:
            # Validate databases exist
            for db_name in database_names:
                if not self.config.database_exists(db_name):
                    logger.error(f"Database '{db_name}' does not exist")
                    return False

            # Create export manifest
            manifest = {
                "export_version": self.EXPORT_VERSION,
                "exported_at": datetime.now().isoformat(),
                "databases": [],
                "config": {
                    "embedding_model": self.config.get("embedding_model"),
                    "chunk_size": self.config.get("chunk_size"),
                    "chunk_overlap": self.config.get("chunk_overlap"),
                    "use_semantic_chunking": self.config.get("use_semantic_chunking", False),
                    "schema_version": self.config.get("schema_version", 1),
                },
            }

            # Create ZIP file
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for db_name in database_names:
                    db_info = self.config.get_database(db_name)
                    db_path = Path(db_info["path"])

                    # Add database metadata to manifest
                    db_manifest = {
                        "name": db_name,
                        "description": db_info.get("description", ""),
                        "doc_count": db_info.get("doc_count", 0),
                        "last_updated": db_info.get("last_updated", ""),
                        "prefix": db_info.get("prefix", ""),
                    }

                    # Add file information
                    files_added = []

                    # Add FAISS index
                    index_file = db_path / "index.faiss"
                    if index_file.exists():
                        arcname = f"{db_name}/index.faiss"
                        zf.write(index_file, arcname)
                        files_added.append("index.faiss")

                    # Add metadata (SQLite or pickle)
                    metadata_db = db_path / "metadata.db"
                    metadata_pkl = db_path / "metadata.pkl"

                    if metadata_db.exists():
                        arcname = f"{db_name}/metadata.db"
                        zf.write(metadata_db, arcname)
                        files_added.append("metadata.db")
                        db_manifest["metadata_format"] = "sqlite"
                    elif metadata_pkl.exists():
                        arcname = f"{db_name}/metadata.pkl"
                        zf.write(metadata_pkl, arcname)
                        files_added.append("metadata.pkl")
                        db_manifest["metadata_format"] = "pickle"

                    # Add manifest (if exists)
                    manifest_db = db_path / "manifest.db"
                    if manifest_db.exists() and include_metadata:
                        arcname = f"{db_name}/manifest.db"
                        zf.write(manifest_db, arcname)
                        files_added.append("manifest.db")

                    # Add BM25 index (if exists)
                    bm25_db = db_path / "bm25.db"
                    if bm25_db.exists() and include_metadata:
                        arcname = f"{db_name}/bm25.db"
                        zf.write(bm25_db, arcname)
                        files_added.append("bm25.db")

                    db_manifest["files"] = files_added
                    manifest["databases"].append(db_manifest)

                # Add cache if requested
                if include_cache:
                    cache_path = Config.get_default_data_dir() / "embedding_cache.db"
                    if cache_path.exists():
                        zf.write(cache_path, "cache/embedding_cache.db")
                        manifest["includes_cache"] = True

                # Write manifest
                manifest_json = json.dumps(manifest, indent=2)
                zf.writestr("manifest.json", manifest_json)

            logger.info(
                f"Exported {len(database_names)} database(s) to {output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error exporting databases: {e}")
            return False

    def import_databases(
        self,
        zip_path: Path,
        overwrite: bool = False,
        validate_compatibility: bool = True,
    ) -> Dict:
        """Import databases from ZIP file with compatibility checks.

        Args:
            zip_path: Path to ZIP file
            overwrite: Overwrite existing databases
            validate_compatibility: Check compatibility before import

        Returns:
            Dictionary with import results
        """
        results = {
            "success": [],
            "skipped": [],
            "failed": [],
            "warnings": [],
        }

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Read manifest
                manifest_data = zf.read("manifest.json")
                manifest = json.loads(manifest_data)

                # Check export version
                export_version = manifest.get("export_version", "1.0")
                if export_version != self.EXPORT_VERSION:
                    results["warnings"].append(
                        f"Export version mismatch: {export_version} != {self.EXPORT_VERSION}"
                    )

                # Validate compatibility
                if validate_compatibility:
                    compat_warnings = self._check_compatibility(manifest)
                    results["warnings"].extend(compat_warnings)

                # Import each database
                for db_manifest in manifest["databases"]:
                    db_name = db_manifest["name"]

                    # Check if exists
                    if self.config.database_exists(db_name) and not overwrite:
                        logger.info(f"Skipping existing database: {db_name}")
                        results["skipped"].append(db_name)
                        continue

                    try:
                        # Create database directory
                        db_path = Config.get_default_data_dir() / "databases" / db_name
                        db_path.mkdir(parents=True, exist_ok=True)

                        # Extract files
                        for file_name in db_manifest.get("files", []):
                            src = f"{db_name}/{file_name}"
                            dst = db_path / file_name
                            with zf.open(src) as src_f:
                                with open(dst, "wb") as dst_f:
                                    dst_f.write(src_f.read())

                        # Add to configuration
                        self.config.add_database(
                            name=db_name,
                            path=str(db_path),
                            description=db_manifest.get("description", ""),
                            doc_count=db_manifest.get("doc_count", 0),
                            prefix=db_manifest.get("prefix", ""),
                        )

                        # Update last_updated
                        self.config.update_database(
                            db_name,
                            last_updated=db_manifest.get("last_updated"),
                        )

                        results["success"].append(db_name)
                        logger.info(f"Imported database: {db_name}")

                    except Exception as e:
                        logger.error(f"Error importing database {db_name}: {e}")
                        results["failed"].append({"name": db_name, "error": str(e)})

                # Import cache if present
                if manifest.get("includes_cache") and "cache/embedding_cache.db" in zf.namelist():
                    cache_dir = Config.get_default_data_dir()
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = cache_dir / "embedding_cache.db"

                    with zf.open("cache/embedding_cache.db") as src_f:
                        with open(cache_path, "wb") as dst_f:
                            dst_f.write(src_f.read())

                    logger.info("Imported embedding cache")

            # Save configuration
            self.config.save()

            return results

        except Exception as e:
            logger.error(f"Error importing databases: {e}")
            results["failed"].append({"error": str(e)})
            return results

    def _check_compatibility(self, manifest: Dict) -> List[str]:
        """Check compatibility between export and current config.

        Args:
            manifest: Export manifest

        Returns:
            List of warning messages
        """
        warnings = []

        export_config = manifest.get("config", {})

        # Check embedding model
        current_model = self.config.get("embedding_model")
        export_model = export_config.get("embedding_model")
        if current_model != export_model:
            warnings.append(
                f"Embedding model mismatch: current='{current_model}', "
                f"export='{export_model}'. Embeddings may be incompatible."
            )

        # Check chunking parameters
        current_chunk_size = self.config.get("chunk_size")
        export_chunk_size = export_config.get("chunk_size")
        if current_chunk_size != export_chunk_size:
            warnings.append(
                f"Chunk size mismatch: current={current_chunk_size}, "
                f"export={export_chunk_size}"
            )

        current_overlap = self.config.get("chunk_overlap")
        export_overlap = export_config.get("chunk_overlap")
        if current_overlap != export_overlap:
            warnings.append(
                f"Chunk overlap mismatch: current={current_overlap}, "
                f"export={export_overlap}"
            )

        # Check semantic chunking
        current_semantic = self.config.get("use_semantic_chunking", False)
        export_semantic = export_config.get("use_semantic_chunking", False)
        if current_semantic != export_semantic:
            warnings.append(
                f"Semantic chunking mismatch: current={current_semantic}, "
                f"export={export_semantic}"
            )

        return warnings

    def get_export_info(self, zip_path: Path) -> Optional[Dict]:
        """Get information about an export without importing.

        Args:
            zip_path: Path to ZIP file

        Returns:
            Export information dictionary or None if invalid
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                manifest_data = zf.read("manifest.json")
                manifest = json.loads(manifest_data)

                return {
                    "export_version": manifest.get("export_version"),
                    "exported_at": manifest.get("exported_at"),
                    "database_count": len(manifest.get("databases", [])),
                    "databases": [
                        {
                            "name": db["name"],
                            "description": db.get("description", ""),
                            "doc_count": db.get("doc_count", 0),
                        }
                        for db in manifest.get("databases", [])
                    ],
                    "config": manifest.get("config", {}),
                    "includes_cache": manifest.get("includes_cache", False),
                }

        except Exception as e:
            logger.error(f"Error reading export info: {e}")
            return None
