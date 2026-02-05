"""Database management for MCP-RAG."""

import json
import signal
import zipfile
from datetime import datetime
from pathlib import Path

import requests

from mcp_agent_rag.config import Config
from mcp_agent_rag.rag import (
    DocumentExtractor,
    OllamaEmbedder,
    VectorDatabase,
    chunk_text,
    clean_text,
    find_files_to_process,
)
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages RAG databases."""

    def __init__(self, config: Config):
        """Initialize database manager.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.embedder = OllamaEmbedder(
            model=config.get("embedding_model", "nomic-embed-text"),
            host=config.get("ollama_host", "http://localhost:11434"),
        )
        self.databases: dict[str, VectorDatabase] = {}
        self._skip_current = False

    def create_database(self, name: str, description: str = "", prefix: str = "") -> bool:
        """Create a new database.

        Args:
            name: Database name
            description: Database description
            prefix: Prefix to prepend to tool names for this database

        Returns:
            True if created successfully
        """
        if self.config.database_exists(name):
            logger.error(f"Database '{name}' already exists")
            return False

        # Create database directory
        db_path = Config.get_default_data_dir() / "databases" / name
        db_path.mkdir(parents=True, exist_ok=True)

        # Add to configuration
        self.config.add_database(
            name=name,
            path=str(db_path),
            description=description,
            prefix=prefix,
        )
        self.config.save()

        logger.info(f"Created database '{name}' at {db_path}")
        return True

    def add_documents(
        self,
        database_name: str,
        path: str | None = None,
        url: str | None = None,
        glob_pattern: str | None = None,
        recursive: bool = False,
        skip_existing: bool = False,
    ) -> dict[str, int]:
        """Add documents to database.

        Args:
            database_name: Name of target database
            path: Path to file or directory
            url: URL to download document
            glob_pattern: Glob pattern for file matching
            recursive: Whether to recurse into subdirectories
            skip_existing: Skip files already in database

        Returns:
            Dictionary with counts of processed, skipped, failed files
        """
        if not self.config.database_exists(database_name):
            logger.error(f"Database '{database_name}' does not exist")
            return {"processed": 0, "skipped": 0, "failed": 0}

        # Load or create vector database
        db_info = self.config.get_database(database_name)
        db_path = Path(db_info["path"])
        vector_db = VectorDatabase(db_path)

        # Set up Ctrl+K handler
        self._skip_current = False
        signal.signal(signal.SIGINT, self._skip_handler)

        stats = {"processed": 0, "skipped": 0, "failed": 0}

        try:
            if url:
                # Download and process URL
                files = [self._download_url(url)]
            else:
                # Find files to process
                if glob_pattern and path:
                    search_path = path.rstrip("/*")
                    pattern = glob_pattern
                else:
                    search_path = path or "."
                    pattern = None

                files = find_files_to_process(
                    search_path,
                    recursive=recursive,
                    glob_pattern=pattern,
                    respect_gitignore=True,
                )

            total_files = len(files)
            logger.info(f"About to add {total_files} document(s)")

            for i, file_path in enumerate(files, 1):
                if self._skip_current:
                    logger.info(f"Skipping {file_path} (Ctrl+K pressed)")
                    stats["skipped"] += 1
                    self._skip_current = False
                    continue

                # Get file size
                try:
                    file_size = Path(file_path).stat().st_size
                    file_size_str = self._format_size(file_size)
                except Exception:
                    file_size_str = "unknown size"

                remaining = total_files - i
                logger.info(
                    f"Processing: {Path(file_path).name} ({file_size_str}) - "
                    f"{remaining} document(s) remaining"
                )

                try:
                    # Extract text
                    text = DocumentExtractor.extract_text(file_path)
                    if not text:
                        logger.warning(f"No text extracted from {file_path}")
                        stats["failed"] += 1
                        continue

                    # Clean text
                    text = clean_text(text)

                    # Chunk text
                    chunks = chunk_text(
                        text,
                        chunk_size=self.config.get("chunk_size", 512),
                        overlap=self.config.get("chunk_overlap", 50),
                        metadata={
                            "source": str(file_path),
                            "database": database_name,
                        },
                    )

                    if not chunks:
                        logger.warning(f"No chunks created from {file_path}")
                        stats["failed"] += 1
                        continue

                    # Generate embeddings
                    chunk_texts = [chunk[0] for chunk in chunks]
                    embeddings = self.embedder.embed(chunk_texts)

                    if not embeddings:
                        logger.error(f"Failed to generate embeddings for {file_path}")
                        stats["failed"] += 1
                        continue

                    # Add to vector database with chunk text in metadata
                    chunk_metadata = []
                    for text_chunk, meta in chunks:
                        meta["text"] = text_chunk
                        chunk_metadata.append(meta)
                    vector_db.add(embeddings, chunk_metadata)
                    vector_db.increment_doc_count()

                    stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    stats["failed"] += 1

            # Save database
            vector_db.save()

            # Update configuration
            self.config.update_database(
                database_name,
                doc_count=vector_db.doc_count,
                last_updated=datetime.now().isoformat(),
            )
            self.config.save()

        finally:
            # Reset signal handler
            signal.signal(signal.SIGINT, signal.default_int_handler)

        logger.info(
            f"Summary: {stats['processed']} processed, "
            f"{stats['skipped']} skipped, {stats['failed']} failed"
        )
        return stats

    def _skip_handler(self, signum, frame):
        """Handle Ctrl+K to skip current file."""
        self._skip_current = True

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            Formatted size string
        """
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _download_url(self, url: str) -> Path:
        """Download file from URL.

        Args:
            url: URL to download

        Returns:
            Path to downloaded file
        """
        import tempfile

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        # Create temporary file
        suffix = Path(url).suffix or ".html"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()

        return Path(temp_file.name)

    def list_databases(self) -> dict[str, dict]:
        """List all databases.

        Returns:
            Dictionary of database info
        """
        return self.config.list_databases()

    def load_database(self, name: str) -> VectorDatabase | None:
        """Load a database into memory.

        Args:
            name: Database name

        Returns:
            VectorDatabase instance or None
        """
        if name in self.databases:
            return self.databases[name]

        if not self.config.database_exists(name):
            logger.error(f"Database '{name}' does not exist")
            return None

        db_info = self.config.get_database(name)
        db_path = Path(db_info["path"])

        if not db_path.exists():
            logger.error(f"Database path does not exist: {db_path}")
            return None

        try:
            vector_db = VectorDatabase(db_path)
            self.databases[name] = vector_db
            logger.info(f"Loaded database '{name}'")
            return vector_db
        except Exception as e:
            logger.error(f"Error loading database '{name}': {e}")
            return None

    def load_multiple_databases(self, names: list[str]) -> dict[str, VectorDatabase]:
        """Load multiple databases.

        Args:
            names: List of database names

        Returns:
            Dictionary of loaded databases
        """
        loaded = {}
        for name in names:
            db = self.load_database(name)
            if db:
                loaded[name] = db
        return loaded

    def export_databases(self, database_names: list[str], export_path: str) -> bool:
        """Export one or more databases to a ZIP file.

        Args:
            database_names: List of database names to export
            export_path: Path to output ZIP file

        Returns:
            True if export was successful
        """
        export_file = Path(export_path)

        # Validate all databases exist
        for name in database_names:
            if not self.config.database_exists(name):
                logger.error(f"Database '{name}' does not exist")
                return False

        try:
            with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Create manifest with all database metadata
                manifest = {
                    "version": "1.0",
                    "export_date": datetime.now().isoformat(),
                    "databases": []
                }

                for db_name in database_names:
                    db_info = self.config.get_database(db_name)
                    db_path = Path(db_info["path"])

                    # Verify database files exist
                    index_file = db_path / "index.faiss"
                    metadata_file = db_path / "metadata.pkl"

                    if not index_file.exists() or not metadata_file.exists():
                        logger.error(f"Database '{db_name}' files not found at {db_path}")
                        return False

                    # Add database metadata to manifest
                    manifest["databases"].append({
                        "name": db_name,
                        "description": db_info.get("description", ""),
                        "doc_count": db_info.get("doc_count", 0),
                        "last_updated": db_info.get("last_updated"),
                        "prefix": db_info.get("prefix", ""),
                    })

                    # Add database files to ZIP
                    zipf.write(index_file, f"{db_name}/index.faiss")
                    zipf.write(metadata_file, f"{db_name}/metadata.pkl")

                # Write manifest
                manifest_json = json.dumps(manifest, indent=2)
                zipf.writestr("manifest.json", manifest_json)

            logger.info(f"Exported {len(database_names)} database(s) to {export_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting databases: {e}")
            return False

    def import_databases(self, import_path: str, overwrite: bool = False) -> dict[str, bool]:
        """Import databases from a ZIP file.

        Args:
            import_path: Path to ZIP file to import
            overwrite: Whether to overwrite existing databases

        Returns:
            Dictionary mapping database names to import success status
        """
        import_file = Path(import_path)

        if not import_file.exists():
            logger.error(f"Import file not found: {import_file}")
            return {}

        results = {}

        try:
            with zipfile.ZipFile(import_file, 'r') as zipf:
                # Read and validate manifest
                try:
                    manifest_data = zipf.read("manifest.json")
                    manifest = json.loads(manifest_data)
                except Exception as e:
                    logger.error(f"Error reading manifest: {e}")
                    return {}

                # Validate manifest version
                if manifest.get("version") != "1.0":
                    logger.error("Unsupported manifest version")
                    return {}

                # Import each database
                for db_info in manifest.get("databases", []):
                    db_name = db_info["name"]

                    # Check if database already exists
                    if self.config.database_exists(db_name) and not overwrite:
                        logger.warning(f"Database '{db_name}' already exists, skipping")
                        results[db_name] = False
                        continue

                    try:
                        # Create database directory
                        db_path = Config.get_default_data_dir() / "databases" / db_name
                        db_path.mkdir(parents=True, exist_ok=True)

                        # Extract database files
                        zipf.extract(f"{db_name}/index.faiss", db_path.parent)
                        zipf.extract(f"{db_name}/metadata.pkl", db_path.parent)

                        # Add or update database in config
                        if self.config.database_exists(db_name):
                            self.config.update_database(
                                db_name,
                                description=db_info.get("description", ""),
                                doc_count=db_info.get("doc_count", 0),
                                last_updated=db_info.get("last_updated"),
                                prefix=db_info.get("prefix", ""),
                            )
                        else:
                            self.config.add_database(
                                name=db_name,
                                path=str(db_path),
                                description=db_info.get("description", ""),
                                doc_count=db_info.get("doc_count", 0),
                                prefix=db_info.get("prefix", ""),
                            )
                            self.config.data["databases"][db_name][
                                "last_updated"
                            ] = db_info.get("last_updated")

                        results[db_name] = True
                        logger.info(f"Imported database '{db_name}'")

                    except Exception as e:
                        logger.error(f"Error importing database '{db_name}': {e}")
                        results[db_name] = False

                # Save configuration
                self.config.save()

            successful = sum(results.values())
            failed = len(results) - successful
            logger.info(f"Import completed: {successful} successful, {failed} failed")
            return results

        except Exception as e:
            logger.error(f"Error importing databases: {e}")
            return {}
