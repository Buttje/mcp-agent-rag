"""Database management for MCP-RAG."""

import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
        self.databases: Dict[str, VectorDatabase] = {}
        self._skip_current = False

    def create_database(self, name: str, description: str = "") -> bool:
        """Create a new database.

        Args:
            name: Database name
            description: Database description

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
        )
        self.config.save()

        logger.info(f"Created database '{name}' at {db_path}")
        return True

    def add_documents(
        self,
        database_name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
        glob_pattern: Optional[str] = None,
        recursive: bool = False,
        skip_existing: bool = False,
    ) -> Dict[str, int]:
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
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _download_url(self, url: str) -> Path:
        """Download file from URL.

        Args:
            url: URL to download

        Returns:
            Path to downloaded file
        """
        import tempfile

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Create temporary file
        suffix = Path(url).suffix or ".html"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()

        return Path(temp_file.name)

    def list_databases(self) -> Dict[str, Dict]:
        """List all databases.

        Returns:
            Dictionary of database info
        """
        return self.config.list_databases()

    def load_database(self, name: str) -> Optional[VectorDatabase]:
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

    def load_multiple_databases(self, names: List[str]) -> Dict[str, VectorDatabase]:
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
