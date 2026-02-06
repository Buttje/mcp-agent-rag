"""Enhanced vector database with SQLite metadata and memory-mapped index."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class EnhancedVectorDatabase:
    """FAISS-based vector database with SQLite metadata persistence.
    
    Replaces pickle metadata with SQLite for better stability and querying.
    Supports memory-mapped FAISS indices for efficient multi-DB scenarios.
    """

    def __init__(
        self,
        db_path: Path,
        dimension: int = 768,
        use_mmap: bool = True,
    ):
        """Initialize enhanced vector database.

        Args:
            db_path: Path to database directory
            dimension: Embedding dimension
            use_mmap: Whether to use memory-mapped index
        """
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.use_mmap = use_mmap
        self.index = None
        self.metadata_conn = None
        self.doc_count = 0

        # Create directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize metadata database
        self._init_metadata_db()

        # Initialize or load index
        self._load_or_create_index()

    def _init_metadata_db(self) -> None:
        """Initialize SQLite metadata database."""
        metadata_db = self.db_path / "metadata.db"
        self.metadata_conn = sqlite3.connect(
            str(metadata_db), check_same_thread=False
        )

        # Create metadata table
        self.metadata_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faiss_index INTEGER NOT NULL UNIQUE,
                text TEXT,
                source TEXT,
                chunk_num INTEGER,
                char_start INTEGER,
                char_end INTEGER,
                database TEXT,
                extra_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Create indices for fast lookup
        self.metadata_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faiss_index ON chunk_metadata(faiss_index)"
        )
        self.metadata_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON chunk_metadata(source)"
        )
        self.metadata_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_database ON chunk_metadata(database)"
        )

        # Create document count table
        self.metadata_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS db_stats (
                key TEXT PRIMARY KEY,
                value INTEGER
            )
            """
        )

        self.metadata_conn.commit()

        # Load doc count
        cursor = self.metadata_conn.execute(
            "SELECT value FROM db_stats WHERE key = 'doc_count'"
        )
        row = cursor.fetchone()
        self.doc_count = row[0] if row else 0

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        index_file = self.db_path / "index.faiss"

        if index_file.exists():
            try:
                if self.use_mmap:
                    # Use memory-mapped index for efficient loading
                    self.index = faiss.read_index(str(index_file), faiss.IO_FLAG_MMAP)
                else:
                    self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded index with {self.index.ntotal} vectors (mmap={self.use_mmap})")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create new FAISS index."""
        # Use IndexFlatL2 for exact search
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info(f"Created new index with dimension {self.dimension}")

    def add(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict],
    ) -> None:
        """Add embeddings and metadata to database.

        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dicts for each embedding
        """
        if not embeddings:
            return

        # Convert to numpy array
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Get starting FAISS index
        start_idx = self.index.ntotal

        # Add to FAISS index
        self.index.add(embeddings_np)

        # Add metadata to SQLite
        for i, meta in enumerate(metadata_list):
            faiss_idx = start_idx + i
            text = meta.get("text", "")
            source = meta.get("source", "")
            chunk_num = meta.get("chunk_num", 0)
            char_start = meta.get("char_start")
            char_end = meta.get("char_end")
            database = meta.get("database", "")

            # Store extra metadata as JSON
            extra = {
                k: v
                for k, v in meta.items()
                if k not in ["text", "source", "chunk_num", "char_start", "char_end", "database"]
            }
            extra_json = json.dumps(extra) if extra else None

            self.metadata_conn.execute(
                """
                INSERT INTO chunk_metadata 
                (faiss_index, text, source, chunk_num, char_start, char_end, database, extra_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (faiss_idx, text, source, chunk_num, char_start, char_end, database, extra_json),
            )

        self.metadata_conn.commit()
        logger.info(f"Added {len(embeddings)} vectors to database")

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[float, Dict]]:
        """Search for similar embeddings with optional metadata filtering.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Optional filters (source, database, etc.)

        Returns:
            List of (distance, metadata) tuples
        """
        if self.index.ntotal == 0:
            return []

        # Convert to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)

        # Search - fetch more if filtering
        fetch_k = k * 5 if filters else k
        fetch_k = min(fetch_k, self.index.ntotal)
        distances, indices = self.index.search(query_np, fetch_k)

        # Retrieve metadata from SQLite
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue

            # Get metadata
            cursor = self.metadata_conn.execute(
                """
                SELECT text, source, chunk_num, char_start, char_end, 
                       database, extra_metadata
                FROM chunk_metadata WHERE faiss_index = ?
                """,
                (int(idx),),
            )
            row = cursor.fetchone()

            if row:
                text, source, chunk_num, char_start, char_end, database, extra_json = row
                metadata = {
                    "text": text,
                    "source": source,
                    "chunk_num": chunk_num,
                    "char_start": char_start,
                    "char_end": char_end,
                    "database": database,
                }

                # Add extra metadata
                if extra_json:
                    try:
                        extra = json.loads(extra_json)
                        metadata.update(extra)
                    except:
                        pass

                # Apply filters
                if filters:
                    if not self._matches_filters(metadata, filters):
                        continue

                results.append((float(distance), metadata))

                if len(results) >= k:
                    break

        return results

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters.

        Args:
            metadata: Metadata to check
            filters: Filter criteria

        Returns:
            True if matches
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, (list, tuple)):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    def save(self) -> None:
        """Save index and metadata to disk."""
        index_file = self.db_path / "index.faiss"

        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_file))

            # Metadata is already persisted in SQLite
            # Just update stats
            self.metadata_conn.execute(
                """
                INSERT OR REPLACE INTO db_stats (key, value)
                VALUES ('doc_count', ?)
                """,
                (self.doc_count,),
            )
            self.metadata_conn.commit()

            logger.info(f"Saved database to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            raise

    def get_stats(self) -> Dict:
        """Get database statistics.

        Returns:
            Dictionary with stats
        """
        cursor = self.metadata_conn.execute("SELECT COUNT(*) FROM chunk_metadata")
        chunk_count = cursor.fetchone()[0]

        cursor = self.metadata_conn.execute(
            "SELECT COUNT(DISTINCT source) FROM chunk_metadata"
        )
        source_count = cursor.fetchone()[0]

        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "doc_count": self.doc_count,
            "chunk_count": chunk_count,
            "source_count": source_count,
            "dimension": self.dimension,
        }

    def increment_doc_count(self) -> None:
        """Increment document count."""
        self.doc_count += 1

    def close(self) -> None:
        """Close database connections."""
        if self.metadata_conn:
            self.metadata_conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
