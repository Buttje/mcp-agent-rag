"""Content hash-based caching for embeddings."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """SQLite-based cache for embeddings keyed by content hash."""

    def __init__(self, cache_path: Path):
        """Initialize embedding cache.

        Args:
            cache_path: Path to cache database file
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (content_hash, model)
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)
            """
        )
        self.conn.commit()

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash of normalized text.

        Args:
            text: Text to hash

        Returns:
            Hex digest of SHA256 hash
        """
        # Normalize: strip whitespace, lowercase
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, content_hash: str, model: str) -> Optional[List[float]]:
        """Get cached embedding.

        Args:
            content_hash: SHA256 hash of content
            model: Model name used for embedding

        Returns:
            Embedding vector or None if not found
        """
        try:
            cursor = self.conn.execute(
                "SELECT embedding FROM embeddings WHERE content_hash = ? AND model = ?",
                (content_hash, model),
            )
            row = cursor.fetchone()
            if row:
                # Deserialize JSON array
                return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def put(self, content_hash: str, embedding: List[float], model: str) -> None:
        """Store embedding in cache.

        Args:
            content_hash: SHA256 hash of content
            embedding: Embedding vector
            model: Model name used for embedding
        """
        try:
            # Serialize embedding as JSON
            embedding_json = json.dumps(embedding)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (content_hash, embedding, model)
                VALUES (?, ?, ?)
                """,
                (content_hash, embedding_json, model),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        try:
            cursor = self.conn.execute("SELECT COUNT(*), COUNT(DISTINCT model) FROM embeddings")
            count, model_count = cursor.fetchone()
            return {
                "total_embeddings": count,
                "unique_models": model_count,
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"total_embeddings": 0, "unique_models": 0}

    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            self.conn.execute("DELETE FROM embeddings")
            self.conn.commit()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
