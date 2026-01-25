"""Vector database using FAISS."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class VectorDatabase:
    """FAISS-based vector database for document embeddings."""

    def __init__(self, db_path: Path, dimension: int = 768):
        """Initialize vector database.

        Args:
            db_path: Path to database directory
            dimension: Embedding dimension
        """
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.index = None
        self.metadata = []
        self.doc_count = 0

        # Create directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize or load index
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        index_file = self.db_path / "index.faiss"
        metadata_file = self.db_path / "metadata.pkl"

        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data.get("metadata", [])
                    self.doc_count = data.get("doc_count", 0)
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create new FAISS index."""
        # Use IndexFlatL2 for exact search (can upgrade to IndexIVFFlat for larger datasets)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.doc_count = 0
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

        # Add to index
        self.index.add(embeddings_np)

        # Add metadata
        self.metadata.extend(metadata_list)

        logger.info(f"Added {len(embeddings)} vectors to database")

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[Tuple[float, Dict]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query vector
            k: Number of results to return

        Returns:
            List of (distance, metadata) tuples
        """
        if self.index.ntotal == 0:
            return []

        # Convert to numpy array
        query_np = np.array([query_embedding], dtype=np.float32)

        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)

        # Return results with metadata
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append((float(distance), self.metadata[idx]))

        return results

    def save(self) -> None:
        """Save index and metadata to disk."""
        index_file = self.db_path / "index.faiss"
        metadata_file = self.db_path / "metadata.pkl"

        try:
            # Save FAISS index
            faiss.write_index(self.index, str(index_file))

            # Save metadata
            with open(metadata_file, "wb") as f:
                pickle.dump(
                    {
                        "metadata": self.metadata,
                        "doc_count": self.doc_count,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                )

            logger.info(f"Saved database to {self.db_path}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            raise

    def get_stats(self) -> Dict:
        """Get database statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "doc_count": self.doc_count,
            "dimension": self.dimension,
        }

    def increment_doc_count(self) -> None:
        """Increment document count."""
        self.doc_count += 1
