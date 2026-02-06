"""BM25 keyword search for hybrid retrieval.

Simple BM25 implementation without external dependencies.
Uses SQLite FTS5 for efficient full-text search.
"""

import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class BM25Index:
    """BM25 keyword search using SQLite FTS5.
    
    Provides traditional keyword-based search to complement
    vector similarity search in hybrid retrieval.
    """

    def __init__(
        self,
        index_path: Path,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25 index.

        Args:
            index_path: Path to SQLite database
            k1: BM25 parameter (term frequency saturation)
            b: BM25 parameter (document length normalization)
        """
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self.conn = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite FTS5 database."""
        self.conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        
        # Create FTS5 virtual table for full-text search
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS bm25_docs USING fts5(
                text,
                source,
                chunk_num,
                metadata,
                tokenize = 'porter unicode61'
            )
            """
        )
        
        # Create metadata table for BM25 scoring
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bm25_stats (
                rowid INTEGER PRIMARY KEY,
                doc_length INTEGER NOT NULL,
                source TEXT NOT NULL,
                chunk_num INTEGER NOT NULL
            )
            """
        )
        
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm25_stats ON bm25_stats(rowid)"
        )
        
        self.conn.commit()

    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to BM25 index.

        Args:
            documents: List of dicts with text, source, chunk_num, metadata
        """
        try:
            for doc in documents:
                text = doc.get("text", "")
                source = doc.get("source", "")
                chunk_num = doc.get("chunk_num", 0)
                metadata = doc.get("metadata", "{}")
                
                # Insert into FTS5 table
                cursor = self.conn.execute(
                    """
                    INSERT INTO bm25_docs (text, source, chunk_num, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (text, source, chunk_num, metadata),
                )
                
                rowid = cursor.lastrowid
                doc_length = len(text.split())
                
                # Store document stats
                self.conn.execute(
                    """
                    INSERT INTO bm25_stats (rowid, doc_length, source, chunk_num)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rowid, doc_length, source, chunk_num),
                )
            
            self.conn.commit()
            logger.info(f"Added {len(documents)} documents to BM25 index")
        except Exception as e:
            logger.error(f"Error adding documents to BM25 index: {e}")
            self.conn.rollback()
            raise

    def search(self, query: str, k: int = 10) -> List[Tuple[float, Dict]]:
        """Search using BM25 scoring.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (score, document) tuples sorted by score descending
        """
        try:
            # Use FTS5 MATCH for initial filtering
            # FTS5 provides its own ranking via bm25() function
            cursor = self.conn.execute(
                """
                SELECT 
                    bm25_docs.rowid,
                    bm25(bm25_docs, ?, ?) as score,
                    text,
                    source,
                    chunk_num,
                    metadata
                FROM bm25_docs
                WHERE bm25_docs MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (self.k1, self.b, query, k),
            )
            
            results = []
            for row in cursor.fetchall():
                rowid, score, text, source, chunk_num, metadata = row
                
                # Normalize score to positive (FTS5 bm25() returns negative)
                normalized_score = abs(score)
                
                results.append((
                    normalized_score,
                    {
                        "rowid": rowid,
                        "text": text,
                        "source": source,
                        "chunk_num": chunk_num,
                        "metadata": metadata,
                    }
                ))
            
            return results
        except Exception as e:
            logger.error(f"Error searching BM25 index: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get index statistics.

        Returns:
            Dictionary with stats
        """
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM bm25_docs")
            doc_count = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT AVG(doc_length) FROM bm25_stats")
            avg_length = cursor.fetchone()[0] or 0.0
            
            return {
                "document_count": doc_count,
                "avg_document_length": avg_length,
            }
        except Exception as e:
            logger.error(f"Error getting BM25 stats: {e}")
            return {"document_count": 0, "avg_document_length": 0.0}

    def remove_documents(self, source: str) -> int:
        """Remove all documents from a source.

        Args:
            source: Source identifier

        Returns:
            Number of documents removed
        """
        try:
            # Get rowids to remove
            cursor = self.conn.execute(
                "SELECT rowid FROM bm25_stats WHERE source = ?",
                (source,),
            )
            rowids = [row[0] for row in cursor.fetchall()]
            
            if not rowids:
                return 0
            
            # Delete from both tables
            placeholders = ",".join("?" * len(rowids))
            self.conn.execute(
                f"DELETE FROM bm25_docs WHERE rowid IN ({placeholders})",
                rowids,
            )
            self.conn.execute(
                f"DELETE FROM bm25_stats WHERE rowid IN ({placeholders})",
                rowids,
            )
            
            self.conn.commit()
            logger.info(f"Removed {len(rowids)} documents from source {source}")
            return len(rowids)
        except Exception as e:
            logger.error(f"Error removing documents from BM25 index: {e}")
            self.conn.rollback()
            return 0

    def clear(self) -> None:
        """Clear all documents from index."""
        try:
            self.conn.execute("DELETE FROM bm25_docs")
            self.conn.execute("DELETE FROM bm25_stats")
            self.conn.commit()
            logger.info("Cleared BM25 index")
        except Exception as e:
            logger.error(f"Error clearing BM25 index: {e}")
            self.conn.rollback()

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


class HybridRetriever:
    """Hybrid retrieval combining BM25 and vector search.
    
    Merges results from keyword (BM25) and semantic (vector) search
    using normalized score fusion.
    """

    def __init__(
        self,
        vector_db,
        bm25_index: BM25Index,
        alpha: float = 0.5,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_db: Vector database instance
            bm25_index: BM25 index instance
            alpha: Weight for vector search (1-alpha for BM25)
                   0.0 = BM25 only, 1.0 = vector only, 0.5 = equal weight
        """
        self.vector_db = vector_db
        self.bm25_index = bm25_index
        self.alpha = alpha

    def search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 10,
    ) -> List[Tuple[float, Dict]]:
        """Hybrid search combining BM25 and vector similarity.

        Args:
            query: Text query
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (score, document) tuples sorted by combined score
        """
        # Get results from both systems (fetch more for fusion)
        fetch_k = k * 3
        
        # Vector search
        vector_results = self.vector_db.search(query_embedding, k=fetch_k)
        
        # BM25 search
        bm25_results = self.bm25_index.search(query, k=fetch_k)
        
        # Normalize scores to [0, 1]
        vector_scores = self._normalize_scores(
            [score for score, _ in vector_results],
            lower_is_better=True,  # Distance metric
        )
        bm25_scores = self._normalize_scores(
            [score for score, _ in bm25_results],
            lower_is_better=False,  # BM25 score
        )
        
        # Build lookup maps
        vector_docs = {}
        for i, (score, doc) in enumerate(vector_results):
            key = (doc.get("source", ""), doc.get("chunk_num", 0))
            vector_docs[key] = (vector_scores[i], doc)
        
        bm25_docs = {}
        for i, (score, doc) in enumerate(bm25_results):
            key = (doc.get("source", ""), doc.get("chunk_num", 0))
            bm25_docs[key] = (bm25_scores[i], doc)
        
        # Merge scores
        all_keys = set(vector_docs.keys()) | set(bm25_docs.keys())
        merged = []
        
        for key in all_keys:
            v_score, v_doc = vector_docs.get(key, (0.0, None))
            b_score, b_doc = bm25_docs.get(key, (0.0, None))
            
            # Combined score
            combined_score = self.alpha * v_score + (1 - self.alpha) * b_score
            
            # Use document with more complete data
            doc = v_doc if v_doc else b_doc
            
            merged.append((combined_score, doc))
        
        # Sort by combined score and return top k
        merged.sort(key=lambda x: x[0], reverse=True)
        return merged[:k]

    @staticmethod
    def _normalize_scores(
        scores: List[float],
        lower_is_better: bool = False,
    ) -> List[float]:
        """Normalize scores to [0, 1] range.

        Args:
            scores: Raw scores
            lower_is_better: If True, lower scores are better (distance)

        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        normalized = []
        for score in scores:
            norm = (score - min_score) / (max_score - min_score)
            if lower_is_better:
                norm = 1.0 - norm  # Invert for distance metrics
            normalized.append(norm)
        
        return normalized
