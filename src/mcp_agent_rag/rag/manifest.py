"""File manifest for tracking indexed documents and enabling incremental updates."""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class FileManifest:
    """SQLite-based manifest for tracking indexed files.
    
    Maintains file-level metadata:
    - path/URL
    - mtime/etag
    - content hash (SHA256)
    - chunk IDs in FAISS index
    - last indexed timestamp
    """

    def __init__(self, manifest_path: Path):
        """Initialize file manifest.

        Args:
            manifest_path: Path to manifest database file
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        self.conn = sqlite3.connect(str(self.manifest_path), check_same_thread=False)
        
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Files table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                url TEXT,
                content_hash TEXT NOT NULL,
                mtime REAL,
                etag TEXT,
                file_size INTEGER,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        # Chunks table - maps chunks to files
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                faiss_index INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                UNIQUE(file_id, chunk_index)
            )
            """
        )
        
        # Indexes
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_path ON files(path)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_hash ON files(content_hash)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_chunks ON chunks(file_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faiss_index ON chunks(faiss_index)"
        )
        
        self.conn.commit()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA256 hash
        """
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""

    def add_file(
        self,
        path: str,
        content_hash: str,
        mtime: Optional[float] = None,
        url: Optional[str] = None,
        etag: Optional[str] = None,
        file_size: Optional[int] = None,
    ) -> int:
        """Add or update file in manifest.

        Args:
            path: File path or identifier
            content_hash: SHA256 hash of content
            mtime: File modification time
            url: URL if downloaded from web
            etag: ETag if from URL
            file_size: File size in bytes

        Returns:
            File ID in manifest
        """
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO files (path, content_hash, mtime, url, etag, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    mtime = excluded.mtime,
                    url = excluded.url,
                    etag = excluded.etag,
                    file_size = excluded.file_size,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (path, content_hash, mtime, url, etag, file_size),
            )
            file_id = cursor.fetchone()[0]
            self.conn.commit()
            return file_id
        except Exception as e:
            logger.error(f"Error adding file to manifest: {e}")
            self.conn.rollback()
            raise

    def add_chunks(self, file_id: int, chunk_data: List[Dict]) -> None:
        """Add chunks for a file.

        Args:
            file_id: File ID in manifest
            chunk_data: List of dicts with chunk_index, faiss_index, chunk_hash
        """
        try:
            # Delete existing chunks for this file
            self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            
            # Insert new chunks
            for chunk in chunk_data:
                self.conn.execute(
                    """
                    INSERT INTO chunks (file_id, chunk_index, faiss_index, chunk_hash)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        chunk["chunk_index"],
                        chunk["faiss_index"],
                        chunk["chunk_hash"],
                    ),
                )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding chunks to manifest: {e}")
            self.conn.rollback()
            raise

    def get_file(self, path: str) -> Optional[Dict]:
        """Get file metadata from manifest.

        Args:
            path: File path

        Returns:
            File metadata dict or None
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT id, path, url, content_hash, mtime, etag, file_size, 
                       indexed_at, updated_at
                FROM files WHERE path = ?
                """,
                (path,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "path": row[1],
                    "url": row[2],
                    "content_hash": row[3],
                    "mtime": row[4],
                    "etag": row[5],
                    "file_size": row[6],
                    "indexed_at": row[7],
                    "updated_at": row[8],
                }
            return None
        except Exception as e:
            logger.error(f"Error getting file from manifest: {e}")
            return None

    def get_file_chunks(self, file_id: int) -> List[Dict]:
        """Get chunks for a file.

        Args:
            file_id: File ID

        Returns:
            List of chunk dicts
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT id, chunk_index, faiss_index, chunk_hash, created_at
                FROM chunks WHERE file_id = ?
                ORDER BY chunk_index
                """,
                (file_id,),
            )
            return [
                {
                    "id": row[0],
                    "chunk_index": row[1],
                    "faiss_index": row[2],
                    "chunk_hash": row[3],
                    "created_at": row[4],
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Error getting chunks from manifest: {e}")
            return []

    def has_changed(self, path: str, content_hash: str) -> bool:
        """Check if file has changed since last index.

        Args:
            path: File path
            content_hash: Current content hash

        Returns:
            True if file has changed or is new
        """
        file_info = self.get_file(path)
        if not file_info:
            return True  # New file
        return file_info["content_hash"] != content_hash

    def remove_file(self, path: str) -> Set[int]:
        """Remove file and get its FAISS indices for deletion.

        Args:
            path: File path

        Returns:
            Set of FAISS indices to remove
        """
        try:
            # Get file ID
            file_info = self.get_file(path)
            if not file_info:
                return set()

            file_id = file_info["id"]
            
            # Get chunk indices before deletion
            cursor = self.conn.execute(
                "SELECT faiss_index FROM chunks WHERE file_id = ?",
                (file_id,),
            )
            faiss_indices = {row[0] for row in cursor.fetchall()}
            
            # Delete file (cascade deletes chunks)
            self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self.conn.commit()
            
            logger.info(f"Removed file {path} with {len(faiss_indices)} chunks")
            return faiss_indices
        except Exception as e:
            logger.error(f"Error removing file from manifest: {e}")
            self.conn.rollback()
            return set()

    def list_files(self) -> List[Dict]:
        """List all files in manifest.

        Returns:
            List of file metadata dicts
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT id, path, url, content_hash, mtime, file_size, indexed_at
                FROM files
                ORDER BY path
                """
            )
            return [
                {
                    "id": row[0],
                    "path": row[1],
                    "url": row[2],
                    "content_hash": row[3],
                    "mtime": row[4],
                    "file_size": row[5],
                    "indexed_at": row[6],
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Error listing files from manifest: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get manifest statistics.

        Returns:
            Dictionary with stats
        """
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            cursor = self.conn.execute("SELECT SUM(file_size) FROM files")
            total_size = cursor.fetchone()[0] or 0
            
            return {
                "file_count": file_count,
                "chunk_count": chunk_count,
                "total_size_bytes": total_size,
            }
        except Exception as e:
            logger.error(f"Error getting manifest stats: {e}")
            return {"file_count": 0, "chunk_count": 0, "total_size_bytes": 0}

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
