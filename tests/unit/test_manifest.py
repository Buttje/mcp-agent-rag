"""Tests for file manifest."""

import tempfile
from pathlib import Path

import pytest

from mcp_agent_rag.rag.manifest import FileManifest


@pytest.fixture
def manifest():
    """Create a temporary manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.db"
        manifest = FileManifest(manifest_path)
        yield manifest
        manifest.close()


def test_manifest_init():
    """Test manifest initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.db"
        manifest = FileManifest(manifest_path)
        
        assert manifest.manifest_path == manifest_path
        assert manifest.conn is not None
        
        # Check tables exist
        cursor = manifest.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "files" in tables
        assert "chunks" in tables
        
        manifest.close()


def test_compute_file_hash():
    """Test file hash computation."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)
    
    try:
        hash1 = FileManifest.compute_file_hash(temp_path)
        hash2 = FileManifest.compute_file_hash(temp_path)
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    finally:
        temp_path.unlink()


def test_add_file(manifest):
    """Test adding file to manifest."""
    file_id = manifest.add_file(
        path="/test/doc.txt",
        content_hash="abc123",
        mtime=1234567890.0,
        file_size=1024,
    )
    
    assert file_id > 0
    
    # Verify file was added
    file_info = manifest.get_file("/test/doc.txt")
    assert file_info is not None
    assert file_info["path"] == "/test/doc.txt"
    assert file_info["content_hash"] == "abc123"
    assert file_info["mtime"] == 1234567890.0
    assert file_info["file_size"] == 1024


def test_add_file_with_url(manifest):
    """Test adding file from URL."""
    file_id = manifest.add_file(
        path="https://example.com/doc.pdf",
        content_hash="xyz789",
        url="https://example.com/doc.pdf",
        etag="etag123",
        file_size=2048,
    )
    
    assert file_id > 0
    
    file_info = manifest.get_file("https://example.com/doc.pdf")
    assert file_info["url"] == "https://example.com/doc.pdf"
    assert file_info["etag"] == "etag123"


def test_update_file(manifest):
    """Test updating existing file."""
    # Add file
    file_id1 = manifest.add_file(
        path="/test/doc.txt",
        content_hash="old_hash",
        file_size=1024,
    )
    
    # Update file
    file_id2 = manifest.add_file(
        path="/test/doc.txt",
        content_hash="new_hash",
        file_size=2048,
    )
    
    # Should have same ID (update, not insert)
    assert file_id1 == file_id2
    
    # Verify updated
    file_info = manifest.get_file("/test/doc.txt")
    assert file_info["content_hash"] == "new_hash"
    assert file_info["file_size"] == 2048


def test_add_chunks(manifest):
    """Test adding chunks for file."""
    file_id = manifest.add_file(
        path="/test/doc.txt",
        content_hash="abc123",
    )
    
    chunks = [
        {"chunk_index": 0, "faiss_index": 10, "chunk_hash": "chunk0"},
        {"chunk_index": 1, "faiss_index": 11, "chunk_hash": "chunk1"},
        {"chunk_index": 2, "faiss_index": 12, "chunk_hash": "chunk2"},
    ]
    
    manifest.add_chunks(file_id, chunks)
    
    # Retrieve chunks
    retrieved = manifest.get_file_chunks(file_id)
    assert len(retrieved) == 3
    assert retrieved[0]["chunk_index"] == 0
    assert retrieved[0]["faiss_index"] == 10
    assert retrieved[1]["chunk_index"] == 1
    assert retrieved[2]["chunk_index"] == 2


def test_replace_chunks(manifest):
    """Test replacing chunks for file."""
    file_id = manifest.add_file(
        path="/test/doc.txt",
        content_hash="abc123",
    )
    
    # Add initial chunks
    chunks1 = [
        {"chunk_index": 0, "faiss_index": 10, "chunk_hash": "chunk0"},
        {"chunk_index": 1, "faiss_index": 11, "chunk_hash": "chunk1"},
    ]
    manifest.add_chunks(file_id, chunks1)
    
    # Replace with new chunks
    chunks2 = [
        {"chunk_index": 0, "faiss_index": 20, "chunk_hash": "new_chunk0"},
    ]
    manifest.add_chunks(file_id, chunks2)
    
    # Should only have new chunks
    retrieved = manifest.get_file_chunks(file_id)
    assert len(retrieved) == 1
    assert retrieved[0]["faiss_index"] == 20


def test_has_changed(manifest):
    """Test checking if file has changed."""
    path = "/test/doc.txt"
    
    # New file
    assert manifest.has_changed(path, "hash1") is True
    
    # Add file
    manifest.add_file(path, "hash1")
    
    # Same hash - not changed
    assert manifest.has_changed(path, "hash1") is False
    
    # Different hash - changed
    assert manifest.has_changed(path, "hash2") is True


def test_remove_file(manifest):
    """Test removing file and getting FAISS indices."""
    # Add file with chunks
    file_id = manifest.add_file(
        path="/test/doc.txt",
        content_hash="abc123",
    )
    
    chunks = [
        {"chunk_index": 0, "faiss_index": 10, "chunk_hash": "chunk0"},
        {"chunk_index": 1, "faiss_index": 11, "chunk_hash": "chunk1"},
        {"chunk_index": 2, "faiss_index": 12, "chunk_hash": "chunk2"},
    ]
    manifest.add_chunks(file_id, chunks)
    
    # Remove file
    faiss_indices = manifest.remove_file("/test/doc.txt")
    
    # Should return FAISS indices
    assert faiss_indices == {10, 11, 12}
    
    # File should be gone
    assert manifest.get_file("/test/doc.txt") is None
    
    # Chunks should be gone (cascade delete)
    assert len(manifest.get_file_chunks(file_id)) == 0


def test_remove_nonexistent_file(manifest):
    """Test removing file that doesn't exist."""
    faiss_indices = manifest.remove_file("/nonexistent.txt")
    assert faiss_indices == set()


def test_list_files(manifest):
    """Test listing all files."""
    # Empty initially
    assert len(manifest.list_files()) == 0
    
    # Add files
    manifest.add_file("/test/doc1.txt", "hash1", file_size=100)
    manifest.add_file("/test/doc2.txt", "hash2", file_size=200)
    manifest.add_file("/test/doc3.txt", "hash3", file_size=300)
    
    files = manifest.list_files()
    assert len(files) == 3
    
    # Should be sorted by path
    paths = [f["path"] for f in files]
    assert paths == sorted(paths)


def test_get_stats(manifest):
    """Test getting manifest statistics."""
    # Initially empty
    stats = manifest.get_stats()
    assert stats["file_count"] == 0
    assert stats["chunk_count"] == 0
    assert stats["total_size_bytes"] == 0
    
    # Add files and chunks
    file_id1 = manifest.add_file("/test/doc1.txt", "hash1", file_size=100)
    file_id2 = manifest.add_file("/test/doc2.txt", "hash2", file_size=200)
    
    manifest.add_chunks(file_id1, [
        {"chunk_index": 0, "faiss_index": 10, "chunk_hash": "c0"},
        {"chunk_index": 1, "faiss_index": 11, "chunk_hash": "c1"},
    ])
    manifest.add_chunks(file_id2, [
        {"chunk_index": 0, "faiss_index": 20, "chunk_hash": "c0"},
    ])
    
    stats = manifest.get_stats()
    assert stats["file_count"] == 2
    assert stats["chunk_count"] == 3
    assert stats["total_size_bytes"] == 300


def test_context_manager():
    """Test manifest as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.db"
        
        with FileManifest(manifest_path) as manifest:
            manifest.add_file("/test/doc.txt", "hash123")
            stats = manifest.get_stats()
            assert stats["file_count"] == 1
        
        # Should be closed after context exit
        # Verify by opening new connection
        with FileManifest(manifest_path) as manifest:
            file_info = manifest.get_file("/test/doc.txt")
            assert file_info is not None
