"""Tests for vector database."""

import numpy as np
import pytest

from mcp_agent_rag.rag.vector_db import VectorDatabase


def test_vector_db_creation(temp_dir):
    """Test creating vector database."""
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=768)

    assert db.db_path == db_path
    assert db.dimension == 768
    assert db.index is not None
    assert db_path.exists()


def test_vector_db_add_and_search(temp_dir):
    """Test adding vectors and searching."""
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=4)

    # Add vectors
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    metadata = [
        {"text": "first", "source": "test1.txt"},
        {"text": "second", "source": "test2.txt"},
        {"text": "third", "source": "test3.txt"},
    ]

    db.add(embeddings, metadata)
    assert db.index.ntotal == 3

    # Search
    query = [1.0, 0.1, 0.0, 0.0]
    results = db.search(query, k=2)

    assert len(results) == 2
    # First result should be closest to [1,0,0,0]
    assert results[0][1]["text"] == "first"


def test_vector_db_save_and_load(temp_dir):
    """Test saving and loading database."""
    db_path = temp_dir / "testdb"

    # Create and save
    db1 = VectorDatabase(db_path, dimension=4)
    embeddings = [[1.0, 0.0, 0.0, 0.0]]
    metadata = [{"text": "test"}]
    db1.add(embeddings, metadata)
    db1.increment_doc_count()
    db1.save()

    # Load
    db2 = VectorDatabase(db_path, dimension=4)
    assert db2.index.ntotal == 1
    assert db2.doc_count == 1
    assert len(db2.metadata) == 1


def test_vector_db_empty_search(temp_dir):
    """Test searching empty database."""
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=4)

    results = db.search([1.0, 0.0, 0.0, 0.0], k=5)
    assert len(results) == 0


def test_vector_db_stats(temp_dir):
    """Test database statistics."""
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=4)

    stats = db.get_stats()
    assert stats["total_vectors"] == 0
    assert stats["doc_count"] == 0
    assert stats["dimension"] == 4

    # Add some data
    db.add([[1.0, 0.0, 0.0, 0.0]], [{"text": "test"}])
    db.increment_doc_count()

    stats = db.get_stats()
    assert stats["total_vectors"] == 1
    assert stats["doc_count"] == 1
