"""Tests for embedding cache."""

import tempfile
from pathlib import Path

import pytest

from mcp_agent_rag.rag.cache import EmbeddingCache


def test_cache_init():
    """Test cache initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = EmbeddingCache(cache_path)
        
        assert cache.cache_path == cache_path
        assert cache.conn is not None
        
        cache.close()


def test_compute_hash():
    """Test content hash computation."""
    text1 = "Hello World"
    text2 = "hello world"  # Different case
    text3 = "  hello world  "  # Extra whitespace
    
    hash1 = EmbeddingCache.compute_hash(text1)
    hash2 = EmbeddingCache.compute_hash(text2)
    hash3 = EmbeddingCache.compute_hash(text3)
    
    # All should normalize to same hash
    assert hash1 == hash2 == hash3
    assert len(hash1) == 64  # SHA256 hex digest length


def test_cache_put_and_get():
    """Test storing and retrieving embeddings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = EmbeddingCache(cache_path)
        
        text = "Test text"
        content_hash = EmbeddingCache.compute_hash(text)
        embedding = [0.1, 0.2, 0.3, 0.4]
        model = "test-model"
        
        # Store embedding
        cache.put(content_hash, embedding, model)
        
        # Retrieve embedding
        retrieved = cache.get(content_hash, model)
        
        assert retrieved == embedding
        cache.close()


def test_cache_miss():
    """Test cache miss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = EmbeddingCache(cache_path)
        
        # Try to get non-existent embedding
        retrieved = cache.get("nonexistent_hash", "test-model")
        
        assert retrieved is None
        cache.close()


def test_cache_model_isolation():
    """Test that different models don't share cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = EmbeddingCache(cache_path)
        
        text = "Test text"
        content_hash = EmbeddingCache.compute_hash(text)
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]
        
        # Store same content with different models
        cache.put(content_hash, embedding1, "model1")
        cache.put(content_hash, embedding2, "model2")
        
        # Retrieve with each model
        retrieved1 = cache.get(content_hash, "model1")
        retrieved2 = cache.get(content_hash, "model2")
        
        assert retrieved1 == embedding1
        assert retrieved2 == embedding2
        assert retrieved1 != retrieved2
        
        cache.close()


def test_cache_stats():
    """Test cache statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = EmbeddingCache(cache_path)
        
        # Initially empty
        stats = cache.get_stats()
        assert stats["total_embeddings"] == 0
        assert stats["unique_models"] == 0
        
        # Add some embeddings
        cache.put("hash1", [0.1, 0.2], "model1")
        cache.put("hash2", [0.3, 0.4], "model1")
        cache.put("hash3", [0.5, 0.6], "model2")
        
        stats = cache.get_stats()
        assert stats["total_embeddings"] == 3
        assert stats["unique_models"] == 2
        
        cache.close()


def test_cache_clear():
    """Test clearing cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = EmbeddingCache(cache_path)
        
        # Add some embeddings
        cache.put("hash1", [0.1, 0.2], "model1")
        cache.put("hash2", [0.3, 0.4], "model1")
        
        stats = cache.get_stats()
        assert stats["total_embeddings"] == 2
        
        # Clear cache
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["total_embeddings"] == 0
        
        cache.close()


def test_cache_context_manager():
    """Test cache as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        
        with EmbeddingCache(cache_path) as cache:
            cache.put("hash1", [0.1, 0.2], "model1")
            stats = cache.get_stats()
            assert stats["total_embeddings"] == 1
        
        # Cache should be closed after context exit
        # Verify by opening a new connection
        with EmbeddingCache(cache_path) as cache:
            retrieved = cache.get("hash1", "model1")
            assert retrieved == [0.1, 0.2]
