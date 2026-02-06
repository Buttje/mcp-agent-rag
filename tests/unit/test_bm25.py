"""Tests for BM25 and hybrid retrieval."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_agent_rag.rag.bm25 import BM25Index, HybridRetriever


@pytest.fixture
def bm25_index():
    """Create temporary BM25 index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "bm25.db"
        index = BM25Index(index_path)
        yield index
        index.close()


def test_bm25_init():
    """Test BM25 index initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "bm25.db"
        index = BM25Index(index_path, k1=2.0, b=0.8)
        
        assert index.index_path == index_path
        assert index.k1 == 2.0
        assert index.b == 0.8
        assert index.conn is not None
        
        index.close()


def test_bm25_add_documents(bm25_index):
    """Test adding documents to BM25 index."""
    documents = [
        {
            "text": "Python is a programming language",
            "source": "doc1.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
        {
            "text": "JavaScript is also a programming language",
            "source": "doc2.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
    ]
    
    bm25_index.add_documents(documents)
    
    stats = bm25_index.get_stats()
    assert stats["document_count"] == 2


def test_bm25_search(bm25_index):
    """Test BM25 search."""
    documents = [
        {
            "text": "Python is a high-level programming language",
            "source": "doc1.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
        {
            "text": "Python is used for web development",
            "source": "doc1.txt",
            "chunk_num": 1,
            "metadata": "{}",
        },
        {
            "text": "JavaScript runs in the browser",
            "source": "doc2.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
    ]
    
    bm25_index.add_documents(documents)
    
    # Search for "Python"
    results = bm25_index.search("Python", k=10)
    
    assert len(results) == 2  # Should find 2 Python documents
    
    # Results should have scores
    for score, doc in results:
        assert score > 0
        assert "Python" in doc["text"]


def test_bm25_search_no_results(bm25_index):
    """Test BM25 search with no matching documents."""
    documents = [
        {
            "text": "Python programming",
            "source": "doc1.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
    ]
    
    bm25_index.add_documents(documents)
    
    results = bm25_index.search("nonexistent query", k=10)
    assert len(results) == 0


def test_bm25_remove_documents(bm25_index):
    """Test removing documents from BM25 index."""
    documents = [
        {
            "text": "Document from source 1",
            "source": "doc1.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
        {
            "text": "Another from source 1",
            "source": "doc1.txt",
            "chunk_num": 1,
            "metadata": "{}",
        },
        {
            "text": "Document from source 2",
            "source": "doc2.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
    ]
    
    bm25_index.add_documents(documents)
    
    # Remove doc1.txt
    removed = bm25_index.remove_documents("doc1.txt")
    assert removed == 2
    
    # Verify stats
    stats = bm25_index.get_stats()
    assert stats["document_count"] == 1


def test_bm25_clear(bm25_index):
    """Test clearing BM25 index."""
    documents = [
        {
            "text": "Test document",
            "source": "doc.txt",
            "chunk_num": 0,
            "metadata": "{}",
        },
    ]
    
    bm25_index.add_documents(documents)
    assert bm25_index.get_stats()["document_count"] == 1
    
    bm25_index.clear()
    assert bm25_index.get_stats()["document_count"] == 0


def test_bm25_context_manager():
    """Test BM25 index as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "bm25.db"
        
        with BM25Index(index_path) as index:
            documents = [
                {
                    "text": "Test",
                    "source": "test.txt",
                    "chunk_num": 0,
                    "metadata": "{}",
                },
            ]
            index.add_documents(documents)
            stats = index.get_stats()
            assert stats["document_count"] == 1


def test_hybrid_retriever_init():
    """Test HybridRetriever initialization."""
    mock_vector_db = MagicMock()
    mock_bm25 = MagicMock()
    
    retriever = HybridRetriever(mock_vector_db, mock_bm25, alpha=0.7)
    
    assert retriever.vector_db == mock_vector_db
    assert retriever.bm25_index == mock_bm25
    assert retriever.alpha == 0.7


def test_hybrid_retriever_search():
    """Test hybrid search combining BM25 and vector results."""
    # Mock vector database
    mock_vector_db = MagicMock()
    mock_vector_db.search.return_value = [
        (0.1, {"text": "doc1", "source": "source1", "chunk_num": 0}),
        (0.3, {"text": "doc2", "source": "source2", "chunk_num": 0}),
    ]
    
    # Mock BM25 index
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        (10.0, {"text": "doc2", "source": "source2", "chunk_num": 0}),
        (5.0, {"text": "doc3", "source": "source3", "chunk_num": 0}),
    ]
    
    retriever = HybridRetriever(mock_vector_db, mock_bm25, alpha=0.5)
    
    query_embedding = [0.1] * 768
    results = retriever.search("test query", query_embedding, k=5)
    
    # Should call both search methods
    mock_vector_db.search.assert_called_once()
    mock_bm25.search.assert_called_once()
    
    # Should return merged results
    assert len(results) > 0
    
    # Results should have scores and documents
    for score, doc in results:
        assert isinstance(score, float)
        assert isinstance(doc, dict)


def test_hybrid_retriever_alpha_vector_only():
    """Test hybrid retriever with alpha=1.0 (vector only)."""
    mock_vector_db = MagicMock()
    mock_vector_db.search.return_value = [
        (0.1, {"text": "doc1", "source": "s1", "chunk_num": 0}),
    ]
    
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        (10.0, {"text": "doc2", "source": "s2", "chunk_num": 0}),
    ]
    
    # alpha=1.0 means vector search only
    retriever = HybridRetriever(mock_vector_db, mock_bm25, alpha=1.0)
    
    results = retriever.search("query", [0.1] * 768, k=5)
    
    # With alpha=1.0, vector results should dominate
    # (exact behavior depends on score normalization)
    assert len(results) > 0


def test_hybrid_retriever_alpha_bm25_only():
    """Test hybrid retriever with alpha=0.0 (BM25 only)."""
    mock_vector_db = MagicMock()
    mock_vector_db.search.return_value = [
        (0.1, {"text": "doc1", "source": "s1", "chunk_num": 0}),
    ]
    
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        (10.0, {"text": "doc2", "source": "s2", "chunk_num": 0}),
    ]
    
    # alpha=0.0 means BM25 only
    retriever = HybridRetriever(mock_vector_db, mock_bm25, alpha=0.0)
    
    results = retriever.search("query", [0.1] * 768, k=5)
    
    # With alpha=0.0, BM25 results should dominate
    assert len(results) > 0


def test_normalize_scores():
    """Test score normalization."""
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Higher is better
    normalized = HybridRetriever._normalize_scores(scores, lower_is_better=False)
    assert normalized[0] == 0.0  # Min
    assert normalized[-1] == 1.0  # Max
    assert all(0.0 <= s <= 1.0 for s in normalized)
    
    # Lower is better (distance)
    normalized = HybridRetriever._normalize_scores(scores, lower_is_better=True)
    assert normalized[0] == 1.0  # Min distance = best
    assert normalized[-1] == 0.0  # Max distance = worst
    assert all(0.0 <= s <= 1.0 for s in normalized)


def test_normalize_scores_equal():
    """Test score normalization with equal scores."""
    scores = [5.0, 5.0, 5.0]
    
    normalized = HybridRetriever._normalize_scores(scores)
    assert all(s == 1.0 for s in normalized)
