"""Tests for reranking functionality."""

import pytest

from mcp_agent_rag.rag.reranker import ChainReranker, MMRReranker, SimpleReranker


def test_simple_reranker_init():
    """Test SimpleReranker initialization."""
    reranker = SimpleReranker()
    assert reranker is not None


def test_simple_reranker_tokenize():
    """Test simple tokenization."""
    reranker = SimpleReranker()
    
    tokens = reranker._tokenize("Hello world, this is a test!")
    assert "Hello" in tokens or "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    # Short tokens should be filtered
    assert "a" not in tokens


def test_simple_reranker_rerank():
    """Test simple reranking."""
    reranker = SimpleReranker()
    
    results = [
        (0.5, {"text": "Python programming language", "source": "doc1"}),
        (0.6, {"text": "Python is great for data science", "source": "doc2"}),
        (0.4, {"text": "JavaScript for web development", "source": "doc3"}),
    ]
    
    reranked = reranker.rerank("Python programming", results, top_k=3)
    
    # Should return all 3 results
    assert len(reranked) == 3
    
    # Should have scores
    for score, doc in reranked:
        assert isinstance(score, float)
        assert isinstance(doc, dict)


def test_simple_reranker_empty():
    """Test reranking with empty results."""
    reranker = SimpleReranker()
    
    reranked = reranker.rerank("query", [], top_k=10)
    assert reranked == []


def test_simple_reranker_top_k():
    """Test reranking respects top_k."""
    reranker = SimpleReranker()
    
    results = [
        (0.1, {"text": f"Document {i}", "source": f"doc{i}"})
        for i in range(20)
    ]
    
    reranked = reranker.rerank("document query", results, top_k=5)
    assert len(reranked) == 5


def test_mmr_reranker_init():
    """Test MMRReranker initialization."""
    reranker = MMRReranker(lambda_param=0.7)
    assert reranker.lambda_param == 0.7


def test_mmr_reranker_rerank():
    """Test MMR reranking for diversity."""
    reranker = MMRReranker(lambda_param=0.5)
    
    # Create results with some duplicates/similar content
    results = [
        (0.9, {"text": "Python programming language", "source": "doc1"}),
        (0.85, {"text": "Python programming tutorial", "source": "doc2"}),  # Similar
        (0.7, {"text": "JavaScript web development", "source": "doc3"}),  # Different
        (0.65, {"text": "Python for beginners", "source": "doc4"}),  # Similar
        (0.6, {"text": "Ruby programming language", "source": "doc5"}),  # Different
    ]
    
    reranked = reranker.rerank("programming", results, top_k=3)
    
    # Should select diverse results
    assert len(reranked) == 3
    
    # First result should be highest scoring
    assert reranked[0][1]["source"] == "doc1"
    
    # Should have selected diverse documents
    sources = {doc["source"] for _, doc in reranked}
    assert len(sources) == 3  # All unique


def test_mmr_reranker_empty():
    """Test MMR with empty results."""
    reranker = MMRReranker()
    
    reranked = reranker.rerank("query", [], top_k=10)
    assert reranked == []


def test_mmr_reranker_fewer_than_k():
    """Test MMR when results < top_k."""
    reranker = MMRReranker()
    
    results = [
        (0.9, {"text": "doc1", "source": "s1"}),
        (0.8, {"text": "doc2", "source": "s2"}),
    ]
    
    reranked = reranker.rerank("query", results, top_k=10)
    assert len(reranked) == 2  # Returns all available


def test_mmr_tokenize():
    """Test MMR tokenization."""
    reranker = MMRReranker()
    
    tokens = reranker._tokenize("Hello World, Testing!")
    assert "hello" in tokens
    assert "world" in tokens
    assert "testing" in tokens


def test_mmr_max_similarity():
    """Test maximum similarity computation."""
    reranker = MMRReranker()
    
    doc = {"text": "Python programming language"}
    selected = [
        (0.9, {"text": "Python tutorial"}),
        (0.8, {"text": "JavaScript development"}),
    ]
    
    max_sim = reranker._max_similarity(doc, selected)
    
    # Should be non-zero (Python appears in both)
    assert max_sim > 0.0
    assert max_sim <= 1.0


def test_chain_reranker_init():
    """Test ChainReranker initialization."""
    simple = SimpleReranker()
    mmr = MMRReranker()
    
    chain = ChainReranker([simple, mmr])
    assert len(chain.rerankers) == 2


def test_chain_reranker_single():
    """Test chain with single reranker."""
    simple = SimpleReranker()
    chain = ChainReranker([simple])
    
    results = [
        (0.9, {"text": "Python programming", "source": "doc1"}),
        (0.8, {"text": "JavaScript coding", "source": "doc2"}),
    ]
    
    reranked = chain.rerank("Python", results, top_k=2)
    assert len(reranked) == 2


def test_chain_reranker_multiple():
    """Test chain with multiple rerankers."""
    simple = SimpleReranker()
    mmr = MMRReranker(lambda_param=0.5)
    
    chain = ChainReranker([simple, mmr])
    
    results = [
        (0.9, {"text": "Python programming language", "source": "doc1"}),
        (0.85, {"text": "Python programming tutorial", "source": "doc2"}),
        (0.7, {"text": "JavaScript web development", "source": "doc3"}),
        (0.65, {"text": "Python for beginners", "source": "doc4"}),
    ]
    
    reranked = chain.rerank("Python programming", results, top_k=2)
    
    # Should apply both rerankers
    assert len(reranked) == 2
    
    # Results should have scores
    for score, doc in reranked:
        assert isinstance(score, float)


def test_chain_reranker_empty():
    """Test chain with empty results."""
    simple = SimpleReranker()
    mmr = MMRReranker()
    chain = ChainReranker([simple, mmr])
    
    reranked = chain.rerank("query", [], top_k=10)
    assert reranked == []


def test_simple_reranker_compute_relevance():
    """Test relevance computation."""
    reranker = SimpleReranker()
    
    query_terms = ["python", "programming"]
    query_set = set(query_terms)
    doc_terms = ["python", "is", "programming", "language"]
    doc_text = "python is a programming language"
    
    relevance = reranker._compute_relevance(
        query_terms, query_set, doc_terms, doc_text
    )
    
    # Should have non-zero relevance (query terms present)
    assert relevance > 0.0
    assert relevance <= 1.0


def test_simple_reranker_phrase_bonus():
    """Test exact phrase match bonus."""
    reranker = SimpleReranker()
    
    query = "python programming"
    
    # Document with exact phrase
    results_with_phrase = [
        (0.5, {"text": "This is about python programming in detail", "source": "doc1"}),
    ]
    
    # Document without exact phrase
    results_without_phrase = [
        (0.5, {"text": "This is about programming and python separately", "source": "doc2"}),
    ]
    
    reranked_with = reranker.rerank(query, results_with_phrase, top_k=1)
    reranked_without = reranker.rerank(query, results_without_phrase, top_k=1)
    
    # Document with exact phrase should score higher
    assert reranked_with[0][0] > reranked_without[0][0]
