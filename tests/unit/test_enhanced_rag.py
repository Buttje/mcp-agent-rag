"""Tests for enhanced RAG and agentic pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_agent_rag.config import Config
from mcp_agent_rag.mcp.enhanced_rag import AgenticRAG, EnhancedRetrieval
from mcp_agent_rag.rag import VectorDatabase


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        "embedding_model": "test-model",
        "ollama_host": "http://localhost:11434",
        "max_context_length": 1000,
    }.get(key, default)
    return config


@pytest.fixture
def mock_databases():
    """Create mock databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_db"
        db = VectorDatabase(db_path, dimension=768)
        
        # Add some test data
        embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        metadata = [
            {"text": "Test chunk 1", "source": "doc1.txt", "chunk_num": 0},
            {"text": "Test chunk 2", "source": "doc1.txt", "chunk_num": 1},
            {"text": "Test chunk 3", "source": "doc2.txt", "chunk_num": 0},
        ]
        db.add(embeddings, metadata)
        
        yield {"test_db": db}


def test_enhanced_retrieval_init(mock_config, mock_databases):
    """Test EnhancedRetrieval initialization."""
    retrieval = EnhancedRetrieval(mock_config, mock_databases)
    
    assert retrieval.config == mock_config
    assert retrieval.databases == mock_databases
    assert retrieval.max_context_length == 1000
    assert retrieval.min_confidence == 0.85  # Default threshold


def test_enhanced_retrieval_custom_confidence(mock_config, mock_databases):
    """Test EnhancedRetrieval with custom confidence threshold."""
    retrieval = EnhancedRetrieval(mock_config, mock_databases, min_confidence=0.90)
    
    assert retrieval.min_confidence == 0.90


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_enhanced_retrieval_get_context(mock_embedder_class, mock_config, mock_databases):
    """Test EnhancedRetrieval context retrieval with confidence scores."""
    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.15] * 768
    mock_embedder_class.return_value = mock_embedder
    
    retrieval = EnhancedRetrieval(mock_config, mock_databases)
    retrieval.embedder = mock_embedder
    
    # Get context
    result = retrieval.get_context("test query", max_results=2)
    
    # Verify structure
    assert "text" in result
    assert "citations" in result
    assert "databases_searched" in result
    assert "average_confidence" in result
    assert "test_db" in result["databases_searched"]
    
    # Verify confidence scores in citations
    if result["citations"]:
        for citation in result["citations"]:
            assert "confidence" in citation
            assert 0 <= citation["confidence"] <= 1
    
    # Verify embedder was called
    mock_embedder.embed_single.assert_called_once_with("test query")


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_enhanced_retrieval_confidence_filtering(
    mock_embedder_class, mock_config, mock_databases
):
    """Test that low-confidence results are filtered out."""
    # Mock embedder
    mock_embedder = MagicMock()
    # Use a query embedding far from stored embeddings to get low confidence
    mock_embedder.embed_single.return_value = [0.9] * 768
    mock_embedder_class.return_value = mock_embedder
    
    # Use very high confidence threshold to ensure filtering
    retrieval = EnhancedRetrieval(mock_config, mock_databases, min_confidence=0.99)
    retrieval.embedder = mock_embedder
    
    result = retrieval.get_context("test query", max_results=5)
    
    # With such high threshold, we expect empty or very few results
    # (depending on actual distances computed)
    assert "citations" in result
    # All returned citations should have confidence >= threshold
    for citation in result["citations"]:
        assert citation["confidence"] >= 0.99


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_enhanced_retrieval_embedding_failure(
    mock_embedder_class, mock_config, mock_databases
):
    """Test handling of embedding failure."""
    # Mock embedder to return None
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = None
    mock_embedder_class.return_value = mock_embedder
    
    retrieval = EnhancedRetrieval(mock_config, mock_databases)
    retrieval.embedder = mock_embedder
    
    result = retrieval.get_context("test query")
    
    # Should return empty result with confidence info
    assert result["text"] == ""
    assert result["citations"] == []
    assert result["databases_searched"] == []
    assert result["average_confidence"] == 0.0


def test_agentic_rag_init(mock_config, mock_databases):
    """Test AgenticRAG initialization with confidence threshold."""
    agentic = AgenticRAG(mock_config, mock_databases, max_iterations=2)
    
    assert agentic.config == mock_config
    assert agentic.databases == mock_databases
    assert agentic.max_iterations == 2
    assert agentic.min_confidence == 0.85  # Default threshold


def test_agentic_rag_custom_confidence(mock_config, mock_databases):
    """Test AgenticRAG with custom confidence threshold."""
    agentic = AgenticRAG(mock_config, mock_databases, min_confidence=0.90)
    
    assert agentic.min_confidence == 0.90


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_agentic_rag_routing(mock_embedder_class, mock_config, mock_databases):
    """Test AgenticRAG routing logic."""
    mock_embedder = MagicMock()
    mock_embedder_class.return_value = mock_embedder
    
    agentic = AgenticRAG(mock_config, mock_databases)
    
    # Test routing
    routing = agentic._route("test query", 1)
    
    assert "databases" in routing
    assert "strategy" in routing
    assert "iteration" in routing
    assert set(routing["databases"]) == set(mock_databases.keys())


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_agentic_rag_retrieval(mock_embedder_class, mock_config, mock_databases):
    """Test AgenticRAG retrieval stage."""
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.15] * 768
    mock_embedder_class.return_value = mock_embedder
    
    agentic = AgenticRAG(mock_config, mock_databases)
    agentic.embedder = mock_embedder
    
    routing = {"databases": ["test_db"], "strategy": "vector_search"}
    results = agentic._retrieve("test query", routing, max_results=2)
    
    assert len(results) > 0
    assert all("score" in r for r in results)
    assert all("database" in r for r in results)


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_agentic_rag_reranking(mock_embedder_class, mock_config, mock_databases):
    """Test AgenticRAG reranking stage."""
    mock_embedder = MagicMock()
    mock_embedder_class.return_value = mock_embedder
    
    agentic = AgenticRAG(mock_config, mock_databases)
    
    # Create test results with different scores
    results = [
        {"score": 0.5, "text": "low"},
        {"score": 0.9, "text": "high"},
        {"score": 0.7, "text": "medium"},
    ]
    
    reranked = agentic._rerank("test query", results)
    
    # Should be sorted by score descending
    assert reranked[0]["score"] == 0.9
    assert reranked[1]["score"] == 0.7
    assert reranked[2]["score"] == 0.5


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_agentic_rag_critic(mock_embedder_class, mock_config, mock_databases):
    """Test AgenticRAG critic evaluation."""
    mock_embedder = MagicMock()
    mock_embedder_class.return_value = mock_embedder
    
    agentic = AgenticRAG(mock_config, mock_databases, max_iterations=3)
    
    # Test with good context
    good_context = {
        "text": "x" * 800,  # Near max length
        "citations": [{"source": f"doc{i}"} for i in range(10)],
        "databases_searched": list(mock_databases.keys()),
    }
    score, should_continue = agentic._critic("test query", good_context, 1)
    
    assert score > 0.5
    assert not should_continue  # Good quality, no need to continue
    
    # Test with poor context
    poor_context = {
        "text": "short",
        "citations": [{"source": "doc1"}],
        "databases_searched": ["test_db"],
    }
    score, should_continue = agentic._critic("test query", poor_context, 1)
    
    assert score < 0.7
    assert should_continue  # Poor quality, should iterate


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_agentic_rag_full_pipeline(mock_embedder_class, mock_config, mock_databases):
    """Test full AgenticRAG pipeline."""
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.15] * 768
    mock_embedder_class.return_value = mock_embedder
    
    agentic = AgenticRAG(mock_config, mock_databases, max_iterations=2)
    agentic.embedder = mock_embedder
    
    result = agentic.get_context("test query", max_results=5)
    
    # Verify result structure
    assert "text" in result
    assert "citations" in result
    assert "databases_searched" in result
    assert "iterations" in result
    assert "quality_score" in result
    
    # Should have executed at least one iteration
    assert result["iterations"] >= 1
    assert result["iterations"] <= 2


@patch("mcp_agent_rag.mcp.enhanced_rag.OllamaEmbedder")
def test_agentic_rag_max_iterations(mock_embedder_class, mock_config, mock_databases):
    """Test that AgenticRAG respects max iterations."""
    mock_embedder = MagicMock()
    mock_embedder.embed_single.return_value = [0.15] * 768
    mock_embedder_class.return_value = mock_embedder
    
    # Set max_iterations to 1
    agentic = AgenticRAG(mock_config, mock_databases, max_iterations=1)
    agentic.embedder = mock_embedder
    
    result = agentic.get_context("test query")
    
    # Should stop after 1 iteration
    assert result["iterations"] == 1
