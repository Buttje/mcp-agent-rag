"""Tests for agentic RAG."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.mcp.agent import AgenticRAG
from mcp_agent_rag.rag.vector_db import VectorDatabase


@pytest.fixture
def agent(test_config, temp_dir):
    """Create agentic RAG instance."""
    # Create test database
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=768)

    # Add test data
    embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    metadata = [
        {"text": "First document about Python programming", "source": "doc1.txt", "chunk_num": 0},
        {"text": "Second document about data science", "source": "doc2.txt", "chunk_num": 0},
        {"text": "Third document about machine learning", "source": "doc3.txt", "chunk_num": 0},
    ]
    db.add(embeddings, metadata)
    db.save()

    databases = {"testdb": db}

    with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
        agent = AgenticRAG(test_config, databases)
        return agent


def test_agent_initialization(agent):
    """Test agent initialization."""
    assert agent is not None
    assert len(agent.databases) == 1
    assert "testdb" in agent.databases


def test_get_context_single_database(agent):
    """Test getting context from single database."""
    with patch.object(agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.15] * 768

        result = agent.get_context("Tell me about Python", max_results=2)

        assert "text" in result
        assert "citations" in result
        assert "databases_searched" in result
        assert "testdb" in result["databases_searched"]


def test_get_context_with_max_results(agent):
    """Test max_results parameter."""
    with patch.object(agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.15] * 768

        result = agent.get_context("test query", max_results=1)

        assert len(result["citations"]) <= 1


def test_get_context_embedding_failure(agent):
    """Test handling embedding failure."""
    with patch.object(agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = None

        result = agent.get_context("test query")

        assert result["text"] == ""
        assert len(result["citations"]) == 0


def test_get_context_max_length(agent):
    """Test respecting max context length."""
    # Set low max length
    agent.max_context_length = 50

    with patch.object(agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.15] * 768

        result = agent.get_context("test query", max_results=10)

        assert len(result["text"]) <= 50


def test_get_context_deduplication(agent):
    """Test deduplication of results."""
    with patch.object(agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.15] * 768

        result = agent.get_context("test query", max_results=5)

        # Check that citations are unique
        citations = result["citations"]
        citation_keys = [(c["source"], c["chunk"]) for c in citations]
        assert len(citation_keys) == len(set(citation_keys))


def test_get_context_search_error(agent):
    """Test handling search error."""
    with patch.object(agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.15] * 768

        # Make search raise an error
        with patch.object(agent.databases["testdb"], "search", side_effect=Exception("Search error")):
            result = agent.get_context("test query")

            # Should handle error gracefully
            assert "text" in result
            assert "citations" in result
