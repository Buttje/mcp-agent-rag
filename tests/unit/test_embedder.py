"""Tests for embedder."""

from unittest.mock import Mock, patch

import pytest
import requests

from mcp_agent_rag.rag.embedder import OllamaEmbedder


@pytest.fixture
def embedder():
    """Create embedder instance."""
    return OllamaEmbedder(model="test-model", host="http://localhost:11434")


def test_embedder_init(embedder):
    """Test embedder initialization."""
    assert embedder.model == "test-model"
    assert embedder.host == "http://localhost:11434"
    assert "api/embed" in embedder.embed_url


def test_embedder_init_with_api_suffix():
    """Test embedder initialization with /api suffix in host."""
    emb = OllamaEmbedder(model="test-model", host="http://localhost:11434/api")
    assert emb.host == "http://localhost:11434"
    assert emb.embed_url == "http://localhost:11434/api/embed"


def test_embedder_init_with_trailing_slash():
    """Test embedder initialization with trailing slash in host."""
    emb = OllamaEmbedder(model="test-model", host="http://localhost:11434/")
    assert emb.host == "http://localhost:11434"
    assert emb.embed_url == "http://localhost:11434/api/embed"


def test_embedder_init_with_api_and_slash():
    """Test embedder initialization with /api/ in host."""
    emb = OllamaEmbedder(model="test-model", host="http://localhost:11434/api/")
    assert emb.host == "http://localhost:11434"
    assert emb.embed_url == "http://localhost:11434/api/embed"


def test_embed_success(embedder, mock_ollama_response):
    """Test successful embedding."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = mock_ollama_response
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        texts = ["test text"]
        embeddings = embedder.embed(texts)

        assert embeddings is not None
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        mock_post.assert_called_once()


def test_embed_single(embedder, mock_ollama_response):
    """Test embedding single text."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = mock_ollama_response
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        embedding = embedder.embed_single("test text")

        assert embedding is not None
        assert isinstance(embedding, list)


def test_embed_connection_error(embedder):
    """Test embedding with connection error."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        embeddings = embedder.embed(["test"])
        assert embeddings is None


def test_embed_timeout(embedder):
    """Test embedding with timeout."""
    with patch("requests.post", side_effect=requests.exceptions.Timeout()):
        embeddings = embedder.embed(["test"])
        assert embeddings is None


def test_check_connection_success(embedder):
    """Test connection check success."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert embedder.check_connection() is True


def test_check_connection_failure(embedder):
    """Test connection check failure."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert embedder.check_connection() is False
