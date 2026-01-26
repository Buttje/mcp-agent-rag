"""Tests for ollama_utils."""

from unittest.mock import Mock, patch

import pytest
import requests

from mcp_agent_rag.rag.ollama_utils import (
    check_ollama_connection,
    fetch_ollama_models,
    normalize_ollama_host,
)


def test_normalize_ollama_host_basic():
    """Test basic host normalization."""
    assert normalize_ollama_host("http://localhost:11434") == "http://localhost:11434"


def test_normalize_ollama_host_trailing_slash():
    """Test host normalization with trailing slash."""
    assert normalize_ollama_host("http://localhost:11434/") == "http://localhost:11434"


def test_normalize_ollama_host_with_api_suffix():
    """Test host normalization with /api suffix."""
    assert normalize_ollama_host("http://localhost:11434/api") == "http://localhost:11434"


def test_normalize_ollama_host_with_whitespace():
    """Test host normalization with whitespace."""
    assert normalize_ollama_host("  http://localhost:11434  ") == "http://localhost:11434"


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_check_ollama_connection_success(mock_get):
    """Test successful Ollama connection."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    success, error = check_ollama_connection("http://localhost:11434")
    
    assert success is True
    assert error == ""
    mock_get.assert_called_once()


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_check_ollama_connection_timeout(mock_get):
    """Test Ollama connection timeout."""
    mock_get.side_effect = requests.exceptions.Timeout()

    success, error = check_ollama_connection("http://localhost:11434")
    
    assert success is False
    assert "timeout" in error.lower()


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_check_ollama_connection_refused(mock_get):
    """Test Ollama connection refused."""
    mock_get.side_effect = requests.exceptions.ConnectionError()

    success, error = check_ollama_connection("http://localhost:11434")
    
    assert success is False
    assert "cannot connect" in error.lower()


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_check_ollama_connection_http_error(mock_get):
    """Test Ollama connection with HTTP error."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    success, error = check_ollama_connection("http://localhost:11434")
    
    assert success is False
    assert "500" in error


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_success(mock_get):
    """Test successful model fetching."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "nomic-embed-text:latest"},
            {"name": "mxbai-embed-large"},
            {"name": "mistral:7b-instruct"},
            {"name": "llama3.2:latest"},
            {"name": "all-minilm"},
        ]
    }
    mock_get.return_value = mock_response

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert error == ""
    assert len(embedding_models) == 3
    assert len(generative_models) == 2
    
    # Check embedding models
    assert "nomic-embed-text" in embedding_models
    assert "mxbai-embed-large" in embedding_models
    assert "all-minilm" in embedding_models
    
    # Check generative models
    assert "mistral:7b-instruct" in generative_models
    assert "llama3.2" in generative_models


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_removes_latest_tag(mock_get):
    """Test that :latest tag is removed from model names."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "mistral:latest"},
        ]
    }
    mock_get.return_value = mock_response

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert error == ""
    assert "mistral" in generative_models
    assert "mistral:latest" not in generative_models


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_timeout(mock_get):
    """Test model fetching with timeout."""
    mock_get.side_effect = requests.exceptions.Timeout()

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert embedding_models == []
    assert generative_models == []
    assert "timeout" in error.lower()


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_connection_error(mock_get):
    """Test model fetching with connection error."""
    mock_get.side_effect = requests.exceptions.ConnectionError()

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert embedding_models == []
    assert generative_models == []
    assert "cannot connect" in error.lower()


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_http_error(mock_get):
    """Test model fetching with HTTP error."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert embedding_models == []
    assert generative_models == []
    assert "500" in error


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_empty_response(mock_get):
    """Test model fetching with empty model list."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": []}
    mock_get.return_value = mock_response

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert error == ""
    assert embedding_models == []
    assert generative_models == []


@patch("mcp_agent_rag.rag.ollama_utils.requests.get")
def test_fetch_ollama_models_categorization(mock_get):
    """Test correct categorization of models."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "bge-large"},
            {"name": "gte-base"},
            {"name": "codellama"},
            {"name": "phi3"},
        ]
    }
    mock_get.return_value = mock_response

    embedding_models, generative_models, error = fetch_ollama_models("http://localhost:11434")
    
    assert error == ""
    assert "bge-large" in embedding_models
    assert "gte-base" in embedding_models
    assert "codellama" in generative_models
    assert "phi3" in generative_models
