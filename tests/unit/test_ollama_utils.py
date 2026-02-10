"""Tests for ollama_utils."""

from unittest.mock import Mock, patch

import pytest
import requests

from mcp_agent_rag.rag.ollama_utils import (
    check_ollama_connection,
    fetch_ollama_models,
    get_model_capabilities,
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


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_success(mock_post):
    """Test successful retrieval of model capabilities."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "details": {
            "capabilities": ["completion", "tools", "thinking"]
        }
    }
    mock_post.return_value = mock_response

    capabilities, error = get_model_capabilities("qwen3:30b")
    
    assert error == ""
    assert len(capabilities) == 3
    assert "completion" in capabilities
    assert "tools" in capabilities
    assert "thinking" in capabilities
    mock_post.assert_called_once()


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_no_thinking(mock_post):
    """Test model without thinking capability."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "details": {
            "capabilities": ["completion", "tools"]
        }
    }
    mock_post.return_value = mock_response

    capabilities, error = get_model_capabilities("mistral:7b-instruct")
    
    assert error == ""
    assert len(capabilities) == 2
    assert "completion" in capabilities
    assert "tools" in capabilities
    assert "thinking" not in capabilities


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_timeout(mock_post):
    """Test model capabilities fetch with timeout."""
    mock_post.side_effect = requests.exceptions.Timeout()

    capabilities, error = get_model_capabilities("qwen3:30b")
    
    assert capabilities == []
    assert "timeout" in error.lower()


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_connection_error(mock_post):
    """Test model capabilities fetch with connection error."""
    mock_post.side_effect = requests.exceptions.ConnectionError()

    capabilities, error = get_model_capabilities("qwen3:30b")
    
    assert capabilities == []
    assert "cannot connect" in error.lower()


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_http_error(mock_post):
    """Test model capabilities fetch with HTTP error."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_post.return_value = mock_response

    capabilities, error = get_model_capabilities("nonexistent:model")
    
    assert capabilities == []
    assert "404" in error


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_no_details(mock_post):
    """Test model with no details field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {}
    mock_post.return_value = mock_response

    capabilities, error = get_model_capabilities("somemodel")
    
    assert error == ""
    assert capabilities == []


@patch("mcp_agent_rag.rag.ollama_utils.requests.post")
def test_get_model_capabilities_no_capabilities_field(mock_post):
    """Test model with details but no capabilities field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "details": {
            "parameter_size": "30B"
        }
    }
    mock_post.return_value = mock_response

    capabilities, error = get_model_capabilities("somemodel")
    
    assert error == ""
    assert capabilities == []
