"""Tests for generator."""

from unittest.mock import Mock, patch

import pytest
import requests

from mcp_agent_rag.rag.generator import OllamaGenerator
from mcp_agent_rag.rag.ollama_utils import normalize_ollama_host


@pytest.fixture
def mock_api_detection():
    """Mock API detection to avoid network calls during tests."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock version endpoint
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.1.14"}
        mock_get.return_value = mock_version_response
        
        yield mock_get, mock_post


@pytest.fixture
def generator(mock_api_detection):
    """Create generator instance."""
    return OllamaGenerator(model="test-model", host="http://localhost:11434")


@pytest.fixture
def generator_legacy():
    """Create generator instance with legacy API mode."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock version endpoint to return old version
        mock_version_response = Mock()
        mock_version_response.status_code = 404  # Old Ollama doesn't have /api/version
        mock_get.return_value = mock_version_response
        
        # Mock /api/chat test to return 404
        mock_chat_response = Mock()
        mock_chat_response.status_code = 404
        mock_post.return_value = mock_chat_response
        
        gen = OllamaGenerator(model="test-model", host="http://localhost:11434")
        yield gen


def test_normalize_ollama_host_basic():
    """Test basic host normalization."""
    assert normalize_ollama_host("http://localhost:11434") == "http://localhost:11434"


def test_normalize_ollama_host_trailing_slash():
    """Test host normalization with trailing slash."""
    assert normalize_ollama_host("http://localhost:11434/") == "http://localhost:11434"


def test_normalize_ollama_host_multiple_trailing_slashes():
    """Test host normalization with multiple trailing slashes."""
    assert normalize_ollama_host("http://localhost:11434///") == "http://localhost:11434"


def test_normalize_ollama_host_with_api_suffix():
    """Test host normalization with /api suffix."""
    assert normalize_ollama_host("http://localhost:11434/api") == "http://localhost:11434"


def test_normalize_ollama_host_with_api_and_slash():
    """Test host normalization with /api/ suffix."""
    assert normalize_ollama_host("http://localhost:11434/api/") == "http://localhost:11434"


def test_normalize_ollama_host_with_whitespace():
    """Test host normalization with whitespace."""
    assert normalize_ollama_host("  http://localhost:11434  ") == "http://localhost:11434"


def test_normalize_ollama_host_with_all_issues():
    """Test host normalization with multiple issues."""
    assert normalize_ollama_host("  http://localhost:11434/api/  ") == "http://localhost:11434"


def test_generator_init(generator):
    """Test generator initialization."""
    assert generator.model == "test-model"
    assert generator.host == "http://localhost:11434"
    assert generator._api_mode == "chat"
    assert "api/chat" in generator.generate_url


def test_generator_init_with_api_suffix(mock_api_detection):
    """Test generator initialization with /api suffix in host."""
    gen = OllamaGenerator(model="test-model", host="http://localhost:11434/api")
    assert gen.host == "http://localhost:11434"
    assert gen.generate_url == "http://localhost:11434/api/chat"


def test_generator_init_with_trailing_slash(mock_api_detection):
    """Test generator initialization with trailing slash in host."""
    gen = OllamaGenerator(model="test-model", host="http://localhost:11434/")
    assert gen.host == "http://localhost:11434"
    assert gen.generate_url == "http://localhost:11434/api/chat"


def test_generator_init_with_api_and_slash(mock_api_detection):
    """Test generator initialization with /api/ in host."""
    gen = OllamaGenerator(model="test-model", host="http://localhost:11434/api/")
    assert gen.host == "http://localhost:11434"
    assert gen.generate_url == "http://localhost:11434/api/chat"


def test_generator_legacy_mode(generator_legacy):
    """Test generator initialization with legacy API mode."""
    assert generator_legacy._api_mode == "generate"
    assert generator_legacy.generate_url == "http://localhost:11434/api/generate"


def test_generate_success(generator):
    """Test successful generation."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Test response"
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = generator.generate("test prompt")

        assert response == "Test response"
        mock_post.assert_called_once()
        
        # Verify the new API format is used
        call_args = mock_post.call_args
        assert "messages" in call_args[1]["json"]
        assert isinstance(call_args[1]["json"]["messages"], list)


def test_generate_with_context(generator):
    """Test generation with context."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Test response"
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = generator.generate("test prompt", context="test context")

        assert response == "Test response"
        # Verify context was included in the call
        call_args = mock_post.call_args
        messages = call_args[1]["json"]["messages"]
        # With context, we should have 2 messages: system + user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "test context" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test prompt"


def test_generate_connection_error(generator):
    """Test generation with connection error."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        response = generator.generate("test prompt")
        assert response is None


def test_generate_timeout(generator):
    """Test generation with timeout."""
    with patch("requests.post", side_effect=requests.exceptions.Timeout()):
        response = generator.generate("test prompt")
        assert response is None


def test_generate_unexpected_response(generator):
    """Test generation with unexpected response format."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = generator.generate("test prompt")
        assert response is None


def test_generate_stream_success(generator):
    """Test successful streaming generation."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'{"message": {"role": "assistant", "content": "Hello "}, "done": false}',
            b'{"message": {"role": "assistant", "content": "world"}, "done": false}',
            b'{"message": {"role": "assistant", "content": "!"}, "done": true}',
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(generator.generate_stream("test prompt"))

        assert len(chunks) == 3
        assert chunks == ["Hello ", "world", "!"]


def test_generate_stream_with_context(generator):
    """Test streaming generation with context."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'{"message": {"role": "assistant", "content": "Test"}, "done": true}',
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        list(generator.generate_stream("test prompt", context="test context"))

        # Verify context was included
        call_args = mock_post.call_args
        assert call_args[1]["json"]["stream"] is True
        messages = call_args[1]["json"]["messages"]
        # With context, we should have 2 messages: system + user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "test context" in messages[0]["content"]


def test_generate_stream_connection_error(generator):
    """Test streaming generation with connection error."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        chunks = list(generator.generate_stream("test prompt"))
        assert None in chunks


def test_check_connection_success(generator):
    """Test connection check success."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert generator.check_connection() is True


def test_check_connection_failure(generator):
    """Test connection check failure."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert generator.check_connection() is False


def test_generate_legacy_mode(generator_legacy):
    """Test generation with legacy /api/generate mode."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Legacy response"
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = generator_legacy.generate("test prompt")

        assert response == "Legacy response"
        mock_post.assert_called_once()
        
        # Verify the legacy API format is used (prompt instead of messages)
        call_args = mock_post.call_args
        assert "prompt" in call_args[1]["json"]
        assert "messages" not in call_args[1]["json"]
        assert call_args[1]["json"]["prompt"] == "test prompt"


def test_generate_stream_legacy_mode(generator_legacy):
    """Test streaming generation with legacy /api/generate mode."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello ", "done": false}',
            b'{"response": "world", "done": false}',
            b'{"response": "!", "done": true}',
        ]
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        chunks = list(generator_legacy.generate_stream("test prompt"))

        assert len(chunks) == 3
        assert chunks == ["Hello ", "world", "!"]


def test_api_mode_detection_new_version():
    """Test API mode detection with new Ollama version."""
    with patch("requests.get") as mock_get:
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.2.0"}
        mock_get.return_value = mock_version_response
        
        gen = OllamaGenerator(model="test-model", host="http://localhost:11434")
        assert gen._api_mode == "chat"


def test_api_mode_detection_boundary_version():
    """Test API mode detection with boundary version 0.1.14."""
    with patch("requests.get") as mock_get:
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.1.14"}
        mock_get.return_value = mock_version_response
        
        gen = OllamaGenerator(model="test-model", host="http://localhost:11434")
        assert gen._api_mode == "chat"


def test_api_mode_detection_old_version():
    """Test API mode detection with old Ollama version."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        mock_version_response = Mock()
        mock_version_response.status_code = 200
        mock_version_response.json.return_value = {"version": "0.1.13"}
        mock_get.return_value = mock_version_response
        
        # Mock /api/chat test to return 404
        mock_chat_response = Mock()
        mock_chat_response.status_code = 404
        mock_post.return_value = mock_chat_response
        
        gen = OllamaGenerator(model="test-model", host="http://localhost:11434")
        assert gen._api_mode == "generate"
