"""Tests for agno Ollama patch."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.utils.agno_ollama_patch import (
    apply_agno_ollama_patch,
    is_ollama_reasoning_model_patched,
)


def test_is_ollama_reasoning_model_patched_non_ollama():
    """Test that non-Ollama models return False."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "OpenAI"
    mock_model.id = "gpt-4"
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is False


@patch("mcp_agent_rag.utils.agno_ollama_patch.get_model_capabilities")
def test_is_ollama_reasoning_model_patched_known_model(mock_get_capabilities):
    """Test that known reasoning models return True without API call."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "qwen3:30b"
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is True
    # Verify API was NOT called since it's a known model
    mock_get_capabilities.assert_not_called()


def test_is_ollama_reasoning_model_patched_deepseek():
    """Test that deepseek-r1 model is recognized."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "deepseek-r1:latest"
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is True


@patch("mcp_agent_rag.utils.agno_ollama_patch.get_model_capabilities")
def test_is_ollama_reasoning_model_patched_with_thinking_capability(mock_get_capabilities):
    """Test that models with thinking capability are recognized via API."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "somemodel:latest"
    mock_model.api_base = "http://localhost:11434"
    
    # Mock API response indicating thinking capability
    mock_get_capabilities.return_value = (["completion", "tools", "thinking"], "")
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is True
    mock_get_capabilities.assert_called_once_with("somemodel:latest", "http://localhost:11434")


@patch("mcp_agent_rag.utils.agno_ollama_patch.get_model_capabilities")
def test_is_ollama_reasoning_model_patched_without_thinking_capability(mock_get_capabilities):
    """Test that models without thinking capability return False."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "mistral:7b-instruct"
    mock_model.api_base = "http://localhost:11434"
    
    # Mock API response without thinking capability
    mock_get_capabilities.return_value = (["completion", "tools"], "")
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is False
    mock_get_capabilities.assert_called_once()


@patch("mcp_agent_rag.utils.agno_ollama_patch.get_model_capabilities")
def test_is_ollama_reasoning_model_patched_api_error(mock_get_capabilities):
    """Test that API errors result in False."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "somemodel:latest"
    mock_model.api_base = "http://localhost:11434"
    
    # Mock API error
    mock_get_capabilities.return_value = ([], "Connection error")
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is False


@patch("mcp_agent_rag.utils.agno_ollama_patch.get_model_capabilities")
def test_is_ollama_reasoning_model_patched_exception(mock_get_capabilities):
    """Test that exceptions result in False."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "somemodel:latest"
    mock_model.api_base = "http://localhost:11434"
    
    # Mock exception
    mock_get_capabilities.side_effect = Exception("Unexpected error")
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    assert result is False


@patch("mcp_agent_rag.utils.agno_ollama_patch.get_model_capabilities")
def test_is_ollama_reasoning_model_patched_default_host(mock_get_capabilities):
    """Test that default host is used when api_base is not set."""
    mock_model = Mock()
    mock_model.__class__.__name__ = "Ollama"
    mock_model.id = "somemodel:latest"
    mock_model.api_base = None
    
    mock_get_capabilities.return_value = (["completion"], "")
    
    result = is_ollama_reasoning_model_patched(mock_model)
    
    # Should use default host
    mock_get_capabilities.assert_called_once_with("somemodel:latest", "http://localhost:11434")


def test_apply_agno_ollama_patch():
    """Test that the patch can be applied successfully."""
    # This test verifies the patch function doesn't crash
    # We can't easily test the actual patching without importing agno
    result = apply_agno_ollama_patch()
    
    # If agno is installed, it should succeed; if not, it should fail gracefully
    assert isinstance(result, bool)
