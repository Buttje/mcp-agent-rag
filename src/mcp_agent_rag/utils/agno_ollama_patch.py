"""Patch for agno library to detect Ollama models with thinking capability.

This module provides a monkey-patch for the agno library's is_ollama_reasoning_model
function to properly detect Ollama models with native "thinking" capability by
querying the Ollama API instead of relying on hardcoded model names.

This is a temporary workaround until the fix is contributed to the agno library.
"""

from agno.models.base import Model

from mcp_agent_rag.rag.ollama_utils import get_model_capabilities


# Known reasoning models that agno originally supported
# These are checked first for backwards compatibility and performance
KNOWN_REASONING_MODELS = ["qwq", "deepseek-r1", "qwen2.5-coder", "openthinker", "qwen3"]


def is_ollama_reasoning_model_patched(reasoning_model: Model) -> bool:
    """Check if an Ollama model supports native reasoning/thinking.
    
    This is an enhanced version of agno's is_ollama_reasoning_model that:
    1. Checks the hardcoded list of known reasoning models (for backwards compatibility)
    2. Queries the Ollama API to check for "thinking" capability
    
    Args:
        reasoning_model: The model to check
        
    Returns:
        True if the model supports native reasoning, False otherwise
    """
    # First check if it's an Ollama model
    if reasoning_model.__class__.__name__ != "Ollama":
        return False
    
    # Check hardcoded list for backwards compatibility and performance
    if any(model_id in reasoning_model.id for model_id in KNOWN_REASONING_MODELS):
        return True
    
    # Query Ollama API for model capabilities
    try:
        # Get Ollama host from model if available, otherwise use default
        ollama_host = getattr(reasoning_model, 'api_base', 'http://localhost:11434')
        if not ollama_host:
            ollama_host = 'http://localhost:11434'
            
        capabilities, error = get_model_capabilities(reasoning_model.id, ollama_host)
        
        if error:
            # If we can't check capabilities, fall back to False
            # to avoid false positives
            return False
        
        # Check if "thinking" capability is present
        return "thinking" in capabilities
        
    except Exception:
        # If capability check fails, return False
        return False


def apply_agno_ollama_patch():
    """Apply the monkey-patch to agno's Ollama reasoning detection.
    
    This function replaces agno's is_ollama_reasoning_model with our enhanced
    version that can detect thinking capability from Ollama API.
    
    Call this function early in your application startup (before creating any agents)
    to enable automatic detection of Ollama models with thinking capability.
    """
    try:
        import agno.reasoning.ollama as ollama_module
        
        # Store original function for reference
        if not hasattr(ollama_module, '_original_is_ollama_reasoning_model'):
            ollama_module._original_is_ollama_reasoning_model = (
                ollama_module.is_ollama_reasoning_model
            )
        
        # Replace with patched version
        ollama_module.is_ollama_reasoning_model = is_ollama_reasoning_model_patched
        
        return True
        
    except Exception as e:
        # If patching fails, log but don't crash
        import logging
        logging.warning(f"Failed to apply agno Ollama patch: {e}")
        return False
