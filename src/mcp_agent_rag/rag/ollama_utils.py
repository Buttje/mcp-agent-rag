"""Ollama utility functions."""

import requests
from typing import Dict, List, Tuple


# Known embedding model name patterns
EMBEDDING_MODEL_PATTERNS = [
    "embed", "nomic", "mxbai", "all-minilm", "bge", "gte"
]


def normalize_ollama_host(host: str) -> str:
    """Normalize Ollama host URL.
    
    Args:
        host: Ollama host URL
        
    Returns:
        Normalized host URL without /api suffix or trailing slashes
    """
    # Strip whitespace
    host = host.strip()
    # Remove trailing slashes
    host = host.rstrip("/")
    # Remove /api suffix if present to avoid double /api/api/ in URLs
    if host.endswith("/api"):
        host = host[:-4]
    return host


def fetch_ollama_models(host: str, timeout: int = 5) -> Tuple[List[str], List[str], str]:
    """Fetch available models from Ollama server and categorize them.
    
    Args:
        host: Ollama host URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (embedding_models, generative_models, error_message)
        If successful, error_message is empty. If failed, model lists are empty.
    """
    try:
        normalized_host = normalize_ollama_host(host)
        response = requests.get(f"{normalized_host}/api/tags", timeout=timeout)
        
        if response.status_code != 200:
            return [], [], f"Failed to fetch models: HTTP {response.status_code}"
        
        data = response.json()
        models = data.get("models", [])
        
        embedding_models = []
        generative_models = []
        
        for model in models:
            model_name = model.get("name", "")
            # Remove :latest tag for cleaner display
            display_name = model_name.replace(":latest", "")
            
            # Check if it's an embedding model using patterns
            is_embedding = any(pattern in model_name.lower() for pattern in EMBEDDING_MODEL_PATTERNS)
            
            if is_embedding:
                embedding_models.append(display_name)
            else:
                generative_models.append(display_name)
        
        return embedding_models, generative_models, ""
        
    except requests.exceptions.Timeout:
        return [], [], f"Connection timeout to {host}"
    except requests.exceptions.ConnectionError:
        return [], [], f"Cannot connect to {host}. Is Ollama running?"
    except Exception as e:
        return [], [], f"Error fetching models: {str(e)}"


def check_ollama_connection(host: str, timeout: int = 5) -> Tuple[bool, str]:
    """Test connection to Ollama server.
    
    Args:
        host: Ollama host URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        normalized_host = normalize_ollama_host(host)
        response = requests.get(f"{normalized_host}/api/tags", timeout=timeout)
        
        if response.status_code == 200:
            return True, ""
        else:
            return False, f"Server returned HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, f"Connection timeout to {host}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {host}. Is Ollama running?"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_model_capabilities(model_name: str, host: str = "http://localhost:11434", 
                          timeout: int = 10) -> Tuple[List[str], str]:
    """Fetch model capabilities from Ollama server.
    
    Args:
        model_name: Name of the model (e.g., "qwen3:30b", "mistral:7b-instruct")
        host: Ollama host URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (capabilities_list, error_message)
        If successful, capabilities is a list like ["completion", "tools", "thinking"]
        and error_message is empty. If failed, capabilities is empty.
    """
    try:
        normalized_host = normalize_ollama_host(host)
        
        # Use POST request with JSON body as per Ollama API specification
        response = requests.post(
            f"{normalized_host}/api/show",
            json={"name": model_name},
            timeout=timeout
        )
        
        if response.status_code != 200:
            return [], f"Failed to get model info: HTTP {response.status_code}"
        
        data = response.json()
        
        # Extract capabilities from model info
        # The capabilities are in the "details" -> "capabilities" field
        capabilities = []
        if "details" in data and isinstance(data["details"], dict):
            if "capabilities" in data["details"]:
                caps = data["details"]["capabilities"]
                if isinstance(caps, list):
                    capabilities = caps
        
        return capabilities, ""
        
    except requests.exceptions.Timeout:
        return [], f"Connection timeout to {host}"
    except requests.exceptions.ConnectionError:
        return [], f"Cannot connect to {host}. Is Ollama running?"
    except Exception as e:
        return [], f"Error fetching model capabilities: {str(e)}"
