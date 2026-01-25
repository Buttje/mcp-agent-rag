"""Ollama utility functions."""


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
