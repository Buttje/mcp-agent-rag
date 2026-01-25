"""Chat generation utilities using Ollama."""

import json

import requests

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


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


class OllamaGenerator:
    """Generate chat responses using Ollama."""

    def __init__(
        self,
        model: str = "mistral:7b-instruct",
        host: str = "http://localhost:11434",
    ):
        """Initialize Ollama generator.

        Args:
            model: Chat model name
            host: Ollama host URL
        """
        self.model = model
        self.host = normalize_ollama_host(host)
        self.generate_url = f"{self.host}/api/generate"

    def generate(self, prompt: str, context: str = "") -> str | None:
        """Generate a response for the given prompt.

        Args:
            prompt: User prompt
            context: Additional context to include

        Returns:
            Generated response or None if failed
        """
        try:
            # Build full prompt with context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()

            if "response" in result:
                return result["response"]
            else:
                logger.error(f"Unexpected response format from Ollama: {result}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    def generate_stream(self, prompt: str, context: str = ""):
        """Generate a streaming response for the given prompt.

        Args:
            prompt: User prompt
            context: Additional context to include

        Yields:
            Response chunks as they arrive
        """
        try:
            # Build full prompt with context if provided
            full_prompt = prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            yield None
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield None

    def check_connection(self) -> bool:
        """Check if Ollama is accessible.

        Returns:
            True if connection successful
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
