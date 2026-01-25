"""Chat generation utilities using Ollama."""

import json

import requests

from mcp_agent_rag.rag.ollama_utils import normalize_ollama_host
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


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
        self.generate_url = f"{self.host}/api/chat"

    def _build_messages(self, prompt: str, context: str = "") -> list[dict]:
        """Build messages array for chat API.

        Args:
            prompt: User prompt
            context: Additional context to include

        Returns:
            List of message objects
        """
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{context}\n\nUse the above context to answer the following question."
            })
        messages.append({
            "role": "user",
            "content": prompt
        })
        return messages

    def generate(self, prompt: str, context: str = "") -> str | None:
        """Generate a response for the given prompt.

        Args:
            prompt: User prompt
            context: Additional context to include

        Returns:
            Generated response or None if failed
        """
        try:
            messages = self._build_messages(prompt, context)

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()

            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
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
            messages = self._build_messages(prompt, context)

            payload = {
                "model": self.model,
                "messages": messages,
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
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
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
