"""Chat generation utilities using Ollama."""

import json
from typing import Literal

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
        self._api_mode: Literal["chat", "generate"] | None = None
        self._detect_api_mode()

    def _detect_api_mode(self):
        """Detect which API mode to use based on Ollama version.

        Tries /api/chat first (v0.1.14+), falls back to /api/generate for older versions.
        """
        # First, try to get version info
        try:
            response = requests.get(f"{self.host}/api/version", timeout=5)
            if response.status_code == 200:
                version_data = response.json()
                version = version_data.get("version", "")
                logger.info(f"Detected Ollama version: {version}")
                # Parse version to check if >= 0.1.14
                try:
                    parts = version.split(".")
                    if len(parts) >= 3:
                        major = int(parts[0])
                        minor = int(parts[1])
                        patch = int(parts[2].split("-")[0])
                        if (
                            major > 0
                            or (major == 0 and minor > 1)
                            or (major == 0 and minor == 1 and patch >= 14)
                        ):
                            self._api_mode = "chat"
                            logger.info("Using /api/chat endpoint")
                            return
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse version: {version}")
        except Exception as e:
            logger.debug(f"Could not get Ollama version: {e}")

        # Try /api/chat endpoint directly
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            }
            response = requests.post(
                f"{self.host}/api/chat",
                json=test_payload,
                timeout=5,
            )
            if response.status_code != 404:
                self._api_mode = "chat"
                logger.info("Using /api/chat endpoint (detected via test request)")
                return
        except requests.exceptions.Timeout:
            # Timeout is okay, means endpoint exists but is slow
            self._api_mode = "chat"
            logger.info("Using /api/chat endpoint (endpoint exists)")
            return
        except Exception as e:
            logger.debug(f"Could not test /api/chat: {e}")

        # Fall back to /api/generate for older versions
        self._api_mode = "generate"
        logger.info("Using /api/generate endpoint (fallback for older Ollama versions)")

    @property
    def generate_url(self) -> str:
        """Get the appropriate generation URL based on API mode."""
        if self._api_mode == "chat":
            return f"{self.host}/api/chat"
        else:
            return f"{self.host}/api/generate"

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
                "content": (
                    f"Context:\n{context}\n\n"
                    "Use the above context to answer the following question."
                ),
            })
        messages.append({
            "role": "user",
            "content": prompt
        })
        return messages

    def _build_prompt_text(self, prompt: str, context: str = "") -> str:
        """Build a single prompt string for /api/generate.

        Args:
            prompt: User prompt
            context: Additional context to include

        Returns:
            Formatted prompt string
        """
        if context:
            return (
                f"Context:\n{context}\n\n"
                "Use the above context to answer the following question.\n\n"
                f"Question: {prompt}"
            )
        return prompt

    def _build_payload(self, prompt: str, context: str = "", stream: bool = False) -> dict:
        """Build the request payload based on API mode.

        Args:
            prompt: User prompt
            context: Additional context to include
            stream: Whether to stream the response

        Returns:
            Request payload dictionary
        """
        if self._api_mode == "chat":
            messages = self._build_messages(prompt, context)
            return {
                "model": self.model,
                "messages": messages,
                "stream": stream,
            }
        else:
            # For /api/generate, use prompt parameter
            prompt_text = self._build_prompt_text(prompt, context)
            payload = {
                "model": self.model,
                "prompt": prompt_text,
                "stream": stream,
            }
            # Add system message if context is provided
            if context:
                payload["system"] = (
                    "You are a helpful assistant. "
                    "Use the provided context to answer questions accurately."
                )
            return payload

    def generate(self, prompt: str, context: str = "") -> str | None:
        """Generate a response for the given prompt.

        Args:
            prompt: User prompt
            context: Additional context to include

        Returns:
            Generated response or None if failed
        """
        try:
            payload = self._build_payload(prompt, context, stream=False)

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()

            # Handle response based on API mode
            if self._api_mode == "chat":
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"]
            else:
                # For /api/generate, response is in "response" field
                if "response" in result:
                    return result["response"]

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
            payload = self._build_payload(prompt, context, stream=True)

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

                    # Handle response based on API mode
                    if self._api_mode == "chat":
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    else:
                        # For /api/generate, response chunk is in "response" field
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
