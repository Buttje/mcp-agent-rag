"""Embedding utilities using Ollama."""

import json
from typing import List, Optional

import requests

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        """Initialize Ollama embedder.

        Args:
            model: Embedding model name
            host: Ollama host URL
        """
        self.model = model
        self.host = host.rstrip("/")
        self.embed_url = f"{self.host}/api/embed"

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors or None if failed
        """
        try:
            # Ollama /api/embed accepts a list of texts
            payload = {
                "model": self.model,
                "input": texts,
            }

            response = requests.post(
                self.embed_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()

            # Ollama returns {"embeddings": [[...], [...]]}
            if "embeddings" in result:
                return result["embeddings"]
            elif "embedding" in result:
                # Single embedding returned
                return [result["embedding"]]
            else:
                logger.error(f"Unexpected response format from Ollama: {result}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None

    def embed_single(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        embeddings = self.embed([text])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None

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
