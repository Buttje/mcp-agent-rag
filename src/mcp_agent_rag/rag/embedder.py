"""Embedding utilities using Ollama."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from mcp_agent_rag.config import Config
from mcp_agent_rag.rag.cache import EmbeddingCache
from mcp_agent_rag.rag.ollama_utils import normalize_ollama_host
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    """Generate embeddings using Ollama with caching support."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        cache_path: Optional[Path] = None,
        batch_size: int = 32,
    ):
        """Initialize Ollama embedder.

        Args:
            model: Embedding model name
            host: Ollama host URL
            cache_path: Path to cache database (optional)
            batch_size: Batch size for embeddings
        """
        self.model = model
        self.host = normalize_ollama_host(host)
        self.embed_url = f"{self.host}/api/embed"
        self.batch_size = batch_size
        
        # Initialize cache if path provided
        if cache_path:
            self.cache = EmbeddingCache(cache_path)
        else:
            # Default cache location
            default_cache = Config.get_default_data_dir() / "embedding_cache.db"
            self.cache = EmbeddingCache(default_cache)

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
                timeout=300,
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

    def embed_batch_with_cache(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], int, int]:
        """Generate embeddings for texts with caching.

        Args:
            texts: List of texts to embed

        Returns:
            Tuple of (embeddings, cache_hits, cache_misses)
        """
        embeddings = []
        cache_hits = 0
        cache_misses = 0
        texts_to_embed = []
        text_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            content_hash = EmbeddingCache.compute_hash(text)
            cached = self.cache.get(content_hash, self.model)
            
            if cached:
                embeddings.append((i, cached))
                cache_hits += 1
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
                cache_misses += 1

        # Batch embed uncached texts
        if texts_to_embed:
            # Process in batches
            for batch_start in range(0, len(texts_to_embed), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[batch_start:batch_end]
                
                new_embeddings = self.embed(batch_texts)
                if not new_embeddings:
                    logger.error("Failed to generate embeddings for batch")
                    # Fill with None for missing embeddings
                    new_embeddings = [None] * len(batch_texts)
                
                # Store in cache and result list
                for j, (text, embedding) in enumerate(zip(batch_texts, new_embeddings)):
                    if embedding:
                        content_hash = EmbeddingCache.compute_hash(text)
                        self.cache.put(content_hash, embedding, self.model)
                        idx = text_indices[batch_start + j]
                        embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        final_embeddings = [emb for _, emb in embeddings]
        
        logger.info(
            f"Embedding batch: {cache_hits} cache hits, {cache_misses} cache misses"
        )
        
        return final_embeddings, cache_hits, cache_misses

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

    def get_cache_stats(self):
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return self.cache.get_stats()
