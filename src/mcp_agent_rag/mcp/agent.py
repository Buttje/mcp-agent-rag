"""Agentic RAG implementation."""

from typing import Dict, List

from mcp_agent_rag.config import Config
from mcp_agent_rag.rag import OllamaEmbedder, VectorDatabase
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class AgenticRAG:
    """Agentic RAG for intelligent context retrieval."""

    def __init__(self, config: Config, databases: Dict[str, VectorDatabase]):
        """Initialize agentic RAG.

        Args:
            config: Configuration instance
            databases: Dictionary of loaded databases
        """
        self.config = config
        self.databases = databases
        self.embedder = OllamaEmbedder(
            model=config.get("embedding_model", "nomic-embed-text"),
            host=config.get("ollama_host", "http://localhost:11434"),
        )
        self.max_context_length = config.get("max_context_length", 4000)

    def get_context(self, prompt: str, max_results: int = 5) -> Dict:
        """Get context for prompt using agentic RAG.

        Args:
            prompt: User prompt
            max_results: Maximum results per database

        Returns:
            Dictionary with context text, citations, and databases searched
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(prompt)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return {
                "text": "",
                "citations": [],
                "databases_searched": [],
            }

        # Search all active databases
        all_results = []
        databases_searched = []

        for db_name, db in self.databases.items():
            try:
                results = db.search(query_embedding, k=max_results)
                for distance, metadata in results:
                    all_results.append({
                        "database": db_name,
                        "distance": distance,
                        "text": metadata.get("text", ""),
                        "source": metadata.get("source", ""),
                        "chunk_num": metadata.get("chunk_num", 0),
                        "metadata": metadata,
                    })
                databases_searched.append(db_name)
                logger.info(f"Found {len(results)} results in database '{db_name}'")
            except Exception as e:
                logger.error(f"Error searching database '{db_name}': {e}")

        # Sort by distance (lower is better)
        all_results.sort(key=lambda x: x["distance"])

        # Deduplicate and aggregate
        context_parts = []
        citations = []
        seen_sources = set()
        total_length = 0

        for result in all_results:
            source = result["source"]
            chunk_num = result["chunk_num"]
            source_key = f"{source}:{chunk_num}"

            # Skip duplicates
            if source_key in seen_sources:
                continue

            # Get chunk text from metadata
            chunk_text = self._get_chunk_text(result["metadata"])
            if not chunk_text:
                continue

            # Check if adding this would exceed limit
            if total_length + len(chunk_text) > self.max_context_length:
                break

            context_parts.append(chunk_text)
            citations.append({
                "source": source,
                "chunk": chunk_num,
                "database": result["database"],
            })
            seen_sources.add(source_key)
            total_length += len(chunk_text)

        # Compose final context
        context_text = "\n\n".join(context_parts)

        return {
            "text": context_text,
            "citations": citations,
            "databases_searched": databases_searched,
        }

    def _get_chunk_text(self, metadata: Dict) -> str:
        """Extract chunk text from metadata.

        For simplicity, we need to reconstruct the chunk text.
        In a production system, you might store the actual text in metadata.

        Args:
            metadata: Chunk metadata

        Returns:
            Chunk text
        """
        # In our implementation, we don't store the actual chunk text in metadata
        # This is a simplified version that would need to be enhanced
        # For now, return a placeholder or try to get from metadata
        return metadata.get("text", "")
