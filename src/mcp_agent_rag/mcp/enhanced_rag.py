"""Enhanced retrieval with Router → Retriever → Reranker → Critic pipeline.

This module implements a true "agentic" retrieval system with:
- Router: Determines which databases and strategies to use
- Retriever: Fetches relevant chunks from vector database  
- Reranker: Scores and reorders results for precision
- Critic: Validates result quality and determines if iteration needed
"""

from typing import Dict, List, Optional, Tuple

from mcp_agent_rag.config import Config
from mcp_agent_rag.rag import OllamaEmbedder, VectorDatabase
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class EnhancedRetrieval:
    """Enhanced retrieval system with multi-stage pipeline.
    
    This implements an improved retrieval pipeline but without a full agent loop.
    For true agentic behavior, use AgenticRAG instead.
    """

    def __init__(self, config: Config, databases: Dict[str, VectorDatabase]):
        """Initialize enhanced retrieval.

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
        """Get context for prompt using enhanced retrieval.

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

        Args:
            metadata: Chunk metadata

        Returns:
            Chunk text
        """
        return metadata.get("text", "")


class AgenticRAG:
    """Agentic RAG with Router → Retriever → Reranker → Critic pipeline.
    
    This implements a bounded agent loop (max 2-3 iterations) with:
    1. Router: Analyzes query and selects retrieval strategy
    2. Retriever: Fetches candidate chunks from databases
    3. Reranker: Re-scores results for improved precision
    4. Critic: Evaluates quality and decides on iteration
    """

    def __init__(
        self,
        config: Config,
        databases: Dict[str, VectorDatabase],
        max_iterations: int = 3,
    ):
        """Initialize agentic RAG.

        Args:
            config: Configuration instance
            databases: Dictionary of loaded databases
            max_iterations: Maximum number of retrieval iterations
        """
        self.config = config
        self.databases = databases
        self.max_iterations = max_iterations
        self.embedder = OllamaEmbedder(
            model=config.get("embedding_model", "nomic-embed-text"),
            host=config.get("ollama_host", "http://localhost:11434"),
        )
        self.max_context_length = config.get("max_context_length", 4000)

    def get_context(self, prompt: str, max_results: int = 5) -> Dict:
        """Get context using agentic RAG pipeline.

        Args:
            prompt: User prompt
            max_results: Maximum results per iteration

        Returns:
            Dictionary with context, citations, metadata
        """
        iteration = 0
        best_context = None
        best_score = 0.0
        all_citations = []

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Agentic RAG iteration {iteration}/{self.max_iterations}")

            # Step 1: Router - Analyze query
            query_analysis = self._route(prompt, iteration)

            # Step 2: Retriever - Fetch results
            results = self._retrieve(prompt, query_analysis, max_results)

            # Step 3: Reranker - Score and reorder
            reranked_results = self._rerank(prompt, results)

            # Step 4: Build context
            context = self._build_context(reranked_results)

            # Step 5: Critic - Evaluate quality
            quality_score, should_continue = self._critic(prompt, context, iteration)

            # Track best result
            if quality_score > best_score:
                best_score = quality_score
                best_context = context
                all_citations = context.get("citations", [])

            # Exit if quality is sufficient or max iterations reached
            if not should_continue or iteration >= self.max_iterations:
                break

            # Refine query for next iteration
            prompt = self._refine_query(prompt, context)

        # Return best context found
        if best_context:
            best_context["iterations"] = iteration
            best_context["quality_score"] = best_score
            return best_context

        # Fallback to empty
        return {
            "text": "",
            "citations": [],
            "databases_searched": list(self.databases.keys()),
            "iterations": iteration,
            "quality_score": 0.0,
        }

    def _route(self, prompt: str, iteration: int) -> Dict:
        """Route query to appropriate databases and strategies.

        Args:
            prompt: Query prompt
            iteration: Current iteration number

        Returns:
            Dictionary with routing decisions
        """
        # For now, simple routing: use all databases
        # In future, could use LLM to select databases based on query
        return {
            "databases": list(self.databases.keys()),
            "strategy": "vector_search",
            "iteration": iteration,
        }

    def _retrieve(
        self, prompt: str, routing: Dict, max_results: int
    ) -> List[Dict]:
        """Retrieve results from databases.

        Args:
            prompt: Query prompt
            routing: Routing decisions
            max_results: Max results per database

        Returns:
            List of result dictionaries
        """
        query_embedding = self.embedder.embed_single(prompt)
        if not query_embedding:
            return []

        all_results = []
        for db_name in routing["databases"]:
            if db_name not in self.databases:
                continue

            db = self.databases[db_name]
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
                        "score": 1.0 / (1.0 + distance),  # Convert distance to score
                    })
            except Exception as e:
                logger.error(f"Error retrieving from {db_name}: {e}")

        return all_results

    def _rerank(self, prompt: str, results: List[Dict]) -> List[Dict]:
        """Rerank results for improved precision.

        Args:
            prompt: Query prompt
            results: Retrieved results

        Returns:
            Reranked results
        """
        # Simple reranking by distance/score
        # In future, could use cross-encoder or LLM-based reranking
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _build_context(self, results: List[Dict]) -> Dict:
        """Build context from reranked results.

        Args:
            results: Reranked results

        Returns:
            Context dictionary
        """
        context_parts = []
        citations = []
        seen_sources = set()
        total_length = 0
        databases_searched = set()

        for result in results:
            source = result["source"]
            chunk_num = result["chunk_num"]
            source_key = f"{source}:{chunk_num}"

            if source_key in seen_sources:
                continue

            chunk_text = result["text"]
            if not chunk_text:
                continue

            if total_length + len(chunk_text) > self.max_context_length:
                break

            context_parts.append(chunk_text)
            citations.append({
                "source": source,
                "chunk": chunk_num,
                "database": result["database"],
                "score": result["score"],
            })
            seen_sources.add(source_key)
            total_length += len(chunk_text)
            databases_searched.add(result["database"])

        return {
            "text": "\n\n".join(context_parts),
            "citations": citations,
            "databases_searched": list(databases_searched),
        }

    def _critic(self, prompt: str, context: Dict, iteration: int) -> Tuple[float, bool]:
        """Evaluate result quality and decide if iteration needed.

        Args:
            prompt: Query prompt
            context: Retrieved context
            iteration: Current iteration

        Returns:
            Tuple of (quality_score, should_continue)
        """
        # Simple heuristic-based critic
        # In future, could use LLM to evaluate relevance
        
        # Quality factors:
        # 1. Context length (prefer more content)
        context_text = context.get("text", "")
        length_score = min(len(context_text) / self.max_context_length, 1.0)
        
        # 2. Number of citations (prefer diverse sources)
        num_citations = len(context.get("citations", []))
        citation_score = min(num_citations / 10.0, 1.0)
        
        # 3. Database coverage (prefer multiple databases)
        num_dbs = len(context.get("databases_searched", []))
        db_score = min(num_dbs / len(self.databases), 1.0) if self.databases else 0.0
        
        # Combined quality score
        quality_score = (length_score + citation_score + db_score) / 3.0
        
        # Continue if score is low and haven't hit max iterations
        should_continue = (
            quality_score < 0.7 and
            iteration < self.max_iterations and
            num_citations > 0
        )
        
        logger.info(
            f"Critic evaluation: quality={quality_score:.2f}, "
            f"continue={should_continue}"
        )
        
        return quality_score, should_continue

    def _refine_query(self, original_prompt: str, context: Dict) -> str:
        """Refine query for next iteration.

        Args:
            original_prompt: Original query
            context: Current context

        Returns:
            Refined query
        """
        # Simple refinement: add context-based keywords
        # In future, could use LLM to generate refined query
        
        # For now, just return original (no refinement)
        return original_prompt
