"""Enhanced retrieval with Router → Retriever → Reranker → Critic pipeline.

This module implements a true "agentic" retrieval system with:
- Router: Determines which databases and strategies to use
- Retriever: Fetches relevant chunks from vector database  
- Reranker: Scores and reorders results for precision
- Critic: Validates result quality and determines if iteration needed
"""

import json
from typing import Dict, List, Optional, Tuple

from mcp_agent_rag.config import Config
from mcp_agent_rag.rag import OllamaEmbedder, OllamaGenerator, VectorDatabase
from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class EnhancedRetrieval:
    """Enhanced retrieval system with multi-stage pipeline.
    
    This implements an improved retrieval pipeline but without a full agent loop.
    For true agentic behavior, use AgenticRAG instead.
    """

    def __init__(
        self,
        config: Config,
        databases: Dict[str, VectorDatabase],
        min_confidence: float = 0.85,
    ):
        """Initialize enhanced retrieval.

        Args:
            config: Configuration instance
            databases: Dictionary of loaded databases
            min_confidence: Minimum confidence score (0-1) for results to be included.
                Results below this threshold are discarded as low quality.
        """
        self.config = config
        self.databases = databases
        self.embedder = OllamaEmbedder(
            model=config.get("embedding_model", "nomic-embed-text"),
            host=config.get("ollama_host", "http://localhost:11434"),
        )
        self.max_context_length = config.get("max_context_length", 4000)
        self.min_confidence = min_confidence

    def get_context(self, prompt: str, max_results: int = 5) -> Dict:
        """Get context for prompt using enhanced retrieval.

        Args:
            prompt: User prompt
            max_results: Maximum results per database

        Returns:
            Dictionary with context text, citations, and databases searched.
            Includes confidence scores for each result. Only results with
            confidence >= min_confidence threshold are included.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_single(prompt)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return {
                "text": "",
                "citations": [],
                "databases_searched": [],
                "average_confidence": 0.0,
            }

        # Search all active databases
        all_results = []
        databases_searched = []

        for db_name, db in self.databases.items():
            try:
                results = db.search(query_embedding, k=max_results)
                for distance, metadata in results:
                    # Convert distance to confidence score (0-1 range)
                    # Lower distance = higher confidence
                    confidence = 1.0 / (1.0 + distance)
                    
                    # Filter by minimum confidence threshold
                    if confidence < self.min_confidence:
                        logger.debug(
                            f"Discarding result with confidence {confidence:.2f} "
                            f"< threshold {self.min_confidence:.2f}"
                        )
                        continue
                    
                    all_results.append({
                        "database": db_name,
                        "distance": distance,
                        "confidence": confidence,
                        "text": metadata.get("text", ""),
                        "source": metadata.get("source", ""),
                        "chunk_num": metadata.get("chunk_num", 0),
                        "metadata": metadata,
                    })
                databases_searched.append(db_name)
                logger.info(f"Found {len(results)} results in database '{db_name}'")
            except Exception as e:
                logger.error(f"Error searching database '{db_name}': {e}")

        # Sort by confidence (higher is better)
        all_results.sort(key=lambda x: x["confidence"], reverse=True)

        # Deduplicate and aggregate
        context_parts = []
        citations = []
        seen_sources = set()
        total_length = 0
        confidence_sum = 0.0
        confidence_count = 0

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
                "confidence": result["confidence"],
            })
            seen_sources.add(source_key)
            total_length += len(chunk_text)
            confidence_sum += result["confidence"]
            confidence_count += 1

        # Compose final context
        context_text = "\n\n".join(context_parts)
        average_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0

        return {
            "text": context_text,
            "citations": citations,
            "databases_searched": databases_searched,
            "average_confidence": average_confidence,
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
    
    The agent prioritizes RAG database retrieval and discards low-confidence
    results (below 85% confidence threshold).
    """
    
    # Constants for text length limits
    MAX_CONTEXT_PREVIEW_LENGTH = 500  # Max length for context preview in iterations
    MAX_AUGMENTATION_CONTEXT_LENGTH = 2000  # Max context length for final augmentation

    def __init__(
        self,
        config: Config,
        databases: Dict[str, VectorDatabase],
        max_iterations: int = 3,
        min_confidence: float = 0.85,
    ):
        """Initialize agentic RAG.

        Args:
            config: Configuration instance
            databases: Dictionary of loaded databases
            max_iterations: Maximum number of retrieval iterations
            min_confidence: Minimum confidence score (0-1) for results to be included.
                Results below this threshold are discarded as low quality.
        """
        self.config = config
        self.databases = databases
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.embedder = OllamaEmbedder(
            model=config.get("embedding_model", "nomic-embed-text"),
            host=config.get("ollama_host", "http://localhost:11434"),
        )
        self.generator = OllamaGenerator(
            model=config.get("generative_model", "mistral:7b-instruct"),
            host=config.get("ollama_host", "http://localhost:11434"),
        )
        self.max_context_length = config.get("max_context_length", 4000)
        
        # Load inference probability thresholds from config
        self.query_inference_threshold = config.get("query_inference_threshold", 0.80)
        self.iteration_confidence_threshold = config.get("iteration_confidence_threshold", 0.90)
        self.final_augmentation_threshold = config.get("final_augmentation_threshold", 0.80)

    def get_context(self, prompt: str, max_results: int = 5) -> Dict:
        """Get context using agentic RAG pipeline.
        
        Uses LLM-based ReAct pattern for intelligent database querying.

        Args:
            prompt: User prompt
            max_results: Maximum results per iteration

        Returns:
            Dictionary with context, citations, metadata
        """
        # Use LLM-based agentic flow
        return self.get_context_with_llm(prompt, max_results)

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
            List of result dictionaries with confidence scores
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
                    # Convert distance to confidence score
                    confidence = 1.0 / (1.0 + distance)
                    
                    # Filter by minimum confidence threshold
                    if confidence < self.min_confidence:
                        logger.debug(
                            f"Discarding result with confidence {confidence:.2f} "
                            f"< threshold {self.min_confidence:.2f}"
                        )
                        continue
                    
                    all_results.append({
                        "database": db_name,
                        "distance": distance,
                        "text": metadata.get("text", ""),
                        "source": metadata.get("source", ""),
                        "chunk_num": metadata.get("chunk_num", 0),
                        "metadata": metadata,
                        "score": confidence,
                        "confidence": confidence,
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
            Context dictionary with confidence scores
        """
        context_parts = []
        citations = []
        seen_sources = set()
        total_length = 0
        databases_searched = set()
        confidence_sum = 0.0
        confidence_count = 0

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
                "confidence": result.get("confidence", result["score"]),
            })
            seen_sources.add(source_key)
            total_length += len(chunk_text)
            databases_searched.add(result["database"])
            confidence_sum += result.get("confidence", result["score"])
            confidence_count += 1

        average_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0

        return {
            "text": "\n\n".join(context_parts),
            "citations": citations,
            "databases_searched": list(databases_searched),
            "average_confidence": average_confidence,
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

    def _build_database_tools(self) -> List[Dict]:
        """Build tool definitions for RAG database capabilities.
        
        Returns:
            List of tool definitions for LLM
        """
        tools = []
        for db_name, db in self.databases.items():
            db_info = self.config.get_database(db_name)
            description = db_info.get("description", f"Database: {db_name}") if db_info else f"Database: {db_name}"
            doc_count = db_info.get("doc_count", 0) if db_info else 0
            
            tools.append({
                "type": "function",
                "function": {
                    "name": f"query_{db_name.replace('-', '_').replace(' ', '_')}",
                    "description": f"{description}. Contains {doc_count} documents. Use this to retrieve information from the {db_name} database.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information in the database"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        return tools

    def _execute_tool_call(self, tool_call: Dict, max_results: int = 5) -> Dict:
        """Execute a tool call (database query) from the LLM.
        
        Args:
            tool_call: Tool call from LLM with function name and arguments
            max_results: Maximum results to return
            
        Returns:
            Dictionary with query results
        """
        function_name = tool_call.get("function", {}).get("name", "")
        arguments = tool_call.get("function", {}).get("arguments", {})
        
        # Parse arguments if they're a JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool arguments: {arguments}")
                return {"error": "Invalid arguments format"}
        
        # Extract database name from function name (query_<db_name>)
        if not function_name.startswith("query_"):
            return {"error": f"Unknown function: {function_name}"}
        
        db_name_normalized = function_name[6:]  # Remove "query_" prefix
        
        # Find matching database
        db_name = None
        for name in self.databases.keys():
            if name.replace('-', '_').replace(' ', '_') == db_name_normalized:
                db_name = name
                break
        
        if not db_name or db_name not in self.databases:
            return {"error": f"Database not found: {db_name_normalized}"}
        
        query = arguments.get("query", "")
        if not query:
            return {"error": "No query provided"}
        
        # Execute the query
        query_embedding = self.embedder.embed_single(query)
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        db = self.databases[db_name]
        try:
            results = db.search(query_embedding, k=max_results)
            
            # Format results
            formatted_results = []
            for distance, metadata in results:
                confidence = 1.0 / (1.0 + distance)
                
                # Filter by minimum confidence
                if confidence < self.min_confidence:
                    continue
                
                formatted_results.append({
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", ""),
                    "chunk": metadata.get("chunk_num", 0),
                    "confidence": confidence,
                })
            
            return {
                "database": db_name,
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results),
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {"error": str(e)}

    def _build_system_prompt(self, tools: List[Dict]) -> str:
        """Build system prompt for LLM with tool descriptions.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            System prompt string
        """
        tool_descriptions = [
            {'name': tool['function']['name'], 'description': tool['function']['description']} 
            for tool in tools
        ]
        
        return f"""You are a helpful AI assistant with access to multiple knowledge databases. 
Your task is to help answer the user's question by querying the appropriate databases.

Instructions:
1. Analyze the user's question carefully
2. Use the available database query tools to retrieve relevant information
3. After receiving results, evaluate if you have enough information to answer the question
4. If there are gaps in the information or the answer is incomplete, generate additional queries
5. Continue until you have sufficient information or determine that the information cannot be found

Available databases and their tools:
{json.dumps(tool_descriptions, indent=2)}

When you have gathered enough information to provide a complete answer, respond with your analysis.
If you need more information to fully answer the question, use the query tools."""

    def _format_retrieved_item(self, item: Dict) -> str:
        """Format a retrieved data item for context display.
        
        Args:
            item: Retrieved data item with 'database' and 'text' keys
            
        Returns:
            Formatted string for display
        """
        text = item['text']
        database = item['database']
        
        if len(text) > self.MAX_CONTEXT_PREVIEW_LENGTH:
            # Truncate at word boundary
            truncated = text[:self.MAX_CONTEXT_PREVIEW_LENGTH]
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]
            return f"From {database}: {truncated}..."
        else:
            return f"From {database}: {text}"

    def get_context_with_llm(self, prompt: str, max_results: int = 5) -> Dict:
        """Get context using LLM-based agentic RAG flow.
        
        This implements the ReAct pattern:
        1. LLM receives user prompt and database tool descriptions
        2. LLM generates queries for RAG databases using tools
        3. Agent retrieves information from databases
        4. LLM evaluates information gaps and generates more queries
        5. Process repeats until information is complete or max iterations reached
        6. LLM generates final prompt augmentation
        
        Args:
            prompt: User prompt
            max_results: Maximum results per database query
            
        Returns:
            Dictionary with augmented context, citations, and metadata
        """
        logger.info("Starting LLM-based agentic RAG flow")
        
        # Build tool definitions for databases
        tools = self._build_database_tools()
        
        if not tools:
            logger.warning("No databases available for querying")
            return {
                "text": "",
                "citations": [],
                "databases_searched": [],
                "average_confidence": 0.0,
                "iterations": 0,
            }
        
        # Build system prompt
        system_prompt = self._build_system_prompt(tools)

        # Track all retrieved information
        all_retrieved_data = []
        all_citations = []
        databases_searched = set()
        iteration = 0
        
        # Iterative retrieval loop
        conversation_history = []
        current_prompt = prompt
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"LLM-based RAG iteration {iteration}/{self.max_iterations}")
            
            # Build context from previous retrievals
            context = ""
            if all_retrieved_data:
                # Format retrieved items with proper truncation
                formatted_items = [
                    self._format_retrieved_item(item) 
                    for item in all_retrieved_data[:5]  # Limit to 5 most recent items
                ]
                context = "Previously retrieved information:\n" + "\n\n".join(formatted_items)
            
            # Call LLM with tools
            response = self.generator.generate_with_tools(
                prompt=current_prompt,
                tools=tools,
                context=context,
                system_prompt=system_prompt,
            )
            
            if not response:
                logger.error("Failed to get LLM response")
                break
            
            tool_calls = response.get("tool_calls", [])
            
            # If no tool calls, LLM thinks it has enough information
            if not tool_calls:
                logger.info("LLM did not request more information")
                break
            
            # Execute tool calls
            new_data_retrieved = False
            for tool_call in tool_calls:
                result = self._execute_tool_call(tool_call, max_results)
                
                if "error" in result:
                    logger.error(f"Tool execution error: {result['error']}")
                    continue
                
                # Store retrieved data
                db_name = result.get("database", "")
                databases_searched.add(db_name)
                
                for item in result.get("results", []):
                    all_retrieved_data.append({
                        "database": db_name,
                        "text": item["text"],
                        "source": item["source"],
                        "chunk": item["chunk"],
                        "confidence": item["confidence"],
                    })
                    all_citations.append({
                        "source": item["source"],
                        "chunk": item["chunk"],
                        "database": db_name,
                        "confidence": item["confidence"],
                    })
                    new_data_retrieved = True
            
            # If no new data was retrieved, stop
            if not new_data_retrieved:
                logger.info("No new data retrieved, stopping iteration")
                break
            
            # Update prompt for next iteration (ask LLM to evaluate gaps)
            current_prompt = f"""Based on the information retrieved so far, evaluate:
1. Do we have sufficient information to answer the original question: "{prompt}"?
2. Are there any information gaps that need to be filled?
3. What additional queries (if any) should we make?

If the information is complete and you can answer the question, stop querying.
If there are gaps or you need more specific information, use the query tools to retrieve it."""
        
        # Build final context
        context_parts = []
        seen_sources = set()
        total_length = 0
        confidence_sum = 0.0
        confidence_count = 0
        
        # Sort by confidence
        all_retrieved_data.sort(key=lambda x: x["confidence"], reverse=True)
        
        for item in all_retrieved_data:
            source_key = f"{item['source']}:{item['chunk']}"
            if source_key in seen_sources:
                continue
            
            text = item["text"]
            if total_length + len(text) > self.max_context_length:
                break
            
            context_parts.append(text)
            seen_sources.add(source_key)
            total_length += len(text)
            confidence_sum += item["confidence"]
            confidence_count += 1
        
        # Generate final augmentation
        final_context = "\n\n".join(context_parts)
        average_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        
        # Ask LLM to create prompt augmentation
        if final_context:
            # Truncate context at word boundary for augmentation
            truncated_context = final_context[:self.MAX_AUGMENTATION_CONTEXT_LENGTH]
            if len(final_context) > self.MAX_AUGMENTATION_CONTEXT_LENGTH:
                last_space = truncated_context.rfind(' ')
                if last_space > 0:
                    truncated_context = truncated_context[:last_space]
            
            augmentation_prompt = f"""Based on the following retrieved information, create a high-quality prompt augmentation 
that enriches the original user prompt with relevant context. 

Original prompt: {prompt}

Retrieved information:
{truncated_context}

Create a concise, well-supported augmentation that helps answer the question."""
            
            augmentation = self.generator.generate(augmentation_prompt, "")
            if augmentation:
                final_context = f"Context augmentation:\n{augmentation}\n\nDetailed information:\n{final_context}"
        
        logger.info(f"Completed LLM-based RAG flow: {iteration} iterations, {len(all_citations)} citations")
        
        return {
            "text": final_context,
            "citations": all_citations,
            "databases_searched": list(databases_searched),
            "average_confidence": average_confidence,
            "iterations": iteration,
        }
