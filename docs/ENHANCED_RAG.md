# Enhanced Retrieval vs Agentic RAG

This document explains the difference between `EnhancedRetrieval` and `AgenticRAG` classes in the MCP-RAG system.

## EnhancedRetrieval

`EnhancedRetrieval` is an improved single-pass retrieval system that:
- Generates embeddings for queries
- Searches across multiple databases in parallel
- Deduplicates results
- Aggregates context up to a maximum length
- Returns results with citations

**Use when:** You want fast, single-shot retrieval without iterative refinement.

## AgenticRAG

`AgenticRAG` implements a true agentic pipeline with bounded iterations (2-3 max):

### Pipeline Stages:

1. **Router** - Analyzes the query and determines:
   - Which databases to search
   - What retrieval strategy to use
   - How to approach the query

2. **Retriever** - Fetches candidate chunks:
   - Generates query embeddings
   - Searches selected databases
   - Returns raw results with scores

3. **Reranker** - Improves result precision:
   - Re-scores results for relevance
   - Reorders based on query-document similarity
   - Can use cross-encoder models (future enhancement)

4. **Critic** - Evaluates quality and decides on iteration:
   - Assesses context quality (length, diversity, coverage)
   - Determines if results are sufficient
   - Decides whether to iterate with refined query
   - Maximum 3 iterations to prevent runaway loops

### Quality Metrics:

The Critic evaluates results based on:
- **Context Length**: Prefers fuller context (up to max_context_length)
- **Citation Diversity**: Prefers results from multiple sources
- **Database Coverage**: Prefers searching across multiple databases

### Iteration Logic:

- **Iteration 1**: Initial retrieval with original query
- **Iteration 2-3**: Refined retrieval if quality score < 0.7
- **Exit**: When quality ≥ 0.7 OR max iterations reached

**Use when:** You need the highest quality retrieval and can afford the extra latency of multiple iterations.

## Performance Comparison

| Feature | EnhancedRetrieval | AgenticRAG |
|---------|-------------------|------------|
| Latency | ~100-200ms | ~300-600ms (1-3 iterations) |
| Quality | Good | Better (iterative refinement) |
| Complexity | Simple | Multi-stage pipeline |
| Use Case | Fast queries | High-quality retrieval |

## Future Enhancements

Both systems can be enhanced with:
- **Hybrid Retrieval**: BM25 + vector search
- **Advanced Reranking**: Cross-encoder models, LLM-based scoring
- **Query Refinement**: LLM-based query reformulation
- **Semantic Chunking**: Structure-aware text splitting
- **MMR Sampling**: Maximal Marginal Relevance for diversity
