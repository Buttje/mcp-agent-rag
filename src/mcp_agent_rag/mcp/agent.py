"""Agentic RAG implementation.

This module provides backward compatibility by re-exporting the new
EnhancedRetrieval class as AgenticRAG. For the true agentic pipeline
with Router → Retriever → Reranker → Critic, use the AgenticRAG class
from enhanced_rag module.
"""

from mcp_agent_rag.mcp.enhanced_rag import AgenticRAG as TrueAgenticRAG
from mcp_agent_rag.mcp.enhanced_rag import EnhancedRetrieval

# For backward compatibility, AgenticRAG defaults to EnhancedRetrieval
# To use true agentic behavior, explicitly import TrueAgenticRAG
AgenticRAG = EnhancedRetrieval

__all__ = ["AgenticRAG", "TrueAgenticRAG", "EnhancedRetrieval"]
