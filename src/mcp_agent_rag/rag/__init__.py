"""RAG package initialization."""

from mcp_agent_rag.rag.archive_extractor import ArchiveExtractor
from mcp_agent_rag.rag.embedder import OllamaEmbedder
from mcp_agent_rag.rag.extractor import DocumentExtractor, find_files_to_process
from mcp_agent_rag.rag.generator import OllamaGenerator
from mcp_agent_rag.rag.ollama_utils import normalize_ollama_host
from mcp_agent_rag.rag.text_processor import chunk_text, clean_text
from mcp_agent_rag.rag.vector_db import VectorDatabase

__all__ = [
    "ArchiveExtractor",
    "OllamaEmbedder",
    "OllamaGenerator",
    "DocumentExtractor",
    "find_files_to_process",
    "chunk_text",
    "clean_text",
    "VectorDatabase",
    "normalize_ollama_host",
]
