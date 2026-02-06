"""RAG package initialization."""

from mcp_agent_rag.rag.archive_extractor import ArchiveExtractor
from mcp_agent_rag.rag.bm25 import BM25Index, HybridRetriever
from mcp_agent_rag.rag.cache import EmbeddingCache
from mcp_agent_rag.rag.embedder import OllamaEmbedder
from mcp_agent_rag.rag.enhanced_vector_db import EnhancedVectorDatabase
from mcp_agent_rag.rag.extractor import DocumentExtractor, find_files_to_process
from mcp_agent_rag.rag.generator import OllamaGenerator
from mcp_agent_rag.rag.manifest import FileManifest
from mcp_agent_rag.rag.metrics import MetricsCollector, get_metrics, reset_metrics
from mcp_agent_rag.rag.ocr_processor import OCRProcessor
from mcp_agent_rag.rag.ollama_utils import normalize_ollama_host
from mcp_agent_rag.rag.parallel_processor import ParallelProcessor
from mcp_agent_rag.rag.reranker import ChainReranker, MMRReranker, SimpleReranker
from mcp_agent_rag.rag.semantic_chunker import SemanticChunker
from mcp_agent_rag.rag.text_processor import chunk_text, clean_text
from mcp_agent_rag.rag.vector_db import VectorDatabase

__all__ = [
    "ArchiveExtractor",
    "BM25Index",
    "ChainReranker",
    "EmbeddingCache",
    "EnhancedVectorDatabase",
    "FileManifest",
    "HybridRetriever",
    "MetricsCollector",
    "MMRReranker",
    "OCRProcessor",
    "OllamaEmbedder",
    "OllamaGenerator",
    "ParallelProcessor",
    "SimpleReranker",
    "SemanticChunker",
    "DocumentExtractor",
    "find_files_to_process",
    "chunk_text",
    "clean_text",
    "VectorDatabase",
    "get_metrics",
    "normalize_ollama_host",
    "reset_metrics",
]
