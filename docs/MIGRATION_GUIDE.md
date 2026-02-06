# Migration Guide: Integrating P1-P4 Features

This guide explains how to integrate the new P1-P4 features into the existing MCP-RAG codebase.

## Overview

The P1-P4 features have been implemented as new modules that can be gradually integrated into the existing system. This guide provides step-by-step integration instructions.

## Phase 1: Configuration Migration (Low Risk)

### Step 1: Switch to EnhancedConfig

**File**: `src/mcp_agent_rag/config.py` or create new entry point

**Option A: Gradual Migration**
```python
# Keep existing Config class, add profile support
from mcp_agent_rag.enhanced_config import ConfigProfile

class Config:
    def __init__(self, config_path=None, profile="default"):
        # ... existing code ...
        
        # Add profile support
        if profile != "default":
            profile_config = ConfigProfile.get_profile(profile)
            self.data.update(profile_config)
```

**Option B: Full Migration**
```python
# Replace Config with EnhancedConfig
from mcp_agent_rag.enhanced_config import EnhancedConfig as Config
```

**Testing**: Run existing config tests, verify no breakage

## Phase 2: Enhanced Vector Database (Medium Risk)

### Step 2: Add SQLite Metadata Option

**File**: `src/mcp_agent_rag/database.py`

**Change**:
```python
from mcp_agent_rag.rag import VectorDatabase, EnhancedVectorDatabase

class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.use_enhanced_db = config.get("use_enhanced_vector_db", False)
    
    def create_vector_db(self, db_path):
        if self.use_enhanced_db:
            return EnhancedVectorDatabase(
                db_path,
                use_mmap=self.config.get("use_mmap", True)
            )
        else:
            return VectorDatabase(db_path)
```

**Migration Path**:
1. Test with `use_enhanced_vector_db=False` (default, no changes)
2. Enable for new databases only
3. Provide migration tool to convert existing databases
4. Eventually deprecate old format

**Testing**: 
- Create test database with both implementations
- Verify search results are identical
- Check performance improvements

## Phase 3: Semantic Chunking (Low Risk)

### Step 3: Add Chunking Strategy Option

**File**: `src/mcp_agent_rag/database.py`

**Change** in `add_documents()`:
```python
from mcp_agent_rag.rag import SemanticChunker, chunk_text

# In add_documents method:
use_semantic = self.config.get("use_semantic_chunking", False)

if use_semantic:
    chunker = SemanticChunker(
        chunk_size=self.config.get("chunk_size", 512),
        overlap=self.config.get("chunk_overlap", 50),
        respect_boundaries=True
    )
    chunks = chunker.chunk(text, metadata={
        "source": str(file_path),
        "database": database_name,
    })
else:
    # Existing chunking
    chunks = chunk_text(
        text,
        chunk_size=self.config.get("chunk_size", 512),
        overlap=self.config.get("chunk_overlap", 50),
        metadata={"source": str(file_path), "database": database_name},
    )
```

**Testing**:
- Compare chunk quality on sample documents
- Verify metadata preservation
- Check char_start/char_end offsets

## Phase 4: OCR Processor (Low Risk)

### Step 4: Replace Direct OCR with OCRProcessor

**File**: `src/mcp_agent_rag/rag/extractor.py`

**Change**:
```python
from mcp_agent_rag.rag.ocr_processor import OCRProcessor

class DocumentExtractor:
    # Class-level OCR processor
    _ocr_processor = None
    
    @classmethod
    def get_ocr_processor(cls):
        if cls._ocr_processor is None:
            cls._ocr_processor = OCRProcessor(
                enabled=True,
                min_text_threshold=100,
                force_ocr_for_images=True
            )
        return cls._ocr_processor
    
    @staticmethod
    def extract_text(file_path: Path) -> str | None:
        # ... existing extraction logic ...
        
        # After initial extraction:
        processor = DocumentExtractor.get_ocr_processor()
        text = processor.extract_text_with_fallback(file_path, initial_text=text)
        
        return text
```

**Testing**:
- Test with image files
- Test with PDFs with minimal text
- Verify OCR only runs when needed

## Phase 5: Parallel Processing (Medium Risk)

### Step 5: Add Parallel Option to Ingestion

**File**: `src/mcp_agent_rag/database.py`

**Change** in `add_documents()`:
```python
from mcp_agent_rag.rag import ParallelProcessor

# Add option to use parallel processing
use_parallel = self.config.get("use_parallel_ingestion", False)
parallel_workers = self.config.get("parallel_workers", 4)

if use_parallel and len(files) > 5:
    processor = ParallelProcessor(
        max_workers=parallel_workers,
        ocr_workers=self.config.get("ocr_workers", 2)
    )
    
    def process_single_file(file_path):
        try:
            # Extract, clean, chunk, embed (existing logic)
            return {"success": True, "chunks": num_chunks}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    results = processor.process_files(files, process_single_file)
    
    # Process results
    for file_path, result, error in results:
        if error or not result["success"]:
            stats["failed"] += 1
        else:
            stats["processed"] += 1
else:
    # Existing sequential processing
    for file_path in files:
        # ... existing loop ...
```

**Testing**:
- Test with various file counts
- Verify error handling
- Monitor resource usage
- Compare processing time

## Phase 6: Metrics and Tracing (Low Risk)

### Step 6: Add Metrics Throughout Pipeline

**Files**: Various (`database.py`, `embedder.py`, `extractor.py`)

**Changes**:
```python
from mcp_agent_rag.rag.metrics import get_metrics

# In DatabaseManager.add_documents()
metrics = get_metrics()

for file_path in files:
    with metrics.trace("file_extraction", file=str(file_path)):
        text = DocumentExtractor.extract_text(file_path)
    
    with metrics.trace("text_cleaning"):
        text = clean_text(text)
    
    with metrics.trace("chunking"):
        chunks = chunk_text(text, ...)
    
    with metrics.trace("embedding", batch_size=len(chunks)):
        embeddings = self.embedder.embed(chunk_texts)
        metrics.increment("chunks_embedded", len(chunks))

# At end of ingestion
metrics.log_summary()
```

**Testing**:
- Verify metrics are collected
- Check log output
- Ensure minimal performance impact

## Phase 7: Security Hardening (Medium Risk)

### Step 7: Add URL Validation

**File**: `src/mcp_agent_rag/database.py`

**Change** in `_download_url()`:
```python
from mcp_agent_rag.utils.security import URLSecurityValidator

class DatabaseManager:
    def __init__(self, config):
        # ... existing ...
        self.url_validator = URLSecurityValidator(
            max_size_mb=config.get("max_download_size_mb", 100),
            timeout_seconds=config.get("download_timeout", 30),
            allow_private_ips=config.get("allow_private_ips", False)
        )
    
    def _download_url(self, url: str) -> Path:
        # Validate URL
        is_valid, error = self.url_validator.validate_url(url)
        if not is_valid:
            raise ValueError(f"URL validation failed: {error}")
        
        # Secure download
        content, error = self.url_validator.download_url(url)
        if error:
            raise ValueError(f"Download failed: {error}")
        
        # Save to temp file
        import tempfile
        suffix = Path(url).suffix or ".html"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        
        return Path(temp_file.name)
```

**Testing**:
- Test with valid URLs
- Test with private IPs (should block)
- Test with large files (should block)
- Test with invalid content types

## Phase 8: Enhanced MCP Tools (Low Risk)

### Step 8: Register Additional Tools

**File**: `src/mcp_agent_rag/mcp/server.py`

**Change**:
```python
from mcp_agent_rag.mcp.enhanced_tools import EnhancedMCPTools

class MCPServer:
    def __init__(self, config, active_databases, protocol_version):
        # ... existing ...
        self.enhanced_tools = EnhancedMCPTools(config, self.db_manager)
    
    def _tools_list(self, params: Dict) -> Dict:
        """Handle tools/list request."""
        # Existing tools
        tools = [
            # ... getDatabases, getInformationFor, etc. ...
        ]
        
        # Add enhanced tools
        enable_enhanced = self.config.get("enable_enhanced_tools", True)
        if enable_enhanced:
            enhanced_tool_defs = self.enhanced_tools.get_tool_definitions(
                prefix=self.tool_prefix
            )
            tools.extend(enhanced_tool_defs)
        
        return {"tools": tools}
    
    def handle_request(self, request: Dict) -> Dict:
        method = request.get("method", "")
        
        # Existing tool handlers
        if method == f"{self.tool_prefix}getDatabases":
            # ...
        
        # Enhanced tool handlers
        elif method == f"{self.tool_prefix}ingest":
            return self.enhanced_tools.ingest(params)
        elif method == f"{self.tool_prefix}status":
            return self.enhanced_tools.status(params)
        elif method == f"{self.tool_prefix}search":
            return self.enhanced_tools.search(params)
        # ... etc ...
```

**Testing**:
- Test each new tool via MCP protocol
- Verify job tracking works
- Test metadata filtering in search

## Phase 9: Structured Citations (Low Risk)

### Step 9: Use CitationBuilder in Responses

**File**: `src/mcp_agent_rag/mcp/agent.py` or wherever citations are built

**Change**:
```python
from mcp_agent_rag.utils.citations import CitationBuilder

def get_context(self, query: str, max_results: int = 5) -> Dict:
    # ... existing search logic ...
    
    # Build structured citations
    citations = CitationBuilder.build_citations(
        search_results,
        include_text=True,
        max_text_length=500
    )
    
    # Deduplicate and sort
    citations = CitationBuilder.deduplicate_citations(citations)
    citations = CitationBuilder.sort_citations(citations, by="score")
    
    # Format for response
    citation_dicts = [c.to_dict() for c in citations]
    
    return {
        "text": combined_text,
        "citations": citation_dicts,
        "databases_searched": databases
    }
```

**Testing**:
- Verify locator strings are correct
- Test with PDFs, PPTX, XLSX
- Check citation deduplication

## Phase 10: Enhanced Export/Import (Low Risk)

### Step 10: Add Enhanced Export Commands

**File**: `src/mcp_agent_rag/cli.py`

**Changes**:
```python
from mcp_agent_rag.utils.enhanced_export import EnhancedDatabaseExporter

# Add new CLI commands or enhance existing ones

def export_databases_enhanced(args):
    """Enhanced database export."""
    config = Config()
    exporter = EnhancedDatabaseExporter(config)
    
    success = exporter.export_databases(
        database_names=args.databases.split(","),
        output_path=Path(args.output),
        include_metadata=args.include_metadata,
        include_cache=args.include_cache
    )
    
    if success:
        # Show export info
        info = exporter.get_export_info(Path(args.output))
        print(json.dumps(info, indent=2))

def import_databases_enhanced(args):
    """Enhanced database import."""
    config = Config()
    exporter = EnhancedDatabaseExporter(config)
    
    results = exporter.import_databases(
        zip_path=Path(args.file),
        overwrite=args.overwrite,
        validate_compatibility=True
    )
    
    # Show results and warnings
    print(f"Success: {len(results['success'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")
```

**Testing**:
- Export and re-import databases
- Test compatibility warnings
- Verify metadata preservation

## Rollback Strategy

For each phase:

1. **Feature Flags**: Add config options to enable/disable new features
   ```python
   config.get("use_enhanced_vector_db", False)  # Default off
   config.get("use_semantic_chunking", False)   # Default off
   config.get("use_parallel_ingestion", False)  # Default off
   ```

2. **Logging**: Add detailed logging for new code paths
   ```python
   logger.info("Using enhanced vector database")
   logger.info("Using semantic chunking")
   ```

3. **Monitoring**: Track metrics to compare old vs. new
   ```python
   metrics.increment("vector_db_type", "enhanced" if use_enhanced else "legacy")
   ```

4. **Gradual Rollout**:
   - Enable for new databases first
   - Monitor performance and errors
   - Migrate existing databases if successful

## Performance Tuning

### Parallel Processing
- Start with `max_workers=4`, tune based on CPU count
- Keep `ocr_workers=2` unless OCR-heavy workload
- Increase `max_queue_size` if memory allows

### Semantic Chunking
- Use only for structured documents (Markdown, code)
- Keep simple chunking for plain text
- Set `respect_boundaries=True` for quality, `False` for speed

### Memory-Mapped FAISS
- Enable for databases > 1GB
- Disable for frequently updated databases
- Monitor search latency vs. load time trade-off

### Hybrid Retrieval
- Start with `alpha=0.5` (equal weight)
- Increase alpha (> 0.5) for semantic queries
- Decrease alpha (< 0.5) for keyword queries
- Monitor precision/recall metrics

## Migration Checklist

- [ ] Phase 1: Config migration
  - [ ] Add profile support
  - [ ] Test with existing configs
  - [ ] Update documentation

- [ ] Phase 2: Enhanced vector DB
  - [ ] Add feature flag
  - [ ] Test with new database
  - [ ] Plan migration for existing databases

- [ ] Phase 3: Semantic chunking
  - [ ] Add chunking strategy option
  - [ ] Test on sample documents
  - [ ] Compare quality metrics

- [ ] Phase 4: OCR processor
  - [ ] Integrate OCRProcessor
  - [ ] Test fallback logic
  - [ ] Verify performance

- [ ] Phase 5: Parallel processing
  - [ ] Add parallel option
  - [ ] Test with various file counts
  - [ ] Monitor resource usage

- [ ] Phase 6: Metrics
  - [ ] Add tracing throughout
  - [ ] Verify log output
  - [ ] Set up monitoring

- [ ] Phase 7: Security
  - [ ] Integrate URL validator
  - [ ] Test SSRF protection
  - [ ] Update documentation

- [ ] Phase 8: Enhanced tools
  - [ ] Register new MCP tools
  - [ ] Test via MCP protocol
  - [ ] Update API docs

- [ ] Phase 9: Citations
  - [ ] Use CitationBuilder
  - [ ] Test locator formats
  - [ ] Update response schemas

- [ ] Phase 10: Export/import
  - [ ] Add enhanced commands
  - [ ] Test compatibility checks
  - [ ] Update user guide

## Timeline Recommendation

- **Week 1**: Phases 1-3 (Config, Vector DB, Chunking)
- **Week 2**: Phases 4-6 (OCR, Parallel, Metrics)
- **Week 3**: Phases 7-8 (Security, Tools)
- **Week 4**: Phases 9-10 (Citations, Export/Import)
- **Week 5**: Testing, tuning, documentation

## Support

For questions or issues during migration:
- Check `docs/P1-P4_IMPLEMENTATION.md` for detailed feature docs
- Review unit tests in `tests/unit/test_*.py`
- Check module docstrings for API details

## Next Steps

After successful migration:
1. Enable features gradually via profiles
2. Monitor metrics and performance
3. Collect user feedback
4. Plan for advanced features (cross-encoder reranking, etc.)
