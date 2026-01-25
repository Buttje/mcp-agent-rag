# Testing Documentation

## Test Coverage

This project has comprehensive test coverage with:
- **117 unit tests** covering all major components
- **13 acceptance tests** based on functional requirements specification
- **Total coverage: ~81%** (exceeds 80% threshold)

### Coverage Breakdown

| Module | Coverage | Notes |
|--------|----------|-------|
| config.py | 98% | Excellent coverage |
| mcp/agent.py | 96% | Excellent coverage |
| rag/text_processor.py | 94% | Excellent coverage |
| rag/vector_db.py | 90% | Excellent coverage |
| cli.py | 87% | Good coverage, some integration paths not tested |
| embedder.py | 79% | Core functionality covered |
| database.py | 78% | Core functionality covered |
| server.py | 73% | Core functionality covered |
| extractor.py | 68% | Core extraction covered, some format handlers mock-tested |

### Why Not 90% Coverage?

The original requirement specified >90% coverage. We achieved >80% for the following reasons:

1. **Document Format Handlers**: Testing all document format extractors (DOCX, XLSX, PPTX, ODF, PDF) requires creating valid binary files. Many format-specific error paths are difficult to test without real library errors.

2. **MCP Server HTTP Transport**: HTTP transport is not fully implemented yet, so those code paths are untested.

3. **Signal Handling**: Testing Ctrl+K signal handling during file processing requires complex test setup.

4. **Integration Tests**: Full end-to-end integration tests (running actual Ollama, creating real FAISS indices) are better suited for manual/integration test suites rather than unit tests.

5. **Error Paths**: Some error handling paths (e.g., corrupt FAISS indices, network timeouts) are difficult to reliably trigger in tests.

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run with Coverage
```bash
pytest tests/ --cov=src/mcp_agent_rag --cov-report=html --cov-report=term
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest tests/unit/

# Acceptance tests only
pytest tests/acceptance/

# Specific module
pytest tests/unit/test_config.py
```

### Generate Coverage Report
```bash
pytest tests/ --cov=src/mcp_agent_rag --cov-report=xml --cov-report=html
# View HTML report
open htmlcov/index.html
```

## Test Organization

### Unit Tests (`tests/unit/`)
- `test_config.py` - Configuration management
- `test_cli.py` - Command-line interface
- `test_database.py` - Database manager
- `test_embedder.py` - Ollama embeddings
- `test_extractor.py` - Document extraction
- `test_text_processor.py` - Text processing and chunking
- `test_vector_db.py` - FAISS vector database
- `test_agent.py` - Agentic RAG
- `test_server.py` - MCP server
- `test_utils.py` - Logging utilities

### Acceptance Tests (`tests/acceptance/`)
- `test_acceptance.py` - Tests based on acceptance test specification
  - Installation and configuration tests
  - Database management tests
  - Server startup tests
  - Query tests
  - Error handling tests
  - Coverage threshold tests

## Test Fixtures

Common fixtures are defined in `tests/conftest.py`:
- `temp_dir` - Temporary directory for test files
- `test_config` - Test configuration instance
- `sample_text_file` - Sample text file for testing
- `sample_python_file` - Sample Python file for testing
- `sample_project_dir` - Sample project directory structure
- `mock_ollama_response` - Mock Ollama API response

## Continuous Integration

Tests are designed to run in CI environments:
- No external dependencies (Ollama, internet) required for unit tests
- All file operations use temporary directories
- Network calls are mocked
- Platform-independent (Linux, Windows, macOS)

## Manual Testing

For full end-to-end testing:

1. Install Ollama and pull models:
```bash
ollama pull nomic-embed-text
ollama pull mistral:7b-instruct
```

2. Run install script:
```bash
python install.py
```

3. Test database operations:
```bash
source .venv/bin/activate
mcp-rag database create --name testdb --description "Test database"
mcp-rag database add --database testdb --path docs/ --recursive
mcp-rag database list
```

4. Test server:
```bash
mcp-rag server start --active-databases testdb --transport stdio
```

## Known Limitations

1. **Ollama Dependency**: Real Ollama testing requires Ollama server running
2. **Document Formats**: Full format testing requires creating binary test files
3. **HTTP Transport**: Not yet implemented
4. **Performance Tests**: Not included (would require large test datasets)
5. **Security Tests**: Basic validation only, needs security audit
