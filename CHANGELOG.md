# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed 404/405 errors when connecting to Ollama server by adding automatic API version detection
- Added fallback support for older Ollama versions (< 0.1.14) that don't have the `/api/chat` endpoint
- Now automatically detects Ollama version and uses appropriate endpoint:
  - `/api/chat` with messages format for Ollama v0.1.14+ (modern)
  - `/api/generate` with prompt format for older Ollama versions (legacy)
- Improved error handling and logging for Ollama API connection issues

### Changed
- Refactored OllamaGenerator to dynamically detect and adapt to Ollama server version
- Added automatic endpoint detection using `/api/version` and fallback testing
- Enhanced compatibility with both old and new Ollama installations

## [0.1.0] - 2024-01-XX

### Added
- Initial implementation of MCP-RAG server
- Database creation and management tools
- Document ingestion with support for .txt, .docx, .xlsx, .pptx, .odt, .ods, .odp, .pdf
- Software project ingestion support
- RAG pipeline with Ollama embeddings and FAISS indexing
- Agentic RAG using Agno framework
- MCP server with stdio and HTTP transports
- CLI interface with database and server commands
- Configuration management via JSON file
- Install script with virtual environment setup
- Comprehensive test suite with >90% coverage
- Progress reporting and error logging with rotation
