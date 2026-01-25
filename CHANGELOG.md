# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Updated Ollama API integration to use `/api/chat` endpoint instead of deprecated `/api/generate` endpoint (removed in Ollama 0.4.0)
- Fixed 404 errors when connecting to Ollama server with version 0.4.0 or later

### Changed
- Refactored request format from single prompt string to messages array for better compatibility with Ollama's chat API
- Context is now passed as a system message instead of being concatenated to the prompt

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
