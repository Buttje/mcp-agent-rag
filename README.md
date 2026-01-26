# MCP-RAG: Model Context Protocol Server with Retrieval Augmented Generation

A Python implementation of a Model Context Protocol (MCP) server that provides retrieval-augmented generation (RAG) capabilities using Ollama embeddings and FAISS vector storage.

## Features

- **Multi-Format Document Support**: Ingest .txt, .docx, .xlsx, .pptx, .odt, .ods, .odp, .pdf, HTML, and source code files
- **Smart RAG Pipeline**: Automatic text extraction, cleaning, chunking with overlap, and embedding generation
- **Vector Database**: FAISS-based indexing with metadata persistence
- **Agentic RAG**: Intelligent context retrieval with deduplication and citation tracking
- **Interactive Chat Client**: Built-in CLI chat interface for natural conversations with your documents
- **MCP Compliance**: Full JSON-RPC server implementing Model Context Protocol
- **CLI Interface**: Easy-to-use command-line tools for database and server management
- **Cross-Platform**: Works on Windows 10/11 and Ubuntu 22.04 LTS
- **Comprehensive Testing**: 117 tests with 81% coverage

## Requirements

- Python 3.10 or higher
- Ollama (for embeddings and generation)
- ~1GB disk space for models

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Buttje/mcp-agent-rag.git
cd mcp-agent-rag

# Run installation script
python install.py

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Verify installation
mcp-rag --help
```

### Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

Pull required models:
```bash
ollama pull nomic-embed-text
ollama pull mistral:7b-instruct
```

## Usage

### Create a Database

```bash
mcp-rag database create --name mydb --description "My document collection"
```

### Add Documents

Add from a directory:
```bash
mcp-rag database add --database mydb --path ~/documents --recursive
```

Add from URL:
```bash
mcp-rag database add --database mydb --url https://example.com/doc.pdf
```

Add specific file types:
```bash
mcp-rag database add --database mydb --path ~/code --glob "*.py" --recursive
```

### List Databases

```bash
mcp-rag database list
```

### Start MCP Server

```bash
mcp-rag server start --active-databases mydb --transport stdio
```

### Interactive Chat Client

The **mcp-rag-cli** uses the AGNO agent framework and communicates with an MCP server:

```bash
mcp-rag-cli
```

The CLI client will:
1. Display all available databases
2. Let you select which database(s) to use (single, multiple, or all)
3. **Start an MCP server** with the selected databases
4. Initialize an AGNO agent with MCP tools
5. Start an interactive chat session where you can ask questions
6. Query the MCP server to retrieve relevant context from your documents
7. Properly shut down the MCP server when you exit

**Example Session:**
```
$ mcp-rag-cli

Available databases:
  1. mydb (42 documents)
     My document collection

Select database(s) (number, name, or 'all'): 1

Starting MCP server...
MCP server started successfully!

Chat started! Type 'quit', 'exit', or '/q' to exit
======================================================================

You: What is Python used for?
Searching databases...
Assistant: Python is used for web development...

Sources:
  - /path/to/python-intro.txt

You: quit
Goodbye!
```

## MCP Tools

The server exposes the following MCP tools:

### database/create
Create a new database with a unique name.

**Parameters:**
- `name` (string, required): Database name
- `description` (string, optional): Database description

### database/add
Add documents to an existing database.

**Parameters:**
- `database_name` (string, required): Target database name
- `path` (string, optional): File or directory path
- `url` (string, optional): URL to download
- `glob` (string, optional): Glob pattern for file matching
- `recursive` (boolean, optional): Traverse subdirectories
- `skip_existing` (boolean, optional): Skip already-indexed files

### database/list
List all databases with metadata.

### query/get_data
Retrieve context for a user's prompt using agentic RAG.

**Parameters:**
- `prompt` (string, required): The query
- `max_results` (integer, optional, default: 5): Maximum results per database

## Configuration

Configuration is stored in `~/.mcp-agent-rag/config.json`:

```json
{
  "embedding_model": "nomic-embed-text",
  "generative_model": "mistral:7b-instruct",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "ollama_host": "http://localhost:11434",
  "log_level": "INFO",
  "max_context_length": 4000,
  "databases": {}
}
```

## Testing

Run the test suite:
```bash
pytest tests/
```

With coverage report:
```bash
pytest tests/ --cov=src/mcp_agent_rag --cov-report=html
open htmlcov/index.html
```

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Project Structure

```
mcp-agent-rag/
├── install.py              # Installation script
├── pyproject.toml          # Project configuration
├── CHANGELOG.md            # Version history
├── TESTING.md              # Testing documentation
├── src/
│   └── mcp_agent_rag/
│       ├── cli.py          # Command-line interface
│       ├── config.py       # Configuration management
│       ├── database.py     # Database manager
│       ├── mcp/            # MCP server and agentic RAG
│       ├── rag/            # RAG pipeline components
│       └── utils/          # Utilities
└── tests/
    ├── unit/               # Unit tests
    └── acceptance/         # Acceptance tests
```

## Supported File Formats

- **Text**: .txt, .md
- **Office**: .docx, .xlsx, .pptx
- **OpenDocument**: .odt, .ods, .odp
- **PDF**: .pdf
- **Web**: .html, .htm
- **Source Code**: .py, .c, .cpp, .h, .hpp, .cs, .go, .rs, .java, .js, .ts, .sh, .bat, .ps1, .s, .asm
- **Build Files**: Makefile, CMakeLists.txt

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Acknowledgments

- [Ollama](https://ollama.ai) for embeddings and generation
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Model Context Protocol](https://modelcontextprotocol.io) specification
