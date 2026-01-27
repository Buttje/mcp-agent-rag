# MCP-RAG: Model Context Protocol Server with Retrieval Augmented Generation

A Python implementation of a Model Context Protocol (MCP) server that provides retrieval-augmented generation (RAG) capabilities using Ollama embeddings and FAISS vector storage.

## Features

- **Multi-Format Document Support**: Ingest .txt, .docx, .xlsx, .pptx, .odt, .ods, .odp, .pdf, HTML, and source code files
- **Smart RAG Pipeline**: Automatic text extraction, cleaning, chunking with overlap, and embedding generation
- **Vector Database**: FAISS-based indexing with metadata persistence
- **Agentic RAG**: Intelligent context retrieval with deduplication and citation tracking
- **Interactive Chat Client**: Built-in CLI chat interface for natural conversations with your documents
- **MCP Compliance**: Full JSON-RPC server implementing Model Context Protocol (2025-11-25 and 2024-11-05)
- **CODY Compatibility**: Optional `--cody` flag for Sourcegraph CODY integration using MCP protocol version 2024-11-05
- **Multiple Transport Protocols**: Support for stdio, HTTP, and SSE transports
- **Required MCP Tools**: `getDatabases()`, `getInformationFor()`, `getInformationForDB()`
- **Resource Templates**: Dynamic resource discovery via `resources/templates/list`
- **CLI Interface**: Easy-to-use command-line tools for database and server management
- **Cross-Platform**: Works on Windows 10/11 and Ubuntu 22.04 LTS
- **Comprehensive Testing**: 241+ tests with 73%+ coverage

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
python mcp-rag.py --help
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
python mcp-rag.py database create --name mydb --description "My document collection"
```

### Add Documents

Add from a directory:
```bash
python mcp-rag.py database add --database mydb --path ~/documents --recursive
```

Add from URL:
```bash
python mcp-rag.py database add --database mydb --url https://example.com/doc.pdf
```

Add specific file types:
```bash
python mcp-rag.py database add --database mydb --path ~/code --glob "*.py" --recursive
```

### List Databases

```bash
python mcp-rag.py database list
```

### Start MCP Server

The MCP server supports three transport protocols: stdio, HTTP, and SSE. By default, it uses MCP protocol version 2025-11-25.

**stdio transport (for local/CLI usage):**
```bash
python mcp-rag.py server start --active-databases mydb --transport stdio
```

**HTTP transport (for remote/API usage):**
```bash
python mcp-rag.py server start --active-databases mydb --transport http --host 127.0.0.1 --port 8080
```

**SSE transport (deprecated, for backwards compatibility):**
```bash
python mcp-rag.py server start --active-databases mydb --transport sse --host 127.0.0.1 --port 8080
```

**CODY compatibility mode (uses MCP protocol version 2024-11-05):**
```bash
python mcp-rag.py server start --active-databases mydb --transport stdio --cody
```

You can activate multiple databases by providing a comma-separated list:
```bash
python mcp-rag.py server start --active-databases db1,db2,db3 --transport stdio
```

### Interactive Chat Client

The **mcp-rag-cli.py** uses the AGNO agent framework and communicates with an MCP server:

```bash
python mcp-rag-cli.py
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
$ python mcp-rag-cli.py

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

## MCP Protocol Support

This implementation conforms to the [Model Context Protocol specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25).

### Transport Protocols

- **stdio**: Standard input/output transport for local processes and IDEs
- **HTTP**: HTTP POST transport for remote/concurrent access with CORS support
- **SSE**: Server-Sent Events transport (deprecated, provided for backwards compatibility)

All transports use JSON-RPC 2.0 for message formatting.

### MCP Tools

The server implements these MCP-compliant tools as specified in the requirements:

#### getDatabases()
Returns the list of activated databases in the MCP RAG server.

**Parameters:** None

**Returns:**
```json
{
  "databases": [
    {
      "name": "mydb",
      "description": "My document collection",
      "doc_count": 42,
      "last_updated": "2024-01-15T10:30:00",
      "path": "/path/to/db"
    }
  ],
  "count": 1
}
```

#### getInformationFor(Prompt)
Returns information by scanning through all activated databases using vector similarity search.

**Parameters:**
- `prompt` (string, required): The query/prompt to search for
- `max_results` (integer, optional): Maximum results per database (default: 5)

**Returns:**
```json
{
  "prompt": "What is Python?",
  "context": "Relevant text chunks...",
  "citations": [
    {"source": "python.txt", "chunk": 0, "database": "mydb"}
  ],
  "databases_searched": ["mydb", "otherdb"]
}
```

#### getInformationForDB(Prompt, DatabaseName)
Returns information by scanning just the named database.

**Parameters:**
- `prompt` (string, required): The query/prompt to search for
- `database_name` (string, required): Name of the specific database to search
- `max_results` (integer, optional): Maximum results (default: 5)

**Returns:**
```json
{
  "prompt": "What is Python?",
  "database": "mydb",
  "context": "Relevant text chunks...",
  "citations": [
    {"source": "python.txt", "chunk": 0, "database": "mydb"}
  ]
}
```

### Legacy Tools (also available)

The server also exposes these additional tools for database management:

### database-create
Create a new database with a unique name.

**Parameters:**
- `name` (string, required): Database name
- `description` (string, optional): Database description

### database-add
Add documents to an existing database.

**Parameters:**
- `database_name` (string, required): Target database name
- `path` (string, optional): File or directory path
- `url` (string, optional): URL to download
- `glob` (string, optional): Glob pattern for file matching
- `recursive` (boolean, optional): Traverse subdirectories
- `skip_existing` (boolean, optional): Skip already-indexed files

### database-list
List all databases with metadata.

### query-get_data
Retrieve context for a user's prompt using agentic RAG.

**Parameters:**
- `prompt` (string, required): The query
- `max_results` (integer, optional, default: 5): Maximum results per database

## Usage Examples

### Using MCP Tools with HTTP Transport

Start the server with HTTP transport:
```bash
python mcp-rag.py server start --active-databases mydb,otherdb --transport http --port 8080
```

Call the `getDatabases` tool:
```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getDatabases",
    "params": {}
  }'
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "databases": [
      {"name": "mydb", "description": "My docs", "doc_count": 42},
      {"name": "otherdb", "description": "Other docs", "doc_count": 15}
    ],
    "count": 2
  }
}
```

Call the `getInformationFor` tool to search all databases:
```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "getInformationFor",
    "params": {
      "prompt": "What is Python used for?",
      "max_results": 5
    }
  }'
```

Call the `getInformationForDB` tool to search a specific database:
```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "getInformationForDB",
    "params": {
      "prompt": "Python programming",
      "database_name": "mydb",
      "max_results": 3
    }
  }'
```

### Using MCP Tools via tools/call

All tools can also be invoked via the standard `tools/call` method:

```bash
curl -X POST http://localhost:8080/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "getDatabases",
      "arguments": {}
    }
  }'
```

### Health Check Endpoint

Both HTTP and SSE transports provide a health check endpoint:
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "active_databases": ["mydb", "otherdb"]
}
```

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
├── mcp-rag.py              # Main CLI entry point
├── mcp-rag-cli.py          # Interactive chat CLI entry point
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
