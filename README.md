# MCP-RAG: Model Context Protocol Server with Retrieval Augmented Generation

A Python implementation of a Model Context Protocol (MCP) server that provides retrieval-augmented generation (RAG) capabilities using Ollama embeddings and FAISS vector storage.

## Features

- **Multi-Format Document Support**: Ingest .txt, .docx, .xlsx, .pptx, .odt, .ods, .odp, .pdf, HTML, images, and source code files
- **Image OCR Support**: Extract text from images (PNG, JPEG, GIF, BMP, TIFF, WebP) and embedded images in documents using EasyOCR
- **Smart RAG Pipeline**: Automatic text extraction, cleaning, chunking with overlap, and embedding generation
- **Vector Database**: FAISS-based indexing with metadata persistence
- **Import/Export**: Export and import databases as ZIP files for easy sharing and backup
- **Agentic RAG**: Intelligent context retrieval with deduplication and citation tracking
- **Interactive Chat Client**: Built-in CLI chat interface for natural conversations with your documents
- **MCP Compliance**: Full JSON-RPC server implementing Model Context Protocol (2025-11-25 and 2024-11-05)
- **CODY Compatibility**: Optional `--cody` flag for Sourcegraph CODY integration using MCP protocol version 2024-11-05
- **Multiple Transport Protocols**: Support for stdio, HTTP, and SSE transports
- **Required MCP Tools**: `getDatabases()`, `getInformationFor()`, `getInformationForDB()`
- **Resource Templates**: Dynamic resource discovery via `resources/templates/list`
- **CLI Interface**: Easy-to-use command-line tools for database and server management
- **Cross-Platform**: Works on Windows 10/11 and Ubuntu 22.04 LTS
- **Comprehensive Testing**: 350+ tests with 73%+ coverage
- **GPU Acceleration**: Optional GPU support for faster OCR processing with automatic detection and setup

## Requirements

- Python 3.10 or higher
- Ollama (for embeddings and generation)
- ~1GB disk space for models
- Optional: NVIDIA GPU with CUDA support for accelerated OCR processing

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

### GPU Support (Optional)

The installer automatically detects GPU availability and offers to install PyTorch with GPU support for accelerated OCR processing.

#### Automatic GPU Detection

When you run `python install.py`, the installer will:

1. **Detect GPU Hardware**: Check for NVIDIA GPUs using `nvidia-smi`
2. **Check CUDA Toolkit**: Verify if CUDA toolkit is installed
3. **Offer PyTorch Installation**: Prompt to install PyTorch with CUDA support if GPU is available
4. **Provide Instructions**: Display step-by-step instructions if manual driver installation is needed

#### Manual GPU Setup

If you need to install GPU drivers manually:

**For NVIDIA GPUs on Linux:**
```bash
# Install NVIDIA drivers
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Install CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# Reboot
sudo reboot

# Verify installation
nvidia-smi
nvcc --version

# Re-run installer to install PyTorch with GPU support
python install.py
```

**For NVIDIA GPUs on Windows:**
1. Download NVIDIA drivers from [nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
2. Download CUDA Toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. Install both and restart your computer
4. Re-run `python install.py` to install PyTorch with GPU support

**For Apple Silicon (M1/M2/M3) Macs:**
```bash
# PyTorch with MPS (Metal Performance Shaders) support is automatically installed
# No additional drivers needed
```

#### GPU Configuration

GPU usage is enabled by default when available. You can configure it in `~/.mcp-agent-rag/config.json`:

```json
{
  "gpu_enabled": true,
  "gpu_device": null
}
```

- `gpu_enabled`: Set to `true` to enable GPU usage, `false` to force CPU-only mode
- `gpu_device`: Set to a specific GPU index (e.g., 0, 1) or `null` for automatic selection

#### Performance Benefits

With GPU acceleration:
- **OCR Processing**: 3-10x faster text extraction from images
- **Large Documents**: Faster processing of PDFs and documents with embedded images
- **Batch Operations**: Significantly improved throughput when adding multiple documents

#### Troubleshooting

If GPU is not being detected:
1. Verify drivers are installed: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Test PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check logs for GPU detection messages during OCR initialization

## Usage

### Create a Database

Create a basic database:
```bash
python mcp-rag.py database create --name mydb --description "My document collection"
```

Create a database with a prefix (for multi-server scenarios):
```bash
python mcp-rag.py database create --name mydb --description "My document collection" --prefix "MY"
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

### Export Databases

Export one or more databases to a ZIP file for sharing or backup:

```bash
# Export a single database
python mcp-rag.py database export --databases mydb --output backup.zip

# Export multiple databases
python mcp-rag.py database export --databases "db1,db2,db3" --output multi-backup.zip
```

The exported ZIP file contains:
- A JSON manifest with all database metadata (name, description, doc_count, last_updated, prefix)
- FAISS index files
- Metadata pickle files

### Import Databases

Import databases from a previously exported ZIP file:

```bash
# Import databases (skips existing databases)
python mcp-rag.py database import --file backup.zip

# Import and overwrite existing databases
python mcp-rag.py database import --file backup.zip --overwrite
```

The import command:
- Validates the ZIP file format and manifest version
- Preserves all metadata (description, doc_count, prefix, last_updated)
- Gracefully handles partial failures when importing multiple databases
- Can overwrite existing databases with the `--overwrite` flag

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

The server implements these three MCP-compliant tools as specified in the requirements:

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

### Tool Name Prefixes for Multiple Server Instances

When running multiple instances of this MCP server, each with different databases, you can use **tool name prefixes** to help agents distinguish between server instances. This is particularly useful when an agent needs to interact with multiple MCP-RAG servers simultaneously.

#### Creating Databases with Prefixes

When creating a database, specify a prefix that will be prepended to tool names:

```bash
python mcp-rag.py database create --name python_docs --description "Python documentation" --prefix "PY"
python mcp-rag.py database create --name java_docs --description "Java documentation" --prefix "JAVA"
python mcp-rag.py database create --name cpp_docs --description "C++ documentation" --prefix "CPP"
```

#### How Tool Prefixes Work

When you start the server with multiple databases, their prefixes are combined and prepended to all tool names:

```bash
python mcp-rag.py server start --active-databases python_docs,java_docs --transport stdio
```

This creates tools with the combined prefix `PY_JAVA_`:
- `PY_JAVA_getDatabases`
- `PY_JAVA_getInformationFor`
- `PY_JAVA_getInformationForDB`

#### Example Use Case

Imagine you're building an AI coding assistant that needs access to documentation for multiple programming languages:

1. **Server 1** (Python documentation):
   ```bash
   python mcp-rag.py server start --active-databases python_docs --transport stdio
   ```
   Tools: `PY_getDatabases`, `PY_getInformationFor`, `PY_getInformationForDB`

2. **Server 2** (Java documentation):
   ```bash
   python mcp-rag.py server start --active-databases java_docs --transport stdio
   ```
   Tools: `JAVA_getDatabases`, `JAVA_getInformationFor`, `JAVA_getInformationForDB`

3. **Server 3** (C++ documentation):
   ```bash
   python mcp-rag.py server start --active-databases cpp_docs --transport stdio
   ```
   Tools: `CPP_getDatabases`, `CPP_getInformationFor`, `CPP_getInformationForDB`

The agent can now clearly understand which server instance to query based on the tool prefix:
- Use `PY_getInformationFor("how to use list comprehension")` for Python questions
- Use `JAVA_getInformationFor("how to use streams")` for Java questions
- Use `CPP_getInformationFor("how to use smart pointers")` for C++ questions

#### Alternative Mechanisms

Besides tool name prefixes, you can also distinguish between server instances using:

1. **Different Transport Ports** (HTTP/SSE): Run each server on a different port
   ```bash
   python mcp-rag.py server start --active-databases python_docs --transport http --port 8080
   python mcp-rag.py server start --active-databases java_docs --transport http --port 8081
   ```

2. **Server Metadata**: The `getDatabases` tool returns database names and descriptions, allowing agents to discover what content each server provides

3. **Naming Conventions**: Use descriptive database names that indicate their purpose (e.g., `python_stdlib`, `python_django`, `java_spring`)

### Database Management via CLI

Note: Database management (create, add, list) is performed using the CLI commands:

```bash
# Create a database
python mcp-rag.py database create --name mydb --description "Description" --prefix "PRE"

# Add documents to a database
python mcp-rag.py database add --database mydb --path ~/documents --recursive

# List all databases
python mcp-rag.py database list
```

These operations are NOT exposed as MCP tools to keep the server interface focused on data retrieval.

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
- **Office**: .docx, .xlsx, .pptx (including embedded images with OCR)
- **OpenDocument**: .odt, .ods, .odp
- **PDF**: .pdf (including embedded images with OCR)
- **Web**: .html, .htm
- **Images**: .png, .jpg, .jpeg, .gif, .bmp, .tiff, .tif, .webp (with OCR text extraction)
- **Source Code**: .py, .c, .cpp, .h, .hpp, .cs, .go, .rs, .java, .js, .ts, .sh, .bat, .ps1, .s, .asm
- **Build Files**: Makefile, CMakeLists.txt
- **Archives**: .zip, .7z, .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, .rar

### Image Support

The tool now supports extracting text from images using OCR (Optical Character Recognition). This includes:

- **Standalone Image Files**: PNG, JPEG, GIF, BMP, TIFF, and WebP formats
- **Embedded Images in Documents**: Images within PDF, DOCX, and PPTX files are automatically extracted and processed
- **OCR Technology**: Uses EasyOCR for high-quality text recognition
- **Automatic Processing**: Image text extraction happens automatically when adding documents to the database
- **Performance**: OCR reader is lazily initialized only when needed to minimize startup time

When you add documents containing images or add image files directly, the tool will:
1. Detect images in supported formats
2. Initialize the OCR reader (first time only)
3. Extract text from the images
4. Combine the extracted text with other document content
5. Index everything together for semantic search

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
