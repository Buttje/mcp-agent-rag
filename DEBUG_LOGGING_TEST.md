# Debug Logging Test Documentation

## Overview
This document tests the debug logging functionality in mcp-rag-cli.py.

## Key Changes

1. **Debug Logger Initialization**: When `--debug` flag is used, the MCP client now initializes its own debug logger
2. **Tool Call Logging**: All MCP tool calls (requests and responses) are logged
3. **User Interaction Logging**: User prompts and agent responses are logged
4. **Log File Location**: Debug logs are written to `~/.mcp-agent-rag/debug/debug_YYYYMMDD_HHMMSS.log`

## What Gets Logged

### MCP Tool Calls
- Tool name and arguments (request)
- Response data including context preview, citations, and confidence scores

### User Interactions
- User prompts (questions)
- Agent responses (answers)

### Server Operations (in server subprocess)
- JSON-RPC requests and responses
- RAG database queries and results
- Agent thinking steps (if using reasoning models)

## Testing

To test the debug logging:

```bash
# Create a test database
python mcp-rag.py database create --name testdb --description "Test database"

# Add some documents
echo "This is a test document about Python programming." > test.txt
python mcp-rag.py database add --database testdb --path test.txt

# Start the CLI with debug logging
python mcp-rag-cli.py --debug

# Select the testdb database
# Ask a question like: "What is Python?"
# Type 'quit' to exit

# Check the debug log
ls -l ~/.mcp-agent-rag/debug/
cat ~/.mcp-agent-rag/debug/debug_*.log
```

The log should contain:
1. Debug logging initialization message
2. MCP tool calls and responses
3. User prompts
4. Agent responses
