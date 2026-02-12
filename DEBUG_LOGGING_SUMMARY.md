# Debug Logging Summary

## Issue Resolution
**Problem**: Two debug log files were created when running `mcp-rag-cli.py --debug`:
- One by the chat CLI client process
- One by the MCP server subprocess

**Solution**: Removed debug logging from the chat CLI client. Now only the MCP server creates and maintains the debug log file.

## Debug Logging Location
When running with `--debug` flag:
```bash
python mcp-rag.py server start --active-databases mydb --debug
```

Debug log is created at:
```
~/.mcp-agent-rag/debug/debug_YYYYMMDD_HHMMSS.log
```
Where `YYYYMMDD` is the date (e.g., `20260212`) and `HHMMSS` is the time (e.g., `210500`).

Example: `~/.mcp-agent-rag/debug/debug_20260212_210500.log`

## What Gets Logged

The MCP server's debug log contains all required information:

### 1. Initial Prompt Received by MCP-Server ✓
- **Location**: `server.py:176` - `log_json_rpc_request()`
- **Content**: Full JSON-RPC request including method and parameters
- **Example**:
  ```
  [mcp.server] Received JSON-RPC request from MCP host:
  {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "query-get_data",
      "arguments": {"prompt": "What is Python?"}
    }
  }
  ```

### 2. Prompt Prepared by Agent for LLM (with Tool Descriptions) ✓
- **Location**: `enhanced_rag.py:695-707` - Logs internal tool descriptions
- **Location**: `enhanced_rag.py:745-749` - `log_llm_request()`
- **Content**: 
  - Database tools available to LLM
  - System prompt with tool descriptions
  - User prompt
  - Retrieved context from previous iterations
- **Example**:
  ```
  [agent.tools] Internal database tools created (Database Capabilities):
  {
    "tool_count": 2,
    "tools": [
      {
        "name": "query_database_testdb",
        "description": "Query the testdb database..."
      }
    ]
  }
  
  [llm.request] Sending request to LLM (mistral:7b-instruct):
  {
    "model": "mistral:7b-instruct",
    "prompt": "What is Python?",
    "context": "Previously retrieved information: ..."
  }
  ```

### 3. Response from LLM to Agent ✓
- **Location**: `enhanced_rag.py:765-773` - Logs LLM response with tool calls
- **Location**: `enhanced_rag.py:910-913` - `log_llm_response()`
- **Content**: 
  - LLM's response text
  - Tool calls requested by LLM
  - Number of tool calls
- **Example**:
  ```
  [llm.response] LLM response (iteration 1):
  {
    "response_text": "I need to query the database for Python information",
    "tool_calls_count": 1,
    "tool_calls": [
      {
        "name": "query_database_testdb",
        "arguments": {"query": "Python programming language"}
      }
    ]
  }
  ```

### 4. Queries for RAG Databases and Their Responses ✓
- **Location**: `enhanced_rag.py:85-89` - `log_rag_query()`
- **Location**: `enhanced_rag.py:126-130` - `log_rag_results()`
- **Content**:
  - Database name being queried
  - Query text
  - Query embedding preview
  - Retrieved results with confidence scores
  - Filtered results count
- **Example**:
  ```
  [rag.retrieval] Querying database 'testdb':
  {
    "database": "testdb",
    "query": "Python programming language",
    "embedding_preview": [0.123, -0.456, 0.789, ...]
  }
  
  [rag.retrieval] Retrieved 3 results from 'testdb':
  {
    "database": "testdb",
    "result_count": 3,
    "filtered_count": 2,
    "results": [
      {
        "source": "python_guide.pdf",
        "chunk": 1,
        "confidence": 0.95,
        "text_preview": "Python is a high-level programming language..."
      }
    ]
  }
  ```

### 5. All Following Agent-LLM Communications ✓
- **Location**: `enhanced_rag.py:728-730` - `log_thinking_step()`
- **Location**: `enhanced_rag.py:792-795`, `802-810` - Logs tool execution
- **Location**: `enhanced_rag.py:745-749` - Subsequent LLM requests
- **Location**: `enhanced_rag.py:765-773` - Subsequent LLM responses
- **Content**:
  - Iteration number and status
  - Tool execution requests and results
  - Thinking steps and decisions
  - All prompts and responses in iterative loop
- **Example**:
  ```
  [agent.thinking] Thinking step 'iteration_1': Starting iteration 1/3
  
  [agent.tool_execution] Executing internal tool call:
  {
    "name": "query_database_testdb",
    "arguments": {"query": "Python"}
  }
  
  [agent.tool_result] Tool execution result:
  {
    "database": "testdb",
    "query": "Python",
    "result_count": 5
  }
  
  [llm.request] Sending request to LLM (iteration 2)...
  [llm.response] LLM response (iteration 2)...
  ```

### 6. Final Response Sent Back to MCP Host ✓
- **Location**: `enhanced_rag.py:923` - `log_augmented_prompt()`
- **Location**: `server.py:342-345` - `log_final_response()`
- **Location**: `server.py:230` - `log_json_rpc_response()`
- **Content**:
  - Final augmented context
  - Citations
  - Average confidence
  - Databases searched
  - Full JSON-RPC response
- **Example**:
  ```
  [mcp.agent] Final response:
  {
    "response": "Python is a high-level programming language...",
    "citations": [
      {
        "source": "python_guide.pdf",
        "chunk": 1,
        "database": "testdb",
        "confidence": 0.95
      }
    ]
  }
  
  [mcp.server] Sending JSON-RPC response to MCP host:
  {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "prompt": "What is Python?",
      "context": "Python is a high-level...",
      "citations": [...],
      "databases_searched": ["testdb"],
      "average_confidence": 0.95,
      "iterations": 2
    }
  }
  ```

## Usage

### Starting Server with Debug Logging
```bash
# Direct server start (standalone mode)
python mcp-rag.py server start --active-databases mydb --debug

# Via chat CLI (server subprocess gets --debug flag)
python mcp-rag-cli.py --debug
```

### Viewing Debug Logs
```bash
# List debug logs
ls -lt ~/.mcp-agent-rag/debug/

# View latest debug log
cat ~/.mcp-agent-rag/debug/debug_*.log | tail -100

# Follow debug log in real-time
tail -f ~/.mcp-agent-rag/debug/debug_*.log
```

## Changes Made

### Files Modified
- `src/mcp_agent_rag/chat_cli.py`: Removed debug logger setup and logging calls

### Files Removed
- `tests/unit/test_chat_cli_logging.py`: Tests for removed client-side logging
- `test_debug_logging_integration.py`: Integration test for removed functionality

### Files Unchanged
- `src/mcp_agent_rag/cli.py`: Server still sets up debug logger when `--debug` is used
- `src/mcp_agent_rag/mcp/server.py`: Server still logs JSON-RPC messages
- `src/mcp_agent_rag/mcp/enhanced_rag.py`: AgenticRAG still logs all RAG operations
- `src/mcp_agent_rag/utils/debug_logger.py`: Debug logger implementation unchanged

## Benefits

1. **Single Log File**: Only one debug log file is created per session
2. **Complete Information**: All required debugging information is captured
3. **Proper Architecture**: Logging happens in the appropriate process (server)
4. **Clean Separation**: Client handles UI, server handles logging
5. **Better Performance**: Less overhead from duplicate logging
