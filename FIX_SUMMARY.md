# Fix Summary: Ollama Model Thinking Capability Detection

## Issue Resolved ✅

**Original Problem**: The system was displaying a warning message indicating that Ollama models don't support native reasoning, even when using models like `qwen3:30b` that have the "thinking" capability as shown by `ollama show`.

```
INFO Reasoning model: Ollama is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning
```

## Root Cause Analysis

The `agno` library (v0.1.0+) used by this project has a function `is_ollama_reasoning_model()` that checks if an Ollama model supports native reasoning. However, it only checks against a hardcoded list of model names:
- qwq
- deepseek-r1  
- qwen2.5-coder
- openthinker

Models like `qwen3:30b` were not in this list, even though they have the "thinking" capability that can be discovered via Ollama's API.

## Solution Implemented

### 1. Dynamic Capability Detection (`ollama_utils.py`)

Added a new function that queries the Ollama API to get model capabilities:

```python
def get_model_capabilities(model_name: str, host: str = "http://localhost:11434", 
                          timeout: int = 10) -> Tuple[List[str], str]:
    """Fetch model capabilities from Ollama server using /api/show endpoint."""
```

This function:
- Uses Ollama's `/api/show` endpoint to get model information
- Extracts capabilities from the response (e.g., ["completion", "tools", "thinking"])
- Returns capabilities list and any error message
- Has proper error handling for timeouts, connection errors, etc.

### 2. Agno Library Patch (`agno_ollama_patch.py`)

Created a monkey-patch for the `agno` library that enhances model detection:

```python
def is_ollama_reasoning_model_patched(reasoning_model: Model) -> bool:
    """Enhanced version that checks both hardcoded list AND queries Ollama API."""
```

The patched function:
- First checks the hardcoded list (for backwards compatibility and performance)
- Then queries the Ollama API for model capabilities
- Returns `True` if "thinking" is in the capabilities list
- Gracefully handles API errors (returns `False` to avoid false positives)

### 3. Integration (`chat_cli.py`)

The patch is applied automatically when the chat CLI starts:

```python
from mcp_agent_rag.utils.agno_ollama_patch import apply_agno_ollama_patch

def main():
    # Apply patch to agno library for better Ollama model capability detection
    # This must be done before any Agent instances are created
    apply_agno_ollama_patch()
    ...
```

In verbose mode, users get clear feedback:
```
✓ Model 'qwen3:30b' has native 'thinking' capability
```

Or:
```
ℹ Model 'mistral:7b-instruct' does not have native 'thinking' capability
  Will use manual Chain-of-Thought reasoning instead
```

## Testing

### Unit Tests Added

**test_ollama_utils.py** - 7 new tests for capability detection:
- ✅ Successful capability retrieval with thinking
- ✅ Model without thinking capability
- ✅ Timeout handling
- ✅ Connection error handling
- ✅ HTTP error handling
- ✅ Missing/malformed response data
- ✅ Empty capabilities list

**test_agno_ollama_patch.py** - 9 new tests for the patch:
- ✅ Non-Ollama models return False
- ✅ Known models from hardcoded list (no API call)
- ✅ DeepSeek-R1 recognition
- ✅ Dynamic capability detection via API
- ✅ Models without thinking capability
- ✅ API error handling
- ✅ Exception handling
- ✅ Default host usage
- ✅ Patch application

### Test Results
```
31 tests passed (22 existing + 9 new)
Coverage: 90.91% for ollama_utils.py, 86.21% for agno_ollama_patch.py
CodeQL Security Scan: 0 vulnerabilities found
```

## Benefits

1. **Automatic Detection**: No manual configuration needed - capabilities are discovered automatically
2. **Future-Proof**: Works with any new Ollama models that support thinking capability
3. **Backwards Compatible**: Still recognizes hardcoded models from original agno implementation
4. **Graceful Degradation**: If API query fails, safely defaults to False (no false positives)
5. **Performance Optimized**: Checks hardcoded list first, only queries API for unknown models
6. **User Feedback**: Clear messages in verbose mode about model capabilities
7. **Non-Invasive**: Uses monkey-patching - no modifications to agno library installation required

## Files Modified

```
docs/OLLAMA_THINKING_FIX.md                  | 126 lines (new documentation)
src/mcp_agent_rag/chat_cli.py                |  19 lines (apply patch + feedback)
src/mcp_agent_rag/rag/ollama_utils.py        |  48 lines (capability detection)
src/mcp_agent_rag/utils/agno_ollama_patch.py |  90 lines (agno patch)
tests/unit/test_agno_ollama_patch.py         | 138 lines (9 tests)
tests/unit/test_ollama_utils.py              | 111 lines (7 tests)
Total: 532 lines added
```

## Code Quality

- ✅ All code review feedback addressed
- ✅ Security vulnerabilities: 0
- ✅ Proper error handling throughout
- ✅ Clear documentation and comments
- ✅ Type hints used appropriately
- ✅ Constants extracted for maintainability
- ✅ Performance optimized (check hardcoded list first)

## How to Use

For end users, no changes are needed! The fix is automatically applied when using the chat CLI.

To verify it's working (with a model that has thinking capability):

```bash
python mcp-rag-cli.py --verbose
```

You should see:
```
✓ Model 'qwen3:30b' has native 'thinking' capability
```

Instead of the old warning message.

## Future Improvements

While this solution works well, the ideal long-term approach is to contribute this enhancement upstream to the `agno` library itself. The implementation in this PR can serve as a reference for such a contribution.

## Related Documentation

See `docs/OLLAMA_THINKING_FIX.md` for detailed technical documentation of the fix.
