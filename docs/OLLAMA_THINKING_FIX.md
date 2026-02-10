# Ollama Thinking Capability Detection

## Overview

This fix addresses an issue where the `agno` library does not properly detect Ollama models with native "thinking" capabilities, resulting in the following warning:

```
INFO Reasoning model: Ollama is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning
```

## The Problem

The `agno` library (used for agent functionality) maintains a hardcoded list of Ollama models that support native reasoning:
- qwq
- deepseek-r1
- qwen2.5-coder
- openthinker

However, Ollama supports dynamic model capabilities that can be queried via its API. Models like `qwen3:30b` have the "thinking" capability as shown by `ollama show`:

```bash
$ ollama show qwen3:30b
  Capabilities
    completion
    tools
    thinking
```

But `agno` was not checking these capabilities dynamically, causing it to incorrectly fall back to manual Chain-of-Thought reasoning for models that actually support native thinking.

## The Solution

This fix consists of three components:

### 1. Model Capability Detection (`ollama_utils.py`)

A new function `get_model_capabilities()` that queries the Ollama API to retrieve a model's capabilities:

```python
from mcp_agent_rag.rag.ollama_utils import get_model_capabilities

capabilities, error = get_model_capabilities("qwen3:30b", "http://localhost:11434")
# capabilities = ["completion", "tools", "thinking"]
```

### 2. Agno Library Patch (`agno_ollama_patch.py`)

A monkey-patch for the `agno` library that enhances its `is_ollama_reasoning_model()` function to:
1. Check the hardcoded list (for backwards compatibility)
2. Query the Ollama API for model capabilities
3. Return `True` if the model has "thinking" capability

### 3. Integration (`chat_cli.py`)

The patch is automatically applied when the chat CLI starts:

```python
from mcp_agent_rag.utils.agno_ollama_patch import apply_agno_ollama_patch

# Apply patch early before creating any agents
apply_agno_ollama_patch()
```

## How It Works

1. When the chat CLI starts, it applies the patch to the `agno` library
2. When an agent is initialized with an Ollama model, `agno` checks if it supports native reasoning
3. The patched function queries Ollama's `/api/show` endpoint to get model capabilities
4. If "thinking" is in the capabilities list, the model is recognized as a native reasoning model
5. `agno` uses native reasoning instead of falling back to manual Chain-of-Thought

## Testing

The fix includes comprehensive unit tests:

### Capability Detection Tests (`test_ollama_utils.py`)
- ✅ Successful capability retrieval
- ✅ Models with and without thinking capability
- ✅ Timeout and connection errors
- ✅ HTTP errors
- ✅ Missing or malformed response data

### Patch Tests (`test_agno_ollama_patch.py`)
- ✅ Non-Ollama models
- ✅ Known reasoning models (hardcoded list)
- ✅ Dynamic capability detection via API
- ✅ API errors and exceptions
- ✅ Default host handling
- ✅ Patch application

Run tests:
```bash
python -m pytest tests/unit/test_ollama_utils.py tests/unit/test_agno_ollama_patch.py -v
```

## Benefits

1. **Automatic Detection**: No need to manually configure which models support thinking
2. **Future-Proof**: Works with any new Ollama models that support thinking capability
3. **Backwards Compatible**: Still recognizes hardcoded models from the original `agno` implementation
4. **Graceful Degradation**: If API query fails, defaults to safe behavior (returns False)
5. **Non-Invasive**: Uses monkey-patching so no changes to `agno` library installation are needed

## Example Output

Before the fix:
```
INFO Reasoning model: Ollama is not a native reasoning model, defaulting to manual Chain-of-Thought reasoning
```

After the fix (with verbose mode):
```
✓ Model 'qwen3:30b' has native 'thinking' capability
```

## Contributing to Agno

While this fix provides an immediate solution, the ideal long-term approach is to contribute this enhancement to the `agno` library itself. The patch can serve as a reference implementation for such a contribution.

## Files Changed

- `src/mcp_agent_rag/rag/ollama_utils.py` - Added `get_model_capabilities()` function
- `src/mcp_agent_rag/utils/agno_ollama_patch.py` - Patch implementation
- `src/mcp_agent_rag/chat_cli.py` - Apply patch on startup
- `tests/unit/test_ollama_utils.py` - Tests for capability detection
- `tests/unit/test_agno_ollama_patch.py` - Tests for patch module
