#!/usr/bin/env python3
"""MCP-RAG command-line interface wrapper.

This script provides a cross-platform entry point to mcp-rag CLI.
Usage: python mcp-rag.py [arguments]
"""

if __name__ == "__main__":
    from mcp_agent_rag.cli import main
    main()
