#!/usr/bin/env python3
"""MCP-RAG interactive chat CLI wrapper.

This script provides a cross-platform entry point to the mcp-rag-cli chat client.
Usage: python mcp-rag-cli.py
"""

if __name__ == "__main__":
    from mcp_agent_rag.chat_cli import main
    main()
