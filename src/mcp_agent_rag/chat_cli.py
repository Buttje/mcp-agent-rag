"""Interactive chat CLI with MCP server integration and AGNO agent."""

import json
import subprocess
import sys
import time
from typing import Any

from agno.agent import Agent

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.utils import get_logger, setup_logger


class MCPClient:
    """Client to communicate with MCP server via JSON-RPC."""

    def __init__(self, process: subprocess.Popen):
        """Initialize MCP client with process.

        Args:
            process: The MCP server subprocess
        """
        self.process = process
        self.request_id = 0
        self.logger = get_logger(__name__)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result as dictionary
        """
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        try:
            # Send request with explicit UTF-8 encoding
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode('utf-8'))
            self.process.stdin.flush()

            # Read response line by line with explicit UTF-8 decoding
            # to handle special characters correctly
            response_line = self.process.stdout.readline().decode('utf-8').strip()
            if not response_line:
                raise ConnectionError("MCP server closed connection")

            response = json.loads(response_line)

            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"MCP error: {error.get('message', 'Unknown error')}")

            return response.get("result", {})

        except Exception as e:
            self.logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            raise

    def close(self):
        """Close the MCP client and terminate the server process."""
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def start_mcp_server(config: Config, active_databases: list[str]) -> subprocess.Popen:
    """Start the MCP server as a subprocess.

    Args:
        config: Configuration instance
        active_databases: List of active database names

    Returns:
        The MCP server subprocess
    """
    logger = get_logger(__name__)

    # Build command to start MCP server
    # Use the same Python interpreter as current process
    cmd = [
        sys.executable,
        "-m",
        "mcp_agent_rag.cli",
        "server",
        "start",
        "--active-databases",
        ",".join(active_databases),
        "--transport",
        "stdio",
    ]

    logger.info(f"Starting MCP server with databases: {', '.join(active_databases)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # No bufsize specified - use default buffering which works correctly
            # with binary mode pipes and avoids RuntimeWarning
        )

        # Give server time to start
        time.sleep(1)

        # Check if process is still running
        if process.poll() is not None:
            stderr = process.stderr.read().decode('utf-8')
            raise RuntimeError(f"MCP server failed to start: {stderr}")

        logger.info("MCP server started successfully")
        return process

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


def create_mcp_tool_query_data(mcp_client: MCPClient):
    """Create a tool function for querying data from MCP server.

    Args:
        mcp_client: The MCP client instance

    Returns:
        Tool function that can be used by AGNO agent
    """

    def query_data(prompt: str, max_results: int = 5) -> str:
        """Query data from the MCP server's active databases.

        Args:
            prompt: The query prompt
            max_results: Maximum results to return per database

        Returns:
            Context text with citations
        """
        try:
            result = mcp_client.call_tool(
                "query-get_data", {"prompt": prompt, "max_results": max_results}
            )

            context = result.get("context", "")
            citations = result.get("citations", [])

            # Format response with citations
            response = context
            if citations:
                response += "\n\n**Sources:**\n"
                unique_sources = {}
                for citation in citations:
                    source = citation.get("source", "Unknown")
                    if source not in unique_sources:
                        unique_sources[source] = []
                    unique_sources[source].append(citation.get("chunk", 0))

                for source in unique_sources:
                    response += f"  - {source}\n"

            return response

        except Exception as e:
            return f"Error querying data: {str(e)}"

    return query_data


def main():
    """Main chat CLI entry point with MCP server and AGNO agent."""
    # Load configuration
    config = Config()

    # Setup logging
    log_dir = Config.get_default_data_dir() / "logs"
    log_file = log_dir / "mcp-rag-cli.log"
    setup_logger(log_file=str(log_file), level=config.get("log_level", "INFO"))
    logger = get_logger("mcp-rag-cli")

    print("=" * 70)
    print("MCP-RAG CLI Chat Client")
    print("=" * 70)
    print()

    # List available databases
    db_manager = DatabaseManager(config)
    databases = db_manager.list_databases()

    if not databases:
        print("No databases found. Please create a database first using:")
        print("  python mcp-rag.py database create --name mydb --description 'My docs'")
        print("  python mcp-rag.py database add --database mydb --path /path/to/docs")
        sys.exit(1)

    # Display database selection menu
    print("Available databases:")
    print()
    db_list = list(databases.items())
    for i, (name, info) in enumerate(db_list, 1):
        doc_count = info.get("doc_count", 0)
        description = info.get("description", "")
        print(f"  {i}. {name} ({doc_count} documents)")
        if description:
            print(f"     {description}")
    print()

    # Get user selection
    selected_databases = []
    while not selected_databases:
        try:
            choice = input(
                "Select database(s) (number, name, or 'all' for all databases): "
            ).strip()

            if choice.lower() in ["q", "quit", "exit"]:
                print("Goodbye!")
                sys.exit(0)

            if choice.lower() == "all":
                selected_databases = [name for name, _ in db_list]
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(db_list):
                    selected_databases = [db_list[idx][0]]
                else:
                    print(f"Invalid selection. Please enter 1-{len(db_list)}")
            elif choice in databases:
                selected_databases = [choice]
            else:
                # Try comma-separated list
                choices = [c.strip() for c in choice.split(",")]
                for c in choices:
                    if c.isdigit():
                        idx = int(c) - 1
                        if 0 <= idx < len(db_list):
                            selected_databases.append(db_list[idx][0])
                    elif c in databases:
                        selected_databases.append(c)

                if not selected_databases:
                    print(
                        "Invalid selection. Please enter 1-"
                        f"{len(db_list)}, database names, or 'all'"
                    )

        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

    print()
    print(f"Selected databases: {', '.join(selected_databases)}")
    print()

    # Start MCP server
    print("Starting MCP server...")
    try:
        mcp_process = start_mcp_server(config, selected_databases)
        mcp_client = MCPClient(mcp_process)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        print(f"Error: Failed to start MCP server: {e}", file=sys.stderr)
        sys.exit(1)

    print("MCP server started successfully!")
    print()

    # Create AGNO agent with MCP tool
    print("Initializing agent...")
    try:
        # Create MCP query tool
        query_tool = create_mcp_tool_query_data(mcp_client)

        # Initialize AGNO agent
        # Note: AGNO Agent can work without a model if we only use tools
        # For actual LLM integration, we'd need to configure Ollama integration
        # Currently, we'll use the agent infrastructure but call tools directly
        _agent = Agent(
            name="MCP-RAG Assistant",
            description="An AI assistant that can query document databases via MCP server",
            instructions=[
                "You are a helpful AI assistant with access to document databases.",
                "When users ask questions, use the query_data tool to search relevant information.",
                "Provide clear, accurate answers based on the retrieved context.",
                "Always cite your sources when providing information from the databases.",
            ],
            tools=[query_tool],
            markdown=True,
        )

        print("Agent initialized successfully!")
        print()
        print("Chat started! Type your questions below.")
        print("Commands: 'quit', 'exit', '/q' to exit")
        print("=" * 70)
        print()

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"Error: Failed to initialize agent: {e}", file=sys.stderr)
        mcp_client.close()
        sys.exit(1)

    # Main chat loop
    try:
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ["quit", "exit", "/q"]:
                    print("\nGoodbye!")
                    break

                # Process query with agent
                print()
                # In a real implementation, we would use agent.run() with a configured model
                # For now, we directly call the tool since AGNO requires model configuration
                print("Searching databases...")
                result = query_tool(user_input)
                print("\nAssistant:")
                print(result)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                print(f"\nError: {e}", file=sys.stderr)
                print()

    finally:
        # Clean up
        print("\nShutting down MCP server...")
        mcp_client.close()
        logger.info("MCP-RAG CLI chat client stopped")


if __name__ == "__main__":
    main()
