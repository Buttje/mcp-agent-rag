"""Interactive chat CLI with MCP server integration."""

import json
import subprocess
import sys
import threading
import time
from typing import Any

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.utils import get_logger, setup_logger


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Text colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


# Confidence thresholds for color coding
CONFIDENCE_HIGH = 0.95  # Green
CONFIDENCE_GOOD = 0.90  # Cyan
CONFIDENCE_MIN = 0.85   # Yellow (matches filtering threshold)


def format_confidence(confidence: float) -> str:
    """Format confidence score with color coding.
    
    Args:
        confidence: Confidence score (0-1)
    
    Returns:
        Color-coded confidence string
    """
    percentage = confidence * 100
    if confidence >= CONFIDENCE_HIGH:
        color = Colors.GREEN
    elif confidence >= CONFIDENCE_GOOD:
        color = Colors.CYAN
    elif confidence >= CONFIDENCE_MIN:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    
    return f"{color}{percentage:.1f}%{Colors.RESET}"


def print_citations(citations: list, show_confidence: bool = True):
    """Print citations with color coding.
    
    Args:
        citations: List of citation dictionaries
        show_confidence: Whether to show confidence scores
    """
    if not citations:
        return
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}📚 Sources:{Colors.RESET}")
    for i, citation in enumerate(citations, 1):
        source = citation.get("source", "Unknown")
        chunk = citation.get("chunk", "?")
        database = citation.get("database", "Unknown")
        confidence = citation.get("confidence")
        
        conf_str = ""
        if show_confidence and confidence is not None:
            conf_str = f" {Colors.GRAY}[Confidence: {format_confidence(confidence)}]{Colors.RESET}"
        
        print(f"  {Colors.BOLD}{i}.{Colors.RESET} {source} (chunk {chunk}) "
              f"{Colors.GRAY}from {database}{Colors.RESET}{conf_str}")


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

    def initialize(self, protocol_version: str = "2025-11-25",
                   capabilities: dict[str, Any] | None = None) -> None:
        """Perform MCP initialize handshake and send initialized notification.
        
        Args:
            protocol_version: MCP protocol version to use
            capabilities: Client capabilities dictionary (optional)
        """
        self.request_id += 1
        init_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "capabilities": capabilities or {},
                "clientInfo": {
                    "name": "mcp-rag-cli",
                    "title": "MCP-RAG CLI",
                    "version": "1.0.0",
                    "description": "Interactive chat client for mcp-agent-rag",
                },
            },
        }
        # write and flush request
        self.process.stdin.write((json.dumps(init_request) + "\n"))
        self.process.stdin.flush()
        # read the server's reply
        response_line = self.process.stdout.readline().strip()
        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"MCP initialize error: {response['error']}")
        # after successful init, send the initialized notification (no id)
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        self.process.stdin.write((json.dumps(notification) + "\n"))
        self.process.stdin.flush()

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
            # Send request (text mode, no encoding needed)
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()

            # Read response line by line (text mode, no decoding needed)
            response_line = self.process.stdout.readline().strip()
            if not response_line:
                raise ConnectionError("MCP server closed connection")

            response = json.loads(response_line)

            if "error" in response:
                error = response["error"]
                error_msg = f"MCP error: {error.get('message', 'Unknown error')}"
                raise RuntimeError(error_msg)

            result = response.get("result", {})
            
            # Parse JSON from MCP content array if present
            if "content" in result and isinstance(result["content"], list):
                for content_item in result["content"]:
                    if content_item.get("type") == "text":
                        text = content_item.get("text", "")
                        try:
                            # Try to parse as JSON
                            parsed = json.loads(text)
                            result = parsed
                            break
                        except json.JSONDecodeError:
                            # Not JSON, use as-is
                            result = {"text": text}
                            break

            return result

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


def start_mcp_server(config: Config, active_databases: list[str], debug: bool = False) -> subprocess.Popen:
    """Start the MCP server as a subprocess.

    Args:
        config: Configuration instance
        active_databases: List of active database names
        debug: Enable debug logging on the server

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

    # Add debug flag if enabled
    if debug:
        cmd.append("--debug")

    logger.info(f"Starting MCP server with databases: {', '.join(active_databases)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,          # decode stdout/stderr to str automatically
            encoding="utf-8",
            bufsize=1,          # line buffered
        )

        # Start a daemon thread to read stderr to avoid blocking the server when logs are generated
        def _drain_stderr(pipe, log):
            for line in iter(pipe.readline, ''):
                log.error(f"[server] {line.rstrip()}")

        threading.Thread(target=_drain_stderr, args=(process.stderr, logger), daemon=True).start()

        # Give server time to start
        time.sleep(1)

        # Check if process is still running
        if process.poll() is not None:
            stderr = process.stderr.read()
            raise RuntimeError(f"MCP server failed to start: {stderr}")

        logger.info("MCP server started successfully")
        return process

    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


def main():
    """Main chat CLI entry point with MCP server."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MCP-RAG Interactive Chat Client")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to log file (default: ~/.mcp-agent-rag/logs/mcp-rag-cli.log)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging on the MCP server",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Setup logging
    if args.log:
        log_file = args.log
    else:
        log_dir = Config.get_default_data_dir() / "logs"
        log_file = str(log_dir / "mcp-rag-cli.log")

    setup_logger(log_file=log_file, level=config.get("log_level", "INFO"))
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
        mcp_process = start_mcp_server(config, selected_databases, debug=args.debug)
        mcp_client = MCPClient(mcp_process)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        print(f"Error: Failed to start MCP server: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Perform MCP initialization handshake
        mcp_client.initialize()
    except Exception as e:
        logger.error(f"Failed during MCP initialization: {e}", exc_info=True)
        print(f"Error: MCP initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("MCP server started successfully!")
    print()
    print("Chat started! Type your questions below.")
    print("Commands: 'quit', 'exit', '/q' to exit")
    print("=" * 70)
    print()

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

                # Query the MCP server using getInformationFor tool
                print()
                try:
                    # Call the MCP server's getInformationFor tool
                    result = mcp_client.call_tool(
                        "getInformationFor", 
                        {"prompt": user_input, "max_results": 5}
                    )

                    # Extract response
                    context = result.get("context", "")
                    citations = result.get("citations", [])
                    
                    # Display response
                    print(f"Assistant: {context}")
                    
                    # Show citations if available
                    if citations:
                        print_citations(citations, show_confidence=True)
                    
                    print()

                except Exception as e:
                    logger.error(f"Error querying MCP server: {e}", exc_info=True)
                    print(f"\nError processing query: {e}", file=sys.stderr)
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
