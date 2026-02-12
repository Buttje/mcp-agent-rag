"""Interactive chat CLI with MCP server integration and AGNO agent."""

import json
import subprocess
import sys
import time
from typing import Any

from agno.agent import Agent

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.rag.ollama_utils import get_model_capabilities
from mcp_agent_rag.utils import get_logger, setup_logger
from mcp_agent_rag.utils.agno_ollama_patch import apply_agno_ollama_patch


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
    
    # Background colors
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"


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

    def __init__(self, process: subprocess.Popen, verbose: bool = False):
        """Initialize MCP client with process.

        Args:
            process: The MCP server subprocess
            verbose: Enable verbose output for debugging
        """
        self.process = process
        self.request_id = 0
        self.logger = get_logger(__name__)
        self.verbose = verbose

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

        if self.verbose:
            print(f"\n{Colors.BOLD}{Colors.BLUE}🔧 [MCP Tool Call]{Colors.RESET}")
            print(f"   Tool: {Colors.CYAN}{tool_name}{Colors.RESET}")
            print(f"   Arguments: {Colors.GRAY}{arguments}{Colors.RESET}")

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
                error_msg = f"MCP error: {error.get('message', 'Unknown error')}"
                if self.verbose:
                    print(f"   {Colors.RED}❌ Error: {error_msg}{Colors.RESET}")
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

            if self.verbose:
                # Show a summary of the result with color coding
                if "context" in result:
                    context = result["context"]
                    context_len = len(context)
                    context_preview = context[:150] + "..." if context_len > 150 else context
                    
                    avg_conf = result.get("average_confidence")
                    conf_str = ""
                    if avg_conf is not None:
                        conf_str = f" {Colors.GRAY}[Avg confidence: {format_confidence(avg_conf)}]{Colors.RESET}"
                    
                    print(f"   {Colors.GREEN}✅ Result: Retrieved context ({context_len} chars){Colors.RESET}{conf_str}")
                    print(f"   {Colors.GRAY}Preview: {context_preview}{Colors.RESET}")
                    
                    if "citations" in result:
                        citations = result.get("citations", [])
                        print(f"   {Colors.CYAN}Citations: {len(citations)} sources{Colors.RESET}")
                        
                        # Show citations with confidence
                        if citations and self.verbose:
                            print_citations(citations[:3])  # Show first 3
                            if len(citations) > 3:
                                print(f"   {Colors.GRAY}... and {len(citations) - 3} more{Colors.RESET}")
                    
                    min_threshold = result.get("min_confidence_threshold")
                    if min_threshold is not None:
                        print(f"   {Colors.GRAY}Minimum confidence threshold: {format_confidence(min_threshold)}{Colors.RESET}")
                
                elif "error" in result:
                    print(f"   {Colors.RED}❌ Tool Error: {result['error']}{Colors.RESET}")
                else:
                    print(f"   {Colors.GREEN}✅ Result: {Colors.RESET}{result}")
                print()

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


def create_mcp_tool_query_data(mcp_client: MCPClient, verbose: bool = False):
    """Create a tool function for querying data from MCP server.

    Args:
        mcp_client: The MCP client instance
        verbose: Enable verbose output

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
        if verbose:
            print(f"\n💭 [Agent Decision: Using query_data tool]")
            print(f"   Reason: Need to search document databases for information")
            print(f"   Query: {prompt}")
            print(f"   Max results: {max_results}")

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
    import argparse

    # Apply patch to agno library for better Ollama model capability detection
    # This must be done before any Agent instances are created
    apply_agno_ollama_patch()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MCP-RAG Interactive Chat Client")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to log file (default: ~/.mcp-agent-rag/logs/mcp-rag-cli.log)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output showing thinking process and tool usage",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (includes verbose output)",
    )
    args = parser.parse_args()

    # Enable verbose mode if debug is set (debug implies verbose)
    verbose = args.verbose or args.debug

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
        mcp_client = MCPClient(mcp_process, verbose=verbose)
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
        query_tool = create_mcp_tool_query_data(mcp_client, verbose=verbose)

        # Get the generative model from config
        # The config stores the Ollama model name (e.g., "mistral:7b-instruct")
        # We prefix it with "ollama:" to get the AGNO format (e.g., "ollama:mistral:7b-instruct")
        generative_model = config.get("generative_model", "mistral:7b-instruct")

        # Handle both cases: model with or without "ollama:" prefix
        if not generative_model.startswith("ollama:"):
            model_string = f"ollama:{generative_model}"
        else:
            model_string = generative_model

        # Show model capability information in verbose mode
        if verbose:
            ollama_host = config.get("ollama_host", "http://localhost:11434")
            capabilities, cap_error = get_model_capabilities(generative_model, ollama_host)
            
            has_thinking = "thinking" in capabilities
            
            if has_thinking:
                print(f"✓ Model '{generative_model}' has native 'thinking' capability")
            elif not cap_error:
                print(f"ℹ Model '{generative_model}' does not have native 'thinking' capability")
                print("  Will use manual Chain-of-Thought reasoning instead")
        
        # Initialize AGNO agent with Ollama model
        agent = Agent(
            name="MCP-RAG Assistant",
            model=model_string,
            description="An AI assistant that can query document databases via MCP server",
            instructions=[
                "You are a helpful AI assistant with access to document databases via RAG (Retrieval Augmented Generation).",
                "",
                "CRITICAL INSTRUCTIONS - READ CAREFULLY:",
                "1. ALWAYS prioritize information from the RAG database over your own knowledge.",
                "2. DO NOT invent, fabricate, or assume any information not present in the retrieved context.",
                "3. Only use information from sources with confidence scores >= 85%. Lower confidence results are unreliable and must be discarded.",
                "4. The RAG system filters results by confidence automatically, so all returned results meet the 85% threshold.",
                "5. If the retrieved context does not contain enough information to answer the question, say so explicitly.",
                "6. Never supplement RAG data with your own inference or general knowledge unless explicitly requested.",
                "",
                "WORKFLOW:",
                "1. For every user question, FIRST use the query_data tool to search the databases.",
                "2. Carefully analyze the retrieved context and confidence scores.",
                "3. Base your answer SOLELY on the retrieved information.",
                "4. ALWAYS cite your sources with database names and confidence scores.",
                "5. If no relevant information is found, respond: 'I could not find relevant information in the databases to answer this question.'",
                "",
                "CITATION FORMAT:",
                "- Always mention source documents and their confidence levels",
                "- Example: 'According to [document.pdf] (confidence: 92%), ...'",
                "",
                "WHAT NOT TO DO:",
                "- Do not answer from memory if RAG returns no results",
                "- Do not mix your knowledge with RAG data",
                "- Do not use information below 85% confidence",
                "- Do not make assumptions beyond what the context explicitly states",
            ],
            tools=[query_tool],
            markdown=True,
            reasoning=True,  # Enable ReAct (Reasoning and Act) mechanism
            debug_mode=verbose,  # Enable debug mode for verbose output
        )

        print("Agent initialized successfully!")
        print()
        if verbose:
            print("🔍 Verbose mode enabled - showing thinking process and tool usage")
            print()
        print("Chat started! Type your questions below.")
        print("Commands: 'quit', 'exit', '/q' to exit")
        print("=" * 70)
        print()

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"Error: Failed to initialize agent: {e}", file=sys.stderr)
        print(
            f"\nMake sure Ollama is running and the model '{generative_model}' is "
            "available.",
            file=sys.stderr,
        )
        print(f"You can pull the model with: ollama pull {generative_model}", file=sys.stderr)
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

                # Use the agent to process the query with ReAct mechanism
                try:
                    if verbose:
                        # In verbose mode, show the thinking process
                        print("🤔 [Agent Thinking...]")
                        print()

                    print("Assistant: ", end="", flush=True)
                    response = agent.run(user_input)

                    # Print the agent's response
                    if hasattr(response, 'content'):
                        print(response.content)
                    else:
                        print(str(response))
                    print()

                except Exception as e:
                    logger.error(f"Error running agent: {e}", exc_info=True)
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
