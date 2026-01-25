"""Interactive chat client for MCP-RAG."""

import sys
from typing import Optional

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.mcp import AgenticRAG
from mcp_agent_rag.rag import OllamaEmbedder, OllamaGenerator
from mcp_agent_rag.utils import get_logger, setup_logger


def main():
    """Main chat client entry point."""
    # Load configuration
    config = Config()
    
    # Setup logging
    log_dir = Config.get_default_data_dir() / "logs"
    log_file = log_dir / "mcp-rag-chat.log"
    setup_logger(
        log_file=str(log_file),
        level=config.get("log_level", "INFO"),
    )
    logger = get_logger("mcp-rag-chat")
    
    print("=" * 70)
    print("MCP-RAG Chat Client")
    print("=" * 70)
    print()
    
    # Check Ollama connection
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    embedder = OllamaEmbedder(
        model=config.get("embedding_model", "nomic-embed-text"),
        host=ollama_host,
    )
    
    if not embedder.check_connection():
        print(f"Error: Cannot connect to Ollama at {ollama_host}", file=sys.stderr)
        print("Please ensure Ollama is running.", file=sys.stderr)
        sys.exit(1)
    
    # List available databases
    db_manager = DatabaseManager(config)
    databases = db_manager.list_databases()
    
    if not databases:
        print("No databases found. Please create a database first using:")
        print("  mcp-rag database create --name mydb --description 'My docs'")
        print("  mcp-rag database add --database mydb --path /path/to/docs")
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
    selected_db = None
    while not selected_db:
        try:
            choice = input("Select database (number or name, or 'q' to quit): ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                sys.exit(0)
            
            # Try as number first
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(db_list):
                    selected_db = db_list[idx][0]
                else:
                    print(f"Invalid selection. Please enter 1-{len(db_list)}")
            # Try as database name
            elif choice in databases:
                selected_db = choice
            else:
                print(f"Invalid selection. Please enter 1-{len(db_list)} or a database name")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
    
    print()
    print(f"Loading database: {selected_db}")
    
    # Load the selected database
    vector_db = db_manager.load_database(selected_db)
    if not vector_db:
        print(f"Error: Failed to load database '{selected_db}'", file=sys.stderr)
        sys.exit(1)
    
    # Initialize RAG and generator
    agentic_rag = AgenticRAG(config, {selected_db: vector_db})
    generator = OllamaGenerator(
        model=config.get("generative_model", "mistral:7b-instruct"),
        host=ollama_host,
    )
    
    print(f"Using model: {generator.model}")
    print()
    print("Chat started! Type your questions below.")
    print("Commands: 'quit', 'exit', '/q' to exit")
    print("=" * 70)
    print()
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', '/q']:
                print("\nGoodbye!")
                break
            
            # Get context from RAG
            print("\nSearching...", end="", flush=True)
            context_result = agentic_rag.get_context(user_input, max_results=5)
            print("\r" + " " * 20 + "\r", end="", flush=True)  # Clear "Searching..."
            
            context_text = context_result.get("text", "")
            citations = context_result.get("citations", [])
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            
            response = None
            try:
                # Try streaming response
                response_parts = []
                for chunk in generator.generate_stream(user_input, context_text):
                    if chunk:
                        print(chunk, end="", flush=True)
                        response_parts.append(chunk)
                response = "".join(response_parts) if response_parts else None
            except Exception as e:
                logger.error(f"Streaming failed, falling back to non-streaming: {e}")
                # Fallback to non-streaming
                response = generator.generate(user_input, context_text)
                if response:
                    print(response, end="", flush=True)
            
            print()  # New line after response
            
            # Show citations if any
            if citations:
                print("\nSources:")
                unique_sources = {}
                for citation in citations:
                    source = citation.get("source", "Unknown")
                    if source not in unique_sources:
                        unique_sources[source] = []
                    unique_sources[source].append(citation.get("chunk", 0))
                
                for source, chunks in unique_sources.items():
                    print(f"  - {source}")
            
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


if __name__ == "__main__":
    main()
