"""Command-line interface for MCP-RAG."""

import argparse
import sys
from pathlib import Path

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.mcp import MCPServer
from mcp_agent_rag.utils import get_logger, setup_logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MCP-RAG: Model Context Protocol server with RAG"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to config file (default: {Config.get_default_config_path()})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Database commands
    db_parser = subparsers.add_parser("database", help="Database management")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database commands")

    # database create
    create_parser = db_subparsers.add_parser("create", help="Create a new database")
    create_parser.add_argument("--name", required=True, help="Database name")
    create_parser.add_argument("--description", default="", help="Database description")

    # database add
    add_parser = db_subparsers.add_parser("add", help="Add documents to database")
    add_parser.add_argument("--database", required=True, help="Database name")
    add_parser.add_argument("--path", help="Path to file or directory")
    add_parser.add_argument("--url", help="URL to download")
    add_parser.add_argument("--glob", help="Glob pattern")
    add_parser.add_argument("--recursive", action="store_true", help="Recurse subdirectories")
    add_parser.add_argument("--skip-existing", action="store_true", help="Skip existing files")

    # database list
    list_parser = db_subparsers.add_parser("list", help="List all databases")

    # Server commands
    server_parser = subparsers.add_parser("server", help="Server management")
    server_subparsers = server_parser.add_subparsers(dest="server_command", help="Server commands")

    # server start
    start_parser = server_subparsers.add_parser("start", help="Start MCP server")
    start_parser.add_argument(
        "--active-databases",
        required=True,
        help="Comma-separated list of active databases",
    )
    start_parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport protocol (stdio, http, or sse)",
    )
    start_parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    start_parser.add_argument("--port", type=int, default=8080, help="HTTP port")

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Setup logging
    log_dir = Config.get_default_data_dir() / "logs"
    log_file = log_dir / "mcp-rag.log"
    setup_logger(
        log_file=str(log_file),
        level=config.get("log_level", "INFO"),
    )
    logger = get_logger("mcp-rag")

    # Handle commands
    try:
        if args.command == "database":
            handle_database_command(args, config, logger)
        elif args.command == "server":
            handle_server_command(args, config, logger)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_database_command(args, config: Config, logger):
    """Handle database commands."""
    db_manager = DatabaseManager(config)

    if args.db_command == "create":
        success = db_manager.create_database(args.name, args.description)
        if success:
            db_path = config.get_database_path(args.name)
            print(f"Created database '{args.name}' at {db_path}")
        else:
            print(f"Error: Database '{args.name}' already exists", file=sys.stderr)
            sys.exit(1)

    elif args.db_command == "add":
        if not args.path and not args.url:
            print("Error: Either --path or --url must be specified", file=sys.stderr)
            sys.exit(1)

        stats = db_manager.add_documents(
            database_name=args.database,
            path=args.path,
            url=args.url,
            glob_pattern=args.glob,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
        )

        print(f"\nSummary:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")

    elif args.db_command == "list":
        databases = db_manager.list_databases()
        if not databases:
            print("No databases found")
        else:
            print(f"\n{'Name':<20} {'Docs':<10} {'Description':<40} {'Last Updated':<25}")
            print("-" * 95)
            for name, info in databases.items():
                print(
                    f"{name:<20} "
                    f"{info.get('doc_count', 0):<10} "
                    f"{info.get('description', ''):<40} "
                    f"{info.get('last_updated', 'Never'):<25}"
                )

    else:
        print("Error: Unknown database command", file=sys.stderr)
        sys.exit(1)


def handle_server_command(args, config: Config, logger):
    """Handle server commands."""
    if args.server_command == "start":
        # Parse active databases
        active_databases = [db.strip() for db in args.active_databases.split(",")]

        # Verify databases exist
        for db_name in active_databases:
            if not config.database_exists(db_name):
                print(f"Error: Database '{db_name}' does not exist", file=sys.stderr)
                sys.exit(1)

        # Create and start server
        try:
            server = MCPServer(config, active_databases)

            if args.transport == "stdio":
                server.run_stdio()
            elif args.transport == "http":
                server.run_http(host=args.host, port=args.port)
            elif args.transport == "sse":
                server.run_sse(host=args.host, port=args.port)

        except Exception as e:
            logger.error(f"Failed to start server: {e}", exc_info=True)
            print(f"Error: Failed to start server: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Error: Unknown server command", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
