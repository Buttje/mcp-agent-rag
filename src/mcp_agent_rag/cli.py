"""Command-line interface for MCP-RAG."""

import argparse
import sys

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.mcp import MCPServer
from mcp_agent_rag.utils import get_logger, setup_debug_logger, setup_logger


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
    parser.add_argument(
        "--log",
        default=None,
        help="Path to log file (default: ~/.mcp-agent-rag/logs/mcp-rag.log)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Database commands
    db_parser = subparsers.add_parser("database", help="Database management")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database commands")

    # database create
    create_parser = db_subparsers.add_parser("create", help="Create a new database")
    create_parser.add_argument("--name", required=True, help="Database name")
    create_parser.add_argument("--description", default="", help="Database description")
    create_parser.add_argument(
        "--prefix", default="", help="Prefix for MCP tool names (e.g., 'A1')"
    )

    # database add
    add_parser = db_subparsers.add_parser("add", help="Add documents to database")
    add_parser.add_argument("--database", required=True, help="Database name")
    add_parser.add_argument("--path", help="Path to file or directory")
    add_parser.add_argument("--url", help="URL to download")
    add_parser.add_argument("--glob", help="Glob pattern")
    add_parser.add_argument("--recursive", action="store_true", help="Recurse subdirectories")
    add_parser.add_argument("--skip-existing", action="store_true", help="Skip existing files")

    # database list
    db_subparsers.add_parser("list", help="List all databases")

    # database export
    export_parser = db_subparsers.add_parser("export", help="Export databases to a ZIP file")
    export_parser.add_argument(
        "--databases",
        required=True,
        help="Comma-separated list of database names to export",
    )
    export_parser.add_argument("--output", required=True, help="Output ZIP file path")

    # database import
    import_parser = db_subparsers.add_parser("import", help="Import databases from a ZIP file")
    import_parser.add_argument("--file", required=True, help="ZIP file to import")
    import_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing databases"
    )

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
    start_parser.add_argument(
        "--cody",
        action="store_true",
        help="Use MCP protocol version 2024-11-05 for CODY compatibility (default: 2025-11-25)",
    )
    start_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Setup logging
    if args.log:
        log_file = args.log
    else:
        log_dir = Config.get_default_data_dir() / "logs"
        log_file = str(log_dir / "mcp-rag.log")

    # Determine log level - use DEBUG if --debug flag is set on server start command
    log_level = config.get("log_level", "INFO")
    debug_enabled = False
    if args.command == "server" and getattr(args, "debug", False):
        log_level = "DEBUG"
        debug_enabled = True

    setup_logger(
        log_file=log_file,
        level=log_level,
    )
    logger = get_logger("mcp-rag")
    
    # Setup debug logger if debug is enabled
    if debug_enabled:
        setup_debug_logger(enabled=True)
        logger.info("Debug logging enabled")

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
        success = db_manager.create_database(args.name, args.description, args.prefix)
        if success:
            db_path = config.get_database_path(args.name)
            prefix_msg = f" with prefix '{args.prefix}'" if args.prefix else ""
            print(f"Created database '{args.name}'{prefix_msg} at {db_path}")
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

        print("\nSummary:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")

    elif args.db_command == "list":
        databases = db_manager.list_databases()
        if not databases:
            print("No databases found")
        else:
            name_width = 20
            prefix_width = 10
            docs_width = 10
            desc_width = 30
            updated_width = 25
            separator_length = (
                name_width + prefix_width + docs_width + desc_width + updated_width
            )

            print(
                f"\n{'Name':<{name_width}} "
                f"{'Prefix':<{prefix_width}} "
                f"{'Docs':<{docs_width}} "
                f"{'Description':<{desc_width}} "
                f"{'Last Updated':<{updated_width}}"
            )
            print("-" * separator_length)
            for name, info in databases.items():
                last_updated = info.get('last_updated') or 'Never'
                print(
                    f"{name:<{name_width}} "
                    f"{info.get('prefix', ''):<{prefix_width}} "
                    f"{info.get('doc_count', 0):<{docs_width}} "
                    f"{info.get('description', ''):<{desc_width}} "
                    f"{last_updated:<{updated_width}}"
                )

    elif args.db_command == "export":
        # Parse database names
        database_names = [db.strip() for db in args.databases.split(",")]

        # Perform export
        success = db_manager.export_databases(database_names, args.output)
        if success:
            print(f"Successfully exported {len(database_names)} database(s) to {args.output}")
        else:
            print("Error: Failed to export databases", file=sys.stderr)
            sys.exit(1)

    elif args.db_command == "import":
        # Perform import
        results = db_manager.import_databases(args.file, args.overwrite)

        if not results:
            print("Error: Failed to import databases", file=sys.stderr)
            sys.exit(1)

        # Print results
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]

        print("\nImport Summary:")
        print(f"  Successful: {len(successful)}")
        if successful:
            for name in successful:
                print(f"    - {name}")

        if failed:
            print(f"  Failed/Skipped: {len(failed)}")
            for name in failed:
                print(f"    - {name}")

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
            # Determine protocol version based on --cody flag
            protocol_version = "2024-11-05" if args.cody else "2025-11-25"
            server = MCPServer(config, active_databases, protocol_version=protocol_version)

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
