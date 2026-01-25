#!/usr/bin/env python3
"""Installation script for MCP-RAG."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Install MCP-RAG")
    parser.add_argument(
        "--config",
        help="Path to existing config file to use",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Use defaults without prompting",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MCP-RAG Installation")
    print("=" * 60)

    # Detect OS
    os_name = platform.system()
    print(f"\nDetected OS: {os_name}")

    if os_name not in ["Windows", "Linux", "Darwin"]:
        print(f"Warning: Unsupported OS: {os_name}")

    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)

    # Get project root
    project_root = Path(__file__).parent.resolve()
    print(f"Project root: {project_root}")

    # Create virtual environment
    venv_path = project_root / ".venv"
    if not venv_path.exists():
        print(f"\nCreating virtual environment at {venv_path}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("Virtual environment created")
    else:
        print(f"Virtual environment already exists at {venv_path}")

    # Determine pip path
    if os_name == "Windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    # Upgrade pip
    print("\nUpgrading pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Install dependencies
    print("\nInstalling dependencies...")
    subprocess.run([str(pip_path), "install", "-e", ".[dev]"], cwd=project_root, check=True)
    print("Dependencies installed")

    # Configuration
    if args.config:
        print(f"\nUsing existing config: {args.config}")
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
    else:
        print("\nSetting up configuration...")
        config = create_config(args.no_prompt)
        config_path = Path.home() / ".mcp-agent-rag" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to: {config_path}")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print(f"\nTo use MCP-RAG:")
    if os_name == "Windows":
        print(f"  1. Activate: {venv_path}\\Scripts\\activate")
    else:
        print(f"  1. Activate: source {venv_path}/bin/activate")
    print(f"  2. Run: mcp-rag --help")
    print(f"\nConfig file: {config_path}")


def create_config(no_prompt: bool) -> dict:
    """Create configuration interactively or with defaults.

    Args:
        no_prompt: Use defaults without prompting

    Returns:
        Configuration dictionary
    """
    config = {
        "embedding_model": "nomic-embed-text",
        "generative_model": "mistral:7b-instruct",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "databases": {},
        "ollama_host": "http://localhost:11434",
        "log_level": "INFO",
        "max_context_length": 4000,
    }

    if no_prompt:
        print("Using default configuration")
        return config

    print("\nConfiguration Setup")
    print("-" * 40)

    # Embedding model
    print("\nAvailable embedding models (Ollama):")
    print("  1. nomic-embed-text (default)")
    print("  2. mxbai-embed-large")
    print("  3. all-minilm")
    choice = input("Select embedding model [1]: ").strip() or "1"
    if choice == "2":
        config["embedding_model"] = "mxbai-embed-large"
    elif choice == "3":
        config["embedding_model"] = "all-minilm"

    # Generative model
    print("\nAvailable generative models:")
    print("  1. mistral:7b-instruct (default, Apache 2.0)")
    print("  2. llama3.2 (Meta license)")
    print("  3. gemma2 (Google)")
    choice = input("Select generative model [1]: ").strip() or "1"
    if choice == "2":
        config["generative_model"] = "llama3.2"
    elif choice == "3":
        config["generative_model"] = "gemma2"

    # Chunk size
    chunk_size = input(f"\nChunk size [{config['chunk_size']}]: ").strip()
    if chunk_size:
        config["chunk_size"] = int(chunk_size)

    # Chunk overlap
    chunk_overlap = input(f"Chunk overlap [{config['chunk_overlap']}]: ").strip()
    if chunk_overlap:
        config["chunk_overlap"] = int(chunk_overlap)

    # Ollama host
    ollama_host = input(f"\nOllama host [{config['ollama_host']}]: ").strip()
    if ollama_host:
        config["ollama_host"] = ollama_host

    return config


if __name__ == "__main__":
    main()
