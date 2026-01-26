#!/usr/bin/env python3
"""Installation script for MCP-RAG."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import requests
except ImportError:
    # requests might not be available before installation
    requests = None


# Known embedding model name patterns
EMBEDDING_MODEL_PATTERNS = [
    "embed", "nomic", "mxbai", "all-minilm", "bge", "gte"
]


def normalize_host(host: str) -> str:
    """Normalize Ollama host URL.
    
    Args:
        host: Ollama host URL
        
    Returns:
        Normalized host URL
    """
    host = host.strip().rstrip("/")
    if host.endswith("/api"):
        host = host[:-4]
    return host


def check_ollama_connection(host: str, timeout: int = 5) -> Tuple[bool, str]:
    """Test connection to Ollama server.
    
    Args:
        host: Ollama host URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, error_message)
    """
    if requests is None:
        return False, "requests library not available yet"
        
    try:
        normalized_host = normalize_host(host)
        response = requests.get(f"{normalized_host}/api/tags", timeout=timeout)
        
        if response.status_code == 200:
            return True, ""
        else:
            return False, f"Server returned HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, f"Connection timeout to {host}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {host}. Is Ollama running?"
    except Exception as e:
        return False, f"Error: {str(e)}"


def fetch_ollama_models(host: str, timeout: int = 5) -> Tuple[List[str], List[str], str]:
    """Fetch available models from Ollama server and categorize them.
    
    Args:
        host: Ollama host URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (embedding_models, generative_models, error_message)
        If successful, error_message is empty. If failed, model lists are empty.
    """
    if requests is None:
        return [], [], "requests library not available yet"
        
    try:
        normalized_host = normalize_host(host)
        response = requests.get(f"{normalized_host}/api/tags", timeout=timeout)
        
        if response.status_code != 200:
            return [], [], f"Failed to fetch models: HTTP {response.status_code}"
        
        data = response.json()
        models = data.get("models", [])
        
        embedding_models = []
        generative_models = []
        
        for model in models:
            model_name = model.get("name", "")
            # Remove :latest tag for cleaner display
            display_name = model_name.replace(":latest", "")
            
            # Check if it's an embedding model using patterns
            is_embedding = any(pattern in model_name.lower() for pattern in EMBEDDING_MODEL_PATTERNS)
            
            if is_embedding:
                embedding_models.append(display_name)
            else:
                generative_models.append(display_name)
        
        return embedding_models, generative_models, ""
        
    except requests.exceptions.Timeout:
        return [], [], f"Connection timeout to {host}"
    except requests.exceptions.ConnectionError:
        return [], [], f"Cannot connect to {host}. Is Ollama running?"
    except Exception as e:
        return [], [], f"Error fetching models: {str(e)}"


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

    # Ollama host - ask first before fetching models
    print("\n1. Ollama Server Configuration")
    print("-" * 40)
    ollama_host = input(f"Ollama host URL [{config['ollama_host']}]: ").strip()
    if ollama_host:
        config["ollama_host"] = ollama_host
    
    # Test connection and fetch available models
    print(f"\nTesting connection to {config['ollama_host']}...")
    success, error = check_ollama_connection(config['ollama_host'])
    
    embedding_models = []
    generative_models = []
    
    if success:
        print("✓ Connection successful!")
        print("Fetching available models...")
        embedding_models, generative_models, error = fetch_ollama_models(config['ollama_host'])
        
        if error:
            print(f"Warning: {error}")
            print("Will use default model options.")
        else:
            print(f"Found {len(embedding_models)} embedding model(s) and {len(generative_models)} generative model(s)")
    else:
        print(f"⚠ Warning: {error}")
        print("Will use default model options. You can change them later in the config.")
    
    # Embedding model
    print("\n2. Embedding Model Selection")
    print("-" * 40)
    if embedding_models:
        print("Available embedding models on your Ollama server:")
        for i, model in enumerate(embedding_models, 1):
            default_marker = " (default)" if model == config['embedding_model'] else ""
            print(f"  {i}. {model}{default_marker}")
        
        choice = input(f"Select embedding model [1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(embedding_models):
                config["embedding_model"] = embedding_models[idx]
        except (ValueError, IndexError):
            print(f"Invalid choice, using default: {config['embedding_model']}")
    else:
        # Fallback to hardcoded options
        print("Available embedding models (default options):")
        print("  1. nomic-embed-text (default)")
        print("  2. mxbai-embed-large")
        print("  3. all-minilm")
        choice = input("Select embedding model [1]: ").strip() or "1"
        if choice == "2":
            config["embedding_model"] = "mxbai-embed-large"
        elif choice == "3":
            config["embedding_model"] = "all-minilm"

    # Generative model
    print("\n3. Generative Model Selection")
    print("-" * 40)
    if generative_models:
        print("Available generative models on your Ollama server:")
        for i, model in enumerate(generative_models, 1):
            default_marker = " (default)" if model == config['generative_model'] else ""
            print(f"  {i}. {model}{default_marker}")
        
        choice = input(f"Select generative model [1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(generative_models):
                config["generative_model"] = generative_models[idx]
        except (ValueError, IndexError):
            print(f"Invalid choice, using default: {config['generative_model']}")
    else:
        # Fallback to hardcoded options
        print("Available generative models (default options):")
        print("  1. mistral:7b-instruct (default, Apache 2.0)")
        print("  2. llama3.2 (Meta license)")
        print("  3. gemma2 (Google)")
        choice = input("Select generative model [1]: ").strip() or "1"
        if choice == "2":
            config["generative_model"] = "llama3.2"
        elif choice == "3":
            config["generative_model"] = "gemma2"

    # Chunk size
    print("\n4. Text Processing Configuration")
    print("-" * 40)
    chunk_size = input(f"Chunk size [{config['chunk_size']}]: ").strip()
    if chunk_size:
        try:
            config["chunk_size"] = int(chunk_size)
        except ValueError:
            print(f"Invalid value, using default: {config['chunk_size']}")

    # Chunk overlap
    chunk_overlap = input(f"Chunk overlap [{config['chunk_overlap']}]: ").strip()
    if chunk_overlap:
        try:
            config["chunk_overlap"] = int(chunk_overlap)
        except ValueError:
            print(f"Invalid value, using default: {config['chunk_overlap']}")

    return config


if __name__ == "__main__":
    main()
