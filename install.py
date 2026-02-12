#!/usr/bin/env python3
"""Installation script for MCP-RAG."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    # requests might not be available before installation
    requests = None


# Known embedding model name patterns
EMBEDDING_MODEL_PATTERNS = [
    "embed", "nomic", "mxbai", "all-minilm", "bge", "gte"
]


def safe_input(prompt: str, default: str = "") -> str:
    """Safely get user input with error handling.
    
    Args:
        prompt: The prompt to display to the user
        default: Default value to return if input fails
        
    Returns:
        User input or default value
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\nNo input received, using default: {default}")
        return default


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


def check_ollama_connection(host: str, timeout: int = 5) -> tuple[bool, str]:
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


def fetch_ollama_models(host: str, timeout: int = 5) -> tuple[list[str], list[str], str]:
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
            is_embedding = any(
                pattern in model_name.lower() for pattern in EMBEDDING_MODEL_PATTERNS
            )

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


def check_and_setup_gpu(python_path: Path, pip_path: Path, no_prompt: bool) -> dict:
    """Check for GPU availability and optionally install PyTorch with GPU support.

    Args:
        python_path: Path to Python executable
        pip_path: Path to pip executable
        no_prompt: Whether to skip prompts

    Returns:
        Dictionary with setup results
    """
    result = {
        "gpu_enabled": False,
        "pytorch_installed": False,
        "manual_install_needed": False,
    }

    print("\nChecking for GPU availability...")

    # Check if PyTorch is already installed
    pytorch_check = subprocess.run(
        [str(python_path), "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
    )

    pytorch_already_installed = pytorch_check.returncode == 0

    if pytorch_already_installed:
        print("✓ PyTorch is already installed")

        # Check CUDA availability
        cuda_check = subprocess.run(
            [str(python_path), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
        )

        if cuda_check.returncode == 0 and "True" in cuda_check.stdout:
            print("✓ CUDA is available - GPU support enabled")
            result["gpu_enabled"] = True
            result["pytorch_installed"] = True
            return result
        else:
            print("○ PyTorch installed but CUDA not available")
            result["pytorch_installed"] = True
            if no_prompt:
                print("  Continuing with CPU-only mode")
                return result
    else:
        print("○ PyTorch not installed")

    # Check for NVIDIA GPU
    nvidia_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )

    has_nvidia_gpu = nvidia_check.returncode == 0

    if has_nvidia_gpu:
        print("\n✓ NVIDIA GPU detected:")
        gpu_info_lines = nvidia_check.stdout.strip().split('\n')
        for line in gpu_info_lines:
            print(f"  {line}")

        # Check if CUDA toolkit is installed
        cuda_available = _check_cuda_toolkit()

        if not cuda_available:
            print("\n⚠ CUDA Toolkit not detected")
            print(
                "  GPU drivers are installed, but CUDA Toolkit is required "
                "for PyTorch GPU support"
            )

            if no_prompt:
                print("  Skipping GPU setup (--no-prompt specified)")
                result["manual_install_needed"] = True
                _print_gpu_install_instructions()
                return result

            print("\nOptions:")
            print("  1. Install PyTorch with CPU-only support (recommended for now)")
            print("  2. View instructions for manual CUDA Toolkit installation")
            print("  3. Skip PyTorch installation")

            choice = safe_input("\nSelect option [1]: ", "1") or "1"

            if choice == "2":
                _print_gpu_install_instructions()
                result["manual_install_needed"] = True
                print("\n⚠ Installation paused. Please install CUDA Toolkit and re-run installer.")
                sys.exit(0)
            elif choice == "3":
                print("Skipping PyTorch installation")
                return result
            # Fall through to option 1
        else:
            print("✓ CUDA Toolkit detected")

            if no_prompt:
                # Automatically install PyTorch with GPU support
                print("\nInstalling PyTorch with CUDA support...")
                return _install_pytorch_gpu(python_path, pip_path, result)

            print("\nDo you want to install PyTorch with GPU support?")
            print("  This will enable GPU acceleration for image OCR processing")
            choice = safe_input("Install PyTorch with GPU support? [Y/n]: ", "n").lower()

            if choice in ['', 'y', 'yes']:
                return _install_pytorch_gpu(python_path, pip_path, result)
            else:
                print("Skipping GPU setup - continuing with CPU-only mode")
                return result
    else:
        print("\n○ No NVIDIA GPU detected")

        # Check for Apple Silicon
        os_name = platform.system()
        if os_name == "Darwin":
            # Check for Apple Silicon
            arch = platform.machine()
            if arch == "arm64":
                print("✓ Apple Silicon detected")
                if not pytorch_already_installed:
                    prompt = "Install PyTorch with MPS support? [Y/n]: "
                    user_input = safe_input(prompt, 'n').lower() if not no_prompt else 'y'
                    if no_prompt or user_input in ['', 'y', 'yes']:
                        return _install_pytorch_mps(python_path, pip_path, result)

        print("Continuing with CPU-only mode")

    return result


def _check_cuda_toolkit() -> bool:
    """Check if CUDA toolkit is installed.

    Returns:
        True if CUDA toolkit is available
    """
    try:
        # Try nvcc command
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _install_pytorch_gpu(python_path: Path, pip_path: Path, result: dict) -> dict:
    """Install PyTorch with GPU support.

    Args:
        python_path: Path to Python executable
        pip_path: Path to pip executable
        result: Result dictionary to update

    Returns:
        Updated result dictionary
    """
    print("\nInstalling PyTorch with CUDA support...")
    print("This may take several minutes...")

    try:
        subprocess.run(
            [
                str(pip_path),
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu121"
            ],
            check=True,
        )

        print("✓ PyTorch with CUDA support installed successfully")
        result["pytorch_installed"] = True
        result["gpu_enabled"] = True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        print("  Continuing with CPU-only mode")

    return result


def _install_pytorch_mps(python_path: Path, pip_path: Path, result: dict) -> dict:
    """Install PyTorch with MPS support for Apple Silicon.

    Args:
        python_path: Path to Python executable
        pip_path: Path to pip executable
        result: Result dictionary to update

    Returns:
        Updated result dictionary
    """
    print("\nInstalling PyTorch with MPS support...")
    print("This may take several minutes...")

    try:
        subprocess.run(
            [str(pip_path), "install", "torch", "torchvision", "torchaudio"],
            check=True,
        )

        print("✓ PyTorch with MPS support installed successfully")
        result["pytorch_installed"] = True
        result["gpu_enabled"] = True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        print("  Continuing with CPU-only mode")

    return result


def _print_gpu_install_instructions():
    """Print GPU installation instructions."""
    os_name = platform.system()

    print("\n" + "=" * 70)
    print("GPU DRIVER AND CUDA TOOLKIT INSTALLATION INSTRUCTIONS")
    print("=" * 70)

    if os_name == "Linux":
        print("""
For NVIDIA GPUs on Linux:

1. Install NVIDIA drivers:
   # Ubuntu/Debian:
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt update
   sudo apt install nvidia-driver-535  # or latest version

2. Install CUDA Toolkit:
   # Download from: https://developer.nvidia.com/cuda-downloads
   # Or use package manager:
   sudo apt install nvidia-cuda-toolkit

3. Reboot your system:
   sudo reboot

4. Verify installation:
   nvidia-smi
   nvcc --version

5. Re-run this installer to install PyTorch with GPU support
""")
    elif os_name == "Windows":
        print("""
For NVIDIA GPUs on Windows:

1. Download and install NVIDIA drivers:
   Visit: https://www.nvidia.com/Download/index.aspx
   Select your GPU model and download the driver

2. Download and install CUDA Toolkit:
   Visit: https://developer.nvidia.com/cuda-downloads
   Download and run the installer for Windows

3. Restart your computer

4. Verify installation:
   Open Command Prompt and run:
   nvidia-smi
   nvcc --version

5. Re-run this installer to install PyTorch with GPU support
""")
    else:
        print("""
Please visit the following resources:

- NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- PyTorch installation: https://pytorch.org/get-started/locally/
""")

    print("=" * 70)


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

    # Check for GPU and offer PyTorch installation
    print("\n" + "=" * 60)
    print("GPU Detection and PyTorch Setup")
    print("=" * 60)

    gpu_setup_result = check_and_setup_gpu(python_path, pip_path, args.no_prompt)

    # Configuration
    config_path = Path.home() / ".mcp-agent-rag" / "config.json"

    if args.config:
        print(f"\nUsing existing config: {args.config}")
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists():
            print(f"\nExisting configuration found at: {config_path}")
            print("Preserving existing configuration...")
            with open(config_path) as f:
                config = json.load(f)

            # Add new agentic RAG fields if they don't exist
            needs_save = False
            if "query_inference_threshold" not in config:
                config["query_inference_threshold"] = 0.80
                needs_save = True
            if "iteration_confidence_threshold" not in config:
                config["iteration_confidence_threshold"] = 0.90
                needs_save = True
            if "final_augmentation_threshold" not in config:
                config["final_augmentation_threshold"] = 0.80
                needs_save = True

            if needs_save:
                print("Adding new agentic RAG configuration fields...")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print("Configuration updated")
            else:
                print("Configuration preserved")
        else:
            print("\nSetting up configuration...")
            # Pass actual GPU status from setup, defaulting to True to auto-detect at runtime
            gpu_enabled = gpu_setup_result.get("gpu_enabled", True)
            config = create_config(args.no_prompt, gpu_enabled=gpu_enabled)

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Configuration saved to: {config_path}")

    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)
    print("\nTo use MCP-RAG:")
    if os_name == "Windows":
        print(f"  1. Activate: {venv_path}\\Scripts\\activate")
    else:
        print(f"  1. Activate: source {venv_path}/bin/activate")
    print("  2. Run: mcp-rag --help")
    print(f"\nConfig file: {config_path}")

    # Display GPU setup summary
    if gpu_setup_result.get("gpu_enabled"):
        print("\n✓ GPU support enabled")
        if gpu_setup_result.get("pytorch_installed"):
            print("✓ PyTorch with GPU support installed")
    elif gpu_setup_result.get("manual_install_needed"):
        print("\n⚠ Manual GPU driver installation required")
        print("  Please follow the instructions above and re-run the installer")
    else:
        print("\n○ CPU-only mode (GPU not available or not configured)")



def create_config(no_prompt: bool, gpu_enabled: bool = True) -> dict:
    """Create configuration interactively or with defaults.

    Args:
        no_prompt: Use defaults without prompting
        gpu_enabled: Enable GPU usage when available (default True)

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
        "gpu_enabled": gpu_enabled,
        "gpu_device": None,
        # Agentic RAG inference probability thresholds
        "query_inference_threshold": 0.80,  # Inference threshold for generating RAG queries
        "iteration_confidence_threshold": 0.90,  # Threshold for accepting information completeness
        "final_augmentation_threshold": 0.80,  # Inference threshold for final prompt augmentation
    }

    if no_prompt:
        print("Using default configuration")
        return config

    print("\nConfiguration Setup")
    print("-" * 40)

    # Ollama host - ask first before fetching models
    print("\n1. Ollama Server Configuration")
    print("-" * 40)
    ollama_host = safe_input(f"Ollama host URL [{config['ollama_host']}]: ", "")
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
            print(
                f"Found {len(embedding_models)} embedding model(s) and "
                f"{len(generative_models)} generative model(s)"
            )
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

        choice = safe_input("Select embedding model [1]: ", "1") or "1"
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
        choice = safe_input("Select embedding model [1]: ", "1") or "1"
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

        choice = safe_input("Select generative model [1]: ", "1") or "1"
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
        choice = safe_input("Select generative model [1]: ", "1") or "1"
        if choice == "2":
            config["generative_model"] = "llama3.2"
        elif choice == "3":
            config["generative_model"] = "gemma2"

    # Chunk size
    print("\n4. Text Processing Configuration")
    print("-" * 40)
    chunk_size = safe_input(f"Chunk size [{config['chunk_size']}]: ", "")
    if chunk_size:
        try:
            config["chunk_size"] = int(chunk_size)
        except ValueError:
            print(f"Invalid value, using default: {config['chunk_size']}")

    # Chunk overlap
    chunk_overlap = safe_input(f"Chunk overlap [{config['chunk_overlap']}]: ", "")
    if chunk_overlap:
        try:
            config["chunk_overlap"] = int(chunk_overlap)
        except ValueError:
            print(f"Invalid value, using default: {config['chunk_overlap']}")

    return config


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during installation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
