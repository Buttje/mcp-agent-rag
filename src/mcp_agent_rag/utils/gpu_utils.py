"""GPU detection and configuration utilities."""

import platform
import subprocess
from pathlib import Path
from typing import Optional

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class GPUInfo:
    """Information about GPU availability and capabilities."""

    def __init__(
        self,
        available: bool = False,
        device_count: int = 0,
        device_names: list[str] = None,
        cuda_version: Optional[str] = None,
        driver_version: Optional[str] = None,
        pytorch_available: bool = False,
        pytorch_cuda_available: bool = False,
    ):
        """Initialize GPU information.

        Args:
            available: Whether GPU is available for use
            device_count: Number of GPU devices
            device_names: List of GPU device names
            cuda_version: CUDA version string
            driver_version: GPU driver version
            pytorch_available: Whether PyTorch is installed
            pytorch_cuda_available: Whether PyTorch has CUDA support
        """
        self.available = available
        self.device_count = device_count
        self.device_names = device_names or []
        self.cuda_version = cuda_version
        self.driver_version = driver_version
        self.pytorch_available = pytorch_available
        self.pytorch_cuda_available = pytorch_cuda_available

    def __repr__(self) -> str:
        """String representation of GPU info."""
        return (
            f"GPUInfo(available={self.available}, "
            f"device_count={self.device_count}, "
            f"devices={self.device_names}, "
            f"cuda={self.cuda_version}, "
            f"driver={self.driver_version})"
        )


def detect_gpu() -> GPUInfo:
    """Detect available GPU hardware and drivers.

    Returns:
        GPUInfo object with detection results
    """
    logger.info("Detecting GPU availability...")

    # Try to import and check PyTorch
    pytorch_available = False
    pytorch_cuda_available = False
    device_count = 0
    device_names = []
    cuda_version = None

    try:
        import torch

        pytorch_available = True
        pytorch_cuda_available = torch.cuda.is_available()

        if pytorch_cuda_available:
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            cuda_version = torch.version.cuda

            logger.info(
                f"PyTorch detected: CUDA available with {device_count} device(s)"
            )
            for i, name in enumerate(device_names):
                logger.info(f"  Device {i}: {name}")
        else:
            logger.info("PyTorch detected but CUDA not available")

    except ImportError:
        logger.info("PyTorch not installed")
    except Exception as e:
        logger.warning(f"Error checking PyTorch GPU availability: {e}")

    # Try to detect NVIDIA driver version
    driver_version = _detect_nvidia_driver()

    # Determine overall GPU availability
    # GPU is available if we have CUDA support in PyTorch
    available = pytorch_cuda_available and device_count > 0

    return GPUInfo(
        available=available,
        device_count=device_count,
        device_names=device_names,
        cuda_version=cuda_version,
        driver_version=driver_version,
        pytorch_available=pytorch_available,
        pytorch_cuda_available=pytorch_cuda_available,
    )


def _detect_nvidia_driver() -> Optional[str]:
    """Detect NVIDIA driver version using nvidia-smi.

    Returns:
        Driver version string or None if not detected
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            if version:
                logger.info(f"NVIDIA driver version: {version}")
                return version

    except FileNotFoundError:
        logger.debug("nvidia-smi not found")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi command timed out")
    except Exception as e:
        logger.debug(f"Could not detect NVIDIA driver: {e}")

    return None


def get_gpu_installation_instructions() -> str:
    """Get GPU driver installation instructions for the current platform.

    Returns:
        Formatted installation instructions
    """
    os_name = platform.system()

    instructions = [
        "=" * 70,
        "GPU DRIVER INSTALLATION INSTRUCTIONS",
        "=" * 70,
        "",
    ]

    if os_name == "Linux":
        instructions.extend(
            [
                "For NVIDIA GPUs on Linux:",
                "",
                "1. Check your GPU model:",
                "   lspci | grep -i nvidia",
                "",
                "2. Add NVIDIA repository and install drivers:",
                "   # Ubuntu/Debian:",
                "   sudo add-apt-repository ppa:graphics-drivers/ppa",
                "   sudo apt update",
                "   sudo apt install nvidia-driver-535  # or latest version",
                "",
                "3. Install CUDA Toolkit (required for PyTorch GPU support):",
                "   # Download from: https://developer.nvidia.com/cuda-downloads",
                "   # Or use package manager:",
                "   sudo apt install nvidia-cuda-toolkit",
                "",
                "4. Install PyTorch with CUDA support:",
                "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "",
                "5. Reboot your system:",
                "   sudo reboot",
                "",
                "6. Verify installation:",
                "   nvidia-smi",
                "   python -c 'import torch; print(torch.cuda.is_available())'",
                "",
            ]
        )

    elif os_name == "Windows":
        instructions.extend(
            [
                "For NVIDIA GPUs on Windows:",
                "",
                "1. Check your GPU in Device Manager (Win+X, then select Device Manager)",
                "",
                "2. Download and install NVIDIA drivers:",
                "   Visit: https://www.nvidia.com/Download/index.aspx",
                "   Select your GPU model and download the driver",
                "",
                "3. Download and install CUDA Toolkit:",
                "   Visit: https://developer.nvidia.com/cuda-downloads",
                "   Download and run the installer for Windows",
                "",
                "4. Install PyTorch with CUDA support:",
                "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "",
                "5. Restart your computer",
                "",
                "6. Verify installation:",
                "   nvidia-smi",
                "   python -c \"import torch; print(torch.cuda.is_available())\"",
                "",
            ]
        )

    elif os_name == "Darwin":  # macOS
        instructions.extend(
            [
                "For Apple Silicon (M1/M2/M3) Macs:",
                "",
                "Apple Silicon Macs use Metal Performance Shaders (MPS) instead of CUDA.",
                "PyTorch supports MPS for GPU acceleration on Apple Silicon.",
                "",
                "1. Ensure you have Python 3.8 or later",
                "",
                "2. Install PyTorch with MPS support:",
                "   pip install torch torchvision torchaudio",
                "",
                "3. Verify MPS availability:",
                "   python -c 'import torch; print(torch.backends.mps.is_available())'",
                "",
                "Note: NVIDIA GPUs are not supported on modern Macs.",
                "",
            ]
        )

    else:
        instructions.extend(
            [
                f"Platform: {os_name}",
                "",
                "Please refer to the following resources for GPU driver installation:",
                "",
                "1. NVIDIA GPUs:",
                "   https://www.nvidia.com/Download/index.aspx",
                "   https://developer.nvidia.com/cuda-downloads",
                "",
                "2. PyTorch installation:",
                "   https://pytorch.org/get-started/locally/",
                "",
            ]
        )

    instructions.extend(
        [
            "=" * 70,
            "ADDITIONAL RESOURCES",
            "=" * 70,
            "",
            "- PyTorch installation guide: https://pytorch.org/get-started/locally/",
            "- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads",
            "- NVIDIA drivers: https://www.nvidia.com/Download/index.aspx",
            "",
        ]
    )

    return "\n".join(instructions)


def recommend_pytorch_installation() -> tuple[str, str]:
    """Recommend PyTorch installation command for current platform.

    Returns:
        Tuple of (command, description)
    """
    os_name = platform.system()

    # Detect if NVIDIA GPU is present
    has_nvidia = _detect_nvidia_driver() is not None

    if os_name == "Darwin":  # macOS
        return (
            "pip install torch torchvision torchaudio",
            "PyTorch with MPS support for Apple Silicon",
        )
    elif has_nvidia:
        return (
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "PyTorch with CUDA 12.1 support",
        )
    else:
        return (
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "PyTorch CPU-only version (no GPU support)",
        )


def check_pytorch_installed() -> bool:
    """Check if PyTorch is installed.

    Returns:
        True if PyTorch is available
    """
    try:
        import torch

        return True
    except ImportError:
        return False
