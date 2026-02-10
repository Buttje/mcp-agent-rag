"""Tests for GPU detection utilities."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.utils.gpu_utils import (
    GPUInfo,
    check_pytorch_installed,
    detect_gpu,
    get_gpu_installation_instructions,
    recommend_pytorch_installation,
)


class TestGPUInfo:
    """Test GPUInfo class."""

    def test_gpu_info_creation(self):
        """Test creating GPUInfo object."""
        info = GPUInfo(
            available=True,
            device_count=2,
            device_names=["GPU 0", "GPU 1"],
            cuda_version="12.1",
            driver_version="535.123",
            pytorch_available=True,
            pytorch_cuda_available=True,
        )

        assert info.available is True
        assert info.device_count == 2
        assert len(info.device_names) == 2
        assert info.cuda_version == "12.1"
        assert info.driver_version == "535.123"

    def test_gpu_info_repr(self):
        """Test GPUInfo string representation."""
        info = GPUInfo(available=True, device_count=1, device_names=["Test GPU"])
        repr_str = repr(info)

        assert "GPUInfo" in repr_str
        assert "available=True" in repr_str
        assert "device_count=1" in repr_str


class TestDetectGPU:
    """Test GPU detection."""

    @patch("mcp_agent_rag.utils.gpu_utils._detect_nvidia_driver")
    def test_detect_gpu_no_pytorch(self, mock_driver):
        """Test GPU detection when PyTorch is not installed."""
        mock_driver.return_value = None

        with patch.dict("sys.modules", {"torch": None}):
            info = detect_gpu()

        assert info.pytorch_available is False
        assert info.pytorch_cuda_available is False
        assert info.available is False
        assert info.device_count == 0

    @patch("mcp_agent_rag.utils.gpu_utils._detect_nvidia_driver")
    def test_detect_gpu_with_cuda(self, mock_driver):
        """Test GPU detection with CUDA available."""
        mock_driver.return_value = "535.123"

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = lambda i: f"GPU {i}"
        mock_torch.version.cuda = "12.1"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = detect_gpu()

        assert info.pytorch_available is True
        assert info.pytorch_cuda_available is True
        assert info.available is True
        assert info.device_count == 2
        assert len(info.device_names) == 2
        assert info.cuda_version == "12.1"
        assert info.driver_version == "535.123"

    @patch("mcp_agent_rag.utils.gpu_utils._detect_nvidia_driver")
    def test_detect_gpu_cpu_only(self, mock_driver):
        """Test GPU detection with CPU-only PyTorch."""
        mock_driver.return_value = None

        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = detect_gpu()

        assert info.pytorch_available is True
        assert info.pytorch_cuda_available is False
        assert info.available is False
        assert info.device_count == 0


class TestNvidiaDriver:
    """Test NVIDIA driver detection."""

    @patch("subprocess.run")
    def test_detect_nvidia_driver_success(self, mock_run):
        """Test successful NVIDIA driver detection."""
        from mcp_agent_rag.utils.gpu_utils import _detect_nvidia_driver

        mock_run.return_value = Mock(returncode=0, stdout="535.123\n")

        version = _detect_nvidia_driver()

        assert version == "535.123"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_detect_nvidia_driver_not_found(self, mock_run):
        """Test NVIDIA driver detection when nvidia-smi not found."""
        from mcp_agent_rag.utils.gpu_utils import _detect_nvidia_driver

        mock_run.side_effect = FileNotFoundError()

        version = _detect_nvidia_driver()

        assert version is None

    @patch("subprocess.run")
    def test_detect_nvidia_driver_timeout(self, mock_run):
        """Test NVIDIA driver detection timeout."""
        from mcp_agent_rag.utils.gpu_utils import _detect_nvidia_driver

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        version = _detect_nvidia_driver()

        assert version is None


class TestInstallationInstructions:
    """Test installation instructions."""

    @patch("platform.system")
    def test_instructions_linux(self, mock_system):
        """Test instructions for Linux."""
        mock_system.return_value = "Linux"

        instructions = get_gpu_installation_instructions()

        assert "Linux" in instructions
        assert "apt" in instructions or "NVIDIA" in instructions
        assert "CUDA" in instructions

    @patch("platform.system")
    def test_instructions_windows(self, mock_system):
        """Test instructions for Windows."""
        mock_system.return_value = "Windows"

        instructions = get_gpu_installation_instructions()

        assert "Windows" in instructions
        assert "NVIDIA" in instructions
        assert "CUDA" in instructions

    @patch("platform.system")
    def test_instructions_macos(self, mock_system):
        """Test instructions for macOS."""
        mock_system.return_value = "Darwin"

        instructions = get_gpu_installation_instructions()

        assert "Apple Silicon" in instructions or "MPS" in instructions


class TestPyTorchInstallation:
    """Test PyTorch installation recommendations."""

    @patch("platform.system")
    @patch("mcp_agent_rag.utils.gpu_utils._detect_nvidia_driver")
    def test_recommend_pytorch_with_cuda(self, mock_driver, mock_system):
        """Test PyTorch recommendation with CUDA."""
        mock_system.return_value = "Linux"
        mock_driver.return_value = "535.123"

        command, description = recommend_pytorch_installation()

        assert "torch" in command
        assert "cu" in command or "CUDA" in description

    @patch("platform.system")
    @patch("mcp_agent_rag.utils.gpu_utils._detect_nvidia_driver")
    def test_recommend_pytorch_cpu_only(self, mock_driver, mock_system):
        """Test PyTorch recommendation for CPU-only."""
        mock_system.return_value = "Linux"
        mock_driver.return_value = None

        command, description = recommend_pytorch_installation()

        assert "torch" in command
        assert "cpu" in command.lower() or "CPU" in description

    @patch("platform.system")
    def test_recommend_pytorch_macos(self, mock_system):
        """Test PyTorch recommendation for macOS."""
        mock_system.return_value = "Darwin"

        command, description = recommend_pytorch_installation()

        assert "torch" in command
        assert "MPS" in description or "Apple" in description


class TestCheckPyTorch:
    """Test PyTorch installation check."""

    def test_check_pytorch_not_installed(self):
        """Test when PyTorch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            # Force ImportError
            import sys

            torch_module = sys.modules.get("torch")
            if torch_module:
                sys.modules["torch"] = None

            result = check_pytorch_installed()

            # The function should handle the ImportError
            assert isinstance(result, bool)

    def test_check_pytorch_installed(self):
        """Test when PyTorch is installed."""
        mock_torch = Mock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = check_pytorch_installed()

            assert result is True
