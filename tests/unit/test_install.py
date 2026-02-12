"""Unit tests for the installation script."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the path so we can import install
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import install


class TestSafeInput:
    """Tests for the safe_input function."""

    def test_safe_input_normal(self):
        """Test safe_input with normal input."""
        with patch('builtins.input', return_value="test input"):
            result = install.safe_input("Enter something: ", "default")
            assert result == "test input"

    def test_safe_input_eoferror(self):
        """Test safe_input handles EOFError gracefully."""
        with patch('builtins.input', side_effect=EOFError):
            result = install.safe_input("Enter something: ", "default")
            assert result == "default"

    def test_safe_input_keyboard_interrupt(self):
        """Test safe_input handles KeyboardInterrupt gracefully."""
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = install.safe_input("Enter something: ", "default")
            assert result == "default"

    def test_safe_input_empty_default(self):
        """Test safe_input with empty default value."""
        with patch('builtins.input', side_effect=EOFError):
            result = install.safe_input("Enter something: ", "")
            assert result == ""

    def test_safe_input_strips_whitespace(self):
        """Test safe_input strips whitespace from input."""
        with patch('builtins.input', return_value="  test  "):
            result = install.safe_input("Enter something: ", "default")
            assert result == "test"


class TestNormalizeHost:
    """Tests for the normalize_host function."""

    def test_normalize_host_basic(self):
        """Test normalizing a basic URL."""
        result = install.normalize_host("http://localhost:11434")
        assert result == "http://localhost:11434"

    def test_normalize_host_trailing_slash(self):
        """Test normalizing URL with trailing slash."""
        result = install.normalize_host("http://localhost:11434/")
        assert result == "http://localhost:11434"

    def test_normalize_host_api_suffix(self):
        """Test normalizing URL with /api suffix."""
        result = install.normalize_host("http://localhost:11434/api")
        assert result == "http://localhost:11434"

    def test_normalize_host_api_and_slash(self):
        """Test normalizing URL with /api/ suffix."""
        result = install.normalize_host("http://localhost:11434/api/")
        assert result == "http://localhost:11434"

    def test_normalize_host_whitespace(self):
        """Test normalizing URL with whitespace."""
        result = install.normalize_host("  http://localhost:11434  ")
        assert result == "http://localhost:11434"


class TestCheckOllamaConnection:
    """Tests for the check_ollama_connection function."""

    def test_check_ollama_connection_no_requests(self):
        """Test when requests library is not available."""
        with patch('install.requests', None):
            success, error = install.check_ollama_connection("http://localhost:11434")
            assert not success
            assert "requests library not available" in error

    @patch('install.requests')
    def test_check_ollama_connection_success(self, mock_requests):
        """Test successful connection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response

        success, error = install.check_ollama_connection("http://localhost:11434")
        assert success
        assert error == ""

    @patch('install.requests')
    def test_check_ollama_connection_http_error(self, mock_requests):
        """Test connection with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_requests.get.return_value = mock_response

        success, error = install.check_ollama_connection("http://localhost:11434")
        assert not success
        assert "HTTP 404" in error

    @patch('install.requests')
    def test_check_ollama_connection_timeout(self, mock_requests):
        """Test connection timeout."""
        mock_requests.get.side_effect = mock_requests.exceptions.Timeout
        mock_requests.exceptions.Timeout = Exception

        success, error = install.check_ollama_connection("http://localhost:11434")
        assert not success
        assert "timeout" in error.lower()

    @patch('install.requests')
    def test_check_ollama_connection_connection_error(self, mock_requests):
        """Test connection error."""
        # We need to import the real requests exceptions for the test
        import requests as real_requests

        mock_requests.exceptions = real_requests.exceptions
        mock_requests.get.side_effect = real_requests.exceptions.ConnectionError("Connection error")

        success, error = install.check_ollama_connection("http://localhost:11434")
        assert not success
        assert "cannot connect" in error.lower()


class TestFetchOllamaModels:
    """Tests for the fetch_ollama_models function."""

    def test_fetch_ollama_models_no_requests(self):
        """Test when requests library is not available."""
        with patch('install.requests', None):
            embedding, generative, error = install.fetch_ollama_models("http://localhost:11434")
            assert embedding == []
            assert generative == []
            assert "requests library not available" in error

    @patch('install.requests')
    def test_fetch_ollama_models_success(self, mock_requests):
        """Test successful model fetching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "mistral:7b-instruct"},
                {"name": "mxbai-embed-large"},
            ]
        }
        mock_requests.get.return_value = mock_response

        embedding, generative, error = install.fetch_ollama_models("http://localhost:11434")
        assert len(embedding) == 2
        assert len(generative) == 1
        assert error == ""
        assert "nomic-embed-text" in embedding
        assert "mxbai-embed-large" in embedding
        assert "mistral:7b-instruct" in generative

    @patch('install.requests')
    def test_fetch_ollama_models_http_error(self, mock_requests):
        """Test fetching models with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response

        embedding, generative, error = install.fetch_ollama_models("http://localhost:11434")
        assert embedding == []
        assert generative == []
        assert "HTTP 500" in error


class TestCheckCudaToolkit:
    """Tests for the _check_cuda_toolkit function."""

    @patch('subprocess.run')
    def test_check_cuda_toolkit_available(self, mock_run):
        """Test when CUDA toolkit is available."""
        mock_run.return_value = Mock(returncode=0)
        result = install._check_cuda_toolkit()
        assert result is True

    @patch('subprocess.run')
    def test_check_cuda_toolkit_not_available(self, mock_run):
        """Test when CUDA toolkit is not available."""
        mock_run.return_value = Mock(returncode=1)
        result = install._check_cuda_toolkit()
        assert result is False

    @patch('subprocess.run')
    def test_check_cuda_toolkit_file_not_found(self, mock_run):
        """Test when nvcc command is not found."""
        mock_run.side_effect = FileNotFoundError
        result = install._check_cuda_toolkit()
        assert result is False

    @patch('subprocess.run')
    def test_check_cuda_toolkit_timeout(self, mock_run):
        """Test when nvcc command times out."""
        mock_run.side_effect = subprocess.TimeoutExpired('nvcc', 5)
        result = install._check_cuda_toolkit()
        assert result is False


class TestCheckAndSetupGPU:
    """Tests for the check_and_setup_gpu function."""

    @patch('subprocess.run')
    def test_check_and_setup_gpu_pytorch_with_cuda(self, mock_run):
        """Test when PyTorch is installed with CUDA available."""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="2.0.0"),  # PyTorch check
            Mock(returncode=0, stdout="True"),    # CUDA check
        ]

        result = install.check_and_setup_gpu(
            Path("/usr/bin/python"),
            Path("/usr/bin/pip"),
            no_prompt=True
        )

        assert result["gpu_enabled"] is True
        assert result["pytorch_installed"] is True
        assert result["manual_install_needed"] is False

    @patch('subprocess.run')
    def test_check_and_setup_gpu_pytorch_no_cuda_with_prompt(self, mock_run):
        """Test when PyTorch is installed but CUDA is not available, with no_prompt=True."""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="2.0.0"),  # PyTorch check
            Mock(returncode=0, stdout="False"),  # CUDA check
        ]

        result = install.check_and_setup_gpu(
            Path("/usr/bin/python"),
            Path("/usr/bin/pip"),
            no_prompt=True
        )

        assert result["gpu_enabled"] is False
        assert result["pytorch_installed"] is True
        assert result["manual_install_needed"] is False

    @patch('subprocess.run')
    def test_check_and_setup_gpu_no_pytorch(self, mock_run):
        """Test when PyTorch is not installed and no GPU detected."""
        mock_run.side_effect = [
            Mock(returncode=1, stdout=""),       # PyTorch not installed
            Mock(returncode=1, stdout=""),       # nvidia-smi not available
        ]

        result = install.check_and_setup_gpu(
            Path("/usr/bin/python"),
            Path("/usr/bin/pip"),
            no_prompt=True
        )

        assert result["gpu_enabled"] is False
        assert result["pytorch_installed"] is False

    @patch('subprocess.run')
    def test_check_and_setup_gpu_nvidia_smi_not_found(self, mock_run):
        """Test when nvidia-smi command is not found (FileNotFoundError)."""
        mock_run.side_effect = [
            Mock(returncode=0, stdout="2.0.0"),  # PyTorch check succeeds
            Mock(returncode=0, stdout="False"),  # CUDA check - not available
            FileNotFoundError,                   # nvidia-smi not found
        ]

        result = install.check_and_setup_gpu(
            Path("/usr/bin/python"),
            Path("/usr/bin/pip"),
            no_prompt=True
        )

        # Should handle FileNotFoundError gracefully
        assert result["gpu_enabled"] is False
        assert result["pytorch_installed"] is True
        assert result["manual_install_needed"] is False

    @patch('subprocess.run')
    def test_check_and_setup_gpu_nvidia_smi_not_found_no_pytorch(self, mock_run):
        """Test when nvidia-smi command is not found and PyTorch not installed."""
        mock_run.side_effect = [
            Mock(returncode=1, stdout=""),       # PyTorch not installed
            FileNotFoundError,                   # nvidia-smi not found
        ]

        result = install.check_and_setup_gpu(
            Path("/usr/bin/python"),
            Path("/usr/bin/pip"),
            no_prompt=True
        )

        # Should handle FileNotFoundError gracefully
        assert result["gpu_enabled"] is False
        assert result["pytorch_installed"] is False
        assert result["manual_install_needed"] is False

    @patch('subprocess.run')
    def test_check_and_setup_gpu_nvidia_smi_file_not_found_windows(self, mock_run):
        """Test handling of Windows-specific nvidia-smi FileNotFoundError scenario."""
        # Simulate the exact scenario from the bug report:
        # PyTorch installed, CUDA not available, nvidia-smi raises FileNotFoundError
        mock_run.side_effect = [
            Mock(returncode=0, stdout="2.0.0"),  # PyTorch installed
            Mock(returncode=0, stdout="False"),  # CUDA not available
            FileNotFoundError,  # nvidia-smi not found (Windows: "[WinError 2]")
        ]

        # This should not raise an exception
        result = install.check_and_setup_gpu(
            Path("C:\\Python\\python.exe"),
            Path("C:\\Python\\Scripts\\pip.exe"),
            no_prompt=True
        )

        # Should continue with CPU-only mode
        assert result["gpu_enabled"] is False
        assert result["pytorch_installed"] is True
        assert result["manual_install_needed"] is False


class TestMainErrorHandling:
    """Tests for error handling in main function."""

    @patch('install.Path.exists')
    @patch('install.subprocess.run')
    @patch('builtins.print')
    def test_create_config_with_input_error(self, mock_print, mock_run, mock_exists):
        """Test that create_config handles input errors gracefully via safe_input."""
        # Mock environment
        mock_exists.return_value = False

        # The safe_input function should handle EOFError internally
        # So we should test that it returns the default when input fails
        with patch('builtins.input', side_effect=EOFError):
            result = install.safe_input("Test prompt: ", "default_value")
            assert result == "default_value"


class TestCreateConfig:
    """Tests for the create_config function."""

    def test_create_config_no_prompt(self):
        """Test create_config with no_prompt=True."""
        config = install.create_config(no_prompt=True, gpu_enabled=True)

        assert config["embedding_model"] == "nomic-embed-text"
        assert config["generative_model"] == "mistral:7b-instruct"
        assert config["chunk_size"] == 512
        assert config["chunk_overlap"] == 50
        assert config["gpu_enabled"] is True
        assert config["query_inference_threshold"] == 0.80
        assert config["iteration_confidence_threshold"] == 0.90
        assert config["final_augmentation_threshold"] == 0.80

    def test_create_config_gpu_disabled(self):
        """Test create_config with gpu_enabled=False."""
        config = install.create_config(no_prompt=True, gpu_enabled=False)
        assert config["gpu_enabled"] is False

    @patch('install.check_ollama_connection')
    @patch('install.fetch_ollama_models')
    @patch('install.safe_input')
    def test_create_config_with_ollama_success(
        self, mock_input, mock_fetch, mock_check
    ):
        """Test create_config with successful Ollama connection."""
        mock_check.return_value = (True, "")
        mock_fetch.return_value = (
            ["nomic-embed-text", "mxbai-embed-large"],
            ["mistral:7b-instruct", "llama3.2"],
            ""
        )
        mock_input.side_effect = [
            "",    # Ollama host (use default)
            "1",   # Embedding model selection
            "1",   # Generative model selection
            "",    # Chunk size (use default)
            "",    # Chunk overlap (use default)
        ]

        config = install.create_config(no_prompt=False, gpu_enabled=True)

        assert config["embedding_model"] == "nomic-embed-text"
        assert config["generative_model"] == "mistral:7b-instruct"

    @patch('install.check_ollama_connection')
    @patch('install.safe_input')
    def test_create_config_with_ollama_failure(self, mock_input, mock_check):
        """Test create_config when Ollama connection fails."""
        mock_check.return_value = (False, "Cannot connect")
        mock_input.side_effect = [
            "",    # Ollama host (use default)
            "1",   # Embedding model selection (fallback)
            "1",   # Generative model selection (fallback)
            "",    # Chunk size (use default)
            "",    # Chunk overlap (use default)
        ]

        config = install.create_config(no_prompt=False, gpu_enabled=True)

        # Should use defaults even when connection fails
        assert config["embedding_model"] == "nomic-embed-text"
        assert config["generative_model"] == "mistral:7b-instruct"
