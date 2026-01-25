"""Test fixtures and utilities."""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from mcp_agent_rag.config import Config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create test configuration."""
    config_path = temp_dir / "config.json"
    config = Config(str(config_path))
    return config


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample text file for testing.\nIt has multiple lines.")
    return file_path


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create sample Python file."""
    file_path = temp_dir / "sample.py"
    file_path.write_text("""
def hello():
    '''Hello function.'''
    print("Hello, world!")

if __name__ == "__main__":
    hello()
""")
    return file_path


@pytest.fixture
def sample_project_dir(temp_dir: Path) -> Path:
    """Create sample project directory."""
    project_dir = temp_dir / "project"
    project_dir.mkdir()

    # Create files
    (project_dir / "main.py").write_text("print('Main')")
    (project_dir / "utils.py").write_text("def util(): pass")

    # Create subdirectory
    sub_dir = project_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "module.py").write_text("def module_func(): pass")

    # Create .gitignore
    (project_dir / ".gitignore").write_text("*.pyc\n__pycache__\n")

    return project_dir


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "embeddings": [
            [0.1, 0.2, 0.3, 0.4] * 192  # 768 dimensions
        ]
    }
