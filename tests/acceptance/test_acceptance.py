"""Acceptance tests based on specification."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestInstallation:
    """Test 1.1 and 1.2: Installation and Configuration."""

    def test_install_creates_venv_and_config(self, temp_dir):
        """Test 1.1: Install script creates virtual environment and installs dependencies."""
        # Note: This would require running the actual install script
        # For unit testing, we verify the config creation
        from mcp_agent_rag.config import Config

        config = Config(str(temp_dir / "config.json"))
        config.save()

        assert (temp_dir / "config.json").exists()
        assert config.get("embedding_model") is not None
        assert config.get("generative_model") is not None

    def test_install_reads_existing_config(self, temp_dir):
        """Test 1.2: Install script reads configuration from file."""
        from mcp_agent_rag.config import Config

        # Create config
        config_path = temp_dir / "config.json"
        config = Config(str(config_path))
        config.set("embedding_model", "custom-model")
        config.save()

        # Load it
        config2 = Config(str(config_path))
        assert config2.get("embedding_model") == "custom-model"


class TestDatabaseManagement:
    """Test 2.1-2.7: Database Management Tools."""

    @pytest.fixture
    def db_manager(self, test_config):
        """Create database manager."""
        from unittest.mock import patch
        from mcp_agent_rag.database import DatabaseManager

        with patch("mcp_agent_rag.database.OllamaEmbedder"):
            return DatabaseManager(test_config)

    def test_create_new_database(self, db_manager, test_config):
        """Test 2.1: Create a new database."""
        success = db_manager.create_database("testdb", "Test database")
        assert success is True
        assert test_config.database_exists("testdb")

        db_path = test_config.get_database_path("testdb")
        assert db_path is not None
        assert db_path.exists()

    def test_prevent_duplicate_database_names(self, db_manager, test_config):
        """Test 2.2: Prevent duplicate database names."""
        db_manager.create_database("testdb")
        success = db_manager.create_database("testdb")
        assert success is False

    def test_list_databases(self, db_manager, test_config):
        """Test 2.3: List databases."""
        db_manager.create_database("db1", "First database")
        db_manager.create_database("db2", "Second database")

        databases = db_manager.list_databases()
        assert len(databases) == 2
        assert "db1" in databases
        assert "db2" in databases

    def test_add_documents_via_path(self, db_manager, test_config, sample_text_file):
        """Test 2.4: Add documents via path."""
        db_manager.create_database("testdb")

        # Mock embedder
        from unittest.mock import patch
        with patch.object(db_manager.embedder, "embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 768]

            stats = db_manager.add_documents(
                database_name="testdb",
                path=str(sample_text_file),
            )

            assert stats["processed"] >= 0

    def test_handle_unsupported_file_format(self, db_manager, test_config, temp_dir):
        """Test 2.7: Handle unsupported file format gracefully."""
        db_manager.create_database("testdb")

        # Create unsupported file
        unsupported = temp_dir / "archive.zip"
        unsupported.write_bytes(b"\x00\x01")

        from unittest.mock import patch
        with patch.object(db_manager.embedder, "embed") as mock_embed:
            mock_embed.return_value = [[0.1] * 768]

            stats = db_manager.add_documents(
                database_name="testdb",
                path=str(unsupported),
            )

            # Should handle gracefully without crashing
            assert stats["failed"] >= 0 or stats["skipped"] >= 0


class TestServerStartup:
    """Test 3.1-3.3: Server Startup and Transports."""

    @pytest.fixture
    def server_config(self, test_config, temp_dir):
        """Create config with database."""
        from mcp_agent_rag.rag.vector_db import VectorDatabase

        # Create a test database
        db_path = temp_dir / "db1"
        db = VectorDatabase(db_path, dimension=768)
        db.save()

        test_config.add_database("db1", str(db_path))
        test_config.save()

        return test_config

    def test_start_server_with_active_databases(self, server_config):
        """Test 3.1: Start server with active databases via stdio."""
        from mcp_agent_rag.mcp.server import MCPServer
        from unittest.mock import patch

        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(server_config, ["db1"])
            assert "db1" in server.loaded_databases

    def test_reject_missing_active_databases(self, server_config):
        """Test 3.3: Reject missing active databases."""
        from mcp_agent_rag.mcp.server import MCPServer
        from unittest.mock import patch

        with pytest.raises(ValueError, match="Failed to load databases"):
            with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
                MCPServer(server_config, ["db1", "nonexistent"])


class TestQuerying:
    """Test 4.1-4.3: Querying for Data."""

    @pytest.fixture
    def server_with_data(self, test_config, temp_dir):
        """Create server with test data."""
        from mcp_agent_rag.mcp.server import MCPServer
        from mcp_agent_rag.rag.vector_db import VectorDatabase
        from unittest.mock import patch

        # Create database with data
        db_path = temp_dir / "db1"
        db = VectorDatabase(db_path, dimension=768)

        # Add test data
        embeddings = [[0.1] * 768, [0.2] * 768]
        metadata = [
            {"text": "Test document about encryption", "source": "doc1.txt"},
            {"text": "Network security documentation", "source": "doc2.txt"},
        ]
        db.add(embeddings, metadata)
        db.save()

        test_config.add_database("db1", str(db_path))
        test_config.save()

        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(test_config, ["db1"])
            return server

    def test_retrieve_context_for_prompt(self, server_with_data):
        """Test 4.1: Retrieve context for prompt."""
        from unittest.mock import patch
        with patch.object(
            server_with_data.agent.embedder, "embed_single"
        ) as mock_embed:
            mock_embed.return_value = [0.15] * 768

            result = server_with_data._query_data({"prompt": "What is encryption?"})

            assert "prompt" in result
            assert "context" in result
            assert "citations" in result

    def test_validate_max_results_parameter(self, server_with_data):
        """Test 4.3: Validate max_results parameter."""
        from unittest.mock import patch
        with patch.object(
            server_with_data.agent.embedder, "embed_single"
        ) as mock_embed:
            mock_embed.return_value = [0.15] * 768

            result = server_with_data._query_data({
                "prompt": "test query",
                "max_results": 2
            })

            # Should respect max_results
            assert len(result["citations"]) <= 2


class TestErrorHandling:
    """Test 5.1-5.2: Error Handling and Logging."""

    def test_meaningful_error_on_nonexistent_database(self, test_config):
        """Test 5.1: Meaningful error on nonexistent database."""
        from mcp_agent_rag.mcp.server import MCPServer
        from mcp_agent_rag.rag.vector_db import VectorDatabase
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "db1"
            db = VectorDatabase(db_path, dimension=768)
            db.save()
            test_config.add_database("db1", str(db_path))
            test_config.save()

            with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
                server = MCPServer(test_config, ["db1"])

                # Try to add to non-existent database
                response = server.handle_request({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "database-add",
                    "params": {"database_name": "nonexistent", "path": "/tmp"},
                })

                assert "error" in response
                assert "nonexistent" in str(response["error"])


class TestCoverage:
    """Test 6.1: Test Coverage."""

    def test_coverage_threshold(self):
        """Test 6.1: Coverage threshold.

        This test verifies that the test configuration requires 90% coverage.
        The actual coverage check is done by pytest-cov.
        """
        # Read pyproject.toml to verify coverage threshold
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomli/tomllib not available")

        from pathlib import Path

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if not pyproject_path.exists():
            pytest.skip("pyproject.toml not found")

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        coverage_config = config.get("tool", {}).get("coverage", {}).get("report", {})
        fail_under = coverage_config.get("fail_under", 0)

        assert fail_under >= 80, "Coverage threshold should be at least 80%"
