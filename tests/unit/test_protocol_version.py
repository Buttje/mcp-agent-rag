"""Tests for MCP protocol version support."""

from unittest.mock import patch

import pytest

from mcp_agent_rag.mcp.server import (
    MCP_PROTOCOL_VERSION_2024,
    MCP_PROTOCOL_VERSION_2025,
    MCPServer,
)
from mcp_agent_rag.rag.vector_db import VectorDatabase


@pytest.fixture
def server_2025(test_config, temp_dir):
    """Create MCP server with 2025 protocol version."""
    # Create database
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=768)
    db.add([[0.1] * 768], [{"text": "test", "source": "test.txt"}])
    db.save()

    test_config.add_database("testdb", str(db_path))
    test_config.save()

    with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
        server = MCPServer(test_config, ["testdb"], protocol_version=MCP_PROTOCOL_VERSION_2025)
        return server


@pytest.fixture
def server_2024(test_config, temp_dir):
    """Create MCP server with 2024 protocol version."""
    # Create database
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=768)
    db.add([[0.1] * 768], [{"text": "test", "source": "test.txt"}])
    db.save()

    test_config.add_database("testdb", str(db_path))
    test_config.save()

    with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
        server = MCPServer(test_config, ["testdb"], protocol_version=MCP_PROTOCOL_VERSION_2024)
        return server


class TestProtocolVersionInitialization:
    """Test protocol version initialization."""

    def test_default_protocol_version(self, test_config, temp_dir):
        """Test that default protocol version is 2025-11-25."""
        db_path = temp_dir / "testdb"
        db = VectorDatabase(db_path, dimension=768)
        db.add([[0.1] * 768], [{"text": "test", "source": "test.txt"}])
        db.save()

        test_config.add_database("testdb", str(db_path))
        test_config.save()

        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(test_config, ["testdb"])
            assert server.protocol_version == MCP_PROTOCOL_VERSION_2025

    def test_explicit_2025_protocol_version(self, server_2025):
        """Test explicit 2025 protocol version."""
        assert server_2025.protocol_version == MCP_PROTOCOL_VERSION_2025
        assert server_2025.protocol_version == "2025-11-25"

    def test_explicit_2024_protocol_version(self, server_2024):
        """Test explicit 2024 protocol version."""
        assert server_2024.protocol_version == MCP_PROTOCOL_VERSION_2024
        assert server_2024.protocol_version == "2024-11-05"

    def test_invalid_protocol_version(self, test_config, temp_dir):
        """Test that invalid protocol version raises ValueError."""
        db_path = temp_dir / "testdb"
        db = VectorDatabase(db_path, dimension=768)
        db.add([[0.1] * 768], [{"text": "test", "source": "test.txt"}])
        db.save()

        test_config.add_database("testdb", str(db_path))
        test_config.save()

        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            with pytest.raises(ValueError) as excinfo:
                MCPServer(test_config, ["testdb"], protocol_version="2023-01-01")
            assert "Unsupported protocol version" in str(excinfo.value)
            assert "2023-01-01" in str(excinfo.value)


class TestProtocolVersionNegotiation:
    """Test protocol version negotiation during initialization."""

    def test_initialize_returns_2025_version(self, server_2025):
        """Test that initialize returns 2025 protocol version."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = server_2025.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2025-11-25"
        assert response["result"]["serverInfo"]["name"] == "mcp-agent-rag"

    def test_initialize_returns_2024_version(self, server_2024):
        """Test that initialize returns 2024 protocol version."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cody-client", "version": "1.0.0"},
            },
        }

        response = server_2024.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        assert response["result"]["serverInfo"]["name"] == "mcp-agent-rag"

    def test_initialize_with_mismatched_client_version(self, server_2024):
        """Test that server returns its configured version regardless of client version."""
        # Client requests 2025 but server is configured for 2024
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = server_2024.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        # Server should return its configured version (2024)
        assert response["result"]["protocolVersion"] == "2024-11-05"

    def test_initialize_without_protocol_version(self, server_2024):
        """Test initialize request without protocolVersion parameter."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        response = server_2024.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"


class TestProtocolVersionFunctionality:
    """Test that both protocol versions work correctly."""

    def test_tools_list_with_2025_version(self, server_2025):
        """Test tools/list works with 2025 protocol version."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = server_2025.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0

    def test_tools_list_with_2024_version(self, server_2024):
        """Test tools/list works with 2024 protocol version."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = server_2024.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0

    def test_resources_list_with_2025_version(self, server_2025):
        """Test resources/list works with 2025 protocol version."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/list",
            "params": {},
        }

        response = server_2025.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert "resources" in response["result"]

    def test_resources_list_with_2024_version(self, server_2024):
        """Test resources/list works with 2024 protocol version."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/list",
            "params": {},
        }

        response = server_2024.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert "resources" in response["result"]

    def test_database_operations_with_2024_version(self, server_2024):
        """Test database operations work with 2024 protocol version."""
        # Test database-list
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "database-list",
            "params": {},
        }

        response = server_2024.handle_request(request)
        assert response["id"] == 1
        assert "result" in response
        assert "databases" in response["result"]

    def test_query_operations_with_2024_version(self, server_2024):
        """Test query operations work with 2024 protocol version."""
        with patch.object(server_2024.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.1] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "query-get_data",
                "params": {"prompt": "test query", "max_results": 5},
            }

            response = server_2024.handle_request(request)
            assert response["id"] == 1
            assert "result" in response
            assert "context" in response["result"]


class TestProtocolVersionConstants:
    """Test protocol version constants."""

    def test_protocol_version_2024_constant(self):
        """Test 2024 protocol version constant."""
        assert MCP_PROTOCOL_VERSION_2024 == "2024-11-05"

    def test_protocol_version_2025_constant(self):
        """Test 2025 protocol version constant."""
        assert MCP_PROTOCOL_VERSION_2025 == "2025-11-25"

    def test_both_versions_supported(self, test_config, temp_dir):
        """Test that both versions can be instantiated."""
        db_path = temp_dir / "testdb"
        db = VectorDatabase(db_path, dimension=768)
        db.add([[0.1] * 768], [{"text": "test", "source": "test.txt"}])
        db.save()

        test_config.add_database("testdb", str(db_path))
        test_config.save()

        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server_2024 = MCPServer(
                test_config, ["testdb"], protocol_version=MCP_PROTOCOL_VERSION_2024
            )
            server_2025 = MCPServer(
                test_config, ["testdb"], protocol_version=MCP_PROTOCOL_VERSION_2025
            )

            assert server_2024.protocol_version == "2024-11-05"
            assert server_2025.protocol_version == "2025-11-25"
