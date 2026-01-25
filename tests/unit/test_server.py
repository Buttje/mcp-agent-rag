"""Tests for MCP server."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.mcp.server import MCPServer
from mcp_agent_rag.rag.vector_db import VectorDatabase


@pytest.fixture
def server(test_config, temp_dir):
    """Create MCP server with test database."""
    # Create database
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=768)
    db.add([[0.1] * 768], [{"text": "test", "source": "test.txt"}])
    db.save()

    test_config.add_database("testdb", str(db_path))
    test_config.save()

    with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
        server = MCPServer(test_config, ["testdb"])
        return server


def test_server_initialization(server):
    """Test server initialization."""
    assert server is not None
    assert "testdb" in server.loaded_databases
    assert server.agent is not None


def test_handle_database_create(server):
    """Test handling database/create request."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "database/create",
        "params": {"name": "newdb", "description": "New database"},
    }

    response = server.handle_request(request)
    assert response["id"] == 1
    assert "result" in response
    assert response["result"]["success"] is True


def test_handle_database_list(server):
    """Test handling database/list request."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "database/list",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 2
    assert "result" in response
    assert "databases" in response["result"]


def test_handle_query_get_data(server):
    """Test handling query/get_data request."""
    with patch.object(server.agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "query/get_data",
            "params": {"prompt": "test query", "max_results": 5},
        }

        response = server.handle_request(request)
        assert response["id"] == 3
        assert "result" in response
        assert "context" in response["result"]


def test_handle_resources_list(server):
    """Test handling resources/list request."""
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "resources/list",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 4
    assert "result" in response
    assert "resources" in response["result"]


def test_handle_tools_list(server):
    """Test handling tools/list request."""
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/list",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 5
    assert "result" in response
    assert "tools" in response["result"]
    assert len(response["result"]["tools"]) > 0


def test_handle_tools_call(server):
    """Test handling tools/call request."""
    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "database/list",
            "arguments": {},
        },
    }

    response = server.handle_request(request)
    assert response["id"] == 6
    assert "result" in response


def test_handle_unknown_method(server):
    """Test handling unknown method."""
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "unknown/method",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 7
    assert "error" in response
    assert response["error"]["code"] == -32601


def test_handle_error_in_request(server):
    """Test handling error during request processing."""
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "database/create",
        "params": {},  # Missing required 'name' parameter
    }

    response = server.handle_request(request)
    assert response["id"] == 8
    assert "error" in response


def test_create_database_already_exists(server):
    """Test creating database that already exists."""
    # testdb already exists
    request = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "database/create",
        "params": {"name": "testdb"},
    }

    response = server.handle_request(request)
    assert response["id"] == 9
    assert "error" in response


def test_add_documents_nonexistent_database(server):
    """Test adding documents to non-existent database."""
    request = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "database/add",
        "params": {"database_name": "nonexistent", "path": "/tmp"},
    }

    response = server.handle_request(request)
    assert response["id"] == 10
    assert "error" in response
