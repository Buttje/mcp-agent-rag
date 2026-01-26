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
    """Test handling database-create request."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "database-create",
        "params": {"name": "newdb", "description": "New database"},
    }

    response = server.handle_request(request)
    assert response["id"] == 1
    assert "result" in response
    assert response["result"]["success"] is True


def test_handle_database_list(server):
    """Test handling database-list request."""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "database-list",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 2
    assert "result" in response
    assert "databases" in response["result"]


def test_handle_query_get_data(server):
    """Test handling query-get_data request."""
    with patch.object(server.agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "query-get_data",
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
            "name": "database-list",
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
        "method": "database-create",
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
        "method": "database-create",
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
        "method": "database-add",
        "params": {"database_name": "nonexistent", "path": "/tmp"},
    }

    response = server.handle_request(request)
    assert response["id"] == 10
    assert "error" in response


def test_handle_initialize(server):
    """Test handling initialize request."""
    request = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "TestClient", "version": "1.0.0"},
        },
    }

    response = server.handle_request(request)
    assert response["id"] == 11
    assert "result" in response
    assert response["result"]["protocolVersion"] == "2025-11-25"
    assert "capabilities" in response["result"]
    assert "serverInfo" in response["result"]
    assert response["result"]["serverInfo"]["name"] == "mcp-agent-rag"


def test_handle_initialized_notification(server):
    """Test handling initialized notification."""
    request = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }

    response = server.handle_request(request)
    # Notifications should return None
    assert response is None


def test_initialize_minimal_params(server):
    """Test initialize with minimal parameters."""
    request = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "initialize",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 12
    assert "result" in response
    assert "protocolVersion" in response["result"]
    assert "capabilities" in response["result"]
    assert "serverInfo" in response["result"]


def test_custom_notification_handling(server):
    """Test that any notification with notifications/ prefix returns None."""
    request = {
        "jsonrpc": "2.0",
        "method": "notifications/custom_event",
        "params": {"data": "test"},
    }

    response = server.handle_request(request)
    # Notifications should return None
    assert response is None


def test_handle_resources_templates_list(server):
    """Test handling resources/templates/list request."""
    request = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "resources/templates/list",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 13
    assert "result" in response
    assert "resourceTemplates" in response["result"]
    templates = response["result"]["resourceTemplates"]
    assert isinstance(templates, list)
    assert len(templates) > 0
    
    # Check that templates have required fields
    for template in templates:
        assert "uriTemplate" in template
        assert "name" in template
        assert "description" in template
        assert "mimeType" in template
        # Verify URI template contains parameter placeholders
        assert "{" in template["uriTemplate"]


def test_resources_templates_structure(server):
    """Test that resource templates follow MCP specification structure."""
    result = server._list_resource_templates({})
    
    assert "resourceTemplates" in result
    templates = result["resourceTemplates"]
    
    # Should have at least database query and info templates
    assert len(templates) >= 2
    
    # Check for specific expected templates
    uri_templates = [t["uriTemplate"] for t in templates]
    assert any("database://{database_name}/query" in uri for uri in uri_templates)
    assert any("database://{database_name}/info" in uri for uri in uri_templates)


def test_handle_prompts_list(server):
    """Test handling prompts/list request."""
    request = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "prompts/list",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 13
    assert "result" in response
    assert "prompts" in response["result"]
    prompts = response["result"]["prompts"]
    assert isinstance(prompts, list)
    # Currently returns empty list, which is valid per MCP spec


def test_prompts_list_structure(server):
    """Test that prompts list follows MCP specification structure."""
    result = server._list_prompts({})
    
    assert "prompts" in result
    prompts = result["prompts"]
    assert isinstance(prompts, list)
    # Empty list is valid - server doesn't define custom prompts yet


def test_handle_get_databases(server):
    """Test handling getDatabases request."""
    request = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "getDatabases",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 14
    assert "result" in response
    assert "databases" in response["result"]
    assert "count" in response["result"]
    assert response["result"]["count"] == 1
    assert response["result"]["databases"][0]["name"] == "testdb"


def test_get_databases_via_tools_call(server):
    """Test calling getDatabases via tools/call method."""
    request = {
        "jsonrpc": "2.0",
        "id": 15,
        "method": "tools/call",
        "params": {
            "name": "getDatabases",
            "arguments": {},
        },
    }

    response = server.handle_request(request)
    assert response["id"] == 15
    assert "result" in response
    assert "databases" in response["result"]
    assert response["result"]["count"] == 1


def test_handle_get_information_for(server):
    """Test handling getInformationFor request."""
    with patch.object(server.agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = {
            "jsonrpc": "2.0",
            "id": 16,
            "method": "getInformationFor",
            "params": {"prompt": "test query", "max_results": 5},
        }

        response = server.handle_request(request)
        assert response["id"] == 16
        assert "result" in response
        assert "context" in response["result"]
        assert "citations" in response["result"]
        assert "databases_searched" in response["result"]


def test_get_information_for_via_tools_call(server):
    """Test calling getInformationFor via tools/call method."""
    with patch.object(server.agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = {
            "jsonrpc": "2.0",
            "id": 17,
            "method": "tools/call",
            "params": {
                "name": "getInformationFor",
                "arguments": {"prompt": "test query"},
            },
        }

        response = server.handle_request(request)
        assert response["id"] == 17
        assert "result" in response
        assert "context" in response["result"]


def test_handle_get_information_for_db(server):
    """Test handling getInformationForDB request."""
    with patch.object(server.agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = {
            "jsonrpc": "2.0",
            "id": 18,
            "method": "getInformationForDB",
            "params": {
                "prompt": "test query",
                "database_name": "testdb",
                "max_results": 5,
            },
        }

        response = server.handle_request(request)
        assert response["id"] == 18
        assert "result" in response
        assert "context" in response["result"]
        assert "citations" in response["result"]
        assert "database" in response["result"]
        assert response["result"]["database"] == "testdb"


def test_get_information_for_db_via_tools_call(server):
    """Test calling getInformationForDB via tools/call method."""
    with patch.object(server.agent.embedder, "embed_single") as mock_embed:
        mock_embed.return_value = [0.1] * 768

        request = {
            "jsonrpc": "2.0",
            "id": 19,
            "method": "tools/call",
            "params": {
                "name": "getInformationForDB",
                "arguments": {"prompt": "test query", "database_name": "testdb"},
            },
        }

        response = server.handle_request(request)
        assert response["id"] == 19
        assert "result" in response
        assert "context" in response["result"]
        assert "database" in response["result"]


def test_get_information_for_db_invalid_database(server):
    """Test getInformationForDB with invalid database name."""
    request = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "getInformationForDB",
        "params": {"prompt": "test query", "database_name": "nonexistent"},
    }

    response = server.handle_request(request)
    assert response["id"] == 20
    assert "error" in response
    assert "not in active databases" in response["error"]["message"]


def test_get_information_for_missing_prompt(server):
    """Test getInformationFor with missing prompt parameter."""
    request = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "getInformationFor",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 21
    assert "error" in response
    assert "prompt" in response["error"]["message"].lower()


def test_get_information_for_db_missing_database_name(server):
    """Test getInformationForDB with missing database_name parameter."""
    request = {
        "jsonrpc": "2.0",
        "id": 22,
        "method": "getInformationForDB",
        "params": {"prompt": "test query"},
    }

    response = server.handle_request(request)
    assert response["id"] == 22
    assert "error" in response
    assert "database_name" in response["error"]["message"].lower()


def test_handle_logging_set_level(server):
    """Test handling logging/setLevel request."""
    request = {
        "jsonrpc": "2.0",
        "id": 23,
        "method": "logging/setLevel",
        "params": {"level": "info"},
    }

    response = server.handle_request(request)
    assert response["id"] == 23
    assert "result" in response
    assert response["result"] == {}


def test_logging_set_level_invalid(server):
    """Test logging/setLevel with invalid level."""
    request = {
        "jsonrpc": "2.0",
        "id": 24,
        "method": "logging/setLevel",
        "params": {"level": "invalid_level"},
    }

    response = server.handle_request(request)
    assert response["id"] == 24
    assert "error" in response
    assert "invalid" in response["error"]["message"].lower()


def test_logging_set_level_missing_param(server):
    """Test logging/setLevel with missing level parameter."""
    request = {
        "jsonrpc": "2.0",
        "id": 25,
        "method": "logging/setLevel",
        "params": {},
    }

    response = server.handle_request(request)
    assert response["id"] == 25
    assert "error" in response
    assert "level" in response["error"]["message"].lower()


def test_logging_set_level_all_valid_levels(server):
    """Test logging/setLevel with all valid levels."""
    valid_levels = ["debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"]
    
    for i, level in enumerate(valid_levels):
        request = {
            "jsonrpc": "2.0",
            "id": 26 + i,
            "method": "logging/setLevel",
            "params": {"level": level},
        }

        response = server.handle_request(request)
        assert response["id"] == 26 + i
        assert "result" in response
        assert response["result"] == {}
