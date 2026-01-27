"""Tests for new MCP tools: getDatabases, getInformationFor, getInformationForDB."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.mcp.server import MCPServer
from mcp_agent_rag.rag.vector_db import VectorDatabase


@pytest.fixture
def multi_db_server(test_config, temp_dir):
    """Create MCP server with multiple test databases."""
    # Create first database
    db1_path = temp_dir / "db1"
    db1 = VectorDatabase(db1_path, dimension=768)
    db1.add(
        [[0.1] * 768, [0.2] * 768],
        [
            {"text": "Python is a programming language", "source": "python.txt", "chunk_num": 0},
            {"text": "Python is used for web development", "source": "python.txt", "chunk_num": 1},
        ],
    )
    db1.save()
    test_config.add_database("db1", str(db1_path), description="Python database", doc_count=1)

    # Create second database
    db2_path = temp_dir / "db2"
    db2 = VectorDatabase(db2_path, dimension=768)
    db2.add(
        [[0.3] * 768, [0.4] * 768],
        [
            {"text": "JavaScript is a scripting language", "source": "js.txt", "chunk_num": 0},
            {"text": "JavaScript runs in browsers", "source": "js.txt", "chunk_num": 1},
        ],
    )
    db2.save()
    test_config.add_database("db2", str(db2_path), description="JavaScript database", doc_count=1)

    test_config.save()

    with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
        server = MCPServer(test_config, ["db1", "db2"])
        return server


class TestGetDatabases:
    """Tests for getDatabases tool."""

    def test_get_databases_success(self, multi_db_server):
        """Test getDatabases returns list of active databases."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getDatabases",
            "params": {},
        }

        response = multi_db_server.handle_request(request)
        
        assert response["id"] == 1
        assert "result" in response
        assert "databases" in response["result"]
        assert response["result"]["count"] == 2
        
        databases = response["result"]["databases"]
        assert len(databases) == 2
        
        # Check database details
        db_names = [db["name"] for db in databases]
        assert "db1" in db_names
        assert "db2" in db_names
        
        # Verify structure
        for db in databases:
            assert "name" in db
            assert "description" in db
            assert "doc_count" in db
            assert "last_updated" in db
            assert "path" in db

    def test_get_databases_via_tools_call(self, multi_db_server):
        """Test getDatabases via tools/call."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "getDatabases",
                "arguments": {},
            },
        }

        response = multi_db_server.handle_request(request)
        
        assert response["id"] == 2
        assert "result" in response
        assert "databases" in response["result"]

    def test_get_databases_in_tools_list(self, multi_db_server):
        """Test getDatabases appears in tools/list."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
            "params": {},
        }

        response = multi_db_server.handle_request(request)
        
        assert "result" in response
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        
        assert "getDatabases" in tool_names
        
        # Find getDatabases tool and verify its schema
        get_db_tool = next(t for t in tools if t["name"] == "getDatabases")
        assert "description" in get_db_tool
        assert "inputSchema" in get_db_tool
        assert get_db_tool["inputSchema"]["type"] == "object"


class TestGetInformationFor:
    """Tests for getInformationFor tool."""

    def test_get_information_for_success(self, multi_db_server):
        """Test getInformationFor searches all active databases."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768  # Close to first database embeddings

            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getInformationFor",
                "params": {"prompt": "What is Python?"},
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 1
            assert "result" in response
            result = response["result"]
            
            assert "prompt" in result
            assert result["prompt"] == "What is Python?"
            assert "context" in result
            assert "citations" in result
            assert "databases_searched" in result
            
            # Should search both databases
            assert len(result["databases_searched"]) == 2
            assert "db1" in result["databases_searched"]
            assert "db2" in result["databases_searched"]

    def test_get_information_for_missing_prompt(self, multi_db_server):
        """Test getInformationFor with missing prompt parameter."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "getInformationFor",
            "params": {},
        }

        response = multi_db_server.handle_request(request)
        
        assert response["id"] == 2
        assert "error" in response
        assert "prompt" in response["error"]["message"].lower()

    def test_get_information_for_with_max_results(self, multi_db_server):
        """Test getInformationFor with custom max_results."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "getInformationFor",
                "params": {"prompt": "programming languages", "max_results": 10},
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 3
            assert "result" in response
            assert "context" in response["result"]

    def test_get_information_for_via_tools_call(self, multi_db_server):
        """Test getInformationFor via tools/call."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "getInformationFor",
                    "arguments": {"prompt": "test query"},
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 4
            assert "result" in response

    def test_get_information_for_in_tools_list(self, multi_db_server):
        """Test getInformationFor appears in tools/list."""
        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/list",
            "params": {},
        }

        response = multi_db_server.handle_request(request)
        
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        
        assert "getInformationFor" in tool_names
        
        # Verify schema
        tool = next(t for t in tools if t["name"] == "getInformationFor")
        assert "prompt" in tool["inputSchema"]["properties"]
        assert "max_results" in tool["inputSchema"]["properties"]
        assert "prompt" in tool["inputSchema"]["required"]


class TestGetInformationForDB:
    """Tests for getInformationForDB tool."""

    def test_get_information_for_db_success(self, multi_db_server):
        """Test getInformationForDB searches specific database."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "What is Python?",
                    "database_name": "db1",
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 1
            assert "result" in response
            result = response["result"]
            
            assert "prompt" in result
            assert result["prompt"] == "What is Python?"
            assert "database" in result
            assert result["database"] == "db1"
            assert "context" in result
            assert "citations" in result
            
            # Citations should only be from db1
            for citation in result["citations"]:
                assert citation["database"] == "db1"

    def test_get_information_for_db_missing_prompt(self, multi_db_server):
        """Test getInformationForDB with missing prompt parameter."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "getInformationForDB",
            "params": {"database_name": "db1"},
        }

        response = multi_db_server.handle_request(request)
        
        assert response["id"] == 2
        assert "error" in response
        assert "prompt" in response["error"]["message"].lower()

    def test_get_information_for_db_missing_database(self, multi_db_server):
        """Test getInformationForDB with missing database_name parameter."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "getInformationForDB",
            "params": {"prompt": "test"},
        }

        response = multi_db_server.handle_request(request)
        
        assert response["id"] == 3
        assert "error" in response
        assert "database_name" in response["error"]["message"].lower()

    def test_get_information_for_db_nonexistent_database(self, multi_db_server):
        """Test getInformationForDB with non-existent database."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "test",
                    "database_name": "nonexistent",
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 4
            assert "error" in response
            assert "nonexistent" in response["error"]["message"]

    def test_get_information_for_db_inactive_database(self, multi_db_server, temp_dir):
        """Test getInformationForDB with database not in active list."""
        # Create a third database but don't activate it
        db3_path = temp_dir / "db3"
        db3 = VectorDatabase(db3_path, dimension=768)
        db3.add([[0.5] * 768], [{"text": "test", "source": "test.txt", "chunk_num": 0}])
        db3.save()
        multi_db_server.config.add_database("db3", str(db3_path))
        multi_db_server.config.save()

        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "test",
                    "database_name": "db3",
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 5
            assert "error" in response
            assert "active" in response["error"]["message"].lower()

    def test_get_information_for_db_with_max_results(self, multi_db_server):
        """Test getInformationForDB with custom max_results."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "Python",
                    "database_name": "db1",
                    "max_results": 1,
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 6
            assert "result" in response
            result = response["result"]
            
            # With max_results=1, should get at most 1 citation
            assert len(result["citations"]) <= 1

    def test_get_information_for_db_embedding_failure(self, multi_db_server):
        """Test getInformationForDB when embedding fails."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = None  # Simulate embedding failure

            request = {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "test",
                    "database_name": "db1",
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 7
            assert "result" in response
            result = response["result"]
            
            # Should return empty results
            assert result["context"] == ""
            assert result["citations"] == []

    def test_get_information_for_db_via_tools_call(self, multi_db_server):
        """Test getInformationForDB via tools/call."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            request = {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "getInformationForDB",
                    "arguments": {
                        "prompt": "test",
                        "database_name": "db1",
                    },
                },
            }

            response = multi_db_server.handle_request(request)
            
            assert response["id"] == 8
            assert "result" in response

    def test_get_information_for_db_in_tools_list(self, multi_db_server):
        """Test getInformationForDB appears in tools/list."""
        request = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/list",
            "params": {},
        }

        response = multi_db_server.handle_request(request)
        
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        
        assert "getInformationForDB" in tool_names
        
        # Verify schema
        tool = next(t for t in tools if t["name"] == "getInformationForDB")
        assert "prompt" in tool["inputSchema"]["properties"]
        assert "database_name" in tool["inputSchema"]["properties"]
        assert "max_results" in tool["inputSchema"]["properties"]
        assert "prompt" in tool["inputSchema"]["required"]
        assert "database_name" in tool["inputSchema"]["required"]


class TestToolsIntegration:
    """Integration tests for all new tools."""

    def test_all_tools_listed(self, multi_db_server):
        """Test that all required tools are in tools/list."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }

        response = multi_db_server.handle_request(request)
        tools = response["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        
        # Check all three tools are present
        assert "getDatabases" in tool_names
        assert "getInformationFor" in tool_names
        assert "getInformationForDB" in tool_names
        
        # Should have exactly 3 tools (removed 4 deprecated tools)
        assert len(tools) == 3

    def test_workflow_get_dbs_then_query(self, multi_db_server):
        """Test workflow: get databases, then query specific one."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            # Step 1: Get list of databases
            request1 = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getDatabases",
                "params": {},
            }
            response1 = multi_db_server.handle_request(request1)
            
            assert "result" in response1
            databases = response1["result"]["databases"]
            assert len(databases) > 0
            
            # Step 2: Query first database
            first_db = databases[0]["name"]
            request2 = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "test query",
                    "database_name": first_db,
                },
            }
            response2 = multi_db_server.handle_request(request2)
            
            assert "result" in response2
            assert response2["result"]["database"] == first_db

    def test_compare_search_all_vs_specific(self, multi_db_server):
        """Test comparing search across all DBs vs specific DB."""
        with patch.object(multi_db_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.15] * 768

            # Search all databases
            request_all = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getInformationFor",
                "params": {"prompt": "programming"},
            }
            response_all = multi_db_server.handle_request(request_all)
            
            # Search specific database
            request_specific = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getInformationForDB",
                "params": {
                    "prompt": "programming",
                    "database_name": "db1",
                },
            }
            response_specific = multi_db_server.handle_request(request_specific)
            
            # Both should succeed
            assert "result" in response_all
            assert "result" in response_specific
            
            # All search should have multiple databases
            assert len(response_all["result"]["databases_searched"]) == 2
            
            # Specific search should have only one database
            assert response_specific["result"]["database"] == "db1"
