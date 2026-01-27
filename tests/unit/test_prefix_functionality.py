"""Tests for prefix functionality in MCP tools."""

from unittest.mock import patch

import pytest

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager
from mcp_agent_rag.mcp.server import MCPServer
from mcp_agent_rag.rag.vector_db import VectorDatabase


@pytest.fixture
def config_with_prefixes(test_config, temp_dir):
    """Create config with databases that have prefixes."""
    # Create first database with prefix A1
    db1_path = temp_dir / "db1"
    db1 = VectorDatabase(db1_path, dimension=768)
    db1.add(
        [[0.1] * 768],
        [{"text": "Test data for db1", "source": "test1.txt", "chunk_num": 0}],
    )
    db1.save()
    test_config.add_database("db1", str(db1_path), description="DB1", prefix="A1")

    # Create second database with prefix B1
    db2_path = temp_dir / "db2"
    db2 = VectorDatabase(db2_path, dimension=768)
    db2.add(
        [[0.2] * 768],
        [{"text": "Test data for db2", "source": "test2.txt", "chunk_num": 0}],
    )
    db2.save()
    test_config.add_database("db2", str(db2_path), description="DB2", prefix="B1")

    # Create third database with prefix A2
    db3_path = temp_dir / "db3"
    db3 = VectorDatabase(db3_path, dimension=768)
    db3.add(
        [[0.3] * 768],
        [{"text": "Test data for db3", "source": "test3.txt", "chunk_num": 0}],
    )
    db3.save()
    test_config.add_database("db3", str(db3_path), description="DB3", prefix="A2")

    test_config.save()
    return test_config


class TestPrefixInConfig:
    """Tests for prefix storage in config."""

    def test_add_database_with_prefix(self, test_config):
        """Test adding a database with a prefix."""
        test_config.add_database(
            name="test_db",
            path="/path/to/db",
            description="Test database",
            prefix="TEST"
        )
        
        db_config = test_config.get_database("test_db")
        assert db_config is not None
        assert db_config["prefix"] == "TEST"

    def test_add_database_without_prefix(self, test_config):
        """Test adding a database without a prefix defaults to empty string."""
        test_config.add_database(
            name="test_db",
            path="/path/to/db",
            description="Test database"
        )
        
        db_config = test_config.get_database("test_db")
        assert db_config is not None
        assert db_config["prefix"] == ""

    def test_update_database_prefix(self, test_config):
        """Test updating a database's prefix."""
        test_config.add_database(
            name="test_db",
            path="/path/to/db",
            description="Test database",
            prefix="OLD"
        )
        
        test_config.update_database("test_db", prefix="NEW")
        
        db_config = test_config.get_database("test_db")
        assert db_config["prefix"] == "NEW"


class TestPrefixInServer:
    """Tests for prefix functionality in MCP server."""

    def test_single_prefix_tool_names(self, config_with_prefixes, temp_dir):
        """Test that tools get prefixed correctly with a single database."""
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(config_with_prefixes, ["db1"])
            
            # Check that prefix was set correctly
            assert server.tool_prefix == "A1_"
            
            # Get tools list
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
            
            response = server.handle_request(request)
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]
            
            # All tools should be prefixed with A1_
            assert "A1_getDatabases" in tool_names
            assert "A1_getInformationFor" in tool_names
            assert "A1_getInformationForDB" in tool_names

    def test_multiple_prefix_tool_names(self, config_with_prefixes, temp_dir):
        """Test that tools get combined prefix from multiple databases."""
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(config_with_prefixes, ["db1", "db2", "db3"])
            
            # Check that combined prefix was created
            assert server.tool_prefix == "A1_B1_A2_"
            
            # Get tools list
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
            
            response = server.handle_request(request)
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]
            
            # All tools should be prefixed with A1_B1_A2_
            assert "A1_B1_A2_getDatabases" in tool_names
            assert "A1_B1_A2_getInformationFor" in tool_names
            assert "A1_B1_A2_getInformationForDB" in tool_names

    def test_no_prefix_tool_names(self, test_config, temp_dir):
        """Test that tools work without prefix when databases have no prefix."""
        # Create database without prefix
        db_path = temp_dir / "db_no_prefix"
        db = VectorDatabase(db_path, dimension=768)
        db.add(
            [[0.1] * 768],
            [{"text": "Test data", "source": "test.txt", "chunk_num": 0}],
        )
        db.save()
        test_config.add_database("db_no_prefix", str(db_path), description="No prefix")
        test_config.save()
        
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(test_config, ["db_no_prefix"])
            
            # Check that no prefix was set
            assert server.tool_prefix == ""
            
            # Get tools list
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
            
            response = server.handle_request(request)
            tools = response["result"]["tools"]
            tool_names = [tool["name"] for tool in tools]
            
            # Tools should not be prefixed
            assert "getDatabases" in tool_names
            assert "getInformationFor" in tool_names
            assert "getInformationForDB" in tool_names

    def test_call_prefixed_tool(self, config_with_prefixes, temp_dir):
        """Test calling a tool with prefixed name."""
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(config_with_prefixes, ["db1"])
            
            # Call prefixed getDatabases
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "A1_getDatabases",
                    "arguments": {},
                },
            }
            
            response = server.handle_request(request)
            
            # Should succeed
            assert "result" in response
            # Check MCP-compliant format
            import json
            assert "content" in response["result"]
            assert isinstance(response["result"]["content"], list)
            assert response["result"]["isError"] is False
            data = json.loads(response["result"]["content"][0]["text"])
            assert "databases" in data

    def test_call_prefixed_tool_with_arguments(self, config_with_prefixes, temp_dir):
        """Test calling a prefixed tool with arguments."""
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(config_with_prefixes, ["db1", "db2"])
            
            with patch.object(server.agent.embedder, "embed_single") as mock_embed:
                mock_embed.return_value = [0.15] * 768
                
                # Call prefixed getInformationFor
                request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "A1_B1_getInformationFor",
                        "arguments": {"prompt": "test query"},
                    },
                }
                
                response = server.handle_request(request)
                
                # Should succeed
                assert "result" in response
                # Check MCP-compliant format
                import json
                assert "content" in response["result"]
                assert isinstance(response["result"]["content"], list)
                assert response["result"]["isError"] is False
                data = json.loads(response["result"]["content"][0]["text"])
                assert "context" in data

    def test_mixed_prefix_databases(self, test_config, temp_dir):
        """Test server with mix of databases with and without prefixes."""
        # Create one database with prefix
        db1_path = temp_dir / "db_with_prefix"
        db1 = VectorDatabase(db1_path, dimension=768)
        db1.add(
            [[0.1] * 768],
            [{"text": "Test data", "source": "test1.txt", "chunk_num": 0}],
        )
        db1.save()
        test_config.add_database("db_with_prefix", str(db1_path), prefix="PRE")
        
        # Create one database without prefix
        db2_path = temp_dir / "db_without_prefix"
        db2 = VectorDatabase(db2_path, dimension=768)
        db2.add(
            [[0.2] * 768],
            [{"text": "Test data", "source": "test2.txt", "chunk_num": 0}],
        )
        db2.save()
        test_config.add_database("db_without_prefix", str(db2_path), prefix="")
        test_config.save()
        
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(test_config, ["db_with_prefix", "db_without_prefix"])
            
            # Only the database with prefix should contribute
            assert server.tool_prefix == "PRE_"

    def test_duplicate_prefix_handling(self, test_config, temp_dir):
        """Test that duplicate prefixes are deduplicated while preserving order."""
        # Create three databases with some duplicate prefixes
        db1_path = temp_dir / "db1"
        db1 = VectorDatabase(db1_path, dimension=768)
        db1.add([[0.1] * 768], [{"text": "Test", "source": "test.txt", "chunk_num": 0}])
        db1.save()
        test_config.add_database("db1", str(db1_path), prefix="A1")
        
        db2_path = temp_dir / "db2"
        db2 = VectorDatabase(db2_path, dimension=768)
        db2.add([[0.2] * 768], [{"text": "Test", "source": "test.txt", "chunk_num": 0}])
        db2.save()
        test_config.add_database("db2", str(db2_path), prefix="B1")
        
        db3_path = temp_dir / "db3"
        db3 = VectorDatabase(db3_path, dimension=768)
        db3.add([[0.3] * 768], [{"text": "Test", "source": "test.txt", "chunk_num": 0}])
        db3.save()
        test_config.add_database("db3", str(db3_path), prefix="A1")  # Duplicate A1
        test_config.save()
        
        with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
            server = MCPServer(test_config, ["db1", "db2", "db3"])
            
            # Should have A1_B1_ (not A1_B1_A1_)
            assert server.tool_prefix == "A1_B1_"


class TestDatabaseManagerWithPrefix:
    """Tests for DatabaseManager with prefix support."""

    def test_create_database_with_prefix(self, test_config):
        """Test creating a database with a prefix."""
        db_manager = DatabaseManager(test_config)
        
        success = db_manager.create_database(
            name="test_db",
            description="Test database",
            prefix="TST"
        )
        
        assert success
        
        db_config = test_config.get_database("test_db")
        assert db_config is not None
        assert db_config["prefix"] == "TST"

    def test_create_database_without_prefix(self, test_config):
        """Test creating a database without a prefix."""
        db_manager = DatabaseManager(test_config)
        
        success = db_manager.create_database(
            name="test_db",
            description="Test database"
        )
        
        assert success
        
        db_config = test_config.get_database("test_db")
        assert db_config is not None
        assert db_config["prefix"] == ""
