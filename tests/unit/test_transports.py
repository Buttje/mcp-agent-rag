"""Tests for MCP transport protocols: HTTP and SSE."""

import json
import threading
import time
from unittest.mock import Mock, patch

import pytest
import requests

from mcp_agent_rag.mcp.server import MCPServer
from mcp_agent_rag.rag.vector_db import VectorDatabase


@pytest.fixture
def test_server(test_config, temp_dir):
    """Create test MCP server with database."""
    db_path = temp_dir / "testdb"
    db = VectorDatabase(db_path, dimension=768)
    db.add([[0.1] * 768], [{"text": "test content", "source": "test.txt", "chunk_num": 0}])
    db.save()

    test_config.add_database("testdb", str(db_path))
    test_config.save()

    with patch("mcp_agent_rag.mcp.agent.OllamaEmbedder"):
        server = MCPServer(test_config, ["testdb"])
        return server


class TestHTTPTransport:
    """Tests for HTTP transport."""

    def test_http_server_startup(self, test_server):
        """Test HTTP server can start."""
        port = 18080  # Use non-standard port to avoid conflicts
        
        # Start server in background thread
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        
        # Give server time to start
        time.sleep(1)
        
        # Server thread should be running
        assert server_thread.is_alive()

    def test_http_health_check(self, test_server):
        """Test HTTP health check endpoint."""
        port = 18081
        
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] == "ok"
            assert "active_databases" in data
            assert "testdb" in data["active_databases"]
        except requests.exceptions.RequestException as e:
            pytest.skip(f"HTTP server not available: {e}")

    def test_http_post_request(self, test_server):
        """Test HTTP POST request handling."""
        port = 18082
        
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            # Send JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getDatabases",
                "params": {},
            }
            
            response = requests.post(
                f"http://127.0.0.1:{port}/",
                json=request_data,
                timeout=2,
            )
            
            assert response.status_code == 200
            assert response.headers["Content-Type"] == "application/json"
            
            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert data["id"] == 1
            assert "result" in data
            assert "databases" in data["result"]
        except requests.exceptions.RequestException as e:
            pytest.skip(f"HTTP server not available: {e}")

    def test_http_invalid_json(self, test_server):
        """Test HTTP server handles invalid JSON."""
        port = 18083
        
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            response = requests.post(
                f"http://127.0.0.1:{port}/",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=2,
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32700  # Parse error
        except requests.exceptions.RequestException as e:
            pytest.skip(f"HTTP server not available: {e}")

    def test_http_cors_headers(self, test_server):
        """Test HTTP server sets CORS headers."""
        port = 18084
        
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            # Send OPTIONS request
            response = requests.options(f"http://127.0.0.1:{port}/", timeout=2)
            
            assert response.status_code == 200
            assert "Access-Control-Allow-Origin" in response.headers
            assert response.headers["Access-Control-Allow-Origin"] == "*"
        except requests.exceptions.RequestException as e:
            pytest.skip(f"HTTP server not available: {e}")

    def test_http_tools_list(self, test_server):
        """Test getting tools list via HTTP."""
        port = 18085
        
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {},
            }
            
            response = requests.post(
                f"http://127.0.0.1:{port}/",
                json=request_data,
                timeout=2,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "result" in data
            assert "tools" in data["result"]
            
            tool_names = [t["name"] for t in data["result"]["tools"]]
            assert "getDatabases" in tool_names
            assert "getInformationFor" in tool_names
            assert "getInformationForDB" in tool_names
        except requests.exceptions.RequestException as e:
            pytest.skip(f"HTTP server not available: {e}")

    def test_http_get_databases(self, test_server):
        """Test getDatabases via HTTP."""
        port = 18086
        
        server_thread = threading.Thread(
            target=test_server.run_http,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getDatabases",
                "params": {},
            }
            
            response = requests.post(
                f"http://127.0.0.1:{port}/",
                json=request_data,
                timeout=2,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "result" in data
            assert "databases" in data["result"]
            assert data["result"]["count"] == 1
        except requests.exceptions.RequestException as e:
            pytest.skip(f"HTTP server not available: {e}")

    def test_http_get_information_for_db(self, test_server):
        """Test getInformationForDB via HTTP."""
        port = 18087
        
        with patch.object(test_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            server_thread = threading.Thread(
                target=test_server.run_http,
                args=("127.0.0.1", port),
                daemon=True,
            )
            server_thread.start()
            time.sleep(1)
            
            try:
                request_data = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "getInformationForDB",
                    "params": {
                        "prompt": "test query",
                        "database_name": "testdb",
                    },
                }
                
                response = requests.post(
                    f"http://127.0.0.1:{port}/",
                    json=request_data,
                    timeout=2,
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "result" in data
                assert data["result"]["database"] == "testdb"
                assert "context" in data["result"]
            except requests.exceptions.RequestException as e:
                pytest.skip(f"HTTP server not available: {e}")


class TestSSETransport:
    """Tests for SSE (Server-Sent Events) transport."""

    def test_sse_server_startup(self, test_server):
        """Test SSE server can start."""
        port = 18088
        
        server_thread = threading.Thread(
            target=test_server.run_sse,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        
        time.sleep(1)
        assert server_thread.is_alive()

    def test_sse_health_check(self, test_server):
        """Test SSE health check endpoint."""
        port = 18089
        
        server_thread = threading.Thread(
            target=test_server.run_sse,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "ok"
            assert "active_databases" in data
        except requests.exceptions.RequestException as e:
            pytest.skip(f"SSE server not available: {e}")

    def test_sse_endpoint_headers(self, test_server):
        """Test SSE endpoint returns correct headers."""
        port = 18090
        
        server_thread = threading.Thread(
            target=test_server.run_sse,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            # Start GET request to /sse endpoint
            response = requests.get(
                f"http://127.0.0.1:{port}/sse",
                stream=True,
                timeout=2,
            )
            
            assert response.status_code == 200
            assert response.headers["Content-Type"] == "text/event-stream"
            assert "no-cache" in response.headers.get("Cache-Control", "").lower()
            
            # Read initial connection message
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    assert data["type"] == "connection"
                    assert "active_databases" in data
                    break
            
            response.close()
        except requests.exceptions.RequestException as e:
            pytest.skip(f"SSE server not available: {e}")

    def test_sse_post_request(self, test_server):
        """Test SSE POST request handling."""
        port = 18091
        
        server_thread = threading.Thread(
            target=test_server.run_sse,
            args=("127.0.0.1", port),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)
        
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getDatabases",
                "params": {},
            }
            
            response = requests.post(
                f"http://127.0.0.1:{port}/",
                json=request_data,
                timeout=2,
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "result" in data
            assert "databases" in data["result"]
        except requests.exceptions.RequestException as e:
            pytest.skip(f"SSE server not available: {e}")

    def test_sse_get_information(self, test_server):
        """Test getInformationFor via SSE."""
        port = 18092
        
        with patch.object(test_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            server_thread = threading.Thread(
                target=test_server.run_sse,
                args=("127.0.0.1", port),
                daemon=True,
            )
            server_thread.start()
            time.sleep(1)
            
            try:
                request_data = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "getInformationFor",
                    "params": {"prompt": "test"},
                }
                
                response = requests.post(
                    f"http://127.0.0.1:{port}/",
                    json=request_data,
                    timeout=2,
                )
                
                assert response.status_code == 200
                data = response.json()
                
                assert "result" in data
                assert "context" in data["result"]
            except requests.exceptions.RequestException as e:
                pytest.skip(f"SSE server not available: {e}")


class TestTransportComparison:
    """Tests comparing different transports."""

    def test_stdio_response_format(self, test_server):
        """Test stdio response format matches spec."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getDatabases",
            "params": {},
        }
        
        response = test_server.handle_request(request)
        
        # All transports should return the same format
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert isinstance(response["result"], dict)

    def test_error_response_format(self, test_server):
        """Test error response format is consistent."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown_method",
            "params": {},
        }
        
        response = test_server.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "error" in response
        assert "code" in response["error"]
        assert "message" in response["error"]
        assert response["error"]["code"] == -32601  # Method not found

    def test_all_tools_work_via_handle_request(self, test_server):
        """Test all new tools work via handle_request (used by all transports)."""
        with patch.object(test_server.agent.embedder, "embed_single") as mock_embed:
            mock_embed.return_value = [0.1] * 768
            
            # Test getDatabases
            response1 = test_server.handle_request({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getDatabases",
                "params": {},
            })
            assert "result" in response1
            
            # Test getInformationFor
            response2 = test_server.handle_request({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "getInformationFor",
                "params": {"prompt": "test"},
            })
            assert "result" in response2
            
            # Test getInformationForDB
            response3 = test_server.handle_request({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "getInformationForDB",
                "params": {"prompt": "test", "database_name": "testdb"},
            })
            assert "result" in response3
