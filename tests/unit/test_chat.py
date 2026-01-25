"""Tests for chat module."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.chat import main


def test_chat_main_no_databases(test_config, capsys):
    """Test chat main with no databases."""
    with patch("mcp_agent_rag.chat.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "log_level": "INFO",
            "ollama_host": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
        }.get(key, default)
        mock_config_class.return_value = mock_config
        
        with patch("mcp_agent_rag.chat.Config.get_default_data_dir") as mock_dir:
            mock_dir.return_value = test_config.config_path.parent
            
            with patch("mcp_agent_rag.chat.setup_logger"):
                with patch("mcp_agent_rag.chat.OllamaEmbedder") as mock_embedder_class:
                    mock_embedder = Mock()
                    mock_embedder.check_connection.return_value = True
                    mock_embedder_class.return_value = mock_embedder
                    
                    with patch("mcp_agent_rag.chat.DatabaseManager") as mock_db_manager_class:
                        mock_db_manager = Mock()
                        mock_db_manager.list_databases.return_value = {}
                        mock_db_manager_class.return_value = mock_db_manager
                        
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        
                        assert exc_info.value.code == 1
                        captured = capsys.readouterr()
                        assert "No databases found" in captured.out


def test_chat_main_ollama_connection_error(test_config, capsys):
    """Test chat main with Ollama connection error."""
    with patch("mcp_agent_rag.chat.Config") as mock_config_class:
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default=None: {
            "log_level": "INFO",
            "ollama_host": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
        }.get(key, default)
        mock_config_class.return_value = mock_config
        
        with patch("mcp_agent_rag.chat.Config.get_default_data_dir") as mock_dir:
            mock_dir.return_value = test_config.config_path.parent
            
            with patch("mcp_agent_rag.chat.setup_logger"):
                with patch("mcp_agent_rag.chat.OllamaEmbedder") as mock_embedder_class:
                    mock_embedder = Mock()
                    mock_embedder.check_connection.return_value = False
                    mock_embedder_class.return_value = mock_embedder
                    
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    
                    assert exc_info.value.code == 1
                    captured = capsys.readouterr()
                    assert "Cannot connect to Ollama" in captured.err
