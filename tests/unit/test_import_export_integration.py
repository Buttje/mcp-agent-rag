"""Integration tests for import/export functionality."""

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_agent_rag.config import Config
from mcp_agent_rag.database import DatabaseManager


@pytest.fixture
def db_manager_with_data(test_config, temp_dir):
    """Create database manager with test data."""
    with patch("mcp_agent_rag.database.OllamaEmbedder"):
        db_manager = DatabaseManager(test_config)
        
        # Create a test database with files
        db_path = temp_dir / "databases" / "testdb"
        db_path.mkdir(parents=True)
        
        # Create realistic FAISS index and metadata files
        (db_path / "index.faiss").write_bytes(b"faiss_index_data_here")
        (db_path / "metadata.pkl").write_bytes(b"metadata_pickle_data")
        
        test_config.add_database(
            name="testdb",
            path=str(db_path),
            description="Test database for integration",
            doc_count=3,
            prefix="TD"
        )
        test_config.data["databases"]["testdb"]["last_updated"] = "2024-01-15T10:30:00"
        test_config.save()
        
        return db_manager


def test_full_export_import_workflow(db_manager_with_data, test_config, temp_dir):
    """Test complete export and import workflow."""
    # Step 1: Export the database
    export_file = temp_dir / "backup.zip"
    success = db_manager_with_data.export_databases(["testdb"], str(export_file))
    assert success is True
    assert export_file.exists()
    
    # Step 2: Verify export file structure
    with zipfile.ZipFile(export_file, 'r') as zipf:
        files = zipf.namelist()
        assert "manifest.json" in files
        assert "testdb/index.faiss" in files
        assert "testdb/metadata.pkl" in files
        
        # Verify manifest content
        manifest = json.loads(zipf.read("manifest.json"))
        assert manifest["version"] == "1.0"
        assert len(manifest["databases"]) == 1
        db_info = manifest["databases"][0]
        assert db_info["name"] == "testdb"
        assert db_info["description"] == "Test database for integration"
        assert db_info["doc_count"] == 3
        assert db_info["prefix"] == "TD"
        assert db_info["last_updated"] == "2024-01-15T10:30:00"
    
    # Step 3: Remove the original database from config (simulate fresh system)
    del test_config.data["databases"]["testdb"]
    test_config.save()
    assert not test_config.database_exists("testdb")
    
    # Step 4: Import the database back
    results = db_manager_with_data.import_databases(str(export_file))
    assert len(results) == 1
    assert results["testdb"] is True
    
    # Step 5: Verify the imported database
    assert test_config.database_exists("testdb")
    imported_db = test_config.get_database("testdb")
    assert imported_db["description"] == "Test database for integration"
    assert imported_db["doc_count"] == 3
    assert imported_db["prefix"] == "TD"
    assert imported_db["last_updated"] == "2024-01-15T10:30:00"
    
    # Step 6: Verify database files exist
    db_path = Path(imported_db["path"])
    assert (db_path / "index.faiss").exists()
    assert (db_path / "metadata.pkl").exists()
    
    # Verify file contents match
    assert (db_path / "index.faiss").read_bytes() == b"faiss_index_data_here"
    assert (db_path / "metadata.pkl").read_bytes() == b"metadata_pickle_data"


def test_export_import_multiple_databases(test_config, temp_dir):
    """Test exporting and importing multiple databases."""
    with patch("mcp_agent_rag.database.OllamaEmbedder"):
        db_manager = DatabaseManager(test_config)
        
        # Create multiple databases
        databases = ["db1", "db2", "db3"]
        for db_name in databases:
            db_path = temp_dir / "databases" / db_name
            db_path.mkdir(parents=True)
            (db_path / "index.faiss").write_bytes(f"{db_name}_index".encode())
            (db_path / "metadata.pkl").write_bytes(f"{db_name}_meta".encode())
            
            test_config.add_database(
                name=db_name,
                path=str(db_path),
                description=f"Database {db_name}",
                doc_count=5,
                prefix=db_name.upper()
            )
        test_config.save()
        
        # Export all databases
        export_file = temp_dir / "multi_backup.zip"
        success = db_manager.export_databases(databases, str(export_file))
        assert success is True
        
        # Clear config
        test_config.data["databases"] = {}
        test_config.save()
        
        # Import all databases
        results = db_manager.import_databases(str(export_file))
        assert len(results) == 3
        assert all(results.values())
        
        # Verify all databases imported correctly
        for db_name in databases:
            assert test_config.database_exists(db_name)
            db_info = test_config.get_database(db_name)
            assert db_info["description"] == f"Database {db_name}"
            assert db_info["doc_count"] == 5


def test_export_import_with_special_characters(test_config, temp_dir):
    """Test export/import with database names containing special characters."""
    with patch("mcp_agent_rag.database.OllamaEmbedder"):
        db_manager = DatabaseManager(test_config)
        
        # Create database with special chars in description
        db_path = temp_dir / "databases" / "special_db"
        db_path.mkdir(parents=True)
        (db_path / "index.faiss").write_bytes(b"data")
        (db_path / "metadata.pkl").write_bytes(b"meta")
        
        special_description = "Test with special chars: éñ中文 & symbols @#$%"
        test_config.add_database(
            name="special_db",
            path=str(db_path),
            description=special_description,
            doc_count=1,
            prefix="SP"
        )
        test_config.save()
        
        # Export
        export_file = temp_dir / "special.zip"
        success = db_manager.export_databases(["special_db"], str(export_file))
        assert success is True
        
        # Remove and import
        del test_config.data["databases"]["special_db"]
        test_config.save()
        
        results = db_manager.import_databases(str(export_file))
        assert results["special_db"] is True
        
        # Verify description preserved
        db_info = test_config.get_database("special_db")
        assert db_info["description"] == special_description


def test_partial_import_failure(test_config, temp_dir):
    """Test import when some databases fail."""
    import zipfile
    
    with patch("mcp_agent_rag.database.OllamaEmbedder"):
        db_manager = DatabaseManager(test_config)
        
        # Create a ZIP with incomplete database (missing metadata file)
        export_file = temp_dir / "partial.zip"
        
        with zipfile.ZipFile(export_file, 'w') as zipf:
            manifest = {
                "version": "1.0",
                "export_date": "2024-01-01T00:00:00",
                "databases": [
                    {
                        "name": "good_db",
                        "description": "Complete database",
                        "doc_count": 5,
                        "last_updated": None,
                        "prefix": ""
                    },
                    {
                        "name": "bad_db",
                        "description": "Incomplete database",
                        "doc_count": 3,
                        "last_updated": None,
                        "prefix": ""
                    }
                ]
            }
            zipf.writestr("manifest.json", json.dumps(manifest))
            
            # Complete database
            zipf.writestr("good_db/index.faiss", b"good_index")
            zipf.writestr("good_db/metadata.pkl", b"good_meta")
            
            # Incomplete database (missing metadata.pkl)
            zipf.writestr("bad_db/index.faiss", b"bad_index")
        
        # Import should handle partial failure gracefully
        results = db_manager.import_databases(str(export_file))
        
        # good_db should succeed, bad_db should fail
        assert results["good_db"] is True
        assert results["bad_db"] is False
        
        # Only good_db should be in config
        assert test_config.database_exists("good_db")
        assert not test_config.database_exists("bad_db")
