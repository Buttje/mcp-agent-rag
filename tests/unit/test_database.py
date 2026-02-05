"""Tests for database manager."""

from unittest.mock import Mock, patch

import pytest

from mcp_agent_rag.database import DatabaseManager


@pytest.fixture
def db_manager(test_config):
    """Create database manager."""
    with patch("mcp_agent_rag.database.OllamaEmbedder"):
        return DatabaseManager(test_config)


def test_create_database(db_manager, test_config):
    """Test creating database."""
    success = db_manager.create_database("testdb", "Test description")
    assert success is True
    assert test_config.database_exists("testdb")

    db_info = test_config.get_database("testdb")
    assert db_info["description"] == "Test description"


def test_create_duplicate_database(db_manager, test_config):
    """Test creating duplicate database."""
    db_manager.create_database("testdb")
    success = db_manager.create_database("testdb")
    assert success is False


def test_list_databases(db_manager, test_config):
    """Test listing databases."""
    db_manager.create_database("db1", "First")
    db_manager.create_database("db2", "Second")

    databases = db_manager.list_databases()
    assert len(databases) == 2
    assert "db1" in databases
    assert "db2" in databases


def test_load_database(db_manager, test_config, temp_dir):
    """Test loading database."""
    # Create database first
    db_path = temp_dir / "testdb"
    db_path.mkdir(parents=True)
    test_config.add_database("testdb", str(db_path))
    test_config.save()

    # Load it
    db = db_manager.load_database("testdb")
    assert db is not None
    assert "testdb" in db_manager.databases


def test_load_nonexistent_database(db_manager):
    """Test loading non-existent database."""
    db = db_manager.load_database("nonexistent")
    assert db is None


def test_load_multiple_databases(db_manager, test_config, temp_dir):
    """Test loading multiple databases."""
    # Create databases
    for name in ["db1", "db2"]:
        db_path = temp_dir / name
        db_path.mkdir(parents=True)
        test_config.add_database(name, str(db_path))
    test_config.save()

    # Load them
    loaded = db_manager.load_multiple_databases(["db1", "db2"])
    assert len(loaded) == 2
    assert "db1" in loaded
    assert "db2" in loaded


def test_format_size(db_manager):
    """Test file size formatting."""
    assert db_manager._format_size(0) == "0.0 B"
    assert db_manager._format_size(512) == "512.0 B"
    assert db_manager._format_size(1024) == "1.0 KB"
    assert db_manager._format_size(1536) == "1.5 KB"
    assert db_manager._format_size(1048576) == "1.0 MB"
    assert db_manager._format_size(1073741824) == "1.0 GB"
    assert db_manager._format_size(1099511627776) == "1.0 TB"


def test_export_single_database(db_manager, test_config, temp_dir):
    """Test exporting a single database."""
    import json
    import zipfile
    from pathlib import Path
    
    # Create a database with files
    db_path = temp_dir / "databases" / "testdb"
    db_path.mkdir(parents=True)
    
    # Create dummy index and metadata files
    (db_path / "index.faiss").write_bytes(b"dummy_faiss_data")
    (db_path / "metadata.pkl").write_bytes(b"dummy_metadata")
    
    test_config.add_database(
        "testdb",
        str(db_path),
        description="Test database",
        doc_count=5,
        prefix="T1"
    )
    test_config.data["databases"]["testdb"]["last_updated"] = "2024-01-01T00:00:00"
    test_config.save()
    
    # Export database
    export_path = temp_dir / "export.zip"
    success = db_manager.export_databases(["testdb"], str(export_path))
    
    assert success is True
    assert export_path.exists()
    
    # Verify ZIP contents
    with zipfile.ZipFile(export_path, 'r') as zipf:
        namelist = zipf.namelist()
        assert "manifest.json" in namelist
        assert "testdb/index.faiss" in namelist
        assert "testdb/metadata.pkl" in namelist
        
        # Verify manifest
        manifest_data = zipf.read("manifest.json")
        manifest = json.loads(manifest_data)
        
        assert manifest["version"] == "1.0"
        assert len(manifest["databases"]) == 1
        assert manifest["databases"][0]["name"] == "testdb"
        assert manifest["databases"][0]["description"] == "Test database"
        assert manifest["databases"][0]["doc_count"] == 5
        assert manifest["databases"][0]["prefix"] == "T1"


def test_export_multiple_databases(db_manager, test_config, temp_dir):
    """Test exporting multiple databases."""
    import zipfile
    
    # Create two databases
    for name in ["db1", "db2"]:
        db_path = temp_dir / "databases" / name
        db_path.mkdir(parents=True)
        (db_path / "index.faiss").write_bytes(b"dummy_data")
        (db_path / "metadata.pkl").write_bytes(b"dummy_metadata")
        test_config.add_database(name, str(db_path), description=f"Database {name}")
    test_config.save()
    
    # Export both databases
    export_path = temp_dir / "export.zip"
    success = db_manager.export_databases(["db1", "db2"], str(export_path))
    
    assert success is True
    
    # Verify ZIP contains both databases
    with zipfile.ZipFile(export_path, 'r') as zipf:
        namelist = zipf.namelist()
        assert "db1/index.faiss" in namelist
        assert "db1/metadata.pkl" in namelist
        assert "db2/index.faiss" in namelist
        assert "db2/metadata.pkl" in namelist


def test_export_nonexistent_database(db_manager, temp_dir):
    """Test exporting a non-existent database."""
    export_path = temp_dir / "export.zip"
    success = db_manager.export_databases(["nonexistent"], str(export_path))
    
    assert success is False
    assert not export_path.exists()


def test_export_database_missing_files(db_manager, test_config, temp_dir):
    """Test exporting a database with missing files."""
    # Create database config but no actual files
    db_path = temp_dir / "databases" / "testdb"
    db_path.mkdir(parents=True)
    
    test_config.add_database("testdb", str(db_path))
    test_config.save()
    
    # Try to export
    export_path = temp_dir / "export.zip"
    success = db_manager.export_databases(["testdb"], str(export_path))
    
    assert success is False


def test_import_single_database(db_manager, test_config, temp_dir):
    """Test importing a single database."""
    import json
    import zipfile
    
    # Create a ZIP file with database
    export_path = temp_dir / "export.zip"
    
    with zipfile.ZipFile(export_path, 'w') as zipf:
        manifest = {
            "version": "1.0",
            "export_date": "2024-01-01T00:00:00",
            "databases": [{
                "name": "importdb",
                "description": "Imported database",
                "doc_count": 10,
                "last_updated": "2024-01-01T00:00:00",
                "prefix": "I1"
            }]
        }
        zipf.writestr("manifest.json", json.dumps(manifest))
        zipf.writestr("importdb/index.faiss", b"dummy_faiss_data")
        zipf.writestr("importdb/metadata.pkl", b"dummy_metadata")
    
    # Import database
    results = db_manager.import_databases(str(export_path))
    
    assert len(results) == 1
    assert results["importdb"] is True
    assert test_config.database_exists("importdb")
    
    db_info = test_config.get_database("importdb")
    assert db_info["description"] == "Imported database"
    assert db_info["doc_count"] == 10
    assert db_info["prefix"] == "I1"


def test_import_multiple_databases(db_manager, test_config, temp_dir):
    """Test importing multiple databases."""
    import json
    import zipfile
    
    # Create ZIP with multiple databases
    export_path = temp_dir / "export.zip"
    
    with zipfile.ZipFile(export_path, 'w') as zipf:
        manifest = {
            "version": "1.0",
            "export_date": "2024-01-01T00:00:00",
            "databases": [
                {"name": "import1", "description": "First", "doc_count": 5, "last_updated": None, "prefix": ""},
                {"name": "import2", "description": "Second", "doc_count": 8, "last_updated": None, "prefix": ""}
            ]
        }
        zipf.writestr("manifest.json", json.dumps(manifest))
        
        for name in ["import1", "import2"]:
            zipf.writestr(f"{name}/index.faiss", b"dummy_data")
            zipf.writestr(f"{name}/metadata.pkl", b"dummy_metadata")
    
    # Import databases
    results = db_manager.import_databases(str(export_path))
    
    assert len(results) == 2
    assert results["import1"] is True
    assert results["import2"] is True
    assert test_config.database_exists("import1")
    assert test_config.database_exists("import2")


def test_import_existing_database_no_overwrite(db_manager, test_config, temp_dir):
    """Test importing over existing database without overwrite flag."""
    import json
    import zipfile
    
    # Create existing database
    db_path = temp_dir / "databases" / "existing"
    db_path.mkdir(parents=True)
    test_config.add_database("existing", str(db_path), description="Original")
    test_config.save()
    
    # Create import ZIP
    export_path = temp_dir / "export.zip"
    with zipfile.ZipFile(export_path, 'w') as zipf:
        manifest = {
            "version": "1.0",
            "export_date": "2024-01-01T00:00:00",
            "databases": [{
                "name": "existing",
                "description": "Updated",
                "doc_count": 5,
                "last_updated": None,
                "prefix": ""
            }]
        }
        zipf.writestr("manifest.json", json.dumps(manifest))
        zipf.writestr("existing/index.faiss", b"new_data")
        zipf.writestr("existing/metadata.pkl", b"new_metadata")
    
    # Import without overwrite
    results = db_manager.import_databases(str(export_path), overwrite=False)
    
    assert results["existing"] is False
    # Original description should be unchanged
    db_info = test_config.get_database("existing")
    assert db_info["description"] == "Original"


def test_import_existing_database_with_overwrite(db_manager, test_config, temp_dir):
    """Test importing over existing database with overwrite flag."""
    import json
    import zipfile
    
    # Create existing database
    db_path = temp_dir / "databases" / "existing"
    db_path.mkdir(parents=True)
    (db_path / "index.faiss").write_bytes(b"old_data")
    (db_path / "metadata.pkl").write_bytes(b"old_metadata")
    test_config.add_database("existing", str(db_path), description="Original")
    test_config.save()
    
    # Create import ZIP
    export_path = temp_dir / "export.zip"
    with zipfile.ZipFile(export_path, 'w') as zipf:
        manifest = {
            "version": "1.0",
            "export_date": "2024-01-01T00:00:00",
            "databases": [{
                "name": "existing",
                "description": "Updated",
                "doc_count": 10,
                "last_updated": "2024-01-01T00:00:00",
                "prefix": "U1"
            }]
        }
        zipf.writestr("manifest.json", json.dumps(manifest))
        zipf.writestr("existing/index.faiss", b"new_data")
        zipf.writestr("existing/metadata.pkl", b"new_metadata")
    
    # Import with overwrite
    results = db_manager.import_databases(str(export_path), overwrite=True)
    
    assert results["existing"] is True
    # Description should be updated
    db_info = test_config.get_database("existing")
    assert db_info["description"] == "Updated"
    assert db_info["doc_count"] == 10
    assert db_info["prefix"] == "U1"


def test_import_invalid_manifest_version(db_manager, temp_dir):
    """Test importing with invalid manifest version."""
    import json
    import zipfile
    
    export_path = temp_dir / "export.zip"
    with zipfile.ZipFile(export_path, 'w') as zipf:
        manifest = {"version": "2.0", "databases": []}
        zipf.writestr("manifest.json", json.dumps(manifest))
    
    results = db_manager.import_databases(str(export_path))
    
    assert len(results) == 0


def test_import_missing_manifest(db_manager, temp_dir):
    """Test importing with missing manifest."""
    import zipfile
    
    export_path = temp_dir / "export.zip"
    with zipfile.ZipFile(export_path, 'w') as zipf:
        zipf.writestr("dummy.txt", "no manifest here")
    
    results = db_manager.import_databases(str(export_path))
    
    assert len(results) == 0


def test_import_nonexistent_file(db_manager, temp_dir):
    """Test importing from non-existent file."""
    results = db_manager.import_databases(str(temp_dir / "nonexistent.zip"))
    
    assert len(results) == 0
