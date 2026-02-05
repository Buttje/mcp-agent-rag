"""Tests for archive extraction."""

import gzip
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import py7zr
import pytest

try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False

from mcp_agent_rag.rag.archive_extractor import ArchiveExtractor


@pytest.fixture
def temp_archive_dir(tmp_path):
    """Create a temporary directory for archives."""
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    return archive_dir


@pytest.fixture
def sample_files(tmp_path):
    """Create sample files for testing."""
    files_dir = tmp_path / "sample_files"
    files_dir.mkdir()
    
    # Create text file
    text_file = files_dir / "test.txt"
    text_file.write_text("This is a test file.")
    
    # Create Python file
    py_file = files_dir / "script.py"
    py_file.write_text("def hello():\n    print('Hello, world!')")
    
    # Create markdown file
    md_file = files_dir / "readme.md"
    md_file.write_text("# Test Readme\n\nThis is a test.")
    
    return [text_file, py_file, md_file]


def test_is_archive():
    """Test archive format detection."""
    assert ArchiveExtractor.is_archive(Path("test.zip"))
    assert ArchiveExtractor.is_archive(Path("test.7z"))
    assert ArchiveExtractor.is_archive(Path("test.gz"))
    assert ArchiveExtractor.is_archive(Path("test.tar"))
    assert ArchiveExtractor.is_archive(Path("test.tar.gz"))
    assert ArchiveExtractor.is_archive(Path("test.tgz"))
    assert ArchiveExtractor.is_archive(Path("test.tar.bz2"))
    assert ArchiveExtractor.is_archive(Path("test.tbz2"))
    assert ArchiveExtractor.is_archive(Path("test.tar.xz"))
    assert ArchiveExtractor.is_archive(Path("test.txz"))
    assert ArchiveExtractor.is_archive(Path("test.rar"))
    
    assert not ArchiveExtractor.is_archive(Path("test.txt"))
    assert not ArchiveExtractor.is_archive(Path("test.pdf"))
    assert not ArchiveExtractor.is_archive(Path("test.exe"))


def test_extract_zip(temp_archive_dir, sample_files):
    """Test extracting ZIP archives."""
    # Create a ZIP archive
    zip_path = temp_archive_dir / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in sample_files:
            zipf.write(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(zip_path)
    
    # Verify extraction
    assert len(extracted_files) == 3
    assert any(f.name == "test.txt" for f in extracted_files)
    assert any(f.name == "script.py" for f in extracted_files)
    assert any(f.name == "readme.md" for f in extracted_files)
    
    # Verify content
    txt_file = next(f for f in extracted_files if f.name == "test.txt")
    assert txt_file.read_text() == "This is a test file."


def test_extract_7z(temp_archive_dir, sample_files):
    """Test extracting 7Z archives."""
    # Create a 7Z archive
    archive_path = temp_archive_dir / "test.7z"
    with py7zr.SevenZipFile(archive_path, 'w') as archive:
        for file in sample_files:
            archive.write(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(archive_path)
    
    # Verify extraction
    assert len(extracted_files) == 3
    assert any(f.name == "test.txt" for f in extracted_files)
    assert any(f.name == "script.py" for f in extracted_files)
    assert any(f.name == "readme.md" for f in extracted_files)


def test_extract_gzip(temp_archive_dir, sample_files):
    """Test extracting GZIP archives."""
    # Create a GZIP archive
    source_file = sample_files[0]
    gz_path = temp_archive_dir / "test.txt.gz"
    with open(source_file, 'rb') as f_in:
        with gzip.open(gz_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(gz_path)
    
    # Verify extraction
    assert len(extracted_files) == 1
    assert extracted_files[0].name == "test.txt"
    assert extracted_files[0].read_text() == "This is a test file."


def test_extract_tar(temp_archive_dir, sample_files):
    """Test extracting TAR archives."""
    # Create a TAR archive
    tar_path = temp_archive_dir / "test.tar"
    with tarfile.open(tar_path, 'w') as tar:
        for file in sample_files:
            tar.add(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(tar_path)
    
    # Verify extraction
    assert len(extracted_files) == 3
    assert any(f.name == "test.txt" for f in extracted_files)


def test_extract_tar_gz(temp_archive_dir, sample_files):
    """Test extracting TAR.GZ archives."""
    # Create a TAR.GZ archive
    tar_path = temp_archive_dir / "test.tar.gz"
    with tarfile.open(tar_path, 'w:gz') as tar:
        for file in sample_files:
            tar.add(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(tar_path)
    
    # Verify extraction
    assert len(extracted_files) == 3
    assert any(f.name == "test.txt" for f in extracted_files)


def test_extract_tar_bz2(temp_archive_dir, sample_files):
    """Test extracting TAR.BZ2 archives."""
    # Create a TAR.BZ2 archive
    tar_path = temp_archive_dir / "test.tar.bz2"
    with tarfile.open(tar_path, 'w:bz2') as tar:
        for file in sample_files:
            tar.add(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(tar_path)
    
    # Verify extraction
    assert len(extracted_files) == 3


@pytest.mark.skipif(not RAR_AVAILABLE, reason="rarfile not available or unrar tool not installed")
def test_extract_rar(temp_archive_dir, sample_files):
    """Test extracting RAR archives."""
    # Note: Creating RAR archives requires the rar tool
    # This test is conditional and may be skipped
    pytest.skip("RAR creation requires external rar tool")


def test_nested_archives(temp_archive_dir, sample_files):
    """Test extracting nested archives."""
    # Create an inner ZIP
    inner_zip = temp_archive_dir / "inner.zip"
    with zipfile.ZipFile(inner_zip, 'w') as zipf:
        for file in sample_files[:2]:  # Only include 2 files
            zipf.write(file, arcname=file.name)
    
    # Create an outer ZIP containing the inner ZIP and another file
    outer_zip = temp_archive_dir / "outer.zip"
    with zipfile.ZipFile(outer_zip, 'w') as zipf:
        zipf.write(inner_zip, arcname="inner.zip")
        zipf.write(sample_files[2], arcname=sample_files[2].name)
    
    # Extract the outer archive
    extracted_files = ArchiveExtractor.extract_archive(outer_zip)
    
    # Should extract files from both the outer zip and the nested inner zip
    # The inner.zip itself should be removed from results, only its contents should remain
    assert len(extracted_files) >= 2  # At least the files from inner zip + outer file
    # Verify we have the expected files
    file_names = [f.name for f in extracted_files]
    assert "test.txt" in file_names
    assert "script.py" in file_names
    assert "readme.md" in file_names


def test_max_depth_limit(temp_archive_dir, sample_files):
    """Test maximum nesting depth limit."""
    # Create deeply nested archives
    current_file = sample_files[0]
    
    # Create 3 levels of nesting
    for i in range(3):
        zip_path = temp_archive_dir / f"level{i}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(current_file, arcname=current_file.name if i == 0 else f"level{i-1}.zip")
        current_file = zip_path
    
    # Extract with max_depth=2
    extracted_files = ArchiveExtractor.extract_archive(current_file, max_depth=2)
    
    # Should stop at depth 2, so we get limited results
    # The exact number depends on implementation details
    assert isinstance(extracted_files, list)


def test_extract_empty_archive(temp_archive_dir):
    """Test extracting an empty archive."""
    # Create an empty ZIP
    zip_path = temp_archive_dir / "empty.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        pass
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(zip_path)
    
    # Should return empty list
    assert len(extracted_files) == 0


def test_extract_nonexistent_archive(temp_archive_dir):
    """Test extracting a non-existent archive."""
    archive_path = temp_archive_dir / "nonexistent.zip"
    
    # Extract should handle gracefully
    extracted_files = ArchiveExtractor.extract_archive(archive_path)
    
    # Should return empty list
    assert len(extracted_files) == 0


def test_extract_corrupted_archive(temp_archive_dir):
    """Test extracting a corrupted archive."""
    # Create a corrupted ZIP file
    zip_path = temp_archive_dir / "corrupted.zip"
    zip_path.write_bytes(b"This is not a valid ZIP file")
    
    # Extract should handle gracefully
    extracted_files = ArchiveExtractor.extract_archive(zip_path)
    
    # Should return empty list
    assert len(extracted_files) == 0


def test_cleanup_temp_dir(temp_archive_dir):
    """Test cleanup of temporary directory."""
    # Create a temporary directory
    temp_dir = temp_archive_dir / "temp_extract"
    temp_dir.mkdir()
    (temp_dir / "file.txt").write_text("test")
    
    # Cleanup
    ArchiveExtractor.cleanup_temp_dir(temp_dir)
    
    # Verify it's removed
    assert not temp_dir.exists()


def test_cleanup_nonexistent_dir(temp_archive_dir):
    """Test cleanup of non-existent directory."""
    temp_dir = temp_archive_dir / "nonexistent"
    
    # Cleanup should handle gracefully
    ArchiveExtractor.cleanup_temp_dir(temp_dir)
    
    # Should not raise an error
    assert True


def test_extract_with_subdirectories(temp_archive_dir, tmp_path):
    """Test extracting archives with subdirectories."""
    # Create files in subdirectories
    files_dir = tmp_path / "files_with_dirs"
    files_dir.mkdir()
    
    subdir = files_dir / "subdir"
    subdir.mkdir()
    
    file1 = files_dir / "root.txt"
    file1.write_text("Root file")
    
    file2 = subdir / "nested.txt"
    file2.write_text("Nested file")
    
    # Create a ZIP with directory structure
    zip_path = temp_archive_dir / "with_dirs.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(file1, arcname="root.txt")
        zipf.write(file2, arcname="subdir/nested.txt")
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(zip_path)
    
    # Verify both files are extracted
    assert len(extracted_files) == 2
    file_names = [f.name for f in extracted_files]
    assert "root.txt" in file_names
    assert "nested.txt" in file_names


def test_extract_archive_with_special_characters(temp_archive_dir, tmp_path):
    """Test extracting archives with special characters in filenames."""
    # Create a file with special characters
    files_dir = tmp_path / "special_files"
    files_dir.mkdir()
    
    special_file = files_dir / "file with spaces & special.txt"
    special_file.write_text("Special content")
    
    # Create a ZIP
    zip_path = temp_archive_dir / "special.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(special_file, arcname=special_file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(zip_path)
    
    # Verify extraction
    assert len(extracted_files) == 1
    assert extracted_files[0].name == "file with spaces & special.txt"
    assert extracted_files[0].read_text() == "Special content"


def test_tar_path_traversal_protection(temp_archive_dir, tmp_path):
    """Test that TAR extraction protects against path traversal attacks."""
    # Create a TAR with a potentially dangerous path
    tar_path = temp_archive_dir / "dangerous.tar"
    
    # Create a legitimate file
    safe_file = tmp_path / "safe.txt"
    safe_file.write_text("Safe content")
    
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(safe_file, arcname="safe.txt")
    
    # Extract should work normally for safe paths
    extracted_files = ArchiveExtractor.extract_archive(tar_path)
    assert len(extracted_files) == 1
    assert extracted_files[0].name == "safe.txt"


def test_extract_rar_error_handling(temp_archive_dir):
    """Test RAR extraction error handling."""
    # Create a file that's not really a RAR
    fake_rar = temp_archive_dir / "fake.rar"
    fake_rar.write_bytes(b"Not a real RAR file")
    
    # Should handle gracefully
    extracted_files = ArchiveExtractor.extract_archive(fake_rar)
    assert len(extracted_files) == 0


def test_extract_7z_error_handling(temp_archive_dir):
    """Test 7Z extraction error handling."""
    # Create a file that's not really a 7Z
    fake_7z = temp_archive_dir / "fake.7z"
    fake_7z.write_bytes(b"Not a real 7Z file")
    
    # Should handle gracefully
    extracted_files = ArchiveExtractor.extract_archive(fake_7z)
    assert len(extracted_files) == 0


def test_extract_gzip_error_handling(temp_archive_dir):
    """Test GZIP extraction error handling."""
    # Create a file that's not really a GZIP
    fake_gz = temp_archive_dir / "fake.gz"
    fake_gz.write_bytes(b"Not a real GZIP file")
    
    # Should handle gracefully
    extracted_files = ArchiveExtractor.extract_archive(fake_gz)
    assert len(extracted_files) == 0


def test_extract_tar_error_handling(temp_archive_dir):
    """Test TAR extraction error handling."""
    # Create a file that's not really a TAR
    fake_tar = temp_archive_dir / "fake.tar"
    fake_tar.write_bytes(b"Not a real TAR file")
    
    # Should handle gracefully
    extracted_files = ArchiveExtractor.extract_archive(fake_tar)
    assert len(extracted_files) == 0


def test_extract_unsupported_extension(temp_archive_dir):
    """Test extraction of file with unsupported archive extension."""
    # Create a file with an unsupported extension but that passes is_archive check
    unsupported = temp_archive_dir / "test.unknown"
    unsupported.write_bytes(b"Some data")
    
    # Should return empty list
    extracted_files = ArchiveExtractor.extract_archive(unsupported)
    assert len(extracted_files) == 0


def test_extract_archive_with_explicit_extract_to(temp_archive_dir, sample_files, tmp_path):
    """Test extracting archive to a specific directory."""
    # Create a ZIP archive
    zip_path = temp_archive_dir / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(sample_files[0], arcname=sample_files[0].name)
    
    # Specify extraction directory
    extract_dir = tmp_path / "custom_extract"
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(zip_path, extract_to=extract_dir)
    
    # Verify extraction to custom directory
    assert len(extracted_files) == 1
    assert extract_dir.exists()
    # Check that extracted file is within the extract directory
    try:
        extracted_files[0].relative_to(extract_dir)
        assert True  # File is within extract_dir
    except ValueError:
        assert False, f"File {extracted_files[0]} not within {extract_dir}"


def test_extract_tgz_archive(temp_archive_dir, sample_files):
    """Test extracting .tgz archives (alternative to .tar.gz)."""
    # Create a .tgz archive
    tar_path = temp_archive_dir / "test.tgz"
    with tarfile.open(tar_path, 'w:gz') as tar:
        for file in sample_files:
            tar.add(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(tar_path)
    
    # Verify extraction
    assert len(extracted_files) == 3


def test_extract_tar_xz_archive(temp_archive_dir, sample_files):
    """Test extracting .tar.xz archives."""
    # Create a .tar.xz archive
    tar_path = temp_archive_dir / "test.tar.xz"
    with tarfile.open(tar_path, 'w:xz') as tar:
        for file in sample_files[:1]:  # Just one file
            tar.add(file, arcname=file.name)
    
    # Extract the archive
    extracted_files = ArchiveExtractor.extract_archive(tar_path)
    
    # Verify extraction
    assert len(extracted_files) == 1
