"""Tests for document extraction."""

from pathlib import Path

import pypdf
import pytest

from mcp_agent_rag.rag.extractor import DocumentExtractor, find_files_to_process


def test_is_supported():
    """Test file format support detection."""
    assert DocumentExtractor.is_supported(Path("test.txt"))
    assert DocumentExtractor.is_supported(Path("test.py"))
    assert DocumentExtractor.is_supported(Path("test.pdf"))
    assert DocumentExtractor.is_supported(Path("test.docx"))
    assert not DocumentExtractor.is_supported(Path("test.exe"))
    assert not DocumentExtractor.is_supported(Path("test.zip"))


def test_extract_text_file(sample_text_file):
    """Test extracting text from plain text file."""
    text = DocumentExtractor.extract_text(sample_text_file)
    assert text is not None
    assert "sample text file" in text
    assert "multiple lines" in text


def test_extract_python_file(sample_python_file):
    """Test extracting text from Python file."""
    text = DocumentExtractor.extract_text(sample_python_file)
    assert text is not None
    assert "def hello()" in text
    assert "print" in text


def test_extract_unsupported_file(temp_dir):
    """Test extracting from unsupported file."""
    unsupported = temp_dir / "test.exe"
    unsupported.write_bytes(b"\x00\x01\x02")
    text = DocumentExtractor.extract_text(unsupported)
    assert text is None


def test_extract_nonexistent_file():
    """Test extracting from non-existent file."""
    text = DocumentExtractor.extract_text(Path("/nonexistent/file.txt"))
    assert text is None


def test_find_files_single_file(sample_text_file):
    """Test finding single file."""
    files = find_files_to_process(str(sample_text_file))
    assert len(files) == 1
    assert files[0] == sample_text_file


def test_find_files_directory(sample_project_dir):
    """Test finding files in directory."""
    files = find_files_to_process(str(sample_project_dir), recursive=False)
    assert len(files) >= 2  # At least main.py and utils.py


def test_find_files_recursive(sample_project_dir):
    """Test finding files recursively."""
    files = find_files_to_process(str(sample_project_dir), recursive=True)
    assert len(files) >= 3  # main.py, utils.py, subdir/module.py


def test_find_files_glob_pattern(sample_project_dir):
    """Test finding files with glob pattern."""
    files = find_files_to_process(
        str(sample_project_dir),
        recursive=True,
        glob_pattern="*.py"
    )
    assert all(f.suffix == ".py" for f in files)


def test_find_files_respect_gitignore(sample_project_dir):
    """Test respecting .gitignore patterns."""
    # Create a .pyc file
    (sample_project_dir / "test.pyc").write_bytes(b"\x00\x01")

    files = find_files_to_process(
        str(sample_project_dir),
        recursive=True,
        respect_gitignore=True
    )

    # .pyc should be ignored
    assert not any(f.suffix == ".pyc" for f in files)


def test_extract_encrypted_pdf(temp_dir):
    """Test extracting text from encrypted PDF file.
    
    This test verifies that the cryptography package is properly installed
    and that encrypted PDFs can be processed without errors about missing cryptography.
    """
    # Create a simple PDF
    pdf_path = temp_dir / "encrypted_test.pdf"
    writer = pypdf.PdfWriter()
    
    # Add a page with some text
    page = pypdf.PageObject.create_blank_page(width=200, height=200)
    writer.add_page(page)
    
    # Encrypt the PDF with a password
    writer.encrypt(user_password="test123", owner_password="owner123", algorithm="AES-256")
    
    # Write the encrypted PDF
    with open(pdf_path, "wb") as f:
        writer.write(f)
    
    # Try to extract text - this should not raise an error about missing cryptography
    # The extraction will fail with "File has not been decrypted" but that's expected
    # The important thing is that it doesn't fail with "cryptography>=3.1 is required"
    text = DocumentExtractor.extract_text(pdf_path)
    
    # The function should handle encrypted PDFs gracefully
    # When cryptography is installed, pypdf can at least detect the encryption
    # Without cryptography, it would fail with "cryptography>=3.1 is required for AES algorithm"
    # So if we get None here, it means the package handled the encrypted PDF properly
    assert text is None  # Expected: None because the file is encrypted and not decrypted
