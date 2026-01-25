"""Additional tests for document extractor to improve coverage."""

from pathlib import Path

import pytest

from mcp_agent_rag.rag.extractor import (
    DocumentExtractor,
    _is_ignored,
    _load_gitignore_patterns,
    find_files_to_process,
)


def test_extract_pdf(temp_dir):
    """Test extracting PDF (mock)."""
    # We can't easily create a valid PDF for testing
    # The function should handle errors gracefully
    pdf_file = temp_dir / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nfake pdf")

    # This will likely fail but shouldn't crash
    text = DocumentExtractor.extract_text(pdf_file)
    # Should return None or empty string on error
    assert text is None or text == ""


def test_extract_docx_error(temp_dir):
    """Test handling DOCX extraction error."""
    docx_file = temp_dir / "test.docx"
    docx_file.write_bytes(b"not a valid docx")

    text = DocumentExtractor.extract_text(docx_file)
    assert text is None


def test_extract_html(temp_dir):
    """Test extracting HTML."""
    html_file = temp_dir / "test.html"
    html_file.write_text("""
    <html>
        <head><title>Test</title></head>
        <body>
            <script>alert('test');</script>
            <p>Hello World</p>
        </body>
    </html>
    """)

    text = DocumentExtractor.extract_text(html_file)
    assert text is not None
    assert "Hello World" in text
    assert "alert" not in text  # Script should be removed


def test_load_gitignore_patterns(temp_dir):
    """Test loading .gitignore patterns."""
    gitignore = temp_dir / ".gitignore"
    gitignore.write_text("*.pyc\n__pycache__\n# comment\n\n.env")

    patterns = _load_gitignore_patterns(temp_dir)
    assert "*.pyc" in patterns
    assert "__pycache__" in patterns
    assert ".env" in patterns
    assert "# comment" not in patterns
    assert "" not in patterns


def test_is_ignored():
    """Test gitignore pattern matching."""
    patterns = ["*.pyc", "__pycache__", "*.log"]

    assert _is_ignored(Path("test.pyc"), patterns)
    assert _is_ignored(Path("dir/__pycache__"), patterns)
    assert _is_ignored(Path("app.log"), patterns)
    assert not _is_ignored(Path("test.py"), patterns)


def test_find_files_no_gitignore(temp_dir):
    """Test finding files without gitignore."""
    (temp_dir / "test.py").write_text("print('test')")
    (temp_dir / "test.pyc").write_bytes(b"\x00\x01")

    # Without respecting gitignore, .pyc should not be found (unsupported)
    files = find_files_to_process(str(temp_dir), respect_gitignore=False)
    py_files = [f for f in files if f.suffix == ".py"]
    assert len(py_files) == 1


def test_extract_all_supported_formats(temp_dir):
    """Test that all supported formats are recognized."""
    supported_exts = [
        ".txt", ".md", ".py", ".c", ".cpp", ".java",
        ".docx", ".pdf", ".html"
    ]

    for ext in supported_exts:
        test_file = temp_dir / f"test{ext}"
        assert DocumentExtractor.is_supported(test_file)
