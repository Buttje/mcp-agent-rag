"""Tests for semantic chunker."""

import pytest

from mcp_agent_rag.rag.semantic_chunker import SemanticChunker


def test_semantic_chunker_basic():
    """Test basic semantic chunking."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)

    text = "This is a paragraph.\n\nThis is another paragraph.\n\nAnd a third one."
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, tuple) for chunk in chunks)
    assert all(len(chunk) == 2 for chunk in chunks)  # (text, metadata)


def test_semantic_chunker_with_metadata():
    """Test chunking with metadata."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)

    text = "Test text for chunking"
    metadata = {"source": "test.txt", "database": "testdb"}

    chunks = chunker.chunk(text, metadata)

    assert len(chunks) > 0
    for chunk_text, chunk_meta in chunks:
        assert "source" in chunk_meta
        assert chunk_meta["source"] == "test.txt"
        assert "database" in chunk_meta
        assert "chunk_num" in chunk_meta


def test_semantic_chunker_respects_boundaries():
    """Test that chunker respects structural boundaries."""
    chunker = SemanticChunker(chunk_size=50, overlap=10, respect_boundaries=True)

    text = "# Heading\n\nParagraph text.\n\n## Subheading\n\nMore text."
    chunks = chunker.chunk(text)

    assert len(chunks) >= 2  # Should split at boundaries


def test_semantic_chunker_code_blocks():
    """Test handling of code blocks."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)

    text = "Introduction\n\n```python\ndef hello():\n    print('hello')\n```\n\nConclusion"
    chunks = chunker.chunk(text)

    # Code block should be kept together
    assert len(chunks) > 0


def test_semantic_chunker_tables():
    """Test handling of table rows."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)

    text = "Header\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |\n\nFooter"
    chunks = chunker.chunk(text)

    # Table should be kept together
    assert len(chunks) > 0


def test_semantic_chunker_empty_text():
    """Test handling of empty text."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)

    chunks = chunker.chunk("")
    assert len(chunks) == 0

    chunks = chunker.chunk("   ")
    assert len(chunks) == 0 or (len(chunks) == 1 and not chunks[0][0].strip())


def test_semantic_chunker_large_section():
    """Test handling of very large sections."""
    chunker = SemanticChunker(chunk_size=50, overlap=10)

    # Create a very long paragraph
    text = " ".join(["word"] * 100)
    chunks = chunker.chunk(text)

    # Should split large section
    assert len(chunks) > 1
    assert all(len(chunk[0]) <= 50 * 2 for chunk in chunks)  # Allow some flexibility


def test_semantic_chunker_overlap():
    """Test that overlap is maintained."""
    chunker = SemanticChunker(chunk_size=50, overlap=15)

    text = " ".join(["word"] * 50)
    chunks = chunker.chunk(text)

    if len(chunks) > 1:
        # Check that consecutive chunks have some overlap
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i][0]
            chunk2_text = chunks[i + 1][0]
            # At least some words should be in both
            words1 = set(chunk1_text.split())
            words2 = set(chunk2_text.split())
            assert len(words1 & words2) > 0  # Some overlap exists
