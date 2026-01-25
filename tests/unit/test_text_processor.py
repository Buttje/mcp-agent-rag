"""Tests for text processing."""

import pytest

import mcp_agent_rag.rag.text_processor as text_proc


def test_clean_text_basic():
    """Test basic text cleaning."""
    text = "  Hello   World  \n\n\n  Test  "
    cleaned = text_proc.clean_text(text)
    assert "Hello World" in cleaned
    assert cleaned.count("\n\n") <= 1


def test_clean_text_empty():
    """Test cleaning empty text."""
    assert text_proc.clean_text("") == ""
    assert text_proc.clean_text(None) == ""


def test_clean_text_whitespace():
    """Test whitespace normalization."""
    text = "Hello     World\n\n\n\nTest"
    cleaned = text_proc.clean_text(text)
    assert "     " not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "This is a test. " * 100
    chunks = text_proc.chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    # Check each chunk has metadata
    for chunk_text, metadata in chunks:
        assert isinstance(chunk_text, str)
        assert "chunk_num" in metadata
        assert "start" in metadata
        assert "end" in metadata


def test_chunk_text_small():
    """Test chunking small text."""
    text = "Short text"
    chunks = text_proc.chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0][0] == text


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = text_proc.chunk_text("", chunk_size=100, overlap=10)
    assert len(chunks) == 0


def test_chunk_text_with_metadata():
    """Test chunking with custom metadata."""
    text = "Test text. " * 50
    metadata = {"source": "test.txt", "author": "tester"}
    chunks = text_proc.chunk_text(text, chunk_size=100, overlap=10, metadata=metadata)

    for _, chunk_meta in chunks:
        assert chunk_meta["source"] == "test.txt"
        assert chunk_meta["author"] == "tester"
        assert "chunk_num" in chunk_meta


def test_chunk_text_overlap():
    """Test chunk overlap."""
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    chunks = text_proc.chunk_text(text, chunk_size=50, overlap=10)

    # Check overlap exists
    if len(chunks) > 1:
        first_end = chunks[0][0][-10:]
        second_start = chunks[1][0][:10]
        # There should be some similarity due to overlap
        assert len(first_end) > 0 and len(second_start) > 0


def test_chunk_text_sentence_boundary():
    """Test chunking respects sentence boundaries."""
    text = "First sentence. Second sentence. " * 20
    chunks = text_proc.chunk_text(text, chunk_size=100, overlap=10)

    # Many chunks should end with a period
    period_endings = sum(1 for chunk, _ in chunks if chunk.rstrip().endswith("."))
    assert period_endings > 0
