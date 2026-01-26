"""Text processing utilities for RAG."""

import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    """Clean extracted text while preserving structure.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\t+", "\t", text)

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    metadata: dict = None,
) -> List[Tuple[str, dict]]:
    """Chunk text into segments with overlap.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        overlap: Number of characters to overlap between chunks
        metadata: Metadata to attach to chunks

    Returns:
        List of (chunk_text, chunk_metadata) tuples
    """
    if not text:
        return []

    if metadata is None:
        metadata = {}

    chunks = []
    start = 0
    chunk_num = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            last_para = chunk.rfind("\n\n")
            if last_para > chunk_size // 2:
                end = start + last_para + 2  # Add 2 for the "\n\n" length
                chunk = text[start:end]
            else:
                # Look for sentence break
                last_period = max(chunk.rfind(". "), chunk.rfind(".\n"))
                if last_period > chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]

        chunk_metadata = {
            **metadata,
            "chunk_num": chunk_num,
            "start": start,
            "end": end,
        }

        chunks.append((chunk.strip(), chunk_metadata))
        chunk_num += 1

        # Move start position with overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks
