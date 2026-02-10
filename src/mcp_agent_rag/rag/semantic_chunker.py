"""Semantic and structure-aware text chunking."""

import re
from typing import Dict, List, Optional, Tuple

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class SemanticChunker:
    """Semantic and structure-aware text chunker.
    
    Preserves document structure like headings, paragraphs, and code blocks.
    Avoids splitting tables and code mid-structure.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        respect_boundaries: bool = True,
    ):
        """Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            respect_boundaries: Whether to respect structural boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_boundaries = respect_boundaries

    def chunk(
        self, text: str, metadata: Optional[Dict] = None
    ) -> List[Tuple[str, Dict]]:
        """Chunk text with semantic awareness.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks

        Returns:
            List of (chunk_text, metadata) tuples
        """
        if not text:
            return []

        metadata = metadata or {}
        chunks = []

        # Detect document structure
        sections = self._split_into_sections(text)

        # Process each section
        current_chunk = ""
        current_start = 0
        chunk_num = 0

        for section in sections:
            section_text = section["text"]
            section_type = section["type"]

            # If section fits in current chunk, add it
            if len(current_chunk) + len(section_text) <= self.chunk_size:
                current_chunk += section_text
            else:
                # Finalize current chunk if not empty
                if current_chunk:
                    chunk_meta = {
                        **metadata,
                        "chunk_num": chunk_num,
                        "char_start": current_start,
                        "char_end": current_start + len(current_chunk),
                    }
                    chunks.append((current_chunk.strip(), chunk_meta))
                    chunk_num += 1

                    # Keep overlap from previous chunk
                    if self.overlap > 0:
                        overlap_text = current_chunk[-self.overlap :]
                        current_chunk = overlap_text + section_text
                        current_start = current_start + len(current_chunk) - self.overlap
                    else:
                        current_chunk = section_text
                        current_start = current_start + len(current_chunk)
                else:
                    current_chunk = section_text

                # Handle very large sections
                if len(current_chunk) > self.chunk_size * 1.5:
                    # Split large section
                    subsections = self._split_large_section(current_chunk)
                    for subsection in subsections[:-1]:
                        chunk_meta = {
                            **metadata,
                            "chunk_num": chunk_num,
                            "char_start": current_start,
                            "char_end": current_start + len(subsection),
                        }
                        chunks.append((subsection.strip(), chunk_meta))
                        chunk_num += 1
                        current_start += len(subsection) - self.overlap

                    current_chunk = subsections[-1] if subsections else ""

        # Add final chunk
        if current_chunk.strip():
            chunk_meta = {
                **metadata,
                "chunk_num": chunk_num,
                "char_start": current_start,
                "char_end": current_start + len(current_chunk),
            }
            chunks.append((current_chunk.strip(), chunk_meta))

        logger.debug(f"Created {len(chunks)} semantic chunks from text")
        return chunks

    def _split_into_sections(self, text: str) -> List[Dict]:
        """Split text into semantic sections.

        Args:
            text: Text to split

        Returns:
            List of section dicts with type and text
        """
        sections = []

        # Patterns for different structures
        patterns = [
            (r"^#{1,6}\s+.+$", "heading"),  # Markdown headings
            (r"^={3,}$", "separator"),  # Separators
            (r"^-{3,}$", "separator"),
            (r"```[\s\S]*?```", "code_block"),  # Code blocks
            (r"^\|.+\|$", "table_row"),  # Table rows
            (r"^\d+\.\s+", "list_item"),  # Numbered lists
            (r"^[-*+]\s+", "list_item"),  # Bullet lists
        ]

        lines = text.split("\n")
        current_section = {"type": "paragraph", "text": ""}
        in_code_block = False
        in_table = False

        for line in lines:
            # Handle code blocks specially
            if line.strip().startswith("```"):
                if in_code_block:
                    # End code block
                    current_section["text"] += line + "\n"
                    sections.append(current_section)
                    current_section = {"type": "paragraph", "text": ""}
                    in_code_block = False
                else:
                    # Start code block
                    if current_section["text"]:
                        sections.append(current_section)
                    current_section = {"type": "code_block", "text": line + "\n"}
                    in_code_block = True
                continue

            if in_code_block:
                current_section["text"] += line + "\n"
                continue

            # Detect line type
            line_type = "paragraph"
            for pattern, ptype in patterns:
                if re.match(pattern, line, re.MULTILINE):
                    line_type = ptype
                    break

            # Handle table rows
            if line_type == "table_row":
                if not in_table:
                    if current_section["text"]:
                        sections.append(current_section)
                    current_section = {"type": "table", "text": line + "\n"}
                    in_table = True
                else:
                    current_section["text"] += line + "\n"
                continue
            elif in_table:
                # End table
                sections.append(current_section)
                current_section = {"type": line_type, "text": line + "\n"}
                in_table = False
                continue

            # Handle structure changes
            if self.respect_boundaries and line_type != current_section["type"]:
                if current_section["text"]:
                    sections.append(current_section)
                current_section = {"type": line_type, "text": line + "\n"}
            else:
                current_section["text"] += line + "\n"

        # Add final section
        if current_section["text"]:
            sections.append(current_section)

        return sections

    def _split_large_section(self, text: str) -> List[str]:
        """Split a large section into smaller chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                    # Add overlap
                    if self.overlap > 0:
                        current = current[-self.overlap :] + sentence + " "
                    else:
                        current = sentence + " "
                else:
                    current = sentence + " "

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text]
