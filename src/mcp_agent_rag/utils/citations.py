"""Structured citations with document locators."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Citation:
    """Structured citation with precise locators.
    
    Supports multiple document types with appropriate locators:
    - PDF: page numbers
    - PPTX: slide numbers
    - XLSX: sheet names and cell ranges
    - General: character offsets
    """

    # Core fields
    source: str  # File path or URL
    text: str  # The actual cited text
    score: float  # Relevance score
    database: str  # Source database

    # Locator fields
    page: Optional[int] = None  # For PDFs
    slide: Optional[int] = None  # For PPTX
    sheet: Optional[str] = None  # For XLSX
    cell_range: Optional[str] = None  # For XLSX (e.g., "A1:D10")
    char_start: Optional[int] = None  # Character offset start
    char_end: Optional[int] = None  # Character offset end
    chunk_num: Optional[int] = None  # Chunk number

    # Additional metadata
    doc_type: Optional[str] = None  # Document type
    section: Optional[str] = None  # Section/heading if available
    metadata: Optional[Dict] = None  # Extra metadata

    def to_dict(self) -> Dict:
        """Convert citation to dictionary.

        Returns:
            Dictionary representation
        """
        result = {
            "source": self.source,
            "text": self.text,
            "score": self.score,
            "database": self.database,
        }

        # Add locator string
        result["locator"] = self.get_locator_string()

        # Add optional fields if present
        if self.page is not None:
            result["page"] = self.page
        if self.slide is not None:
            result["slide"] = self.slide
        if self.sheet is not None:
            result["sheet"] = self.sheet
        if self.cell_range is not None:
            result["cell_range"] = self.cell_range
        if self.char_start is not None:
            result["char_start"] = self.char_start
        if self.char_end is not None:
            result["char_end"] = self.char_end
        if self.chunk_num is not None:
            result["chunk_num"] = self.chunk_num
        if self.doc_type is not None:
            result["doc_type"] = self.doc_type
        if self.section is not None:
            result["section"] = self.section
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def get_locator_string(self) -> str:
        """Get human-readable locator string.

        Returns:
            Locator string (e.g., "pdf:page=12", "xlsx:sheet=Costs!A1:D10")
        """
        # Detect document type from source if not specified
        doc_type = self.doc_type
        if not doc_type:
            suffix = Path(self.source).suffix.lower()
            doc_type = suffix[1:] if suffix else "unknown"

        # Build locator based on available information
        if self.page is not None:
            return f"{doc_type}:page={self.page}"
        elif self.slide is not None:
            return f"{doc_type}:slide={self.slide}"
        elif self.sheet is not None and self.cell_range:
            return f"{doc_type}:sheet={self.sheet}!{self.cell_range}"
        elif self.sheet is not None:
            return f"{doc_type}:sheet={self.sheet}"
        elif self.char_start is not None and self.char_end is not None:
            return f"{doc_type}:chars={self.char_start}-{self.char_end}"
        elif self.chunk_num is not None:
            return f"{doc_type}:chunk={self.chunk_num}"
        else:
            return f"{doc_type}:unknown"

    @staticmethod
    def from_metadata(metadata: Dict, score: float = 0.0) -> "Citation":
        """Create citation from metadata dictionary.

        Args:
            metadata: Metadata from vector database
            score: Relevance score

        Returns:
            Citation instance
        """
        return Citation(
            source=metadata.get("source", ""),
            text=metadata.get("text", ""),
            score=score,
            database=metadata.get("database", ""),
            page=metadata.get("page"),
            slide=metadata.get("slide"),
            sheet=metadata.get("sheet"),
            cell_range=metadata.get("cell_range"),
            char_start=metadata.get("char_start"),
            char_end=metadata.get("char_end"),
            chunk_num=metadata.get("chunk_num"),
            doc_type=metadata.get("doc_type"),
            section=metadata.get("section"),
            metadata={k: v for k, v in metadata.items() 
                     if k not in ["source", "text", "database", "page", "slide", 
                                 "sheet", "cell_range", "char_start", "char_end",
                                 "chunk_num", "doc_type", "section"]},
        )


class CitationBuilder:
    """Helper for building citations from search results."""

    @staticmethod
    def build_citations(
        results: List[tuple], include_text: bool = True, max_text_length: int = 500
    ) -> List[Citation]:
        """Build structured citations from search results.

        Args:
            results: List of (score, metadata) tuples from vector search
            include_text: Whether to include full text in citations
            max_text_length: Maximum length of text snippet

        Returns:
            List of Citation objects
        """
        citations = []

        for distance, metadata in results:
            # Convert distance to score (0-1 range)
            score = 1.0 / (1.0 + distance)

            # Truncate text if needed
            text = metadata.get("text", "")
            if not include_text:
                text = ""
            elif len(text) > max_text_length:
                text = text[:max_text_length] + "..."

            # Create citation
            citation = Citation.from_metadata(metadata, score=score)
            citation.text = text

            citations.append(citation)

        return citations

    @staticmethod
    def deduplicate_citations(
        citations: List[Citation], by_source: bool = True
    ) -> List[Citation]:
        """Remove duplicate citations.

        Args:
            citations: List of citations
            by_source: Deduplicate by source (True) or by text (False)

        Returns:
            Deduplicated list of citations
        """
        seen = set()
        unique = []

        for citation in citations:
            key = citation.source if by_source else citation.text
            if key not in seen:
                seen.add(key)
                unique.append(citation)

        return unique

    @staticmethod
    def sort_citations(
        citations: List[Citation],
        by: str = "score",
        reverse: bool = True,
    ) -> List[Citation]:
        """Sort citations.

        Args:
            citations: List of citations
            by: Sort key (score, source, database)
            reverse: Reverse order (high to low for score)

        Returns:
            Sorted list of citations
        """
        if by == "score":
            return sorted(citations, key=lambda c: c.score, reverse=reverse)
        elif by == "source":
            return sorted(citations, key=lambda c: c.source, reverse=reverse)
        elif by == "database":
            return sorted(citations, key=lambda c: c.database, reverse=reverse)
        else:
            return citations

    @staticmethod
    def group_by_source(citations: List[Citation]) -> Dict[str, List[Citation]]:
        """Group citations by source.

        Args:
            citations: List of citations

        Returns:
            Dictionary mapping source to list of citations
        """
        grouped = {}
        for citation in citations:
            source = citation.source
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(citation)

        return grouped

    @staticmethod
    def format_citations(citations: List[Citation], style: str = "structured") -> str:
        """Format citations for display.

        Args:
            citations: List of citations
            style: Format style (structured, simple, bibtex)

        Returns:
            Formatted citation string
        """
        if style == "structured":
            lines = []
            for i, citation in enumerate(citations, 1):
                lines.append(
                    f"{i}. {citation.source} [{citation.get_locator_string()}] "
                    f"(score: {citation.score:.3f})"
                )
            return "\n".join(lines)

        elif style == "simple":
            sources = [c.source for c in citations]
            return ", ".join(set(sources))

        elif style == "bibtex":
            # Basic BibTeX format (simplified)
            lines = []
            for i, citation in enumerate(citations, 1):
                lines.append(f"@misc{{cite{i},")
                lines.append(f"  title = {{{Path(citation.source).name}}},")
                lines.append(f"  note = {{{citation.get_locator_string()}}}")
                lines.append("}")
            return "\n".join(lines)

        else:
            return str([c.to_dict() for c in citations])
