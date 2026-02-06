"""Tests for structured citations."""

import pytest

from mcp_agent_rag.utils.citations import Citation, CitationBuilder


def test_citation_basic():
    """Test basic citation creation."""
    citation = Citation(
        source="test.pdf",
        text="Sample text",
        score=0.95,
        database="testdb",
    )

    assert citation.source == "test.pdf"
    assert citation.text == "Sample text"
    assert citation.score == 0.95
    assert citation.database == "testdb"


def test_citation_pdf_locator():
    """Test PDF page locator."""
    citation = Citation(
        source="document.pdf",
        text="Text",
        score=0.9,
        database="db",
        page=12,
    )

    locator = citation.get_locator_string()
    assert locator == "pdf:page=12"


def test_citation_pptx_locator():
    """Test PPTX slide locator."""
    citation = Citation(
        source="presentation.pptx",
        text="Text",
        score=0.9,
        database="db",
        slide=7,
    )

    locator = citation.get_locator_string()
    assert locator == "pptx:slide=7"


def test_citation_xlsx_locator():
    """Test XLSX sheet and cell range locator."""
    citation = Citation(
        source="spreadsheet.xlsx",
        text="Text",
        score=0.9,
        database="db",
        sheet="Costs",
        cell_range="A3:D19",
    )

    locator = citation.get_locator_string()
    assert locator == "xlsx:sheet=Costs!A3:D19"


def test_citation_char_offset_locator():
    """Test character offset locator."""
    citation = Citation(
        source="text.txt",
        text="Text",
        score=0.9,
        database="db",
        char_start=100,
        char_end=200,
    )

    locator = citation.get_locator_string()
    assert locator == "txt:chars=100-200"


def test_citation_to_dict():
    """Test citation to dictionary conversion."""
    citation = Citation(
        source="test.pdf",
        text="Sample text",
        score=0.95,
        database="testdb",
        page=5,
    )

    data = citation.to_dict()

    assert data["source"] == "test.pdf"
    assert data["text"] == "Sample text"
    assert data["score"] == 0.95
    assert data["database"] == "testdb"
    assert data["page"] == 5
    assert "locator" in data


def test_citation_from_metadata():
    """Test creating citation from metadata."""
    metadata = {
        "source": "doc.pdf",
        "text": "Content",
        "database": "db1",
        "page": 10,
        "chunk_num": 5,
    }

    citation = Citation.from_metadata(metadata, score=0.85)

    assert citation.source == "doc.pdf"
    assert citation.text == "Content"
    assert citation.database == "db1"
    assert citation.score == 0.85
    assert citation.page == 10
    assert citation.chunk_num == 5


def test_citation_builder_build():
    """Test building citations from search results."""
    results = [
        (0.1, {"source": "doc1.pdf", "text": "Text 1", "database": "db", "page": 1}),
        (0.2, {"source": "doc2.pdf", "text": "Text 2", "database": "db", "page": 2}),
    ]

    citations = CitationBuilder.build_citations(results)

    assert len(citations) == 2
    assert all(isinstance(c, Citation) for c in citations)
    assert citations[0].source == "doc1.pdf"
    assert citations[1].source == "doc2.pdf"


def test_citation_builder_truncate_text():
    """Test text truncation."""
    results = [
        (
            0.1,
            {
                "source": "doc.pdf",
                "text": "A" * 1000,
                "database": "db",
            },
        )
    ]

    citations = CitationBuilder.build_citations(results, max_text_length=100)

    assert len(citations[0].text) <= 103  # 100 + "..."


def test_citation_builder_deduplicate_by_source():
    """Test deduplication by source."""
    citations = [
        Citation("doc1.pdf", "Text 1", 0.9, "db"),
        Citation("doc1.pdf", "Text 2", 0.8, "db"),
        Citation("doc2.pdf", "Text 3", 0.7, "db"),
    ]

    unique = CitationBuilder.deduplicate_citations(citations, by_source=True)

    assert len(unique) == 2
    assert unique[0].source == "doc1.pdf"
    assert unique[1].source == "doc2.pdf"


def test_citation_builder_sort():
    """Test citation sorting."""
    citations = [
        Citation("doc1.pdf", "Text", 0.5, "db"),
        Citation("doc2.pdf", "Text", 0.9, "db"),
        Citation("doc3.pdf", "Text", 0.7, "db"),
    ]

    sorted_citations = CitationBuilder.sort_citations(citations, by="score")

    assert sorted_citations[0].score == 0.9
    assert sorted_citations[1].score == 0.7
    assert sorted_citations[2].score == 0.5


def test_citation_builder_group_by_source():
    """Test grouping by source."""
    citations = [
        Citation("doc1.pdf", "Text 1", 0.9, "db"),
        Citation("doc1.pdf", "Text 2", 0.8, "db"),
        Citation("doc2.pdf", "Text 3", 0.7, "db"),
    ]

    grouped = CitationBuilder.group_by_source(citations)

    assert len(grouped) == 2
    assert len(grouped["doc1.pdf"]) == 2
    assert len(grouped["doc2.pdf"]) == 1


def test_citation_builder_format_structured():
    """Test structured formatting."""
    citations = [
        Citation("doc1.pdf", "Text", 0.95, "db", page=1),
        Citation("doc2.pdf", "Text", 0.85, "db", slide=5),
    ]

    formatted = CitationBuilder.format_citations(citations, style="structured")

    assert "doc1.pdf" in formatted
    assert "doc2.pdf" in formatted
    assert "page=1" in formatted
    assert "slide=5" in formatted


def test_citation_builder_format_simple():
    """Test simple formatting."""
    citations = [
        Citation("doc1.pdf", "Text", 0.95, "db"),
        Citation("doc2.pdf", "Text", 0.85, "db"),
    ]

    formatted = CitationBuilder.format_citations(citations, style="simple")

    assert "doc1.pdf" in formatted
    assert "doc2.pdf" in formatted
