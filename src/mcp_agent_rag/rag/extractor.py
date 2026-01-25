"""Document extraction utilities."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import chardet
import pypdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from odf import text as odf_text
from odf import teletype
from odf.opendocument import load as odf_load
from openpyxl import load_workbook
from pptx import Presentation

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class DocumentExtractor:
    """Extract text from various document formats."""

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".py", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs",
        ".java", ".js", ".ts", ".sh", ".bat", ".ps1", ".s", ".asm",
        ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".pdf", ".html", ".htm"
    }

    @staticmethod
    def is_supported(file_path: Path) -> bool:
        """Check if file format is supported.

        Args:
            file_path: Path to file

        Returns:
            True if supported
        """
        return file_path.suffix.lower() in DocumentExtractor.SUPPORTED_EXTENSIONS

    @staticmethod
    def extract_text(file_path: Path) -> Optional[str]:
        """Extract text from file.

        Args:
            file_path: Path to file

        Returns:
            Extracted text or None if failed
        """
        try:
            suffix = file_path.suffix.lower()

            if suffix in {".txt", ".md", ".py", ".c", ".cpp", ".h", ".hpp", ".cs",
                          ".go", ".rs", ".java", ".js", ".ts", ".sh", ".bat", ".ps1",
                          ".s", ".asm", ".cmake"}:
                return DocumentExtractor._extract_text_file(file_path)
            elif suffix == ".docx":
                return DocumentExtractor._extract_docx(file_path)
            elif suffix == ".xlsx":
                return DocumentExtractor._extract_xlsx(file_path)
            elif suffix == ".pptx":
                return DocumentExtractor._extract_pptx(file_path)
            elif suffix == ".odt":
                return DocumentExtractor._extract_odt(file_path)
            elif suffix == ".ods":
                return DocumentExtractor._extract_ods(file_path)
            elif suffix == ".odp":
                return DocumentExtractor._extract_odp(file_path)
            elif suffix == ".pdf":
                return DocumentExtractor._extract_pdf(file_path)
            elif suffix in {".html", ".htm"}:
                return DocumentExtractor._extract_html(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return None

    @staticmethod
    def _extract_text_file(file_path: Path) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result["encoding"] or "utf-8"
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return ""

    @staticmethod
    def _extract_docx(file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = DocxDocument(str(file_path))
        return "\n".join([para.text for para in doc.paragraphs])

    @staticmethod
    def _extract_xlsx(file_path: Path) -> str:
        """Extract text from XLSX file."""
        wb = load_workbook(str(file_path), read_only=True)
        text_parts = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text_parts.append(f"Sheet: {sheet_name}\n")
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
                if row_text.strip():
                    text_parts.append(row_text)
        return "\n".join(text_parts)

    @staticmethod
    def _extract_pptx(file_path: Path) -> str:
        """Extract text from PPTX file."""
        prs = Presentation(str(file_path))
        text_parts = []
        for i, slide in enumerate(prs.slides):
            text_parts.append(f"Slide {i + 1}:")
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_parts.append(shape.text)
        return "\n".join(text_parts)

    @staticmethod
    def _extract_odt(file_path: Path) -> str:
        """Extract text from ODT file."""
        doc = odf_load(str(file_path))
        paragraphs = doc.getElementsByType(odf_text.P)
        return "\n".join([teletype.extractText(p) for p in paragraphs])

    @staticmethod
    def _extract_ods(file_path: Path) -> str:
        """Extract text from ODS file."""
        doc = odf_load(str(file_path))
        text_parts = []
        from odf.table import Table, TableRow, TableCell
        tables = doc.getElementsByType(Table)
        for table in tables:
            text_parts.append(f"Table: {table.getAttribute('name') or 'Unnamed'}")
            rows = table.getElementsByType(TableRow)
            for row in rows:
                cells = row.getElementsByType(TableCell)
                row_text = "\t".join([teletype.extractText(cell) for cell in cells])
                if row_text.strip():
                    text_parts.append(row_text)
        return "\n".join(text_parts)

    @staticmethod
    def _extract_odp(file_path: Path) -> str:
        """Extract text from ODP file."""
        doc = odf_load(str(file_path))
        from odf.draw import Page as DrawPage
        pages = doc.getElementsByType(DrawPage)
        text_parts = []
        for i, page in enumerate(pages):
            text_parts.append(f"Slide {i + 1}:")
            paragraphs = page.getElementsByType(odf_text.P)
            for p in paragraphs:
                text = teletype.extractText(p)
                if text.strip():
                    text_parts.append(text)
        return "\n".join(text_parts)

    @staticmethod
    def _extract_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            text_parts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"Page {i + 1}:\n{text}")
            return "\n".join(text_parts)

    @staticmethod
    def _extract_html(file_path: Path) -> str:
        """Extract text from HTML file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n", strip=True)


def find_files_to_process(
    path: str,
    recursive: bool = False,
    glob_pattern: Optional[str] = None,
    respect_gitignore: bool = True,
) -> List[Path]:
    """Find files to process.

    Args:
        path: Path to file or directory
        recursive: Whether to recurse into subdirectories
        glob_pattern: Optional glob pattern to match files
        respect_gitignore: Whether to respect .gitignore

    Returns:
        List of file paths to process
    """
    path_obj = Path(path).expanduser().resolve()
    files = []

    # Load gitignore patterns if requested
    gitignore_patterns = []
    if respect_gitignore:
        gitignore_patterns = _load_gitignore_patterns(path_obj if path_obj.is_dir() else path_obj.parent)

    if path_obj.is_file():
        if DocumentExtractor.is_supported(path_obj):
            files.append(path_obj)
    elif path_obj.is_dir():
        if glob_pattern:
            # Use glob pattern
            if recursive:
                files.extend(path_obj.rglob(glob_pattern))
            else:
                files.extend(path_obj.glob(glob_pattern))
        else:
            # Get all supported files
            if recursive:
                for root, dirs, filenames in os.walk(path_obj):
                    root_path = Path(root)
                    # Filter directories based on gitignore
                    if gitignore_patterns:
                        dirs[:] = [d for d in dirs if not _is_ignored(root_path / d, gitignore_patterns)]
                    for filename in filenames:
                        file_path = root_path / filename
                        if DocumentExtractor.is_supported(file_path):
                            if not gitignore_patterns or not _is_ignored(file_path, gitignore_patterns):
                                files.append(file_path)
            else:
                for file_path in path_obj.iterdir():
                    if file_path.is_file() and DocumentExtractor.is_supported(file_path):
                        if not gitignore_patterns or not _is_ignored(file_path, gitignore_patterns):
                            files.append(file_path)

    # Filter out ignored files
    if gitignore_patterns:
        files = [f for f in files if not _is_ignored(f, gitignore_patterns)]

    return sorted(files)


def _load_gitignore_patterns(directory: Path) -> List[str]:
    """Load .gitignore patterns from directory.

    Args:
        directory: Directory to search for .gitignore

    Returns:
        List of patterns
    """
    patterns = []
    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def _is_ignored(path: Path, patterns: List[str]) -> bool:
    """Check if path matches any gitignore pattern.

    Args:
        path: Path to check
        patterns: List of gitignore patterns

    Returns:
        True if path should be ignored
    """
    path_str = str(path)
    for pattern in patterns:
        # Simple pattern matching (simplified version)
        if pattern.startswith("/"):
            pattern = pattern[1:]
        if pattern.endswith("/"):
            if path.is_dir() and path.name == pattern[:-1]:
                return True
        elif "*" in pattern:
            import fnmatch
            if fnmatch.fnmatch(path.name, pattern):
                return True
        elif path.name == pattern:
            return True
        elif pattern in path_str:
            return True
    return False
