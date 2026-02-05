"""Archive extraction utilities for compressed files."""

import gzip
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import py7zr
import rarfile

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)


class ArchiveExtractor:
    """Extract files from compressed archives."""

    ARCHIVE_EXTENSIONS = {
        ".zip", ".7z", ".gz", ".tar", ".tar.gz", ".tgz",
        ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".rar"
    }

    @staticmethod
    def is_archive(file_path: Path) -> bool:
        """Check if file is a supported archive format.

        Args:
            file_path: Path to file

        Returns:
            True if file is a supported archive
        """
        path_str = str(file_path).lower()
        # Check for double extensions like .tar.gz
        for ext in ArchiveExtractor.ARCHIVE_EXTENSIONS:
            if path_str.endswith(ext):
                return True
        return False

    @staticmethod
    def extract_archive(
        archive_path: Path,
        extract_to: Path | None = None,
        max_depth: int = 10,
        current_depth: int = 0
    ) -> list[Path]:
        """Extract files from archive, handling nested archives recursively.

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to (creates temp dir if None)
            max_depth: Maximum recursion depth for nested archives
            current_depth: Current recursion depth

        Returns:
            List of extracted file paths (including files from nested archives)
        """
        if current_depth >= max_depth:
            logger.warning(
                f"Maximum nesting depth ({max_depth}) reached for {archive_path}"
            )
            return []

        # Create extraction directory
        if extract_to is None:
            extract_to = Path(tempfile.mkdtemp(prefix="mcp_rag_archive_"))
            logger.debug(f"Created temporary extraction directory: {extract_to}")
        else:
            extract_to.mkdir(parents=True, exist_ok=True)

        extracted_files = []
        archive_str = str(archive_path).lower()

        try:
            # Determine archive type and extract
            if archive_str.endswith('.zip'):
                extracted_files.extend(
                    ArchiveExtractor._extract_zip(archive_path, extract_to)
                )
            elif archive_str.endswith('.7z'):
                extracted_files.extend(
                    ArchiveExtractor._extract_7z(archive_path, extract_to)
                )
            elif archive_str.endswith('.rar'):
                extracted_files.extend(
                    ArchiveExtractor._extract_rar(archive_path, extract_to)
                )
            elif archive_str.endswith('.gz') and not archive_str.endswith(('.tar.gz', '.tgz')):
                extracted_files.extend(
                    ArchiveExtractor._extract_gzip(archive_path, extract_to)
                )
            elif any(
                archive_str.endswith(ext)
                for ext in [
                    '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz'
                ]
            ):
                extracted_files.extend(
                    ArchiveExtractor._extract_tar(archive_path, extract_to)
                )
            else:
                logger.warning(f"Unsupported archive format: {archive_path}")
                return []

            # Process nested archives
            nested_files = []
            for extracted_file in extracted_files[:]:  # Copy list to iterate safely
                if ArchiveExtractor.is_archive(extracted_file):
                    logger.info(
                        f"Found nested archive: {extracted_file.name} "
                        f"(depth {current_depth + 1})"
                    )
                    # Create subdirectory for nested archive
                    nested_dir = extract_to / f"nested_{extracted_file.stem}"
                    nested_extracted = ArchiveExtractor.extract_archive(
                        extracted_file,
                        nested_dir,
                        max_depth=max_depth,
                        current_depth=current_depth + 1
                    )
                    nested_files.extend(nested_extracted)
                    # Remove the nested archive file itself from results
                    extracted_files.remove(extracted_file)

            extracted_files.extend(nested_files)

            logger.info(
                f"Extracted {len(extracted_files)} file(s) from {archive_path.name} "
                f"(depth {current_depth})"
            )

        except Exception as e:
            logger.error(f"Error extracting archive {archive_path}: {e}")
            return []

        return extracted_files

    @staticmethod
    def _extract_zip(archive_path: Path, extract_to: Path) -> list[Path]:
        """Extract ZIP archive.

        Args:
            archive_path: Path to ZIP file
            extract_to: Directory to extract to

        Returns:
            List of extracted file paths
        """
        extracted_files = []
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                for member in zip_ref.namelist():
                    extracted_path = extract_to / member
                    if extracted_path.is_file():
                        extracted_files.append(extracted_path)
            logger.debug(f"Extracted {len(extracted_files)} files from ZIP: {archive_path.name}")
        except Exception as e:
            logger.error(f"Error extracting ZIP {archive_path}: {e}")
        return extracted_files

    @staticmethod
    def _extract_7z(archive_path: Path, extract_to: Path) -> list[Path]:
        """Extract 7Z archive.

        Args:
            archive_path: Path to 7Z file
            extract_to: Directory to extract to

        Returns:
            List of extracted file paths
        """
        extracted_files = []
        try:
            with py7zr.SevenZipFile(archive_path, 'r') as archive:
                archive.extractall(path=extract_to)
                # Get list of extracted files
                for root, _, files in os.walk(extract_to):
                    for file in files:
                        extracted_files.append(Path(root) / file)
            logger.debug(f"Extracted {len(extracted_files)} files from 7Z: {archive_path.name}")
        except Exception as e:
            logger.error(f"Error extracting 7Z {archive_path}: {e}")
        return extracted_files

    @staticmethod
    def _extract_rar(archive_path: Path, extract_to: Path) -> list[Path]:
        """Extract RAR archive.

        Args:
            archive_path: Path to RAR file
            extract_to: Directory to extract to

        Returns:
            List of extracted file paths
        """
        extracted_files = []
        try:
            with rarfile.RarFile(archive_path, 'r') as rar_ref:
                rar_ref.extractall(extract_to)
                for member in rar_ref.namelist():
                    extracted_path = extract_to / member
                    if extracted_path.is_file():
                        extracted_files.append(extracted_path)
            logger.debug(f"Extracted {len(extracted_files)} files from RAR: {archive_path.name}")
        except Exception as e:
            logger.error(f"Error extracting RAR {archive_path}: {e}")
        return extracted_files

    @staticmethod
    def _extract_gzip(archive_path: Path, extract_to: Path) -> list[Path]:
        """Extract GZIP archive.

        Args:
            archive_path: Path to GZIP file
            extract_to: Directory to extract to

        Returns:
            List of extracted file paths
        """
        extracted_files = []
        try:
            # GZIP typically compresses a single file
            # Remove .gz extension to get original filename
            output_name = archive_path.stem
            if not output_name:
                output_name = "extracted_file"
            output_path = extract_to / output_name

            with gzip.open(archive_path, 'rb') as gz_file:
                with open(output_path, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
            extracted_files.append(output_path)
            logger.debug(f"Extracted GZIP: {archive_path.name} -> {output_name}")
        except Exception as e:
            logger.error(f"Error extracting GZIP {archive_path}: {e}")
        return extracted_files

    @staticmethod
    def _extract_tar(archive_path: Path, extract_to: Path) -> list[Path]:
        """Extract TAR archive (including tar.gz, tar.bz2, tar.xz).

        Args:
            archive_path: Path to TAR file
            extract_to: Directory to extract to

        Returns:
            List of extracted file paths
        """
        extracted_files = []
        try:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                # Security check: ensure no path traversal
                safe_members = []
                for member in tar_ref.getmembers():
                    # Resolve the member path and check it's within extract_to
                    member_path = (extract_to / member.name).resolve()
                    extract_to_resolved = extract_to.resolve()
                    is_safe = (
                        extract_to_resolved in member_path.parents
                        or member_path == extract_to_resolved
                    )
                    if is_safe:
                        safe_members.append(member)
                    else:
                        logger.warning(
                            f"Skipping potentially unsafe path in TAR: {member.name}"
                        )

                tar_ref.extractall(extract_to, members=safe_members)
                for member in safe_members:
                    if member.isfile():
                        extracted_files.append(extract_to / member.name)
            logger.debug(f"Extracted {len(extracted_files)} files from TAR: {archive_path.name}")
        except Exception as e:
            logger.error(f"Error extracting TAR {archive_path}: {e}")
        return extracted_files

    @staticmethod
    def cleanup_temp_dir(temp_dir: Path) -> None:
        """Clean up temporary extraction directory.

        Args:
            temp_dir: Path to temporary directory to remove
        """
        try:
            if temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory {temp_dir}: {e}")
