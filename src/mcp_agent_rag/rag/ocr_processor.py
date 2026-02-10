"""OCR processor with opt-in heuristics and configuration."""

from pathlib import Path
from typing import Optional

from PIL import Image

from mcp_agent_rag.utils import get_logger

logger = get_logger(__name__)

# Global OCR reader instance (lazy loaded)
_ocr_reader = None
_OCR_UNAVAILABLE = object()


class OCRProcessor:
    """OCR processor with intelligent fallback heuristics."""

    # Minimum text length threshold to skip OCR
    MIN_TEXT_THRESHOLD = 100

    # File types that should always use OCR
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(
        self,
        enabled: bool = True,
        min_text_threshold: int = MIN_TEXT_THRESHOLD,
        force_ocr_for_images: bool = True,
    ):
        """Initialize OCR processor.

        Args:
            enabled: Whether OCR is enabled globally
            min_text_threshold: Minimum text length before falling back to OCR
            force_ocr_for_images: Always use OCR for image files
        """
        self.enabled = enabled
        self.min_text_threshold = min_text_threshold
        self.force_ocr_for_images = force_ocr_for_images

    @staticmethod
    def get_ocr_reader():
        """Get or initialize the OCR reader (lazy loading).

        Returns:
            EasyOCR reader instance or None if initialization fails
        """
        global _ocr_reader
        if _ocr_reader is None:
            try:
                import easyocr

                # Check GPU availability
                gpu_available = False
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                except ImportError:
                    pass

                logger.info("Initializing EasyOCR reader (this may take a moment)...")
                if gpu_available:
                    logger.info("GPU detected, using GPU acceleration for OCR")
                    _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                else:
                    logger.info("No GPU detected, using CPU for OCR")
                    _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                logger.error("Image text extraction will be unavailable")
                _ocr_reader = _OCR_UNAVAILABLE
        return _ocr_reader if _ocr_reader is not _OCR_UNAVAILABLE else None

    def should_use_ocr(self, file_path: Path, extracted_text: str = "") -> bool:
        """Determine if OCR should be used for this file.

        Args:
            file_path: Path to file
            extracted_text: Text already extracted (if any)

        Returns:
            True if OCR should be used
        """
        if not self.enabled:
            return False

        suffix = file_path.suffix.lower()

        # Always use OCR for standalone image files if configured
        if suffix in self.IMAGE_EXTENSIONS and self.force_ocr_for_images:
            return True

        # Fallback to OCR if extracted text is insufficient
        if extracted_text and len(extracted_text.strip()) < self.min_text_threshold:
            logger.info(
                f"Extracted text too short ({len(extracted_text)} chars), "
                f"using OCR fallback for {file_path.name}"
            )
            return True

        return False

    def extract_text_from_image(self, image_path: Path) -> Optional[str]:
        """Extract text from image using OCR.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text or None if failed
        """
        if not self.enabled:
            return None

        reader = self.get_ocr_reader()
        if not reader:
            return None

        try:
            # Read and process image
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            # Perform OCR
            result = reader.readtext(image, detail=0, paragraph=True)

            if result:
                return "\n".join(result)
            return None

        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return None

    def extract_text_with_fallback(
        self, file_path: Path, initial_text: str = ""
    ) -> str:
        """Extract text with OCR fallback if needed.

        Args:
            file_path: Path to file
            initial_text: Initially extracted text

        Returns:
            Combined or fallback text
        """
        if not self.should_use_ocr(file_path, initial_text):
            return initial_text

        ocr_text = self.extract_text_from_image(file_path)

        if ocr_text:
            # Combine initial text with OCR text
            if initial_text:
                return f"{initial_text}\n\n[OCR Extracted Text]\n{ocr_text}"
            return ocr_text

        # Return initial text even if OCR failed
        return initial_text

    def get_stats(self) -> dict:
        """Get OCR processor statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "enabled": self.enabled,
            "ocr_available": _ocr_reader is not None
            and _ocr_reader is not _OCR_UNAVAILABLE,
            "min_text_threshold": self.min_text_threshold,
            "force_ocr_for_images": self.force_ocr_for_images,
        }
