"""Tests for OCR warning suppression fix."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mcp_agent_rag.rag.extractor import DocumentExtractor


class TestOCRWarningFix:
    """Test cases for the pin_memory warning fix in EasyOCR initialization."""

    def test_ocr_reader_initialization_suppresses_warning(self):
        """Test that OCR reader initialization suppresses pin_memory warning."""
        # Mock easyocr module to simulate the warning
        mock_easyocr = MagicMock()
        
        def mock_reader_init(*args, **kwargs):
            # Simulate the warning that would be raised by PyTorch DataLoader
            warnings.warn(
                "'pin_memory' argument is set as true but no accelerator is found, "
                "then device pinned memory won't be used.",
                UserWarning
            )
            return MagicMock()
        
        mock_easyocr.Reader = mock_reader_init
        
        # Reset the global OCR reader for this test
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = None
        
        try:
            with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
                # Capture warnings
                with warnings.catch_warnings(record=True) as warning_list:
                    warnings.simplefilter("always")
                    
                    # Initialize OCR reader
                    reader = DocumentExtractor._get_ocr_reader()
                    
                    # Verify reader was created
                    assert reader is not None
                    
                    # Verify the pin_memory warning was suppressed
                    pin_memory_warnings = [
                        w for w in warning_list
                        if "pin_memory" in str(w.message)
                    ]
                    assert len(pin_memory_warnings) == 0, \
                        "pin_memory warning should be suppressed"
        finally:
            # Restore original state
            extractor_module._ocr_reader = original_reader

    def test_ocr_reader_initialization_with_gpu_false(self):
        """Test that OCR reader is initialized with gpu=False."""
        mock_easyocr = MagicMock()
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        # Reset the global OCR reader
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = None
        
        try:
            with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
                reader = DocumentExtractor._get_ocr_reader()
                
                # Verify Reader was called with gpu=False
                mock_easyocr.Reader.assert_called_once()
                call_kwargs = mock_easyocr.Reader.call_args[1]
                assert call_kwargs.get('gpu') is False, \
                    "OCR reader should be initialized with gpu=False"
                assert call_kwargs.get('verbose') is False, \
                    "OCR reader should be initialized with verbose=False"
                
                # Verify reader was returned
                assert reader is mock_reader
        finally:
            extractor_module._ocr_reader = original_reader

    def test_ocr_reader_lazy_loading(self):
        """Test that OCR reader is only initialized once (lazy loading)."""
        mock_easyocr = MagicMock()
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = None
        
        try:
            with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
                # First call
                reader1 = DocumentExtractor._get_ocr_reader()
                
                # Second call
                reader2 = DocumentExtractor._get_ocr_reader()
                
                # Verify Reader was only called once
                assert mock_easyocr.Reader.call_count == 1
                
                # Verify same reader instance is returned
                assert reader1 is reader2
        finally:
            extractor_module._ocr_reader = original_reader

    def test_ocr_reader_initialization_failure_handling(self):
        """Test that OCR reader initialization failure is handled gracefully."""
        mock_easyocr = MagicMock()
        mock_easyocr.Reader.side_effect = Exception("Failed to initialize")
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = None
        
        try:
            with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
                # Should return None on failure
                reader = DocumentExtractor._get_ocr_reader()
                assert reader is None
                
                # Subsequent calls should also return None without retrying
                reader2 = DocumentExtractor._get_ocr_reader()
                assert reader2 is None
                
                # Verify Reader was only called once (not retried)
                assert mock_easyocr.Reader.call_count == 1
        finally:
            extractor_module._ocr_reader = original_reader

    def test_ocr_reader_returns_none_when_unavailable(self):
        """Test that OCR reader returns None when marked as unavailable."""
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        
        try:
            # Mark as unavailable
            extractor_module._ocr_reader = extractor_module._OCR_UNAVAILABLE
            
            # Should return None
            reader = DocumentExtractor._get_ocr_reader()
            assert reader is None
        finally:
            extractor_module._ocr_reader = original_reader

    def test_warning_filter_is_specific(self):
        """Test that the warning filter only suppresses pin_memory warnings."""
        mock_easyocr = MagicMock()
        
        def mock_reader_init(*args, **kwargs):
            # Emit both the pin_memory warning and another warning
            warnings.warn(
                "'pin_memory' argument is set as true but no accelerator is found",
                UserWarning
            )
            warnings.warn(
                "This is a different warning that should not be suppressed",
                UserWarning
            )
            return MagicMock()
        
        mock_easyocr.Reader = mock_reader_init
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = None
        
        try:
            with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
                with warnings.catch_warnings(record=True) as warning_list:
                    warnings.simplefilter("always")
                    
                    reader = DocumentExtractor._get_ocr_reader()
                    
                    # Verify pin_memory warning was suppressed
                    pin_memory_warnings = [
                        w for w in warning_list
                        if "pin_memory" in str(w.message)
                    ]
                    assert len(pin_memory_warnings) == 0
                    
                    # Verify other warnings are NOT suppressed
                    other_warnings = [
                        w for w in warning_list
                        if "different warning" in str(w.message)
                    ]
                    assert len(other_warnings) == 1, \
                        "Other warnings should not be suppressed"
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_image_uses_ocr_reader(self, temp_dir):
        """Test that image extraction uses the OCR reader."""
        img_path = temp_dir / "test_image.png"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path)
        
        # Mock the OCR reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["Sample", "Text"]
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            # Extract text from image
            text = DocumentExtractor._extract_image(img_path)
            
            # Verify OCR reader was used
            assert mock_reader.readtext.called
            assert "Sample" in text
            assert "Text" in text
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_image_without_ocr_reader(self, temp_dir):
        """Test that image extraction handles missing OCR reader gracefully."""
        img_path = temp_dir / "test_image.png"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path)
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = extractor_module._OCR_UNAVAILABLE
        
        try:
            # Extract text from image - should return empty string
            text = DocumentExtractor._extract_image(img_path)
            assert text == ""
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_text_from_image_bytes_with_ocr(self):
        """Test extracting text from image bytes."""
        from io import BytesIO
        
        # Create test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Mock OCR reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["Extracted", "Text"]
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            text = DocumentExtractor._extract_text_from_image_bytes(
                img_data, "test_source"
            )
            
            assert mock_reader.readtext.called
            assert "Extracted" in text
            assert "Text" in text
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_text_from_image_bytes_without_ocr(self):
        """Test extracting text from image bytes without OCR reader."""
        from io import BytesIO
        
        # Create test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = extractor_module._OCR_UNAVAILABLE
        
        try:
            text = DocumentExtractor._extract_text_from_image_bytes(
                img_data, "test_source"
            )
            assert text == ""
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_image_converts_mode(self, temp_dir):
        """Test that image extraction converts non-RGB images."""
        img_path = temp_dir / "test_image_rgba.png"
        img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 255))
        img.save(img_path)
        
        # Mock the OCR reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["Converted"]
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            # Extract text from image with non-standard mode
            text = DocumentExtractor._extract_image(img_path)
            
            # Verify OCR reader was used
            assert mock_reader.readtext.called
            assert "Converted" in text
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_image_error_handling(self, temp_dir):
        """Test that image extraction handles errors gracefully."""
        img_path = temp_dir / "invalid_image.png"
        # Write invalid image data
        img_path.write_bytes(b"not an image")
        
        # Mock the OCR reader
        mock_reader = MagicMock()
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            # Extract text should handle error and return empty string
            text = DocumentExtractor._extract_image(img_path)
            assert text == ""
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_text_from_image_bytes_error_handling(self):
        """Test that image bytes extraction handles errors gracefully."""
        # Invalid image data
        img_data = b"not an image"
        
        # Mock OCR reader
        mock_reader = MagicMock()
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            text = DocumentExtractor._extract_text_from_image_bytes(
                img_data, "test_source"
            )
            assert text == ""
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_image_no_text_found(self, temp_dir):
        """Test that image extraction handles no text found."""
        img_path = temp_dir / "blank_image.png"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path)
        
        # Mock the OCR reader to return empty results
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            # Extract text should return empty string
            text = DocumentExtractor._extract_image(img_path)
            assert text == ""
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_text_from_image_bytes_no_text_found(self):
        """Test that image bytes extraction handles no text found."""
        from io import BytesIO
        
        # Create test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        # Mock OCR reader to return empty results
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            text = DocumentExtractor._extract_text_from_image_bytes(
                img_data, "test_source"
            )
            assert text == ""
        finally:
            extractor_module._ocr_reader = original_reader

    def test_extract_text_from_image_bytes_converts_mode(self):
        """Test that image bytes extraction converts non-RGB images."""
        from io import BytesIO
        
        # Create test image with RGBA mode
        img = Image.new('CMYK', (100, 100), color=(0, 0, 0, 0))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        
        # Mock OCR reader
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ["Converted"]
        
        import mcp_agent_rag.rag.extractor as extractor_module
        original_reader = extractor_module._ocr_reader
        extractor_module._ocr_reader = mock_reader
        
        try:
            text = DocumentExtractor._extract_text_from_image_bytes(
                img_data, "test_source"
            )
            assert "Converted" in text
        finally:
            extractor_module._ocr_reader = original_reader
