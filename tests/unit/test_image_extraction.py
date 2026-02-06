"""Tests for image extraction functionality."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from mcp_agent_rag.rag.extractor import DocumentExtractor


@pytest.fixture
def sample_image_with_text(temp_dir: Path) -> Path:
    """Create a sample PNG image with text."""
    # Create a simple image with text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some text
    text = "Hello World 2024"
    # Use default font
    draw.text((10, 30), text, fill='black')
    
    # Save the image
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_jpeg_image(temp_dir: Path) -> Path:
    """Create a sample JPEG image."""
    img = Image.new('RGB', (300, 200), color='blue')
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "JPEG Test", fill='white')
    
    img_path = temp_dir / "test_image.jpg"
    img.save(img_path, 'JPEG')
    return img_path


@pytest.fixture
def sample_gif_image(temp_dir: Path) -> Path:
    """Create a sample GIF image."""
    img = Image.new('RGB', (200, 150), color='green')
    draw = ImageDraw.Draw(img)
    draw.text((50, 60), "GIF Test", fill='black')
    
    img_path = temp_dir / "test_image.gif"
    img.save(img_path, 'GIF')
    return img_path


@pytest.fixture
def empty_image(temp_dir: Path) -> Path:
    """Create an empty image without text."""
    img = Image.new('RGB', (100, 100), color='red')
    img_path = temp_dir / "empty_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def corrupted_image(temp_dir: Path) -> Path:
    """Create a corrupted image file."""
    img_path = temp_dir / "corrupted.png"
    img_path.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00corrupted data')
    return img_path


def test_is_supported_image_formats():
    """Test that image formats are properly supported."""
    assert DocumentExtractor.is_supported(Path("test.png"))
    assert DocumentExtractor.is_supported(Path("test.jpg"))
    assert DocumentExtractor.is_supported(Path("test.jpeg"))
    assert DocumentExtractor.is_supported(Path("test.gif"))
    assert DocumentExtractor.is_supported(Path("test.bmp"))
    assert DocumentExtractor.is_supported(Path("test.tiff"))
    assert DocumentExtractor.is_supported(Path("test.tif"))
    assert DocumentExtractor.is_supported(Path("test.webp"))


def test_extract_image_with_mock_ocr(sample_image_with_text):
    """Test extracting text from an image using mocked OCR."""
    # Mock the OCR reader
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["Hello World 2024"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(sample_image_with_text)
        
        assert text is not None
        assert "Hello World 2024" in text
        mock_reader.readtext.assert_called_once()


def test_extract_jpeg_with_mock_ocr(sample_jpeg_image):
    """Test extracting text from JPEG image."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["JPEG Test"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(sample_jpeg_image)
        
        assert text is not None
        assert "JPEG Test" in text


def test_extract_gif_with_mock_ocr(sample_gif_image):
    """Test extracting text from GIF image."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["GIF Test"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(sample_gif_image)
        
        assert text is not None
        assert "GIF Test" in text


def test_extract_empty_image(empty_image):
    """Test extracting text from an empty image."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = []  # No text found
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(empty_image)
        
        assert text == ""


def test_extract_image_without_ocr(sample_image_with_text):
    """Test extracting from image when OCR is not available."""
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=None):
        text = DocumentExtractor.extract_text(sample_image_with_text)
        
        # Should return empty string when OCR is not available
        assert text == ""


def test_extract_corrupted_image(corrupted_image):
    """Test handling of corrupted image file."""
    mock_reader = MagicMock()
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(corrupted_image)
        
        # Should handle error gracefully and return empty string
        assert text == ""


def test_extract_text_from_image_bytes():
    """Test extracting text from image bytes."""
    # Create a simple image in memory
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Test", fill='black')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Mock OCR
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["Test"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor._extract_text_from_image_bytes(img_bytes, "test_source")
        
        assert text is not None
        assert "Test" in text


def test_extract_text_from_image_bytes_without_ocr():
    """Test extracting from image bytes when OCR is not available."""
    img_bytes = b'\x89PNG\r\n\x1a\n...'
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=None):
        text = DocumentExtractor._extract_text_from_image_bytes(img_bytes, "test")
        
        assert text == ""


def test_extract_text_from_image_bytes_error():
    """Test error handling when extracting from corrupted image bytes."""
    corrupted_bytes = b'corrupted data'
    
    mock_reader = MagicMock()
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor._extract_text_from_image_bytes(corrupted_bytes, "test")
        
        # Should handle error and return empty string
        assert text == ""


def test_ocr_reader_initialization():
    """Test OCR reader initialization."""
    # Reset global reader
    import mcp_agent_rag.rag.extractor as extractor_module
    extractor_module._ocr_reader = None
    
    # Mock easyocr
    mock_easyocr = MagicMock()
    mock_reader = MagicMock()
    mock_easyocr.Reader.return_value = mock_reader
    
    with patch.dict('sys.modules', {'easyocr': mock_easyocr}):
        reader = DocumentExtractor._get_ocr_reader()
        
        # Should initialize and return reader
        assert reader is not None
        mock_easyocr.Reader.assert_called_once()


def test_ocr_reader_initialization_failure():
    """Test OCR reader initialization failure handling."""
    # Reset global reader
    import mcp_agent_rag.rag.extractor as extractor_module
    extractor_module._ocr_reader = None
    
    # Mock easyocr import to fail
    with patch('mcp_agent_rag.rag.extractor.logger'):
        # Simulate import failure by making the import statement fail
        def mock_import(name, *args, **kwargs):
            if name == 'easyocr':
                raise ImportError("No module named easyocr")
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            reader = DocumentExtractor._get_ocr_reader()
            
            # Should return None on failure
            assert reader is None


def test_ocr_reader_cached():
    """Test that OCR reader is cached after first initialization."""
    # Reset and set a mock reader
    import mcp_agent_rag.rag.extractor as extractor_module
    mock_reader = MagicMock()
    extractor_module._ocr_reader = mock_reader
    
    # Get reader multiple times
    reader1 = DocumentExtractor._get_ocr_reader()
    reader2 = DocumentExtractor._get_ocr_reader()
    
    # Should return same cached instance
    assert reader1 is reader2
    assert reader1 is mock_reader


def test_extract_image_with_multiple_text_blocks(sample_image_with_text):
    """Test extracting image with multiple text blocks."""
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["Line 1", "Line 2", "Line 3"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(sample_image_with_text)
        
        assert "Line 1" in text
        assert "Line 2" in text
        assert "Line 3" in text


def test_image_mode_conversion(temp_dir: Path):
    """Test that images are converted to RGB mode if necessary."""
    # Create an RGBA image
    img = Image.new('RGBA', (200, 100), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "RGBA Test", fill=(0, 0, 0, 255))
    
    img_path = temp_dir / "rgba_image.png"
    img.save(img_path)
    
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["RGBA Test"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(img_path)
        
        assert text is not None
        assert "RGBA Test" in text


def test_extract_bmp_image(temp_dir: Path):
    """Test extracting text from BMP image."""
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "BMP Test", fill='black')
    
    img_path = temp_dir / "test.bmp"
    img.save(img_path, 'BMP')
    
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["BMP Test"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(img_path)
        
        assert "BMP Test" in text


def test_extract_tiff_image(temp_dir: Path):
    """Test extracting text from TIFF image."""
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "TIFF Test", fill='black')
    
    img_path = temp_dir / "test.tiff"
    img.save(img_path, 'TIFF')
    
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["TIFF Test"]
    
    with patch.object(DocumentExtractor, '_get_ocr_reader', return_value=mock_reader):
        text = DocumentExtractor.extract_text(img_path)
        
        assert "TIFF Test" in text
