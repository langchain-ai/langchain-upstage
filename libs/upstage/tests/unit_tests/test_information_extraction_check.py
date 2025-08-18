from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from langchain_upstage.tools.information_extraction_check import create_message


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a simple file with .png extension for testing."""
    image_path = tmp_path / "test_image.png"

    # Simple content - extension is all that matters for validation
    image_path.write_text("fake png content")
    return image_path


class TestCreateMessage:
    """Test create_message function."""

    def test_create_message_with_url_returns_correct_structure(self) -> None:
        """Test that URLs create correct message structure."""
        # Arrange
        url = "https://example.com/image.png"
        supported_extensions = ["png", "jpg"]
        max_file_size = 10 * 1024 * 1024  # 10MB

        # Act
        result = create_message(url, supported_extensions, max_file_size)

        # Assert
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == url

    def test_create_message_with_local_file_returns_base64_structure(
        self, temp_image_file: Path
    ) -> None:
        """Test that local files create base64 message structure."""
        # Arrange
        supported_extensions = ["png", "jpg"]
        max_file_size = 10 * 1024 * 1024  # 10MB

        # Act
        result = create_message(
            str(temp_image_file), supported_extensions, max_file_size
        )

        # Assert
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith(
            "data:application/octet-stream;base64,"
        )

    def test_create_message_with_nonexistent_file_raises_error(self) -> None:
        """Test that nonexistent files raise FileNotFoundError."""
        # Arrange
        supported_extensions = ["png", "jpg"]
        max_file_size = 10 * 1024 * 1024  # 10MB

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            create_message("nonexistent_file.png", supported_extensions, max_file_size)

    @patch("os.path.getsize")
    def test_create_message_with_large_file_raises_error(
        self, mock_getsize: Mock, tmp_path: Path
    ) -> None:
        """Test that files exceeding size limit raise ValueError using mocking."""
        # Arrange
        large_file = tmp_path / "large_image.png"
        large_file.write_text("small content")

        # Mock file size to be larger than limit
        mock_getsize.return_value = 2 * 1024 * 1024  # 2MB

        supported_extensions = ["png", "jpg"]
        max_file_size = 1024 * 1024  # 1MB

        # Act & Assert
        with pytest.raises(ValueError):
            create_message(str(large_file), supported_extensions, max_file_size)

    def test_create_message_with_unsupported_extension_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that unsupported extensions raise ValueError."""
        # Arrange
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("test content")
        supported_extensions = ["png", "jpg"]
        max_file_size = 10 * 1024 * 1024  # 10MB

        # Act & Assert
        with pytest.raises(ValueError):
            create_message(str(unsupported_file), supported_extensions, max_file_size)

    def test_create_message_with_no_extension_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that files without extension raise ValueError."""
        # Arrange
        no_extension_file = tmp_path / "test_image"
        no_extension_file.write_text("test content")
        supported_extensions = ["png", "jpg"]
        max_file_size = 10 * 1024 * 1024  # 10MB

        # Act & Assert
        with pytest.raises(ValueError):
            create_message(str(no_extension_file), supported_extensions, max_file_size)
