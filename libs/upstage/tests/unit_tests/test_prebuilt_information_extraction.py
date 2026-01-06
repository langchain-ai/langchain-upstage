from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from langchain_upstage import UpstagePrebuiltInformationExtraction


@pytest.fixture
def temp_document_file(tmp_path: Path) -> Path:
    """Create a simple file with supported extension for testing."""
    document_path = tmp_path / "test_document.pdf"

    # Simple content - extension is all that matters for validation
    document_path.write_text("fake pdf content")
    return document_path


class TestUpstagePrebuiltInformationExtraction:
    """Test UpstagePrebuiltInformationExtractionTool core functionality."""

    def test_initialization_without_api_key_raises_error(
        self, no_api_key: None
    ) -> None:
        """Test that initialization fails when no API key is provided."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstagePrebuiltInformationExtraction(model="receipt-extraction")

    @patch("langchain_upstage.tools.prebuilt_information_extraction.ChatUpstage")
    def test_invoke_calls_api_with_correct_parameters(
        self, mock_chat_upstage: Mock, temp_document_file: Path
    ) -> None:
        """Test that invoke method calls API with correct parameters."""
        # Arrange
        mock_response = Mock()
        mock_response.content = '{"any": "response", "structure": "doesnt", "matter": true}'
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_upstage.return_value = mock_instance

        tool = UpstagePrebuiltInformationExtraction(model="receipt-extraction")

        # Act
        result = tool.invoke({"filename": str(temp_document_file)})

        # Assert
        assert result == {"any": "response", "structure": "doesnt", "matter": True}
        mock_instance.invoke.assert_called_once()

    def test_invoke_with_unsupported_extension_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that invoke method raises error for unsupported file extensions."""
        # Arrange
        tool = UpstagePrebuiltInformationExtraction(model="receipt-extraction")
        unsupported_file = tmp_path / "document.txt"
        unsupported_file.write_text("test content")

        # Act & Assert
        with pytest.raises(ValueError):
            tool.invoke({"filename": str(unsupported_file)})

    def test_invoke_with_nonexistent_file_raises_error(self) -> None:
        """Test that invoke method raises error for nonexistent files."""
        # Arrange
        tool = UpstagePrebuiltInformationExtraction(model="receipt-extraction")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            tool.invoke({"filename": "nonexistent_file.pdf"})

    @patch("langchain_upstage.tools.prebuilt_information_extraction.ChatUpstage")
    def test_invoke_propagates_api_errors(
        self, mock_chat_upstage: Mock, temp_document_file: Path
    ) -> None:
        """Test that invoke method properly propagates API errors."""
        # Arrange
        api_error = Exception("API service unavailable")
        mock_instance = Mock()
        mock_instance.invoke.side_effect = api_error
        mock_chat_upstage.return_value = mock_instance

        tool = UpstagePrebuiltInformationExtraction(model="receipt-extraction")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            tool.invoke({"filename": str(temp_document_file)})

        assert "Failed to extract information from document" in str(exc_info.value)
