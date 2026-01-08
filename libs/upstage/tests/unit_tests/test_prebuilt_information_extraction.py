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

    @patch("langchain_upstage.tools.prebuilt_information_extraction.httpx.Client")
    def test_invoke_calls_api_with_correct_parameters(
        self, mock_client_class: Mock, temp_document_file: Path, mock_api_key: str
    ) -> None:
        """Test that invoke method calls API with correct parameters."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "any": "response",
            "structure": "doesnt",
            "matter": True,
        }
        mock_response.raise_for_status = Mock()
        mock_client = Mock()
        # Configure context manager behavior
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        tool = UpstagePrebuiltInformationExtraction(
            model="receipt-extraction", api_key=mock_api_key
        )

        # Act
        result = tool.invoke({"filename": str(temp_document_file)})

        # Assert
        assert result == {"any": "response", "structure": "doesnt", "matter": True}
        mock_client.post.assert_called_once()

    def test_invoke_with_unsupported_extension_raises_error(
        self, tmp_path: Path, mock_api_key: str
    ) -> None:
        """Test that invoke method raises error for unsupported file extensions."""
        # Arrange
        tool = UpstagePrebuiltInformationExtraction(
            model="receipt-extraction", api_key=mock_api_key
        )
        unsupported_file = tmp_path / "document.txt"
        unsupported_file.write_text("test content")

        # Act & Assert
        with pytest.raises(ValueError):
            tool.invoke({"filename": str(unsupported_file)})

    def test_invoke_with_nonexistent_file_raises_error(
        self, mock_api_key: str
    ) -> None:
        """Test that invoke method raises error for nonexistent files."""
        # Arrange
        tool = UpstagePrebuiltInformationExtraction(
            model="receipt-extraction", api_key=mock_api_key
        )

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            tool.invoke({"filename": "nonexistent_file.pdf"})

    @patch("langchain_upstage.tools.prebuilt_information_extraction.httpx.Client")
    def test_invoke_propagates_api_errors(
        self, mock_client_class: Mock, temp_document_file: Path, mock_api_key: str
    ) -> None:
        """Test that invoke method properly propagates API errors."""
        # Arrange
        api_error = Exception("API service unavailable")
        mock_client = Mock()
        # Configure context manager behavior
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.side_effect = api_error
        mock_client_class.return_value = mock_client

        tool = UpstagePrebuiltInformationExtraction(
            model="receipt-extraction", api_key=mock_api_key
        )

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            tool.invoke({"filename": str(temp_document_file)})

        assert "Failed to extract information from document" in str(exc_info.value)
