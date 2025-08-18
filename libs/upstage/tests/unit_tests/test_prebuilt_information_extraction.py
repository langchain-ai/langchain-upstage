import os
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from langchain_upstage import UpstagePrebuiltInformationExtraction


@pytest.fixture
def mock_api_key() -> Generator[str, None, None]:
    """Provide a mock API key for testing and restore original after test."""
    original_api_key = os.environ.get("UPSTAGE_API_KEY")
    test_api_key = "test_api_key_12345"
    os.environ["UPSTAGE_API_KEY"] = test_api_key

    yield test_api_key

    if original_api_key is not None:
        os.environ["UPSTAGE_API_KEY"] = original_api_key
    else:
        os.environ.pop("UPSTAGE_API_KEY", None)


@pytest.fixture
def no_api_key() -> Generator[None, None, None]:
    """Temporarily remove UPSTAGE_API_KEY and restore after test."""
    original_api_key = os.environ.get("UPSTAGE_API_KEY")

    # Remove the API key
    if "UPSTAGE_API_KEY" in os.environ:
        del os.environ["UPSTAGE_API_KEY"]

    yield

    # Restore the original API key
    if original_api_key is not None:
        os.environ["UPSTAGE_API_KEY"] = original_api_key
    else:
        os.environ.pop("UPSTAGE_API_KEY", None)


@pytest.fixture
def temp_document_file(tmp_path: Path) -> Path:
    """Create a simple file with supported extension for testing."""
    document_path = tmp_path / "test_document.pdf"

    # Simple content - extension is all that matters for validation
    document_path.write_text("fake pdf content")
    return document_path


class TestUpstagePrebuiltInformationExtraction:
    """Test UpstagePrebuiltInformationExtraction core functionality."""

    def test_initialization_without_api_key_raises_error(
        self, no_api_key: None
    ) -> None:
        """Test that initialization fails when no API key is provided."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstagePrebuiltInformationExtraction(model="receipt-extraction")

    @patch("langchain_upstage.prebuilt_information_extraction.make_request")
    def test_extract_calls_api_with_correct_parameters(
        self, mock_make_request: Mock, temp_document_file: Path
    ) -> None:
        """Test that extract method calls API with correct parameters."""
        # Arrange
        mock_response = {"any": "response", "structure": "doesnt", "matter": True}
        mock_make_request.return_value = mock_response

        model = UpstagePrebuiltInformationExtraction(model="receipt-extraction")

        # Act
        result = model.extract(str(temp_document_file))

        # Assert
        assert result == mock_response
        mock_make_request.assert_called_once()

        # Verify API call parameters
        call_args = mock_make_request.call_args
        assert call_args[1]["data"]["model"] == "receipt-extraction"

    def test_extract_with_unsupported_extension_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that extract method raises error for unsupported file extensions."""
        # Arrange
        model = UpstagePrebuiltInformationExtraction(model="receipt-extraction")
        unsupported_file = tmp_path / "document.txt"
        unsupported_file.write_text("test content")

        # Act & Assert
        with pytest.raises(ValueError):
            model.extract(str(unsupported_file))

    def test_extract_with_nonexistent_file_raises_error(self) -> None:
        """Test that extract method raises error for nonexistent files."""
        # Arrange
        model = UpstagePrebuiltInformationExtraction(model="receipt-extraction")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            model.extract("nonexistent_file.pdf")

    @patch("langchain_upstage.prebuilt_information_extraction.make_request")
    def test_extract_propagates_api_errors(
        self, mock_make_request: Mock, temp_document_file: Path
    ) -> None:
        """Test that extract method properly propagates API errors."""
        # Arrange
        api_error = ValueError("API service unavailable")
        mock_make_request.side_effect = api_error

        model = UpstagePrebuiltInformationExtraction(model="receipt-extraction")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            model.extract(str(temp_document_file))

        assert exc_info.value == api_error
