import json
import os
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from langchain_upstage import UpstageUniversalInformationExtraction


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


class TestUpstageUniversalInformationExtraction:
    """Test UpstageUniversalInformationExtraction core functionality."""

    def test_initialization_without_api_key_raises_error(
        self, no_api_key: None
    ) -> None:
        """Test that initialization fails when no API key is provided."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstageUniversalInformationExtraction()

    @patch("langchain_upstage.universal_information_extraction.make_request")
    def test_extract_calls_api_with_correct_parameters(
        self, mock_make_request: Mock
    ) -> None:
        """Test that extract method calls API with correct parameters."""
        # Arrange
        mock_response = {"any": "response", "structure": "doesnt", "matter": True}
        mock_make_request.return_value = mock_response

        model = UpstageUniversalInformationExtraction(api_key="test_key")
        image_urls = ["https://example.com/invoice.png"]
        response_format = {"type": "object"}

        # Act
        result = model.extract(image_urls=image_urls, response_format=response_format)

        # Assert
        assert result == mock_response
        mock_make_request.assert_called_once()

        # Verify API call parameters
        call_args = mock_make_request.call_args
        json_data = call_args[1]["json"]
        assert json_data["model"] == model.model_name
        assert json_data["response_format"] == response_format

    @patch("langchain_upstage.universal_information_extraction.make_request")
    def test_extract_with_custom_parameters_calls_api_correctly(
        self, mock_make_request: Mock
    ) -> None:
        """Test that extract method calls API for custom configuration."""
        # Arrange
        mock_make_request.return_value = {"response": "data"}

        model = UpstageUniversalInformationExtraction(api_key="test_key")

        # Act
        model.extract(
            image_urls=["https://example.com/doc1.png"],
            response_format={"type": "object"},
            pages_per_chunk=3,
            confidence=False,
            doc_split=True,
            location=True,
        )

        # Assert
        call_args = mock_make_request.call_args
        json_data = call_args[1]["json"]

        assert json_data["chunking"]["pages_per_chunk"] == 3
        assert json_data["confidence"] is False
        assert json_data["doc_split"] is True
        assert json_data["location"] is True

    @patch("langchain_upstage.universal_information_extraction.make_request")
    def test_extract_with_default_parameters_calls_api_correctly(
        self, mock_make_request: Mock
    ) -> None:
        """Test that extract method calls API for default configuration."""
        # Arrange
        mock_make_request.return_value = {"response": "data"}

        model = UpstageUniversalInformationExtraction(api_key="test_key")

        # Act
        model.extract(
            image_urls=[
                "https://example.com/doc2.png",
                "https://example.com/doc3.png",
            ],
            response_format={"type": "array"},
            pages_per_chunk=10,
            confidence=True,
            doc_split=False,
            location=False,
        )

        # Assert
        call_args = mock_make_request.call_args
        json_data = call_args[1]["json"]

        assert json_data["chunking"]["pages_per_chunk"] == 10
        assert json_data["confidence"] is True
        assert json_data["doc_split"] is False
        assert json_data["location"] is False

    @patch("langchain_upstage.universal_information_extraction.make_request")
    def test_generate_schema_calls_api_with_correct_parameters(
        self, mock_make_request: Mock
    ) -> None:
        """Test that generate_schema method calls API with correct parameters."""
        # Arrange
        mock_response = {
            "choices": [
                {"message": {"content": json.dumps({"schema": {"type": "object"}})}}
            ]
        }
        mock_make_request.return_value = mock_response

        model = UpstageUniversalInformationExtraction(api_key="test_key")
        image_urls = ["https://example.com/sample.png"]

        # Act
        result = model.generate_schema(image_urls=image_urls)

        # Assert
        assert result == {"schema": {"type": "object"}}
        mock_make_request.assert_called_once()

        # Verify API call parameters
        call_args = mock_make_request.call_args
        assert call_args[0][1] == model.base_url + "/schema-generation"
        json_data = call_args[1]["json"]
        assert json_data["model"] == model.model_name

    @patch("langchain_upstage.universal_information_extraction.make_request")
    def test_extract_propagates_api_errors(self, mock_make_request: Mock) -> None:
        """Test that extract method properly propagates API errors."""
        # Arrange
        api_error = ValueError("API service unavailable")
        mock_make_request.side_effect = api_error

        model = UpstageUniversalInformationExtraction(api_key="test_key")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            model.extract(
                image_urls=["https://example.com/invoice.png"],
                response_format={"type": "object"},
            )

        assert exc_info.value == api_error

    @patch("langchain_upstage.universal_information_extraction.make_request")
    def test_generate_schema_propagates_api_errors(
        self, mock_make_request: Mock
    ) -> None:
        """Test that generate_schema method properly propagates API errors."""
        # Arrange
        api_error = ValueError("API service unavailable")
        mock_make_request.side_effect = api_error

        model = UpstageUniversalInformationExtraction(api_key="test_key")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            model.generate_schema(image_urls=["https://example.com/sample.png"])

        assert exc_info.value == api_error
