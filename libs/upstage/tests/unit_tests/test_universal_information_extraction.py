from unittest.mock import Mock, patch

import pytest

from langchain_upstage import UpstageUniversalInformationExtraction


class TestUpstageUniversalInformationExtraction:
    """Test UpstageUniversalInformationExtractionTool core functionality."""

    def test_initialization_without_api_key_raises_error(
        self, no_api_key: None
    ) -> None:
        """Test that initialization fails when no API key is provided."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstageUniversalInformationExtraction()

    @patch("langchain_upstage.tools.universal_information_extraction.ChatUpstage")
    def test_invoke_calls_api_with_correct_parameters(
        self, mock_chat_upstage: Mock
    ) -> None:
        """Test that invoke method calls API with correct parameters."""
        # Arrange
        mock_response = Mock()
        mock_response.content = '{"any": "response", "structure": "doesnt", "matter": true}'
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_upstage.return_value = mock_instance

        tool = UpstageUniversalInformationExtraction()
        image_urls = ["https://example.com/invoice.png"]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        }

        # Act
        result = tool.invoke(
            {
                "image_urls": image_urls,
                "response_format": response_format,
            }
        )

        # Assert
        assert result == {"any": "response", "structure": "doesnt", "matter": True}
        mock_instance.invoke.assert_called_once()

    @patch("langchain_upstage.tools.universal_information_extraction.ChatUpstage")
    def test_invoke_with_custom_parameters_calls_api_correctly(
        self, mock_chat_upstage: Mock
    ) -> None:
        """Test that invoke method calls API for custom configuration."""
        # Arrange
        mock_response = Mock()
        mock_response.content = '{"response": "data"}'
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_upstage.return_value = mock_instance

        tool = UpstageUniversalInformationExtraction()

        # Act
        tool.invoke(
            {
                "image_urls": ["https://example.com/doc1.png"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "test", "schema": {"type": "object"}},
                },
                "pages_per_chunk": 3,
                "confidence": False,
                "doc_split": True,
                "location": True,
            }
        )

        # Assert
        call_args = mock_instance.invoke.call_args
        assert call_args is not None
        # Verify extra_body contains the custom parameters
        assert "extra_body" in call_args.kwargs
        extra_body = call_args.kwargs["extra_body"]
        assert extra_body["chunking"]["pages_per_chunk"] == 3
        assert extra_body["confidence"] is False
        assert extra_body["doc_split"] is True
        assert extra_body["location"] is True

    @patch("langchain_upstage.tools.universal_information_extraction.ChatUpstage")
    def test_invoke_with_default_parameters_calls_api_correctly(
        self, mock_chat_upstage: Mock
    ) -> None:
        """Test that invoke method calls API for default configuration."""
        # Arrange
        mock_response = Mock()
        mock_response.content = '{"response": "data"}'
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_response
        mock_chat_upstage.return_value = mock_instance

        tool = UpstageUniversalInformationExtraction()

        # Act
        tool.invoke(
            {
                "image_urls": [
                    "https://example.com/doc2.png",
                    "https://example.com/doc3.png",
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "test", "schema": {"type": "array"}},
                },
                "pages_per_chunk": 10,
                "confidence": True,
                "doc_split": False,
                "location": False,
            }
        )

        # Assert
        call_args = mock_instance.invoke.call_args
        assert call_args is not None
        extra_body = call_args.kwargs.get("extra_body", {})
        assert extra_body["chunking"]["pages_per_chunk"] == 10
        assert extra_body["confidence"] is True
        assert extra_body["doc_split"] is False
        assert extra_body["location"] is False

    @patch("langchain_upstage.tools.universal_information_extraction.ChatUpstage")
    def test_invoke_propagates_api_errors(self, mock_chat_upstage: Mock) -> None:
        """Test that invoke method properly propagates API errors."""
        # Arrange
        api_error = Exception("API service unavailable")
        mock_instance = Mock()
        mock_instance.invoke.side_effect = api_error
        mock_chat_upstage.return_value = mock_instance

        tool = UpstageUniversalInformationExtraction()

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            tool.invoke(
                {
                    "image_urls": ["https://example.com/invoice.png"],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "test", "schema": {"type": "object"}},
                    },
                }
            )

        assert "Failed to extract information from documents" in str(exc_info.value)
