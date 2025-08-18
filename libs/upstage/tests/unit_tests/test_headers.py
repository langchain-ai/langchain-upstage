import os
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from langchain_upstage import ChatUpstage, UpstageEmbeddings


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
def temp_pdf_file(tmp_path: Path) -> Path:
    """Create a minimal PDF file for testing."""
    pdf_path = tmp_path / "test_document.pdf"

    # Minimal PDF content (just enough to be recognized as PDF)
    minimal_pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    pdf_path.write_bytes(minimal_pdf_content)
    return pdf_path


class TestChatUpstageHeaders:
    """Test that ChatUpstage correctly sets and passes default headers."""

    def test_default_headers_are_set_correctly(self, mock_api_key: str) -> None:
        """Test that ChatUpstage sets the correct default headers."""
        # Arrange & Act
        chat_model = ChatUpstage(model="solar-pro2")

        # Assert
        assert chat_model.default_headers == {"x-upstage-client": "langchain"}

    def test_x_upstage_client_header_is_protected(self, mock_api_key: str) -> None:
        """Test that x-upstage-client header cannot be overridden."""
        # Arrange
        custom_headers = {
            "x-upstage-client": "custom-client",
            "x-custom-header": "value",
        }

        # Act
        chat_model = ChatUpstage(model="solar-pro2", default_headers=custom_headers)

        # Assert
        assert chat_model.default_headers is not None
        assert chat_model.default_headers["x-upstage-client"] == "langchain"
        assert chat_model.default_headers["x-custom-header"] == "value"

    def test_custom_headers_without_x_upstage_client(self, mock_api_key: str) -> None:
        """Test that custom headers without x-upstage-client work correctly."""
        # Arrange
        custom_headers = {
            "x-custom-header": "value",
            "x-another-header": "another-value",
        }

        # Act
        chat_model = ChatUpstage(model="solar-pro2", default_headers=custom_headers)

        # Assert
        assert chat_model.default_headers is not None
        assert chat_model.default_headers["x-upstage-client"] == "langchain"
        assert chat_model.default_headers["x-custom-header"] == "value"
        assert chat_model.default_headers["x-another-header"] == "another-value"

    def test_none_default_headers(self, mock_api_key: str) -> None:
        """Test that None default_headers are handled correctly."""
        # Arrange & Act
        chat_model = ChatUpstage(model="solar-pro2", default_headers=None)

        # Assert
        assert chat_model.default_headers is not None
        assert chat_model.default_headers["x-upstage-client"] == "langchain"

    @patch("openai.OpenAI")
    def test_default_headers_passed_to_sync_client(
        self, mock_openai_class: Mock, mock_api_key: str
    ) -> None:
        """Test that default headers are passed to OpenAI sync client."""
        # Arrange
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Act
        ChatUpstage(model="solar-pro2")

        # Assert
        mock_openai_class.assert_called_once()
        call_args = mock_openai_class.call_args[1]  # kwargs
        assert call_args["default_headers"] == {"x-upstage-client": "langchain"}

    @patch("openai.AsyncOpenAI")
    def test_default_headers_passed_to_async_client(
        self, mock_async_openai_class: Mock, mock_api_key: str
    ) -> None:
        """Test that default headers are passed to OpenAI async client."""
        # Arrange
        mock_client = Mock()
        mock_async_openai_class.return_value = mock_client

        # Act
        ChatUpstage(model="solar-pro2")

        # Assert
        mock_async_openai_class.assert_called_once()
        call_args = mock_async_openai_class.call_args[1]  # kwargs
        assert call_args["default_headers"] == {"x-upstage-client": "langchain"}


class TestUpstageEmbeddingsHeaders:
    """Test that UpstageEmbeddings correctly sets and passes default headers."""

    def test_default_headers_are_set_correctly(self, mock_api_key: str) -> None:
        """Test that UpstageEmbeddings sets the correct default headers."""
        # Arrange & Act
        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

        # Assert
        assert embeddings.default_headers == {"x-upstage-client": "langchain"}

    def test_x_upstage_client_header_is_protected(self, mock_api_key: str) -> None:
        """Test that x-upstage-client header cannot be overridden."""
        # Arrange
        custom_headers = {
            "x-upstage-client": "custom-client",
            "x-custom-header": "value",
        }

        # Act
        embeddings = UpstageEmbeddings(
            model="solar-embedding-1-large", default_headers=custom_headers
        )

        # Assert
        assert embeddings.default_headers is not None
        assert embeddings.default_headers["x-upstage-client"] == "langchain"
        assert embeddings.default_headers["x-custom-header"] == "value"

    def test_custom_headers_without_x_upstage_client(self, mock_api_key: str) -> None:
        """Test that custom headers without x-upstage-client work correctly."""
        # Arrange
        custom_headers = {
            "x-custom-header": "value",
            "x-another-header": "another-value",
        }

        # Act
        embeddings = UpstageEmbeddings(
            model="solar-embedding-1-large", default_headers=custom_headers
        )

        # Assert
        assert embeddings.default_headers is not None
        assert embeddings.default_headers["x-upstage-client"] == "langchain"
        assert embeddings.default_headers["x-custom-header"] == "value"
        assert embeddings.default_headers["x-another-header"] == "another-value"

    def test_none_default_headers(self, mock_api_key: str) -> None:
        """Test that None default_headers are handled correctly."""
        # Arrange & Act
        embeddings = UpstageEmbeddings(
            model="solar-embedding-1-large", default_headers=None
        )

        # Assert
        assert embeddings.default_headers is not None
        assert embeddings.default_headers["x-upstage-client"] == "langchain"

    @patch("openai.OpenAI")
    def test_default_headers_passed_to_sync_client(
        self, mock_openai_class: Mock, mock_api_key: str
    ) -> None:
        """Test that default headers are passed to OpenAI sync client."""
        # Arrange
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Act
        UpstageEmbeddings(model="solar-embedding-1-large")

        # Assert
        mock_openai_class.assert_called_once()
        call_args = mock_openai_class.call_args[1]  # kwargs
        assert call_args["default_headers"] == {"x-upstage-client": "langchain"}

    @patch("openai.AsyncOpenAI")
    def test_default_headers_passed_to_async_client(
        self, mock_async_openai_class: Mock, mock_api_key: str
    ) -> None:
        """Test that default headers are passed to OpenAI async client."""
        # Arrange
        mock_client = Mock()
        mock_async_openai_class.return_value = mock_client

        # Act
        UpstageEmbeddings(model="solar-embedding-1-large")

        # Assert
        mock_async_openai_class.assert_called_once()
        call_args = mock_async_openai_class.call_args[1]  # kwargs
        assert call_args["default_headers"] == {"x-upstage-client": "langchain"}


class TestUpstageDocumentParseParserHeaders:
    """Test that UpstageDocumentParseParser correctly sets headers."""

    def test_parser_includes_correct_headers_in_api_request(
        self, mock_api_key: str, temp_pdf_file: Path
    ) -> None:
        """Test that parser includes correct headers when making API requests."""
        # Arrange
        from langchain_upstage.document_parse_parsers import UpstageDocumentParseParser

        parser = UpstageDocumentParseParser(api_key=mock_api_key)

        # Act
        with patch(
            "langchain_upstage.document_parse_parsers.make_request"
        ) as mock_make_request:
            mock_response: dict[str, list] = {"elements": []}
            mock_make_request.return_value = mock_response

            # Use public interface to trigger API request
            from langchain_core.document_loaders import Blob

            blob = Blob.from_path(temp_pdf_file)
            list(parser.lazy_parse(blob))

        # Assert
        mock_make_request.assert_called()
        headers = mock_make_request.call_args[1]["headers"]
        assert headers["x-upstage-client"] == "langchain"
        assert headers["Authorization"] == f"Bearer {mock_api_key}"
