import os
from pathlib import Path
from typing import Generator, get_args
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage.document_parse_parsers import (
    OCR,
    Category,
    OutputFormat,
    SplitType,
)


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


@pytest.fixture
def temp_pdf_file(tmp_path: Path) -> Path:
    """Create a minimal PDF file for testing."""
    pdf_path = tmp_path / "test_document.pdf"

    # Minimal PDF content (just enough to be recognized as PDF)
    minimal_pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    pdf_path.write_bytes(minimal_pdf_content)
    return pdf_path


class TestUpstageDocumentParseLoader:
    """Test UpstageDocumentParseLoader."""

    def test_initialization_without_api_key_raises_error(
        self, temp_pdf_file: Path, no_api_key: None
    ) -> None:
        """Test that initialization fails when no API key is provided."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstageDocumentParseLoader(temp_pdf_file)

    def test_initialization_with_nonexistent_file_raises_error(
        self, mock_api_key: str
    ) -> None:
        """Test that initialization fails with non-existent file."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            UpstageDocumentParseLoader("nonexistent_file.pdf")

    @pytest.mark.parametrize("output_format", get_args(OutputFormat))
    @pytest.mark.parametrize("split", get_args(SplitType))
    @pytest.mark.parametrize("ocr", get_args(OCR))
    @pytest.mark.parametrize("chart_recognition", [True, False])
    @pytest.mark.parametrize("coordinates", [True, False])
    @pytest.mark.parametrize("base64_encoding", [[], ["table"], ["figure", "chart"]])
    def test_initialization_with_all_parameters(
        self,
        temp_pdf_file: Path,
        no_api_key: None,
        output_format: OutputFormat,
        split: SplitType,
        ocr: OCR,
        chart_recognition: bool,
        coordinates: bool,
        base64_encoding: list[Category],
    ) -> None:
        """Test that loader initializes with all parameter combinations."""
        # Act
        loader = UpstageDocumentParseLoader(
            temp_pdf_file,
            api_key="test_key",
            output_format=output_format,
            split=split,
            ocr=ocr,
            chart_recognition=chart_recognition,
            coordinates=coordinates,
            base64_encoding=base64_encoding,
        )

        # Assert
        assert loader.output_format == output_format
        assert loader.split == split
        assert loader.ocr == ocr
        assert loader.chart_recognition == chart_recognition
        assert loader.coordinates == coordinates
        assert loader.base64_encoding == base64_encoding

    def test_merge_and_split_combines_documents_and_metadata(
        self, temp_pdf_file: Path, mock_api_key: str
    ) -> None:
        """Test that merge_and_split correctly combines documents and metadata."""
        # Arrange
        documents = [
            Document(page_content="Content 1", metadata={"page": 1, "source": "doc1"}),
            Document(page_content="Content 2", metadata={"page": 2, "source": "doc2"}),
        ]
        loader = UpstageDocumentParseLoader(temp_pdf_file)

        # Act
        result = loader.merge_and_split(documents)

        # Assert
        assert len(result) == 1
        assert result[0].page_content == "Content 1 Content 2"
        assert result[0].metadata["page"] == [1, 2]
        assert result[0].metadata["source"] == ["doc1", "doc2"]

    def test_merge_and_split_with_splitter_delegates_to_splitter(
        self, temp_pdf_file: Path, mock_api_key: str
    ) -> None:
        """Test that merge_and_split delegates to splitter when provided."""
        # Arrange
        documents = [
            Document(page_content="Content 1", metadata={"page": 1}),
            Document(page_content="Content 2", metadata={"page": 2}),
        ]
        loader = UpstageDocumentParseLoader(temp_pdf_file)

        # Mock splitter
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [
            Document(page_content="Split 1", metadata={"split": 1}),
            Document(page_content="Split 2", metadata={"split": 2}),
        ]

        # Act
        result = loader.merge_and_split(documents, mock_splitter)

        # Assert
        assert len(result) == 2
        mock_splitter.split_documents.assert_called_once_with(documents)
