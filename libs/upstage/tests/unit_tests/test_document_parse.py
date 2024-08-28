import json
from pathlib import Path
from tokenize import Octnumber
from typing import Any, Dict, get_args
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from langchain_upstage.document_parse import UpstageDocumentParseLoader
from langchain_upstage.document_parse_parsers import (
    Category,
    OCRMode,
    OutputFormat,
    SplitType,
)

MOCK_RESPONSE_JSON: Dict[str, Any] = {
    "api": "1.0",
    "model": "layout-analyzer-0.1.0",
    "elements": [{
        "id": 0,
        "coordinates": {
            "x": 74,
            "y": 906,
        },
        "category": "header",
        "content": {
            "html": "arXiv:2103.15348v2",
            "markdown": "arXiv:2103.15348v2",
            "text": "arXiv:2103.15348v2",
        },
        "page": 1,
            "base64_encoding": "string",
        },
    ],
    "content": {
        "text": "arXiv:2103.15348v2LayoutParser Toolkit",
        "html": "arXiv:2103.15348v2",
        "markdown": "arXiv:2103.15348v2",
    },
    "usage": {
        "pages": 1,
    },
}

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_initialization() -> None:
    """Test layout analysis document loader initialization."""
    UpstageDocumentParseLoader(file_path=EXAMPLE_PDF_PATH, api_key="bar")


@pytest.mark.parametrize("output_format", get_args(OutputFormat))
@pytest.mark.parametrize("split", get_args(SplitType))
@pytest.mark.parametrize("ocr", get_args(OCRMode))
@pytest.mark.parametrize("coordinates", [True, False])
@pytest.mark.parametrize("base64_encoding", ["header"])
def test_document_parse_param(
    output_format: OutputFormat,
    split: SplitType,
    ocr: OCRMode,
    coordinates: bool,
    base64_encoding: Category,
) -> None:
    loader = UpstageDocumentParseLoader(
        file_path=EXAMPLE_PDF_PATH,
        api_key="bar",
        ocr=ocr,
        split=split,
        output_format=output_format,
        coordinates=coordinates,
        base64_encoding=[base64_encoding],
    )
    assert loader.output_format == output_format


@patch("requests.post")
@pytest.mark.parametrize("split", get_args(SplitType))
@pytest.mark.parametrize("output_format", get_args(OutputFormat))
def test_document_parse_output(
    mock_post: Mock, output_format: OutputFormat, split: SplitType
) -> None:
    mock_post.return_value = MagicMock(
        status_code=200, json=MagicMock(return_value=MOCK_RESPONSE_JSON)
    )

    loader = UpstageDocumentParseLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_format=output_format,
        split=split,
        api_key="valid_api_key",
    )
    documents = loader.load()

    assert len(documents) == 1
    assert documents[0].page_content == MOCK_RESPONSE_JSON["elements"][0]["content"][output_format]

@patch("requests.post")
def test_request_exception(mock_post: Mock) -> None:
    mock_post.side_effect = requests.RequestException("Mocked request exception")

    loader = UpstageDocumentParseLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_format="html",
        split="page",
        api_key="valid_api_key",
    )

    with TestCase.assertRaises(TestCase(), ValueError) as context:
        loader.load()

    assert "Failed to send request: Mocked request exception" == str(context.exception)
    
@patch("requests.post")
def test_json_decode_error(mock_post: Mock) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    mock_post.return_value = mock_response

    loader = UpstageDocumentParseLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_format="html",
        split="page",
        api_key="valid_api_key",
    )

    with TestCase.assertRaises(TestCase(), ValueError) as context:
        loader.load()

    assert (
        "Failed to decode JSON response: Expecting value: line 1 column 1 (char 0)"
        == str(context.exception)
    )
