from pathlib import Path
from typing import get_args

import pytest

from langchain_upstage.document_parse import UpstageDocumentParseLoader
from langchain_upstage.document_parse_parsers import (
    OCR,
    Category,
    OutputFormat,
    SplitType,
)

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_file_not_found_error() -> None:
    """Test layout analysis error handling."""

    try:
        UpstageDocumentParseLoader(
            file_path="./NOT_EXISTING_FILE.pdf",
        )
        assert False
    except FileNotFoundError:
        assert True


@pytest.mark.parametrize("output_format", get_args(OutputFormat))
@pytest.mark.parametrize("split", get_args(SplitType))
@pytest.mark.parametrize("ocr", get_args(OCR))
@pytest.mark.parametrize("coordinates", [True, False])
@pytest.mark.parametrize("base64_encoding", ["paragraph"])
def test_document_parse(
    output_format: OutputFormat,
    split: SplitType,
    ocr: OCR,
    coordinates: bool,
    base64_encoding: Category,
) -> None:
    loader = UpstageDocumentParseLoader(
        file_path=EXAMPLE_PDF_PATH,
        output_format=output_format,
        split=split,
        ocr=ocr,
        coordinates=coordinates,
        base64_encoding=[base64_encoding],
    )
    documents = loader.load()
    if split == "element" and ocr == "auto":
        assert len(documents) == 14
    if split == "element" and ocr == "force":
        assert len(documents) == 15
    else:
        assert len(documents) == 1
