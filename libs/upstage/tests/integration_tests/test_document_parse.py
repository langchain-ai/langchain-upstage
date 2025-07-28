from pathlib import Path

from langchain_upstage.document_parse import UpstageDocumentParseLoader

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
