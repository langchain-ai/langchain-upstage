"""Test Upstage layout analysis."""

from pathlib import Path
from typing import List, get_args

from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_upstage.layout_analysis import OutputType, SplitType

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_layout_analysis_param() -> None:
    """Test layout analysis document loader initialization."""

    for output_type in get_args(OutputType):
        for split in get_args(SplitType):
            loader = UpstageLayoutAnalysisLoader(
                file_path=EXAMPLE_PDF_PATH,
                output_type=output_type,
                split=split,
            )
            assert loader.output_type == output_type
            assert loader.split == split
            assert loader.file_path == EXAMPLE_PDF_PATH
            assert loader.exclude == ["header", "footer"]

            excludes: List[List[str]] = [[], ["header"], ["header", "footer"]]
            for exclude in excludes:
                loader = UpstageLayoutAnalysisLoader(
                    file_path=EXAMPLE_PDF_PATH,
                    output_type=output_type,
                    split=split,
                    exclude=exclude,
                )
                assert loader.output_type == output_type
                assert loader.split == split
                assert loader.file_path == EXAMPLE_PDF_PATH
                assert loader.exclude == exclude


def test_file_not_found_error() -> None:
    """Test layout analysis error handling."""

    try:
        UpstageLayoutAnalysisLoader(
            file_path="./NOT_EXISTING_FILE.pdf",
        )
        assert False
    except FileNotFoundError:
        assert True


def test_none_split() -> None:
    """Test layout analysis with no split."""

    for output_type in get_args(OutputType):
        loader = UpstageLayoutAnalysisLoader(
            file_path=EXAMPLE_PDF_PATH,
            output_type=output_type,
            split="none",
        )
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].page_content is not None
        assert documents[0].metadata["total_pages"] == 1


def test_element_split() -> None:
    """Test layout analysis with element split."""

    for output_type in get_args(OutputType):
        loader = UpstageLayoutAnalysisLoader(
            file_path=EXAMPLE_PDF_PATH,
            output_type=output_type,
            split="element",
        )
        documents = loader.load()

        assert len(documents) == 13
        for document in documents:
            assert document.page_content is not None
            assert document.metadata["page"] == 1
            assert document.metadata["id"] is not None
            assert document.metadata["bounding_box"] is not None
            assert isinstance(document.metadata["bounding_box"], str)
            assert document.metadata["category"] is not None


def test_page_split() -> None:
    """Test layout analysis with page split."""

    for output_type in get_args(OutputType):
        loader = UpstageLayoutAnalysisLoader(
            file_path=EXAMPLE_PDF_PATH,
            output_type=output_type,
            split="page",
        )
        documents = loader.load()

        assert len(documents) == 1
        for document in documents:
            assert document.page_content is not None
            assert document.metadata["page"] == 1
