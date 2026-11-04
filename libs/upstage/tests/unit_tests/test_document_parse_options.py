"""Tests for the `mode` and `merge_multipage_tables` Document Parse options."""

import io
from pathlib import Path
from typing import Any, Dict, List, get_args
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.document_loaders import Blob
from pypdf import PdfReader, PdfWriter

from langchain_upstage import UpstageDocumentParseLoader, UpstageDocumentParseParser
from langchain_upstage.document_parse_parsers import Mode, SplitType

MAKE_REQUEST = "langchain_upstage.document_parse_parsers.make_request"


def _write_pdf(path: Path, num_pages: int) -> Path:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def pdf_25_pages(tmp_path: Path) -> Path:
    return _write_pdf(tmp_path / "doc25.pdf", 25)


@pytest.fixture
def pdf_150_pages(tmp_path: Path) -> Path:
    return _write_pdf(tmp_path / "doc150.pdf", 150)


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "scan.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n not a real image")
    return path


def _fake_api(calls: List[Dict[str, Any]]) -> Any:
    """Fake `make_request` that records each request and returns one paragraph
    element per page of the chunk it receives, like the real API does."""

    def _make_request(*args: Any, **kwargs: Any) -> dict:
        document = kwargs["files"]["document"]
        try:
            pages = len(PdfReader(io.BytesIO(document.read())).pages)
        except Exception:
            pages = 1
        calls.append({"data": dict(kwargs["data"]), "pages": pages})
        return {
            "elements": [
                {
                    "id": i,
                    "page": i + 1,
                    "category": "paragraph",
                    "content": {"html": f"<p>{i + 1}</p>", "text": "", "markdown": ""},
                }
                for i in range(pages)
            ]
        }

    return _make_request


class TestMode:
    @pytest.mark.parametrize("mode", get_args(Mode))
    @pytest.mark.parametrize("split", get_args(SplitType))
    def test_mode_is_sent_with_every_request(
        self, pdf_25_pages: Path, mode: Mode, split: SplitType
    ) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_25_pages, api_key="test_key", split=split, mode=mode
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            loader.load()

        assert calls
        assert all(call["data"]["mode"] == mode for call in calls)

    @pytest.mark.parametrize("mode", get_args(Mode))
    def test_mode_is_sent_on_lazy_load(self, pdf_25_pages: Path, mode: Mode) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_25_pages, api_key="test_key", split="page", mode=mode
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            docs = list(loader.lazy_load())

        assert len(docs) == 25
        assert len(calls) == 25
        assert all(call["data"]["mode"] == mode for call in calls)

    def test_mode_is_sent_for_non_pdf_files(self, image_file: Path) -> None:
        calls: List[Dict[str, Any]] = []
        parser = UpstageDocumentParseParser(
            api_key="test_key", ocr="force", mode="enhanced"
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            list(parser.lazy_parse(Blob.from_path(image_file)))

        assert len(calls) == 1
        assert calls[0]["data"]["mode"] == "enhanced"

    def test_mode_is_omitted_by_default(self, pdf_25_pages: Path) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(pdf_25_pages, api_key="test_key")
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            loader.load()

        assert loader.mode is None
        assert calls
        assert all("mode" not in call["data"] for call in calls)

    @pytest.mark.parametrize("bad_mode", ["Enhanced", "fast", ""])
    def test_invalid_mode_raises(self, pdf_25_pages: Path, bad_mode: str) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            UpstageDocumentParseParser(api_key="test_key", mode=bad_mode)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid mode"):
            UpstageDocumentParseLoader(
                pdf_25_pages,
                api_key="test_key",
                mode=bad_mode,  # type: ignore[arg-type]
            )


class TestLoaderForwardsOptions:
    def test_chart_recognition_false_reaches_the_api(self, pdf_25_pages: Path) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_25_pages, api_key="test_key", chart_recognition=False
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            loader.load()

        assert loader.parser.chart_recognition is False
        assert calls
        assert all(call["data"]["chart_recognition"] is False for call in calls)

    def test_parser_receives_all_new_options(self, pdf_25_pages: Path) -> None:
        loader = UpstageDocumentParseLoader(
            pdf_25_pages,
            api_key="test_key",
            mode="auto",
            merge_multipage_tables=True,
        )
        assert loader.parser.mode == "auto"
        assert loader.parser.merge_multipage_tables is True


class TestMergeMultipageTables:
    def test_flag_is_omitted_by_default(self, pdf_25_pages: Path) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(pdf_25_pages, api_key="test_key")
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            loader.load()

        assert all("merge_multipage_tables" not in call["data"] for call in calls)

    @pytest.mark.parametrize("split", get_args(SplitType))
    def test_pdf_is_sent_in_one_request_up_to_100_pages(
        self, pdf_25_pages: Path, split: SplitType
    ) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_25_pages,
            api_key="test_key",
            split=split,
            merge_multipage_tables=True,
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            loader.load()

        assert len(calls) == 1
        assert calls[0]["pages"] == 25
        assert calls[0]["data"]["merge_multipage_tables"] is True

    def test_lazy_load_also_uses_large_chunks(self, pdf_25_pages: Path) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_25_pages,
            api_key="test_key",
            split="page",
            merge_multipage_tables=True,
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            docs = list(loader.lazy_load())

        # Without the option, lazy_load sends one request per page.
        assert len(calls) == 1
        assert [doc.metadata["page"] for doc in docs] == list(range(1, 26))

    def test_long_pdf_is_chunked_at_the_sync_limit(self, pdf_150_pages: Path) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_150_pages,
            api_key="test_key",
            split="page",
            merge_multipage_tables=True,
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            docs = loader.load()

        assert [call["pages"] for call in calls] == [100, 50]
        # Page numbers stay global across chunks.
        assert [doc.metadata["page"] for doc in docs] == list(range(1, 151))

    def test_default_chunking_is_unchanged(self, pdf_25_pages: Path) -> None:
        calls: List[Dict[str, Any]] = []
        loader = UpstageDocumentParseLoader(
            pdf_25_pages, api_key="test_key", split="element"
        )
        with patch(MAKE_REQUEST, side_effect=_fake_api(calls)):
            loader.load()

        assert [call["pages"] for call in calls] == [10, 10, 5]


def test_mode_and_merge_together(pdf_25_pages: Path) -> None:
    mock = MagicMock(side_effect=_fake_api([]))
    loader = UpstageDocumentParseLoader(
        pdf_25_pages,
        api_key="test_key",
        mode="enhanced",
        merge_multipage_tables=True,
    )
    with patch(MAKE_REQUEST, mock):
        docs = loader.load()

    assert len(docs) == 1
    data = mock.call_args.kwargs["data"]
    assert data["mode"] == "enhanced"
    assert data["merge_multipage_tables"] is True
    assert data["chart_recognition"] is True
