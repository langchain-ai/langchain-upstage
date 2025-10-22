import io
import json
import logging
import os
from typing import Any, Dict, Iterator, List, Literal, Optional

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from langchain_upstage.tools.response_generator import make_request

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)

DOCUMENT_PARSE_BASE_URL = "https://api.upstage.ai/v1/document-digitization"
DEFAULT_NUM_PAGES = 10
DOCUMENT_PARSE_DEFAULT_MODEL = "document-parse"

OutputFormat = Literal["text", "html", "markdown"]
OCR = Literal["auto", "force"]
SplitType = Literal["none", "page", "element"]
Category = Literal[
    "table",
    "figure",
    "chart",
    "heading1",
    "header",
    "footer",
    "caption",
    "paragraph",
    "equation",
    "list",
    "index",
    "footnote",
]


def parse_output(data: dict, output_format: OutputFormat) -> str:
    """
    Parse the output data based on the specified output type.

    Args:
        data (dict): The data to be parsed.
        output_format (OutputFormat): The output format to parse the element data
                                               into.

    Returns:
        str: The parsed output.

    Raises:
        ValueError: If the output type is invalid.
    """
    content = data["content"]
    if output_format == "text":
        return content["text"]
    elif output_format == "html":
        return content["html"]
    elif output_format == "markdown":
        return content["markdown"]
    else:
        raise ValueError(f"Invalid output type: {output_format}")


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class UpstageDocumentParseParser(BaseBlobParser):
    """Upstage Document Parse Parser.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageDocumentParseParser

            loader = UpstageDocumentParseParser(split="page", output_format="text")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DOCUMENT_PARSE_BASE_URL,
        model: str = DOCUMENT_PARSE_DEFAULT_MODEL,
        split: SplitType = "none",
        chart_recognition: bool = True,
        ocr: OCR = "auto",
        output_format: OutputFormat = "html",
        coordinates: bool = True,
        base64_encoding: Optional[List[Category]] = None,
    ):
        """
        Initializes an instance of the Upstage class.

        Args:
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
            base_url (str, optional): The base URL for accessing the Upstage API.
            model (str): The model to be used for the document parse.
                         Defaults to "document-parse".
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none" (no splitting).
            ocr (OCRMode, optional): Extract text from images in the document using OCR.
                                     If the value is "force", OCR is used to extract
                                     text from an image. If the value is "auto", text is
                                     extracted from a PDF. (An error will occur if the
                                     value is "auto" and the input is NOT in PDF format)
            output_format (OutputFormat, optional): Format of the inference results.
            coordinates (bool, optional): Whether to include the coordinates of the
                                          OCR in the output.
            base64_encoding (List[Category], optional): The category of the elements to
                                                        be encoded in base64.


        """
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY",
            api_key,
            "UPSTAGE_API_KEY",
            os.environ.get("UPSTAGE_API_KEY"),
        )
        self.base_url = base_url
        self.model = model
        self.split = split
        self.chart_recognition = chart_recognition
        self.ocr = ocr
        self.output_format = output_format
        self.coordinates = coordinates
        self.base64_encoding = base64_encoding if base64_encoding is not None else []

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests with x-upstage-client always set to "langchain".

        Returns:
            Dict containing Authorization and x-upstage-client headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "x-upstage-client": "langchain",
        }

    def _get_response(self, files: Dict) -> List:
        """
        Sends a POST request to the API endpoint with the provided files and
        returns the response.

        Args:
            files: the files to be sent in the request.

        Returns:
            dict: The JSON response from the API.

        Raises:
            ValueError: If there is an error in the API call.
        """
        response = make_request(
            "POST",
            self.base_url,
            self.api_key,
            headers=self._get_headers(),
            files=files,
            data={
                "model": self.model,
                "chart_recognition": self.chart_recognition,
                "ocr": self.ocr,
                "output_formats": f"['{self.output_format}']",
                "coordinates": self.coordinates,
                "base64_encoding": json.dumps(self.base64_encoding),
            },
        )

        return response.get("elements", [])

    def _split_and_request(
        self,
        full_docs: PdfReader,
        start_page: int,
        num_pages: int = DEFAULT_NUM_PAGES,
    ) -> List:
        """
        Splits the full pdf document into partial pages and sends a request to the
        server.

        Args:
            full_docs (PdfReader): The full document to be split and requested.
            start_page (int): The starting page number for splitting the document.
            num_pages (int, optional): The number of pages to split the document
                                       into.
                                       Defaults to DEFAULT_NUMBER_OF_PAGE.

        Returns:
            response: The response from the server.
        """
        merger = PdfWriter()
        merger.append(
            full_docs,
            pages=(start_page, min(start_page + num_pages, full_docs.get_num_pages())),
        )

        with io.BytesIO() as buffer:
            merger.write(buffer)
            buffer.seek(0)
            response = self._get_response({"document": buffer})

        return response

    def _element_document(self, elements: Dict, start_page: int = 0) -> Document:
        """
        Converts an elements into a Document object.

        Args:
            elements (Dict) : The elements to convert.
            start_page (int): The starting page number for splitting the document.
                              This number starts from zero.

        Returns:
            A list containing a single Document object.

        """
        metadata = {
            "id": elements["id"],
            "page": elements["page"] + start_page,
            "category": elements["category"],
        }

        if self.coordinates and elements.get("coordinates"):
            metadata["coordinates"] = elements.get("coordinates")
        if self.base64_encoding and elements.get("base64_encoding"):
            metadata["base64_encoding"] = elements.get("base64_encoding")

        return Document(
            page_content=(parse_output(elements, self.output_format)),
            metadata=metadata,
        )

    def _page_document(self, elements: List, start_page: int = 0) -> List[Document]:
        """
        Combines elements with the same page number into a single Document object.

        Args:
            elements: A list of elements containing page numbers.
            start_page: The starting page number for splitting the document.
                This number starts from zero.

        Returns:
            `Document` objects, each representing a page with its content and
                metadata.
        """
        _docs = []
        pages = sorted(set(map(lambda x: x["page"], elements)))

        page_group = [
            [element for element in elements if element["page"] == x] for x in pages
        ]

        for group in page_group:
            page_content = " ".join(
                [parse_output(element, self.output_format) for element in group]
            )

            metadata = {
                "page": group[0]["page"] + start_page,
            }

            if self.base64_encoding:
                base64_encodings = [
                    element.get("base64_encoding")
                    for element in group
                    if element.get("base64_encoding") is not None
                ]
                metadata["base64_encodings"] = base64_encodings

            if self.coordinates:
                coordinates = [
                    element.get("coordinates")
                    for element in group
                    if element.get("coordinates") is not None
                ]
                metadata["coordinates"] = coordinates

            _docs.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        return _docs

    def lazy_parse(self, blob: Blob, is_batch: bool = False) -> Iterator[Document]:
        """
        Lazily parses a document and yields Document objects based on the specified
        split type.

        Args:
            blob (Blob): The input document blob to parse.
            is_batch (bool, optional): Whether to parse the document in batches.
                                       Defaults to False (single page parsing)

        Yields:
            Document: The parsed document object.

        Raises:
            ValueError: If an invalid split type is provided.

        """

        if is_batch:
            num_pages = DEFAULT_NUM_PAGES
        else:
            num_pages = 1

        try:
            full_docs = PdfReader(str(blob.path))
            number_of_pages = full_docs.get_num_pages()
            is_pdf = True
        except PdfReadError:
            number_of_pages = 1
            is_pdf = False
        except Exception as e:
            raise ValueError(f"Failed to read PDF file: {e}")

        if self.split == "none":
            result = ""
            base64_encodings = []
            coordinates = []

            if is_pdf:
                start_page = 0
                num_pages = DEFAULT_NUM_PAGES
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    elements = self._split_and_request(full_docs, start_page, num_pages)
                    for element in elements:
                        result += parse_output(element, self.output_format)
                        if self.base64_encoding:
                            base64_encoding = element.get("base64_encoding")
                            if base64_encoding is not None:
                                base64_encodings.append(base64_encoding)
                        if self.coordinates:
                            coordinate = element.get("coordinates")
                            if coordinate is not None:
                                coordinates.append(coordinate)

                    start_page += num_pages

            else:
                if not blob.path:
                    raise ValueError("Blob path is required for non-PDF files.")

                with open(blob.path, "rb") as f:
                    elements = self._get_response({"document": f})

                for element in elements:
                    result += parse_output(element, self.output_format)

                    if (
                        self.base64_encoding
                        and element.get("base64_encoding") is not None
                    ):
                        base64_encoding = element.get("base64_encoding")
                        if base64_encoding is not None:
                            base64_encodings.append(base64_encoding)
                    if self.coordinates and element.get("coordinates") is not None:
                        coordinate = element.get("coordinates")
                        if coordinate is not None:
                            coordinates.append(coordinate)
            metadata: Dict[str, Any] = {
                "total_pages": number_of_pages,
            }
            if self.coordinates:
                metadata["coordinates"] = coordinates
            if self.base64_encoding:
                metadata["base64_encodings"] = base64_encodings

            yield Document(
                page_content=result,
                metadata=metadata,
            )

        elif self.split == "element":
            if is_pdf:
                start_page = 0
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    elements = self._split_and_request(full_docs, start_page, num_pages)
                    for element in elements:
                        yield self._element_document(element, start_page)

                    start_page += num_pages

            else:
                if not blob.path:
                    raise ValueError("Blob path is required for non-PDF files.")
                with open(blob.path, "rb") as f:
                    elements = self._get_response({"document": f})

                for element in elements:
                    yield self._element_document(element)

        elif self.split == "page":
            if is_pdf:
                start_page = 0
                for _ in range(number_of_pages):
                    if start_page >= number_of_pages:
                        break

                    elements = self._split_and_request(full_docs, start_page, num_pages)
                    yield from self._page_document(elements, start_page)

                    start_page += num_pages
            else:
                if not blob.path:
                    raise ValueError("Blob path is required for non-PDF files.")
                with open(blob.path, "rb") as f:
                    elements = self._get_response({"document": f})

                yield from self._page_document(elements)

        else:
            raise ValueError(f"Invalid split type: {self.split}")
