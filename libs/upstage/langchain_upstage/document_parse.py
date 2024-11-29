import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document

from langchain_upstage.document_parse_parsers import (
    DOCUMENT_PARSE_BASE_URL,
    DOCUMENT_PARSE_DEFAULT_MODEL,
    OCR,
    Category,
    OutputFormat,
    SplitType,
    UpstageDocumentParseParser,
)

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)


def validate_file_path(file_path: Union[str, Path, List[str], List[Path]]) -> None:
    """
    Validates if a file exists at the given file path.

    Args:
        file_path (Union[str, Path, List[str], List[Path]): The file path(s) to be
                                                            validated.

    Raises:
        FileNotFoundError: If the file or any of the files in the list do not exist.
    """
    if isinstance(file_path, list):
        for path in file_path:
            validate_file_path(path)
        return
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


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


class UpstageDocumentParseLoader(BaseLoader):
    """Upstage Document Parse Loader.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageDocumentParseLoader

            file_path = "/PATH/TO/YOUR/FILE.pdf"
            loader = UpstageDocumentParseLoader(
                        file_path, split="page", output_format="text"
                     )
    """

    def __init__(
        self,
        file_path: Union[str, Path, List[str], List[Path]],
        split: SplitType = "none",
        api_key: Optional[str] = None,
        base_url: str = DOCUMENT_PARSE_BASE_URL,
        model: str = DOCUMENT_PARSE_DEFAULT_MODEL,
        ocr: OCR = "auto",
        output_format: OutputFormat = "html",
        coordinates: bool = True,
        base64_encoding: List[Category] = [],
    ):
        """
        Initializes an instance of the Upstage document parse loader.

        Args:
            file_path (Union[str, Path, List[str], List[Path]]): The path to the
                                                                document to be loaded.
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none" (no splitting).
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
            base_url (str, optional): The base URL for accessing the Upstage API.
            model (str): The model to be used for the document parse.
                         Defaults to "document-parse".
            ocr (OCRMode, optional): Extract text from images in the document using
                                      OCR. If the value is "force", OCR is used to
                                      extract text from an image. If the value is
                                      "auto", text is extracted from a PDF. (An error
                                      will occur if the value is "auto" and the input
                                      is NOT in PDF format)
            output_format (OutputFormat, optional): Format of the inference results.
            coordinates (bool, optional): Whether to include the coordinates of the
                                          OCR in the output.
            base64_encoding (List[Category], optional): The category of the elements to
                                                        be encoded in base64.
        """
        self.file_path = file_path
        self.split = split
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY",
            api_key,
            "UPSTAGE_API_KEY",
            os.environ.get("UPSTAGE_API_KEY"),
        )
        self.base_url = base_url
        self.model = model
        self.ocr = ocr
        self.output_format = output_format
        self.coordinates = coordinates
        self.base64_encoding = base64_encoding
        self.parser = UpstageDocumentParseParser(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            split=self.split,
            ocr=self.ocr,
            output_format=self.output_format,
            coordinates=self.coordinates,
            base64_encoding=self.base64_encoding,
        )

        validate_file_path(self.file_path)

    def load(self) -> List[Document]:
        """
        Loads and parses the document using the UpstageDocumentParseParser.

        Returns:
            A list of Document objects representing the parsed layout analysis.
        """

        if isinstance(self.file_path, list):
            result = []

            for file_path in self.file_path:
                blob = Blob.from_path(file_path)
                result.extend(list(self.parser.lazy_parse(blob, is_batch=True)))

            return result

        else:
            blob = Blob.from_path(self.file_path)
            return list(self.parser.lazy_parse(blob, is_batch=True))

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads and parses the document using the UpstageDocumentParseParser.

        Returns:
            An iterator of Document objects representing the parsed layout analysis.
        """

        if isinstance(self.file_path, list):
            for file_path in self.file_path:
                blob = Blob.from_path(file_path)
                yield from self.parser.lazy_parse(blob, is_batch=True)
        else:
            blob = Blob.from_path(self.file_path)

            yield from self.parser.lazy_parse(blob)

    def merge_and_split(
        self, documents: List[Document], splitter: Optional[object] = None
    ) -> List[Document]:
        """
        Merges the page content and metadata of multiple documents into a single
        document, or splits the documents using a custom splitter.

        Args:
            documents (list): A list of Document objects to be merged and split.
            splitter (object, optional): An optional splitter object that implements the
                `split_documents` method. If provided, the documents will be split using
                this splitter. Defaults to None, in which case the documents are merged.

        Returns:
            list: A list of Document objects. If no splitter is provided, a single
            Document object is returned with the merged content and combined metadata.
            If a splitter is provided, the documents are split and a list of Document
            objects is returned.

        Raises:
            AssertionError: If a splitter is provided but it does not implement the
            `split_documents` method.
        """
        if splitter is None:
            merged_content = " ".join([doc.page_content for doc in documents])

            metadatas: Dict[str, Any] = dict()
            for _meta in [doc.metadata for doc in documents]:
                for key, value in _meta.items():
                    if key in metadatas:
                        metadatas[key].append(value)
                    else:
                        metadatas[key] = [value]

            return [Document(page_content=merged_content, metadata=metadatas)]
        else:
            assert hasattr(
                splitter, "split_documents"
            ), "splitter must implement split_documents method"

            return splitter.split_documents(documents)
