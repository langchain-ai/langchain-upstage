import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from langchain_core.document_loaders import BaseLoader, Blob
from langchain_core.documents import Document

from .layout_analysis_parsers import UpstageLayoutAnalysisParser

DEFAULT_PAGE_BATCH_SIZE = 10

OutputType = Literal["text", "html"]
SplitType = Literal["none", "element", "page"]


def validate_api_key(api_key: str) -> None:
    """
    Validates the provided API key.

    Args:
        api_key (str): The API key to be validated.

    Raises:
        ValueError: If the API key is empty or None.

    Returns:
        None
    """
    if not api_key:
        raise ValueError("API Key is required for Upstage Document Loader")


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


class UpstageLayoutAnalysisLoader(BaseLoader):
    """Upstage Layout Analysis.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageLayoutAnalysis

            file_path = "/PATH/TO/YOUR/FILE.pdf"
            loader = UpstageLayoutAnalysis(
                        file_path, split="page", output_type="text"
                     )
    """

    def __init__(
        self,
        file_path: Union[str, Path, List[str], List[Path]],
        output_type: Union[OutputType, dict] = "html",
        split: SplitType = "none",
        api_key: Optional[str] = None,
        use_ocr: Optional[bool] = None,
        exclude: list = ["header", "footer"],
    ):
        """
        Initializes an instance of the Upstage document loader.

        Args:
            file_path (Union[str, Path, List[str], List[Path]): The path to the document
                                                                to be loaded.
            output_type (Union[OutputType, dict], optional): The type of output to be
                                                             generated by the parser.
                                                             Defaults to "html".
            split (SplitType, optional): The type of splitting to be applied.
                                         Defaults to "none" (no splitting).
            api_key (str, optional): The API key for accessing the Upstage API.
                                     Defaults to None, in which case it will be
                                     fetched from the environment variable
                                     `UPSTAGE_API_KEY`.
            use_ocr (bool, optional): Extract text from images in the document using
                                      OCR. If the value is True, OCR is used to extract
                                      text from an image. If the value is False, text is
                                      extracted from a PDF. (An error will occur if the
                                      value is False and the input is NOT in PDF format)
                                      The default value is None, and the default
                                      behavior will be performed based on the API's
                                      policy if no value is specified. Please check https://developers.upstage.ai/docs/apis/layout-analysis#request-body.
            exclude (list, optional): Exclude specific elements from
                                                     the output.
                                                     Defaults to ["header", "footer"].
        """
        self.file_path = file_path
        self.output_type = output_type
        self.split = split
        if deprecated_key := os.environ.get("UPSTAGE_DOCUMENT_AI_API_KEY"):
            warnings.warn(
                "UPSTAGE_DOCUMENT_AI_API_KEY is deprecated."
                "Please use UPSTAGE_API_KEY instead."
            )

        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY", api_key, "UPSTAGE_API_KEY", deprecated_key
        )
        self.use_ocr = use_ocr
        self.exclude = exclude

        validate_file_path(self.file_path)
        validate_api_key(self.api_key)

    def load(self) -> List[Document]:
        """
        Loads and parses the document using the UpstageLayoutAnalysisParser.

        Returns:
            A list of Document objects representing the parsed layout analysis.
        """

        if isinstance(self.file_path, list):
            result = []

            for file_path in self.file_path:
                blob = Blob.from_path(file_path)

                parser = UpstageLayoutAnalysisParser(
                    self.api_key,
                    split=self.split,
                    output_type=self.output_type,
                    use_ocr=self.use_ocr,
                    exclude=self.exclude,
                )
                result.extend(list(parser.lazy_parse(blob, is_batch=True)))

            return result

        else:
            blob = Blob.from_path(self.file_path)

            parser = UpstageLayoutAnalysisParser(
                self.api_key,
                split=self.split,
                output_type=self.output_type,
                use_ocr=self.use_ocr,
                exclude=self.exclude,
            )
            return list(parser.lazy_parse(blob, is_batch=True))

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads and parses the document using the UpstageLayoutAnalysisParser.

        Returns:
            An iterator of Document objects representing the parsed layout analysis.
        """

        if isinstance(self.file_path, list):
            for file_path in self.file_path:
                blob = Blob.from_path(file_path)

                parser = UpstageLayoutAnalysisParser(
                    self.api_key,
                    split=self.split,
                    output_type=self.output_type,
                    use_ocr=self.use_ocr,
                    exclude=self.exclude,
                )
                yield from parser.lazy_parse(blob, is_batch=True)
        else:
            blob = Blob.from_path(self.file_path)

            parser = UpstageLayoutAnalysisParser(
                self.api_key,
                split=self.split,
                output_type=self.output_type,
                use_ocr=self.use_ocr,
                exclude=self.exclude,
            )
            yield from parser.lazy_parse(blob)

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
