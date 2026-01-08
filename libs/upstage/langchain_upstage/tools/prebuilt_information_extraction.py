from __future__ import annotations

import os
from typing import Any, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str
from pydantic import BaseModel, Field, SecretStr

try:
    import httpx
except ImportError:
    raise ImportError(
        "httpx is required for Prebuilt Information Extraction. "
        "Please install it with: pip install httpx"
    )

from langchain_upstage.utils.constants import MEGABYTE

PREBUILT_EXTRACT_BASE_URL = "https://api.upstage.ai/v1/information-extraction"
MODELS = Literal[
    "receipt-extraction",
    "air-waybill-extraction",
    "bill-of-lading-and-shipping-request-extraction",
    "commercial-invoice-and-packing-list-extraction",
    "kr-export-declaration-certificate-extraction",
]
SUPPORTED_EXTENSIONS = [
    "jpeg",
    "png",
    "bmp",
    "pdf",
    "tiff",
    "heic",
    "docx",
    "pptx",
    "xlsx",
    "hwp",
    "hwpx",
]
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * MEGABYTE


class UpstagePrebuiltInformationExtractionInput(BaseModel):
    """Input for the Prebuilt Information Extraction tool."""

    filename: str = Field(
        description="Path to the document file to extract information from. "
        "Supported formats: jpeg, png, bmp, pdf, tiff, heic, docx, pptx, "
        "xlsx, hwp, hwpx. Maximum file size: 50MB."
    )


def _validate_file(file_path: str) -> None:
    """Validate file exists, format, and size.

    Args:
        file_path: Path to the file to validate.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or file size exceeds limit.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.lower().split(".")[-1]
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join([f".{e}" for e in SUPPORTED_EXTENSIONS])
        raise ValueError(
            f"Unsupported file format: .{ext}. Supported formats: {supported}"
        )

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        max_size_mb = MAX_FILE_SIZE_BYTES / MEGABYTE
        raise ValueError(
            f"File too large: {file_size / MEGABYTE:.2f}MB. "
            f"Maximum size: {max_size_mb}MB"
        )


class UpstagePrebuiltInformationExtraction(BaseTool):
    """Tool for extracting structured information from documents using prebuilt models.

    This tool uses Upstage's prebuilt information extraction models for specific
    document types such as receipts, air waybills, bills of lading, etc.

    Available models:
        - receipt-extraction: Extract information from receipts
        - air-waybill-extraction: Extract information from air waybills
        - bill-of-lading-and-shipping-request-extraction: Extract from bills of lading
        - commercial-invoice-and-packing-list-extraction: Extract from invoices and
            packing lists
        - kr-export-declaration-certificate-extraction: Extract from Korean export
            certificates

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Args:
        model: The prebuilt model to use for extraction. Must be one of the
            available models.
        api_key: Optional API key. If not provided, will use UPSTAGE_API_KEY
            environment variable.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstagePrebuiltInformationExtraction

            tool = UpstagePrebuiltInformationExtraction(
                model='receipt-extraction'
            )
            result = tool.invoke({"filename": "/path/to/receipt.pdf"})
    """

    name: str = "prebuilt_information_extraction"
    description: str = (
        "Extract structured information from documents using prebuilt extraction "
        "models. Available models: receipt-extraction, air-waybill-extraction, "
        "bill-of-lading-and-shipping-request-extraction, "
        "commercial-invoice-and-packing-list-extraction, "
        "kr-export-declaration-certificate-extraction. "
        "Supports specific document types: receipts, air waybills, "
        "bills of lading, commercial invoices, packing lists, and Korean export "
        "declaration certificates. "
        "Use this tool when you need to extract information from a specific type of "
        "document that matches one of the prebuilt models."
    )

    model: MODELS
    upstage_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")

    args_schema: Type[BaseModel] = UpstagePrebuiltInformationExtractionInput

    def __init__(
        self,
        model: MODELS,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        upstage_api_key = kwargs.get("upstage_api_key", None)
        if not upstage_api_key:
            # api_key is an explicit parameter, not in kwargs
            upstage_api_key = api_key
        if not upstage_api_key:
            upstage_api_key = kwargs.get("api_key", None)
        if not upstage_api_key:
            upstage_api_key = SecretStr(os.getenv("UPSTAGE_API_KEY", ""))
        upstage_api_key = convert_to_secret_str(upstage_api_key)

        if (
            not upstage_api_key
            or not upstage_api_key.get_secret_value()
            or upstage_api_key.get_secret_value() == ""
        ):
            raise ValueError("UPSTAGE_API_KEY must be set or passed")

        super().__init__(
            model=model,
            upstage_api_key=upstage_api_key,
            **kwargs,
        )
        # Ensure upstage_api_key is set (BaseTool might not preserve it)
        self.upstage_api_key = upstage_api_key

    def _run(
        self,
        filename: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Extract information from a document using the prebuilt extraction model.

        Args:
            filename: Path to the document file to extract information from.
            run_manager: Optional callback manager for tool execution.

        Returns:
            dict: Extracted information as a dictionary with fields, documentType, etc.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file format is unsupported or file size exceeds limit.
            Exception: If API call fails or response parsing fails.
        """
        try:
            # Validate file
            _validate_file(filename)

            # Get API key
            api_key = (
                self.upstage_api_key.get_secret_value()
                if self.upstage_api_key
                else None
            )
            if not api_key:
                raise ValueError("UPSTAGE_API_KEY must be set or passed")

            # Prepare multipart/form-data request
            with open(filename, "rb") as f:
                files = {
                    "document": (
                        os.path.basename(filename),
                        f,
                        "application/octet-stream",
                    )
                }
                data = {"model": self.model}

                # Make HTTP request
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }

                with httpx.Client(timeout=180.0) as client:  # 3 minute timeout
                    response = client.post(
                        PREBUILT_EXTRACT_BASE_URL,
                        headers=headers,
                        files=files,
                        data=data,
                    )
                    response.raise_for_status()
                    return response.json()

        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors as-is
            if run_manager:
                run_manager.on_tool_error(e)
            raise
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors
            error_msg = (
                f"Failed to extract information from document: "
                f"{e.response.status_code} - {e.response.text}"
            )
            if run_manager:
                run_manager.on_tool_error(Exception(error_msg))
            raise Exception(error_msg) from e
        except Exception as e:
            # Wrap other exceptions with context
            error_msg = f"Failed to extract information from document: {str(e)}"
            if run_manager:
                run_manager.on_tool_error(Exception(error_msg))
            raise Exception(error_msg) from e

    async def _arun(
        self,
        filename: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Extract information from a document asynchronously using the prebuilt
        extraction model.

        Args:
            filename: Path to the document file to extract information from.
            run_manager: Optional async callback manager for tool execution.

        Returns:
            dict: Extracted information as a dictionary with fields, documentType, etc.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file format is unsupported or file size exceeds limit.
            Exception: If API call fails or response parsing fails.
        """
        try:
            # Validate file
            _validate_file(filename)

            # Get API key
            api_key = (
                self.upstage_api_key.get_secret_value()
                if self.upstage_api_key
                else None
            )
            if not api_key:
                raise ValueError("UPSTAGE_API_KEY must be set or passed")

            # Prepare multipart/form-data request
            with open(filename, "rb") as f:
                files = {
                    "document": (
                        os.path.basename(filename),
                        f,
                        "application/octet-stream",
                    )
                }
                data = {"model": self.model}

                # Make HTTP request
                headers = {
                    "Authorization": f"Bearer {api_key}",
                }

                async with httpx.AsyncClient(
                    timeout=180.0
                ) as client:  # 3 minute timeout
                    response = await client.post(
                        PREBUILT_EXTRACT_BASE_URL,
                        headers=headers,
                        files=files,
                        data=data,
                    )
                    response.raise_for_status()
                    return response.json()

        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors as-is
            if run_manager:
                await run_manager.on_tool_error(e)
            raise
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors
            error_msg = (
                f"Failed to extract information from document: "
                f"{e.response.status_code} - {e.response.text}"
            )
            if run_manager:
                await run_manager.on_tool_error(Exception(error_msg))
            raise Exception(error_msg) from e
        except Exception as e:
            # Wrap other exceptions with context
            error_msg = f"Failed to extract information from document: {str(e)}"
            if run_manager:
                await run_manager.on_tool_error(Exception(error_msg))
            raise Exception(error_msg) from e
