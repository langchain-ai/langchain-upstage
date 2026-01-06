from __future__ import annotations

import json
import os
from typing import Any, Optional, Type, cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str
from openai import NOT_GIVEN
from pydantic import BaseModel, Field, SecretStr

from langchain_upstage import ChatUpstage
from langchain_upstage.utils.constants import MEGABYTE
from langchain_upstage.utils.file_utils import file_to_base64_message

INFORMATION_EXTRACT_BASE_URL = "https://api.upstage.ai/v1/information-extraction"
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
]
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * MEGABYTE
DEFAULT_PAGES_PER_CHUNK = 5
DEFAULT_CONFIDENCE = True
DEFAULT_DOC_SPLIT = False
DEFAULT_LOCATION = False


class UpstageUniversalInformationExtractionInput(BaseModel):
    """Input for the Universal Information Extraction tool."""

    image_urls: list[str] = Field(
        description=(
            "List of file paths or URLs to images/documents to extract information "
            f"from. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}. "
            f"Maximum file size: {MAX_FILE_SIZE_MB}MB per file. "
            "URLs starting with http:// or https:// are supported."
        )
    )
    response_format: dict = Field(
        description="JSON schema defining the structure of information to extract. "
        "This should be a dict with 'type' and 'json_schema' keys, where 'json_schema' "
        "defines the expected output structure."
    )
    pages_per_chunk: int = Field(
        default=DEFAULT_PAGES_PER_CHUNK,
        description=(
            f"Number of pages to process per chunk. "
            f"Default is {DEFAULT_PAGES_PER_CHUNK}."
        ),
    )
    confidence: bool = Field(
        default=DEFAULT_CONFIDENCE,
        description=(
            f"Whether to include confidence scores in the response. "
            f"Default is {DEFAULT_CONFIDENCE}."
        ),
    )
    doc_split: bool = Field(
        default=DEFAULT_DOC_SPLIT,
        description=f"Whether to split documents. Default is {DEFAULT_DOC_SPLIT}.",
    )
    location: bool = Field(
        default=DEFAULT_LOCATION,
        description=(
            f"Whether to include location information. Default is {DEFAULT_LOCATION}."
        ),
    )


def _file_to_base64(file_path: str) -> dict:
    """Convert file to base64 encoded message format.

    Args:
        file_path: Path to the file or URL to convert.

    Returns:
        dict: Dictionary with 'type' and 'image_url' keys. For URLs, returns the URL directly.
            For local files, returns base64 encoded data.

    Raises:
        FileNotFoundError: If the file does not exist (URLs are skipped).
        ValueError: If the file format is unsupported or file size exceeds limit.
    """
    return file_to_base64_message(
        file_path,
        supported_extensions=SUPPORTED_EXTENSIONS,
        max_file_size_bytes=MAX_FILE_SIZE_BYTES,
        allow_urls=True,  # Universal extraction supports URLs
    )


class UpstageUniversalInformationExtraction(BaseTool):
    """Tool for extracting structured information from any document using universal extraction.

    This tool uses Upstage's universal information extraction model that can extract
    information from any document type based on a provided JSON schema.

    Unlike prebuilt models, this tool requires you to define a JSON schema that specifies
    what information to extract from the document. This makes it flexible for any
    document type but requires more configuration.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Args:
        model: The model to use for extraction. Defaults to "information-extract".
        api_key: Optional API key. If not provided, will use UPSTAGE_API_KEY environment variable.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageUniversalInformationExtractionTool

            tool = UpstageUniversalInformationExtractionTool()

            # Use with a schema
            result = tool.invoke({
                "image_urls": ["/path/to/document.pdf"],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "field_name": {"type": "string"}
                            }
                        }
                    }
                }
            })
    """

    name: str = "universal_information_extraction"
    description: str = (
        "Extract structured information from any document using a universal "
        "extraction model. This tool can extract information from any document "
        "type by providing a JSON schema that defines what information to extract. "
        "Use this tool when you need to extract custom information from documents "
        "that don't match prebuilt models, or when you need flexible extraction "
        "based on a schema you define."
    )

    model: str = Field(default="information-extract")
    upstage_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    api_wrapper: Optional[ChatUpstage] = None

    args_schema: Type[BaseModel] = UpstageUniversalInformationExtractionInput

    def __init__(
        self,
        model: str = "information-extract",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        upstage_api_key = kwargs.get("upstage_api_key", None)
        if not upstage_api_key:
            upstage_api_key = api_key
        if not upstage_api_key:
            upstage_api_key = SecretStr(os.getenv("UPSTAGE_API_KEY", ""))
        upstage_api_key = convert_to_secret_str(upstage_api_key)

        if (
            not upstage_api_key
            or not upstage_api_key.get_secret_value()
            or upstage_api_key.get_secret_value() == ""
        ):
            raise ValueError("UPSTAGE_API_KEY must be set or passed")

        # Universal IE uses OpenAI Chat Completion API format
        # So we can use ChatUpstage with custom base_url
        api_wrapper = ChatUpstage(
            model=model,
            api_key=upstage_api_key,
            base_url=INFORMATION_EXTRACT_BASE_URL,
        )

        super().__init__(
            model=model,
            upstage_api_key=upstage_api_key,
            api_wrapper=api_wrapper,
            **kwargs,
        )

    def _run(
        self,
        image_urls: list[str],
        response_format: dict,
        pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
        confidence: bool = DEFAULT_CONFIDENCE,
        doc_split: bool = DEFAULT_DOC_SPLIT,
        location: bool = DEFAULT_LOCATION,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Extract information from documents using the universal extraction model.

        Args:
            image_urls: List of file paths or URLs to images/documents to extract from.
            response_format: JSON schema defining the structure of information to extract.
                Should be a dict with 'type' and 'json_schema' keys.
            pages_per_chunk: Number of pages to process per chunk. Default is 5.
            confidence: Whether to include confidence scores in the response. Default is True.
            doc_split: Whether to split documents. Default is False.
            location: Whether to include location information. Default is False.
            run_manager: Optional callback manager for tool execution.

        Returns:
            dict: Extracted information as a dictionary. If the response is JSON,
                returns the parsed JSON. Otherwise, returns a dict with 'content' key.

        Raises:
            ValueError: If API wrapper is not initialized.
            FileNotFoundError: If any file does not exist.
            ValueError: If file format is unsupported or file is too large.
            Exception: If API call fails or response parsing fails.
        """
        if self.api_wrapper is None:
            error_msg = "API wrapper not initialized. Tool may not have been properly configured."
            if run_manager:
                run_manager.on_tool_error(ValueError(error_msg))
            raise ValueError(error_msg)

        try:
            # Convert files to base64
            contents = [_file_to_base64(url) for url in image_urls]

            # Prepare messages
            messages = [HumanMessage(content=contents)]

            # Use ChatUpstage to make the API call
            api_wrapper = cast(ChatUpstage, self.api_wrapper)
            response = api_wrapper.invoke(
                messages,
                response_format=response_format,
                stream=NOT_GIVEN,
                extra_body={
                    "chunking": {"pages_per_chunk": pages_per_chunk},
                    "confidence": confidence,
                    "doc_split": doc_split,
                    "location": location,
                },
            )

            # Parse response
            if hasattr(response, "content") and response.content:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"content": response.content}
            return {}
        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors as-is
            if run_manager:
                run_manager.on_tool_error(e)
            raise
        except Exception as e:
            # Wrap other exceptions with context
            error_msg = f"Failed to extract information from documents: {str(e)}"
            if run_manager:
                run_manager.on_tool_error(Exception(error_msg))
            raise Exception(error_msg) from e

    async def _arun(
        self,
        image_urls: list[str],
        response_format: dict,
        pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
        confidence: bool = DEFAULT_CONFIDENCE,
        doc_split: bool = DEFAULT_DOC_SPLIT,
        location: bool = DEFAULT_LOCATION,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Extract information from documents asynchronously using the universal extraction model.

        Args:
            image_urls: List of file paths or URLs to images/documents to extract from.
            response_format: JSON schema defining the structure of information to extract.
                Should be a dict with 'type' and 'json_schema' keys.
            pages_per_chunk: Number of pages to process per chunk. Default is 5.
            confidence: Whether to include confidence scores in the response. Default is True.
            doc_split: Whether to split documents. Default is False.
            location: Whether to include location information. Default is False.
            run_manager: Optional async callback manager for tool execution.

        Returns:
            dict: Extracted information as a dictionary. If the response is JSON,
                returns the parsed JSON. Otherwise, returns a dict with 'content' key.

        Raises:
            ValueError: If API wrapper is not initialized.
            FileNotFoundError: If any file does not exist.
            ValueError: If file format is unsupported or file is too large.
            Exception: If API call fails or response parsing fails.
        """
        if self.api_wrapper is None:
            error_msg = "API wrapper not initialized. Tool may not have been properly configured."
            if run_manager:
                await run_manager.on_tool_error(ValueError(error_msg))
            raise ValueError(error_msg)

        try:
            # Convert files to base64
            contents = [_file_to_base64(url) for url in image_urls]

            # Prepare messages
            messages = [HumanMessage(content=contents)]

            # Use ChatUpstage to make the API call
            api_wrapper = cast(ChatUpstage, self.api_wrapper)
            response = await api_wrapper.ainvoke(
                messages,
                response_format=response_format,
                extra_body={
                    "chunking": {"pages_per_chunk": pages_per_chunk},
                    "confidence": confidence,
                    "doc_split": doc_split,
                    "location": location,
                },
            )

            # Parse response
            if hasattr(response, "content") and response.content:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"content": response.content}
            return {}
        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors as-is
            if run_manager:
                await run_manager.on_tool_error(e)
            raise
        except Exception as e:
            # Wrap other exceptions with context
            error_msg = f"Failed to extract information from documents: {str(e)}"
            if run_manager:
                await run_manager.on_tool_error(Exception(error_msg))
            raise Exception(error_msg) from e

    def generate_schema(self, image_urls: list[str]) -> dict:
        """Generate JSON schema from documents.

        This method analyzes the provided documents and generates a JSON schema
        that can be used for information extraction.

        Args:
            image_urls: List of file paths or URLs to images/documents to analyze.
                Maximum 3 images are supported.

        Returns:
            dict: Generated JSON schema in the format:
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema_name",
                        "schema": {...}
                    }
                }

        Raises:
            ValueError: If API wrapper is not initialized or too many images provided.
            FileNotFoundError: If any file does not exist.
            ValueError: If file format is unsupported or file is too large.
            Exception: If API call fails or response parsing fails.
        """
        if self.api_wrapper is None:
            raise ValueError(
                "API wrapper not initialized. Tool may not have been properly configured."
            )

        if len(image_urls) > 3:
            raise ValueError("Maximum 3 images are supported for schema generation.")

        try:
            # Convert files to base64
            contents = [_file_to_base64(url) for url in image_urls]

            # Prepare messages
            messages = [HumanMessage(content=contents)]

            # Create ChatUpstage instance for schema generation endpoint
            # The endpoint is /schema-generation/chat/completions
            # Get API key from api_wrapper if upstage_api_key is not set
            api_key = self.upstage_api_key
            if not api_key and self.api_wrapper:
                api_key = getattr(self.api_wrapper, "upstage_api_key", None)
            
            if not api_key:
                raise ValueError("API key not available for schema generation")
            
            schema_api_wrapper = ChatUpstage(
                model=self.model,
                api_key=api_key,
                base_url=f"{INFORMATION_EXTRACT_BASE_URL}/schema-generation",
            )

            # Make API call
            response = schema_api_wrapper.invoke(messages)

            # Parse response - content contains stringified JSON schema
            if hasattr(response, "content") and response.content:
                try:
                    # Parse the JSON schema from content
                    schema_content = json.loads(response.content)
                    # Return in the format expected by extract() method
                    return {
                        "type": "json_schema",
                        "json_schema": schema_content
                    }
                except json.JSONDecodeError:
                    # If content is not valid JSON, try to wrap it
                    return {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "generated_schema",
                            "schema": json.loads(response.content)
                        }
                    }
            raise ValueError("Empty response from schema generation API")

        except (FileNotFoundError, ValueError) as e:
            raise
        except Exception as e:
            raise Exception(f"Failed to generate schema: {str(e)}") from e

    async def agenerate_schema(self, image_urls: list[str]) -> dict:
        """Generate JSON schema from documents asynchronously.

        Args:
            image_urls: List of file paths or URLs to images/documents to analyze.
                Maximum 3 images are supported.

        Returns:
            dict: Generated JSON schema.

        Raises:
            ValueError: If API wrapper is not initialized or too many images provided.
            FileNotFoundError: If any file does not exist.
            ValueError: If file format is unsupported or file is too large.
            Exception: If API call fails or response parsing fails.
        """
        if self.api_wrapper is None:
            raise ValueError(
                "API wrapper not initialized. Tool may not have been properly configured."
            )

        if len(image_urls) > 3:
            raise ValueError("Maximum 3 images are supported for schema generation.")

        try:
            # Convert files to base64
            contents = [_file_to_base64(url) for url in image_urls]

            # Prepare messages
            messages = [HumanMessage(content=contents)]

            # Create ChatUpstage instance for schema generation endpoint
            # Get API key from api_wrapper if upstage_api_key is not set
            api_key = self.upstage_api_key
            if not api_key and self.api_wrapper:
                api_key = getattr(self.api_wrapper, "upstage_api_key", None)
            
            if not api_key:
                raise ValueError("API key not available for schema generation")
            
            schema_api_wrapper = ChatUpstage(
                model=self.model,
                api_key=api_key,
                base_url=f"{INFORMATION_EXTRACT_BASE_URL}/schema-generation",
            )

            # Make async API call
            response = await schema_api_wrapper.ainvoke(messages)

            # Parse response
            if hasattr(response, "content") and response.content:
                try:
                    schema_content = json.loads(response.content)
                    return {
                        "type": "json_schema",
                        "json_schema": schema_content
                    }
                except json.JSONDecodeError:
                    return {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "generated_schema",
                            "schema": json.loads(response.content)
                        }
                    }
            raise ValueError("Empty response from schema generation API")

        except (FileNotFoundError, ValueError) as e:
            raise
        except Exception as e:
            raise Exception(f"Failed to generate schema: {str(e)}") from e
