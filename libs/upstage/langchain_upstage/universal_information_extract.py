from __future__ import annotations

import base64
import os
import re
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import openai
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

INFORMATION_EXTRACTION_BASE_URL = "https://api.upstage.ai/v1/information-extraction"
SUPPORTED_EXTENSIONS = [
    "jpeg",
    "png",
    "bmp",
    "pdf",
    "tiff",
    "heic",
    "docx",
    "pptx",
    "xlsx"
]

KILOBYTE = 1024
MEGABYTE = 1024 * KILOBYTE
MAX_FILE_SIZE = 50 * MEGABYTE


def _process_input(input_path):
    if re.match(r"^https?://", input_path):
        return input_path

    if os.path.exists(input_path):
        if os.path.getsize(input_path) > MAX_FILE_SIZE:
            raise ValueError(f"File too large: max {MAX_FILE_SIZE / MEGABYTE}MB")
    else:
        raise FileNotFoundError(f"File not found: {input_path}")

    file_ext = input_path.lower().split('.')[-1]
    if file_ext not in SUPPORTED_EXTENSIONS:
        supported = ', '.join([f".{ext}" for ext in SUPPORTED_EXTENSIONS])
        raise ValueError(f"Unsupported image extension. supported: {supported}")

    try:
        with open(input_path, 'rb') as img_file:
            img_bytes = img_file.read()
            base64_data = base64.b64encode(img_bytes).decode('utf-8')

        return f"data:application/octet-stream;base64,{base64_data}"
    except Exception as e:
        raise ValueError(f"Error occurred while processing the file: {e}")


def create_message(url: str):
    return {
        "type": "image_url",
        "image_url": {
            "url": url
        }
    }


class UpstageUniversalInformationExtraction(BaseModel):
    """UpstageUniversalInformationExtraction Information extraction model.

    To use, set the environment variable `UPSTAGE_API_KEY` with your API key or
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageUniversalInformationExtraction

            model = UpstageUniversalInformationExtraction(model='information-extract')
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(
        default="information-extract",
        alias="model"
    )
    """Model name to use."""
    upstage_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "UPSTAGE_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `UPSTAGE_API_KEY`."
            ),
        ),
        alias="api_key",
    )
    """Automatically inferred from env are `UPSTAGE_API_KEY` if not provided."""
    upstage_api_base: Optional[str] = Field(
        default_factory=from_env(
            "UPSTAGE_API_BASE", default=INFORMATION_EXTRACTION_BASE_URL
        ),
        alias="base_url",
    )
    """Endpoint URL to use."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Upstage embedding API. Can be float, httpx.Timeout or
        None."""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    default_headers: Union[Mapping[str, str], None] = {"x-upstage-client": "langchain"}
    """add trace header."""
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client. Only used for sync invocations. Must specify 
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify 
        http_client as well if you'd like a custom client for sync invocations."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""

        client_params: dict = {
            "api_key": (
                self.upstage_api_key.get_secret_value()
                if self.upstage_api_key
                else None
            ),
            "base_url": self.upstage_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return self

    def information_extract(self, img_paths: list[str], response_format):
        contents = [
            create_message(_process_input(img_path))
            for img_path in img_paths
        ]

        messages = [
            {
                "role": "user",
                "content": contents
            }
        ]

        return self.client.create(
            model=self.model_name,
            messages=messages,
            response_format=response_format
        )
