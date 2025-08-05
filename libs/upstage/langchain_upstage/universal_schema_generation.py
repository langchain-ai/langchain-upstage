from __future__ import annotations

import base64
import json
import os
import re
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import openai
import requests
from langchain_core.utils import get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_upstage.tools.information_extraction_check import MEGABYTE, create_message

SCHEMA_GENERATION_BASE_URL = "https://api.upstage.ai/v1/information-extraction/schema-generation"
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

MAX_FILE_SIZE = 50 * MEGABYTE


class UpstageUniversalSchemaGeneration(BaseModel):
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
    base_url: str = SCHEMA_GENERATION_BASE_URL
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
            "base_url": self.base_url,
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

    def generate(self, img_paths):
        contents = [
            create_message(img_path, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE)
            for img_path in img_paths
        ]

        messages = [
            {
                "role": "user",
                "content": contents
            }
        ]

        response = self.client.create(
            model=self.model_name,
            messages=messages,
        )

        return json.loads(response.choices[0].message.content)