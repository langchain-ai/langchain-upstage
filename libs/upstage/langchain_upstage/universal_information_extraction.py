from __future__ import annotations

import json
import os
from typing import Literal, Optional

from langchain_upstage.tools.information_extraction_check import (
    MEGABYTE,
    create_message,
)
from langchain_upstage.tools.response_generator import make_request
from langchain_upstage.tools.value_retriever import get_from_param_or_env

SCHEMA_GENERATION_ENDPOINT = "/schema-generation"
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

MAX_FILE_SIZE = 50 * MEGABYTE
MAX_IMAGE_COUNT = 3


def _create_system_content(system_content: str) -> dict:
    return {
        "role": "system",
        "content": system_content,
    }


class UpstageUniversalInformationExtraction:
    """UpstageUniversalInformationExtraction Information extraction model.

    To use, set the environment variable `UPSTAGE_API_KEY` with your API key or
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstageUniversalInformationExtraction

            model = UpstageUniversalInformationExtraction(model='information-extract')
    """

    def __init__(
        self,
        model: str = "information-extract",
        api_key: Optional[str] = None,
        base_url: str = "https://api.upstage.ai/v1/information-extraction",
    ):
        self.model_name = model
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY",
            api_key,
            "UPSTAGE_API_KEY",
            os.environ.get("UPSTAGE_API_KEY"),
        )
        self.base_url = base_url

    def extract(
        self,
        image_urls: list[str],
        response_format: dict,
        pages_per_chunk: int = 5,
        confidence: bool = True,
        doc_split: bool = False,
        location: bool = False,
        mode: Literal["standard", "enhanced"] = "standard",
    ) -> dict:
        contents = [
            create_message(image_url, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE)
            for image_url in image_urls
        ]

        messages = [
            {
                "role": "user",
                "content": contents,
            }
        ]

        return make_request(
            "POST",
            self.base_url,
            self.api_key,
            json={
                "model": self.model_name,
                "messages": messages,
                "response_format": response_format,
                "mode": mode,
                "chunking": {
                    "pages_per_chunk": pages_per_chunk,
                },
                "confidence": confidence,
                "doc_split": doc_split,
                "location": location,
            },
        )

    def generate_schema(
        self, image_urls: list[str], system_content: str | None = None
    ) -> dict:
        if len(image_urls) > MAX_IMAGE_COUNT:
            raise ValueError(f"max image count: {MAX_IMAGE_COUNT}")

        contents = []

        if system_content:
            contents.append(_create_system_content(system_content))

        contents.extend(
            [
                create_message(image_url, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE)
                for image_url in image_urls
            ]
        )

        messages = [{"role": "user", "content": contents}]

        response = make_request(
            "POST",
            self.base_url + SCHEMA_GENERATION_ENDPOINT,
            self.api_key,
            json={
                "model": self.model_name,
                "messages": messages,
            },
        )

        return json.loads(response["choices"][0]["message"]["content"])
