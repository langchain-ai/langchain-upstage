from __future__ import annotations

import os
from typing import (
    Literal,
    Optional,
)

from langchain_upstage.tools.information_extraction_check import validate_extension
from langchain_upstage.tools.response_generator import make_request
from langchain_upstage.tools.value_retriever import get_from_param_or_env

INFORMATION_EXTRACT_BASE_URL = "https://api.upstage.ai/v1/information-extraction"
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


class UpstagePrebuiltInformationExtraction:
    """UpstagePrebuiltInformationExtraction Information extraction model.

    To use, set the environment variable `UPSTAGE_API_KEY` with your API key or
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstagePrebuiltInformationExtraction

            model = UpstagePrebuiltInformationExtraction(model='receipt-extraction')
    """

    def __init__(
        self,
        model: MODELS,
        api_key: Optional[str] = None,
        base_url: str = INFORMATION_EXTRACT_BASE_URL,
    ):
        self.model_name = model
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY",
            api_key,
            "UPSTAGE_API_KEY",
            os.environ.get("UPSTAGE_API_KEY"),
        )
        self.base_url = base_url

    def extract(self, filename: str) -> dict:
        validate_extension(filename, SUPPORTED_EXTENSIONS)

        files = {"document": open(filename, "rb")}
        data = {"model": self.model_name}

        return make_request("POST", self.base_url, self.api_key, files=files, data=data)
