from __future__ import annotations

import os
from typing import (
    Literal,
    Optional,
)

import requests

INFORMATION_EXTRACT_BASE_URL = "https://api.upstage.ai/v1/information-extraction"
MODELS = Literal[
    "receipt-extraction",
    "air-waybill-extraction",
    "bill-of-lading-and-shipping-request-extraction",
    "commercial-invoice-and-packing-list-extraction",
    "kr-export-declaration-certificate-extraction",
]


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


class UpstagePrebuiltInformationExtraction:
    """UpstagePrebuiltInformationExtraction Information extraction model.

    To use, set the environment variable `UPSTAGE_API_KEY` with your API key or
    pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import UpstagePrebuiltInformationExtraction

            model = UpstagePrebuiltInformationExtraction(model='information-extract')
    """

    def __init__(
            self,
            model: MODELS,
            api_key: Optional[str] = None,
            base_url: str = INFORMATION_EXTRACT_BASE_URL
    ):
        self.model_name = model
        self.api_key = get_from_param_or_env(
            "UPSTAGE_API_KEY",
            api_key,
            "UPSTAGE_API_KEY",
            os.environ.get("UPSTAGE_API_KEY"),
        )
        self.base_url = base_url

    def information_extract(self, file_path):
        headers = {"Authorization": f"Bearer {self.api_key}"}

        files = {"document": open(file_path, "rb")}
        data = {"model": self.model_name}

        response = requests.post(self.base_url, headers=headers, files=files, data=data)
        return response.json()
