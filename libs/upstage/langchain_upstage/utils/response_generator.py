import json as json_type
from typing import Any, Optional

import requests


def _default_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "x-upstage-client": "langchain",
    }


def make_request(
    method: str,
    url: str,
    api_key: str,
    headers: Optional[dict[str, str]] = None,
    files: Optional[dict[str, Any]] = None,
    json: Optional[dict[str, Any]] = None,
    data: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> dict:
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=(headers or {}) | _default_headers(api_key),
            files=files,
            json=json,
            data=data,
            params=params,
        )

        response.raise_for_status()
        return response.json()

    except requests.HTTPError as e:
        raise ValueError(f"HTTP error: {e.response.text}")
    except requests.RequestException as e:
        # Handle any request-related exceptions
        raise ValueError(f"Failed to send request: {e}")
    except json_type.JSONDecodeError as e:
        # Handle JSON decode errors
        raise ValueError(f"Failed to decode JSON response: {e}")
    except Exception as e:
        # Handle any other exceptions
        raise ValueError(f"An error occurred: {e}")
