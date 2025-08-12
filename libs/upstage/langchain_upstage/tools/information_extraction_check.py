import base64
import os
import re
from typing import List

KILOBYTE = 1024
MEGABYTE = 1024 * KILOBYTE


def _process_input(
    input_url: str, supported_extensions: List[str], max_file_size: int
) -> str:
    if re.match(r"^https?://", input_url):
        return input_url

    if os.path.exists(input_url):
        if os.path.getsize(input_url) > max_file_size:
            raise ValueError(f"File too large: max {max_file_size / MEGABYTE}MB")
    else:
        raise FileNotFoundError(f"File not found: {input_url}")

    validate_extension(input_url, supported_extensions)

    try:
        with open(input_url, "rb") as img_file:
            img_bytes = img_file.read()
            base64_data = base64.b64encode(img_bytes).decode("utf-8")

        return f"data:application/octet-stream;base64,{base64_data}"
    except Exception as e:
        raise ValueError(f"Error occurred while processing the file: {e}")


def validate_extension(input_url: str, supported_extensions: List[str]) -> None:
    file_ext = input_url.lower().split(".")[-1]
    if file_ext not in supported_extensions:
        supported = ", ".join([f".{ext}" for ext in supported_extensions])
        raise ValueError(f"Unsupported image extension. supported: {supported}")


def create_message(
    input_url: str, supported_extensions: List[str], max_file_size: int
) -> dict:
    url = _process_input(input_url, supported_extensions, max_file_size)

    return {"type": "image_url", "image_url": {"url": url}}
