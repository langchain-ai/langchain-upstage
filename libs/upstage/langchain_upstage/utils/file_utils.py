"""Utility functions for file operations."""

from __future__ import annotations

import base64
import os

from langchain_upstage.utils.constants import MEGABYTE


def file_to_base64_message(
    file_path: str,
    supported_extensions: list[str],
    max_file_size_bytes: int,
    allow_urls: bool = True,
) -> dict:
    """Convert file to base64 encoded message format.

    This is a generic utility function that can be used across different modules
    for converting files to the format expected by Upstage APIs.

    Args:
        file_path: Path to the file or URL to convert.
        supported_extensions: List of supported file extensions (e.g., ["pdf", "jpg"]).
        max_file_size_bytes: Maximum file size in bytes.
        allow_urls: Whether to allow URLs. If True, URLs are returned as-is without
            validation. If False, URLs will raise ValueError.

    Returns:
        dict: Dictionary with 'type' and 'image_url' keys. For URLs, returns the URL
            directly. For local files, returns base64 encoded data.

    Raises:
        FileNotFoundError: If the file does not exist (URLs are skipped if allow_urls=True).
        ValueError: If file format is unsupported, file is too large, or URL is not
            allowed when allow_urls=False.
    """

    if file_path.startswith(("http://", "https://")):
        if not allow_urls:
            raise ValueError("URLs are not allowed for this operation")
        return {"type": "image_url", "image_url": {"url": file_path}}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.lower().split(".")[-1]
    if ext not in supported_extensions:
        supported = ", ".join([f".{e}" for e in supported_extensions])
        raise ValueError(
            f"Unsupported file format: .{ext}. Supported formats: {supported}"
        )

    file_size = os.path.getsize(file_path)
    if file_size > max_file_size_bytes:
        max_size_mb = max_file_size_bytes / MEGABYTE
        raise ValueError(
            f"File too large: {file_size / MEGABYTE:.2f}MB. "
            f"Maximum size: {max_size_mb}MB"
        )

    with open(file_path, "rb") as f:
        file_bytes = f.read()
        base64_data = base64.b64encode(file_bytes).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:application/octet-stream;base64,{base64_data}"},
    }

