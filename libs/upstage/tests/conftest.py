import os
from typing import Generator

import pytest


@pytest.fixture
def mock_api_key() -> Generator[str, None, None]:
    """Provide a mock API key for testing and restore original after test."""
    original_api_key = os.environ.get("UPSTAGE_API_KEY")
    test_api_key = "test_api_key_12345"
    os.environ["UPSTAGE_API_KEY"] = test_api_key

    yield test_api_key

    if original_api_key is not None:
        os.environ["UPSTAGE_API_KEY"] = original_api_key
    else:
        os.environ.pop("UPSTAGE_API_KEY", None)


@pytest.fixture
def no_api_key() -> Generator[None, None, None]:
    """Temporarily remove UPSTAGE_API_KEY and restore after test."""
    original_api_key = os.environ.get("UPSTAGE_API_KEY")

    # Remove the API key
    if "UPSTAGE_API_KEY" in os.environ:
        del os.environ["UPSTAGE_API_KEY"]

    yield

    # Restore the original API key
    if original_api_key is not None:
        os.environ["UPSTAGE_API_KEY"] = original_api_key
    else:
        os.environ.pop("UPSTAGE_API_KEY", None)
