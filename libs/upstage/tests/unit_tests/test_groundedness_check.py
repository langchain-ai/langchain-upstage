import os
from typing import Generator
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.documents import Document

from langchain_upstage import UpstageGroundednessCheck


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


@pytest.fixture
def groundedness_tool(mock_api_key: str) -> UpstageGroundednessCheck:
    """Create a groundedness tool instance for testing."""
    return UpstageGroundednessCheck()


@pytest.fixture
def mock_api_wrapper() -> Mock:
    """Create a mock API wrapper for testing."""
    return Mock()


class TestUpstageGroundednessCheckInitialization:
    """Test UpstageGroundednessCheck initialization behavior."""

    def test_initialization_without_api_key_raises_error(
        self, no_api_key: None
    ) -> None:
        """Test that initialization fails when no API key is provided."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstageGroundednessCheck()

    def test_initialization_with_empty_api_key_raises_error(
        self, no_api_key: None
    ) -> None:
        """Test that initialization fails with empty API key."""
        # Act & Assert
        with pytest.raises(ValueError):
            UpstageGroundednessCheck(api_key="")


class TestUpstageGroundednessCheckCoreLogic:
    """Test UpstageGroundednessCheck core business logic."""

    def test_format_documents_as_string(self) -> None:
        """Test the format_documents_as_string function."""
        # Arrange
        from langchain_upstage.tools.groundedness_check import (
            format_documents_as_string,
        )

        docs = [
            Document(page_content="First document content"),
            Document(page_content="Second document content"),
            Document(page_content="Third document content"),
        ]

        # Act
        result = format_documents_as_string(docs)

        # Assert
        expected = (
            "First document content\nSecond document content\nThird document content"
        )
        assert result == expected

    def test_format_documents_as_string_empty_list(self) -> None:
        """Test format_documents_as_string with empty list."""
        # Arrange
        from langchain_upstage.tools.groundedness_check import (
            format_documents_as_string,
        )

        # Act
        result = format_documents_as_string([])

        # Assert
        assert result == ""

    def test_run_with_string_context_calls_api(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that run method calls the API wrapper."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "grounded"
        mock_api_wrapper.invoke.return_value = mock_response
        groundedness_tool.api_wrapper = mock_api_wrapper

        # Act
        result = groundedness_tool.run(
            {"context": "This is the context", "answer": "This is the answer"}
        )

        # Assert
        assert result == "grounded"
        mock_api_wrapper.invoke.assert_called_once()

    def test_run_with_document_list_context_formats_documents(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that run method properly formats document list context."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "notGrounded"
        mock_api_wrapper.invoke.return_value = mock_response
        groundedness_tool.api_wrapper = mock_api_wrapper

        docs = [Document(page_content="First doc"), Document(page_content="Second doc")]

        # Act
        result = groundedness_tool.run(
            {"context": docs, "answer": "This is the answer"}
        )

        # Assert
        assert result == "notGrounded"
        mock_api_wrapper.invoke.assert_called_once()

    def test_run_with_empty_context_calls_api(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that run method handles empty context."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "notSure"
        mock_api_wrapper.invoke.return_value = mock_response
        groundedness_tool.api_wrapper = mock_api_wrapper

        # Act
        result = groundedness_tool.run({"context": "", "answer": "This is the answer"})

        # Assert
        assert result == "notSure"
        mock_api_wrapper.invoke.assert_called_once()

    def test_run_with_empty_answer_calls_api(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that run method handles empty answer."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "notGrounded"
        mock_api_wrapper.invoke.return_value = mock_response
        groundedness_tool.api_wrapper = mock_api_wrapper

        # Act
        result = groundedness_tool.run({"context": "This is the context", "answer": ""})

        # Assert
        assert result == "notGrounded"
        mock_api_wrapper.invoke.assert_called_once()

    async def test_arun_with_string_context_calls_api(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that arun method calls the API wrapper."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "notSure"
        mock_api_wrapper.ainvoke = AsyncMock(return_value=mock_response)
        groundedness_tool.api_wrapper = mock_api_wrapper

        # Act
        result = await groundedness_tool.arun(
            {"context": "This is the context", "answer": "This is the answer"}
        )

        # Assert
        assert result == "notSure"
        mock_api_wrapper.ainvoke.assert_called_once()

    def test_run_returns_api_response_content(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that run method returns the API response content."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "custom_response"
        mock_api_wrapper.invoke.return_value = mock_response
        groundedness_tool.api_wrapper = mock_api_wrapper

        # Act
        result = groundedness_tool.run({"context": "context", "answer": "answer"})

        # Assert
        assert result == "custom_response"
        mock_api_wrapper.invoke.assert_called_once()

    async def test_arun_returns_api_response_content(
        self, groundedness_tool: UpstageGroundednessCheck, mock_api_wrapper: Mock
    ) -> None:
        """Test that arun method returns the API response content."""
        # Arrange
        mock_response = Mock()
        mock_response.content = "async_custom_response"
        mock_api_wrapper.ainvoke = AsyncMock(return_value=mock_response)
        groundedness_tool.api_wrapper = mock_api_wrapper

        # Act
        result = await groundedness_tool.arun(
            {"context": "context", "answer": "answer"}
        )

        # Assert
        assert result == "async_custom_response"
        mock_api_wrapper.ainvoke.assert_called_once()
