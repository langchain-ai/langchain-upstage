from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
)

from langchain_upstage import ChatUpstage

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_initialization() -> None:
    """Test chat model initialization."""
    llm = ChatUpstage()
    assert llm.model_name == "solar-mini"


def test_upstage_model_param() -> None:
    """Test that model parameter can be set using 'model' alias."""
    llm = ChatUpstage(model="foo")
    assert llm.model_name == "foo"


def test_upstage_model_name_param() -> None:
    """Test that model parameter can be set using 'model_name' alias."""
    llm = ChatUpstage(model_name="foo")  # type: ignore[call-arg]
    assert llm.model_name == "foo"


@pytest.fixture
def mock_completion() -> dict:
    return {
        "id": "chatcmpl-7fcZavknQda3SQ",
        "object": "chat.completion",
        "created": 1689989000,
        "model": "solar-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Bab",
                    "name": "KimSolar",
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_upstage_invoke_response(mock_completion: dict) -> None:
    """Test that synchronous invoke returns correct response content."""
    llm = ChatUpstage()

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bab")
        assert res.content == "Bab"


def test_upstage_invoke_api_call(mock_completion: dict) -> None:
    """Test that synchronous invoke calls API with correct parameters."""
    llm = ChatUpstage()

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        llm.invoke("bab")

        # Verify that create was called
        assert mock_create_method.called
        call_args, call_kwargs = mock_create_method.call_args
        assert len(call_args) == 0  # no positional args
        assert "messages" in call_kwargs


def test_upstage_invoke_message_format(mock_completion: dict) -> None:
    """Test that synchronous invoke formats messages correctly."""
    llm = ChatUpstage()

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        llm.invoke("bab")

        call_args, call_kwargs = mock_create_method.call_args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "bab"


def test_upstage_invoke_doc_parsing_response(mock_completion: dict) -> None:
    """Test that invoke with document parsing model returns correct response."""
    llm = ChatUpstage(model="solar-pro2")

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    mock_document = Document(page_content="test document content", metadata={})

    with (
        patch.object(
            llm,
            "client",
            mock_client,
        ),
        patch(
            "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
            return_value=[mock_document],
        ),
    ):
        res = llm.invoke("bab", file_path=EXAMPLE_PDF_PATH)
        assert res.content == "Bab"


def test_upstage_invoke_doc_parsing_loader_called(mock_completion: dict) -> None:
    """Test that invoke with document parsing model calls document loader."""
    llm = ChatUpstage(model="solar-pro2")

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    mock_document = Document(page_content="test document content", metadata={})

    with (
        patch.object(
            llm,
            "client",
            mock_client,
        ),
        patch(
            "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
            return_value=[mock_document],
        ) as mock_loader,
    ):
        llm.invoke("bab", file_path=EXAMPLE_PDF_PATH)

        # Verify document loader was called
        mock_loader.assert_called_once()


def test_upstage_invoke_doc_parsing_message_content(
    mock_completion: dict,
) -> None:
    """Test that invoke with document parsing model includes document content."""
    llm = ChatUpstage(model="solar-pro2")

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    mock_document = Document(page_content="test document content", metadata={})

    with (
        patch.object(
            llm,
            "client",
            mock_client,
        ),
        patch(
            "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
            return_value=[mock_document],
        ),
    ):
        llm.invoke("bab", file_path=EXAMPLE_PDF_PATH)

        # Verify that messages include document content
        assert mock_create_method.called
        call_args, call_kwargs = mock_create_method.call_args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) >= 1
        last_message = call_messages[-1]
        assert last_message["role"] == "user"
        assert "test document content" in last_message["content"]


async def test_upstage_ainvoke_response(mock_completion: dict) -> None:
    """Test that asynchronous invoke returns correct response content."""
    llm = ChatUpstage()

    mock_response = AsyncMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    call_kwargs_captured: dict[str, Any] = {}

    async def mock_create(*args: Any, **kwargs: Any) -> AsyncMock:
        call_kwargs_captured.update(kwargs)
        return mock_response

    mock_with_raw_response = AsyncMock()
    mock_with_raw_response.create = mock_create

    mock_client = AsyncMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.ainvoke("bab")
        assert res.content == "Bab"


async def test_upstage_ainvoke_api_call(mock_completion: dict) -> None:
    """Test that asynchronous invoke calls API with correct parameters."""
    llm = ChatUpstage()

    mock_response = AsyncMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    call_kwargs_captured: dict[str, Any] = {}

    async def mock_create(*args: Any, **kwargs: Any) -> AsyncMock:
        call_kwargs_captured.update(kwargs)
        return mock_response

    mock_with_raw_response = AsyncMock()
    mock_with_raw_response.create = mock_create

    mock_client = AsyncMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        await llm.ainvoke("bab")

        # Verify that create was called with correct parameters
        assert "messages" in call_kwargs_captured


async def test_upstage_ainvoke_message_format(mock_completion: dict) -> None:
    """Test that asynchronous invoke formats messages correctly."""
    llm = ChatUpstage()

    mock_response = AsyncMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    call_kwargs_captured: dict[str, Any] = {}

    async def mock_create(*args: Any, **kwargs: Any) -> AsyncMock:
        call_kwargs_captured.update(kwargs)
        return mock_response

    mock_with_raw_response = AsyncMock()
    mock_with_raw_response.create = mock_create

    mock_client = AsyncMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        await llm.ainvoke("bab")

        call_messages = call_kwargs_captured["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "bab"


async def test_upstage_ainvoke_doc_parsing_response(
    mock_completion: dict,
) -> None:
    """Test that async invoke with document parsing model returns correct response."""
    llm = ChatUpstage(model="solar-pro2")

    mock_response = AsyncMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    call_kwargs_captured: dict[str, Any] = {}

    async def mock_create(*args: Any, **kwargs: Any) -> AsyncMock:
        call_kwargs_captured.update(kwargs)
        return mock_response

    mock_with_raw_response = AsyncMock()
    mock_with_raw_response.create = mock_create

    mock_client = AsyncMock()
    mock_client.with_raw_response = mock_with_raw_response

    mock_document = Document(page_content="test document content", metadata={})

    with (
        patch.object(
            llm,
            "async_client",
            mock_client,
        ),
        patch(
            "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
            return_value=[mock_document],
        ),
    ):
        res = await llm.ainvoke("bab", file_path=EXAMPLE_PDF_PATH)
        assert res.content == "Bab"


async def test_upstage_ainvoke_doc_parsing_loader_called(
    mock_completion: dict,
) -> None:
    """Test that async invoke with document parsing model calls document loader."""
    llm = ChatUpstage(model="solar-pro2")

    mock_response = AsyncMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    call_kwargs_captured: dict[str, Any] = {}

    async def mock_create(*args: Any, **kwargs: Any) -> AsyncMock:
        call_kwargs_captured.update(kwargs)
        return mock_response

    mock_with_raw_response = AsyncMock()
    mock_with_raw_response.create = mock_create

    mock_client = AsyncMock()
    mock_client.with_raw_response = mock_with_raw_response

    mock_document = Document(page_content="test document content", metadata={})

    with (
        patch.object(
            llm,
            "async_client",
            mock_client,
        ),
        patch(
            "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
            return_value=[mock_document],
        ) as mock_loader,
    ):
        await llm.ainvoke("bab", file_path=EXAMPLE_PDF_PATH)

        # Verify document loader was called
        mock_loader.assert_called_once()


async def test_upstage_ainvoke_doc_parsing_message_content(
    mock_completion: dict,
) -> None:
    """Test that async invoke with document parsing model includes document content."""
    llm = ChatUpstage(model="solar-pro2")

    mock_response = AsyncMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    call_kwargs_captured: dict[str, Any] = {}

    async def mock_create(*args: Any, **kwargs: Any) -> AsyncMock:
        call_kwargs_captured.update(kwargs)
        return mock_response

    mock_with_raw_response = AsyncMock()
    mock_with_raw_response.create = mock_create

    mock_client = AsyncMock()
    mock_client.with_raw_response = mock_with_raw_response

    mock_document = Document(page_content="test document content", metadata={})

    with (
        patch.object(
            llm,
            "async_client",
            mock_client,
        ),
        patch(
            "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
            return_value=[mock_document],
        ),
    ):
        await llm.ainvoke("bab", file_path=EXAMPLE_PDF_PATH)

        # Verify that messages include document content
        assert "messages" in call_kwargs_captured
        call_messages = call_kwargs_captured["messages"]
        assert len(call_messages) >= 1
        last_message = call_messages[-1]
        assert last_message["role"] == "user"
        assert "test document content" in last_message["content"]


def test_upstage_invoke_input_name(mock_completion: dict) -> None:
    """Test that input message name is correctly passed to API call."""
    llm = ChatUpstage()

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        messages = [
            HumanMessage(content="Foo", name="Zorba"),
        ]
        llm.invoke(messages)

        call_args, call_kwargs = mock_create_method.call_args
        assert len(call_args) == 0  # no positional args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Foo"
        assert call_messages[0]["name"] == "Zorba"


def test_upstage_invoke_response_name(mock_completion: dict) -> None:
    """Test that response message includes name field."""
    llm = ChatUpstage()

    mock_response = MagicMock()
    mock_response.parse = MagicMock(return_value=mock_completion)

    mock_with_raw_response = MagicMock()
    mock_create_method = MagicMock(return_value=mock_response)
    mock_with_raw_response.create = mock_create_method

    mock_client = MagicMock()
    mock_client.with_raw_response = mock_with_raw_response

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        messages = [
            HumanMessage(content="Foo", name="Zorba"),
        ]
        res = llm.invoke(messages)

        # Check return type has name
        assert res.content == "Bab"
        assert res.name == "KimSolar"


def test_upstage_tokenizer_solar_mini() -> None:
    """Test tokenizer retrieval for solar-mini model."""
    llm = ChatUpstage(model="solar-mini")
    # Verify tokenizer works correctly through public API
    # This will internally call _get_tokenizer() and update tokenizer_name
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="test")])
    assert num_tokens > 0
    # Verify tokenizer_name is set correctly after tokenizer initialization
    assert llm.tokenizer_name == "upstage/solar-1-mini-tokenizer"


def test_upstage_tokenizer_solar_pro2() -> None:
    """Test tokenizer retrieval for solar-pro2 model."""
    llm = ChatUpstage(model="solar-pro2")
    # Verify tokenizer works correctly through public API
    # This will internally call _get_tokenizer() and update tokenizer_name
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="test")])
    assert num_tokens > 0
    # Verify tokenizer_name is set correctly after tokenizer initialization
    assert llm.tokenizer_name == "upstage/solar-pro2-tokenizer"


def test_upstage_tokenizer_default() -> None:
    """Test tokenizer retrieval for default model."""
    llm = ChatUpstage()
    # Verify tokenizer works correctly through public API
    # This will internally call _get_tokenizer() and update tokenizer_name
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="test")])
    assert num_tokens > 0
    # Verify tokenizer_name is set correctly after tokenizer initialization
    # Default model is "solar-mini", so tokenizer should be solar-1-mini-tokenizer
    assert llm.tokenizer_name == "upstage/solar-1-mini-tokenizer"


def test_get_num_tokens_basic() -> None:
    """Test token counting for basic message."""
    llm = ChatUpstage(model="solar-mini")
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="Hello World")])
    # Token count should be positive and reasonable
    # "Hello World" + message formatting overhead (prefix, suffix, role tokens)
    assert num_tokens > 0
    assert num_tokens >= 5  # At least prefix + suffix + some content tokens


def test_get_num_tokens_empty() -> None:
    """Test token counting for empty message."""
    llm = ChatUpstage(model="solar-mini")
    num_tokens_empty = llm.get_num_tokens_from_messages([HumanMessage(content="")])
    assert num_tokens_empty > 0  # Should still have formatting overhead


def test_get_num_tokens_longer_message() -> None:
    """Test that longer messages have more tokens than shorter ones."""
    llm = ChatUpstage(model="solar-mini")
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="Hello World")])
    num_tokens_long = llm.get_num_tokens_from_messages(
        [HumanMessage(content="Hello World " * 10)]
    )
    assert num_tokens_long > num_tokens


def test_get_num_tokens_multiple_messages() -> None:
    """Test that multiple messages have more tokens than single message."""
    llm = ChatUpstage(model="solar-mini")
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="Hello World")])
    num_tokens_multi = llm.get_num_tokens_from_messages(
        [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
        ]
    )
    assert num_tokens_multi > num_tokens


def test_chat_upstage_extra_kwargs_storage() -> None:
    """Test that extra kwargs are saved in model_kwargs."""
    # Using **kwargs to avoid type checker issues while testing runtime behavior
    llm = ChatUpstage(max_tokens=10, **{"foo": 3})  # type: ignore[arg-type]
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}


def test_chat_upstage_extra_kwargs_merge() -> None:
    """Test that extra kwargs are merged with existing model_kwargs."""
    # Using **kwargs to avoid type checker issues while testing runtime behavior
    llm = ChatUpstage(model_kwargs={"bar": 2}, **{"foo": 3})  # type: ignore[arg-type]
    assert llm.model_kwargs == {"foo": 3, "bar": 2}


def test_chat_upstage_extra_kwargs_duplicate_error() -> None:
    """Test that duplicate keys in model_kwargs and extra kwargs raise error."""
    # Using **kwargs to avoid type checker issues while testing runtime behavior
    with pytest.raises(ValueError):
        ChatUpstage(model_kwargs={"foo": 2}, **{"foo": 3})  # type: ignore[arg-type]


def _make_mock_response(message_attrs: dict, response_dict: dict) -> MagicMock:
    """Helper to create a mock openai.BaseModel response for _create_chat_result."""
    mock_message = MagicMock()
    for k, v in message_attrs.items():
        setattr(mock_message, k, v)

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.index = 0
    mock_choice.finish_reason = "stop"
    mock_choice.logprobs = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = response_dict
    return mock_response


def test_reasoning_content_non_streaming() -> None:
    """Test that reasoning_content is extracted from non-streaming responses."""
    import openai

    llm = ChatUpstage()

    response_dict = {
        "id": "chatcmpl-test",
        "model": "solar-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 42."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_response = _make_mock_response(
        {
            "reasoning_content": "Let me think step by step...",
            "content": "The answer is 42.",
        },
        response_dict,
    )

    with patch.object(openai, "BaseModel", type(mock_response)):
        result = llm._create_chat_result(mock_response)

    assert result.generations[0].message.content == "The answer is 42."
    assert (
        result.generations[0].message.additional_kwargs["reasoning_content"]
        == "Let me think step by step..."
    )


def test_reasoning_content_none_not_stored() -> None:
    """Test that None reasoning_content is not stored in additional_kwargs."""
    import openai

    llm = ChatUpstage()

    response_dict = {
        "id": "chatcmpl-test",
        "model": "solar-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response without reasoning.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_response = _make_mock_response(
        {"reasoning_content": None, "content": "Response without reasoning."},
        response_dict,
    )

    with patch.object(openai, "BaseModel", type(mock_response)):
        result = llm._create_chat_result(mock_response)

    assert "reasoning_content" not in result.generations[0].message.additional_kwargs


def test_reasoning_content_model_extra_fallback() -> None:
    """Test reasoning_content extraction from model_extra (OpenRouter format)."""
    import openai

    llm = ChatUpstage()

    response_dict = {
        "id": "chatcmpl-test",
        "model": "solar-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The answer."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    mock_message = MagicMock(
        spec=["content", "role", "function_call", "tool_calls", "audio", "model_extra"]
    )
    mock_message.content = "The answer."
    mock_message.role = "assistant"
    mock_message.function_call = None
    mock_message.tool_calls = None
    mock_message.audio = None
    mock_message.model_extra = {"reasoning": "OpenRouter reasoning text"}

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.index = 0
    mock_choice.finish_reason = "stop"
    mock_choice.logprobs = None
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = response_dict

    with patch.object(openai, "BaseModel", type(mock_response)):
        result = llm._create_chat_result(mock_response)

    assert (
        result.generations[0].message.additional_kwargs["reasoning_content"]
        == "OpenRouter reasoning text"
    )


def test_reasoning_content_streaming() -> None:
    """Test that reasoning_content is extracted from streaming chunks."""
    from langchain_core.messages import AIMessageChunk

    llm = ChatUpstage()

    chunk = {
        "id": "chatcmpl-test",
        "model": "solar-mini",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "partial",
                    "reasoning_content": "thinking step 1",
                },
                "finish_reason": None,
            }
        ],
    }

    result = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert result is not None
    assert result.message.additional_kwargs["reasoning_content"] == "thinking step 1"


def test_reasoning_content_streaming_openrouter() -> None:
    """Test that reasoning field (OpenRouter) is extracted from streaming chunks."""
    from langchain_core.messages import AIMessageChunk

    llm = ChatUpstage()

    chunk = {
        "id": "chatcmpl-test",
        "model": "solar-mini",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "partial",
                    "reasoning": "openrouter thinking",
                },
                "finish_reason": None,
            }
        ],
    }

    result = llm._convert_chunk_to_generation_chunk(chunk, AIMessageChunk, None)

    assert result is not None
    assert (
        result.message.additional_kwargs["reasoning_content"] == "openrouter thinking"
    )
