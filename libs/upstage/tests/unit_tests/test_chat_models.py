import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_message_to_dict,
)

from langchain_upstage import ChatUpstage

EXAMPLE_PDF_PATH = Path(__file__).parent.parent / "examples/solar.pdf"


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatUpstage()


def test_upstage_model_param() -> None:
    llm = ChatUpstage(model="foo")
    assert llm.model_name == "foo"
    llm = ChatUpstage(model_name="foo")  # type: ignore
    assert llm.model_name == "foo"
    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "upstage"


def test_function_dict_to_message_function_message() -> None:
    content = json.dumps({"result": "Example #1"})
    name = "test_function"
    result = _convert_dict_to_message(
        {
            "role": "function",
            "name": name,
            "content": content,
        }
    )
    assert isinstance(result, FunctionMessage)
    assert result.name == name
    assert result.content == content


def test_convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test__convert_dict_to_message_human_with_name() -> None:
    message = {"role": "user", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_ai_with_name() -> None:
    message = {"role": "assistant", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = AIMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_system_with_name() -> None:
    message = {"role": "system", "content": "foo", "name": "test"}
    result = _convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo", name="test")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


def test_convert_dict_to_message_tool() -> None:
    message = {"role": "tool", "content": "foo", "tool_call_id": "bar"}
    result = _convert_dict_to_message(message)
    expected_output = ToolMessage(content="foo", tool_call_id="bar")
    assert result == expected_output
    assert _convert_message_to_dict(expected_output) == message


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


def test_upstage_invoke(mock_completion: dict) -> None:
    llm = ChatUpstage()
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        res = llm.invoke("bab")
        assert res.content == "Bab"
    assert completed


def test_upstage_invoke_with_doc_parsing_model(mock_completion: dict) -> None:
    llm = ChatUpstage(model="solar-pro2")
    mock_client = MagicMock()
    completed = False

    def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "client",
        mock_client,
    ), patch(
        "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
        return_value=[MagicMock(page_content="test")],
    ):
        res = llm.invoke("bab", file_path=EXAMPLE_PDF_PATH)
        assert res.content == "Bab"
    assert completed


async def test_upstage_ainvoke(mock_completion: dict) -> None:
    llm = ChatUpstage()
    mock_client = AsyncMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ):
        res = await llm.ainvoke("bab")
        assert res.content == "Bab"
    assert completed


async def test_upstage_ainvoke_with_doc_parsing_model(mock_completion: dict) -> None:
    llm = ChatUpstage(model="solar-pro2")
    mock_client = AsyncMock()
    completed = False

    async def mock_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal completed
        completed = True
        return mock_completion

    mock_client.create = mock_create
    with patch.object(
        llm,
        "async_client",
        mock_client,
    ), patch(
        "langchain_upstage.chat_models.UpstageDocumentParseLoader.load",
        return_value=[MagicMock(page_content="test")],
    ):
        res = await llm.ainvoke("bab", file_path=EXAMPLE_PDF_PATH)
        assert res.content == "Bab"
    assert completed


def test_upstage_invoke_name(mock_completion: dict) -> None:
    llm = ChatUpstage()

    mock_client = MagicMock()
    mock_client.create.return_value = mock_completion

    with patch.object(
        llm,
        "client",
        mock_client,
    ):
        messages = [
            HumanMessage(content="Foo", name="Zorba"),
        ]
        res = llm.invoke(messages)
        call_args, call_kwargs = mock_client.create.call_args
        assert len(call_args) == 0  # no positional args
        call_messages = call_kwargs["messages"]
        assert len(call_messages) == 1
        assert call_messages[0]["role"] == "user"
        assert call_messages[0]["content"] == "Foo"
        assert call_messages[0]["name"] == "Zorba"

        # check return type has name
        assert res.content == "Bab"
        assert res.name == "KimSolar"


def test_upstage_tokenizer() -> None:
    llm = ChatUpstage(model="solar-mini")
    llm._get_tokenizer()


def test_upstage_tokenizer_get_num_tokens() -> None:
    llm = ChatUpstage(model="solar-mini")
    num_tokens = llm.get_num_tokens_from_messages([HumanMessage(content="Hello World")])
    assert num_tokens == 12


def test_chat_upstage_extra_kwargs() -> None:
    """Test extra kwargs to chat upstage."""
    # Check that foo is saved in extra_kwargs.
    llm = ChatUpstage(foo=3, max_tokens=10)  # type: ignore
    assert llm.max_tokens == 10
    assert llm.model_kwargs == {"foo": 3}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatUpstage(foo=3, model_kwargs={"bar": 2})  # type: ignore
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatUpstage(foo=3, model_kwargs={"foo": 2})  # type: ignore
