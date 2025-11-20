from __future__ import annotations

import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import Field, SecretStr, model_validator
from tokenizers import Tokenizer
from typing_extensions import Self

from langchain_upstage.document_parse import UpstageDocumentParseLoader

DOC_PARSING_MODEL = ["solar-pro2"]
SOLAR_TOKENIZERS = {
    "solar-pro2": "upstage/solar-pro2-tokenizer",
    "solar-mini": "upstage/solar-1-mini-tokenizer",
}

# Constants for header management
UPSTAGE_CLIENT_HEADER = "x-upstage-client"
UPSTAGE_CLIENT_VALUE = "langchain"
DEFAULT_HEADERS = {UPSTAGE_CLIENT_HEADER: UPSTAGE_CLIENT_VALUE}


class ChatUpstage(BaseChatOpenAI):
    """ChatUpstage chat model.

    To use, you should have the environment variable `UPSTAGE_API_KEY`
    set with your API key or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_upstage import ChatUpstage

            model = ChatUpstage()
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"upstage_api_key": "UPSTAGE_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "upstage"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.upstage_api_base:
            attributes["upstage_api_base"] = self.upstage_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "upstage-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "upstage"
        return params

    model_name: str = Field(default="solar-mini", alias="model")
    """Model name to use."""
    upstage_api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "UPSTAGE_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `UPSTAGE_API_KEY`."
            ),
        ),
        alias="api_key",
    )
    """Automatically inferred from env are `UPSTAGE_API_KEY` if not provided."""
    upstage_api_base: Optional[str] = Field(
        default_factory=from_env(
            "UPSTAGE_API_BASE", default="https://api.upstage.ai/v1/solar"
        ),
        alias="base_url",
    )
    """Base URL path for API requests, leave blank if not using a proxy or service 
    emulator."""
    openai_api_key: Optional[SecretStr] = Field(default=None)
    """openai api key is not supported for upstage. use `upstage_api_key` instead."""
    openai_api_base: Optional[str] = Field(default=None)
    """openai api base is not supported for upstage. use `upstage_api_base` instead."""
    openai_organization: Optional[str] = Field(default=None)
    """openai organization is not supported for upstage."""
    tiktoken_model_name: Optional[str] = None
    """tiktoken is not supported for upstage."""
    tokenizer_name: Optional[str] = "upstage/solar-pro2-tokenizer"
    """huggingface tokenizer name. Solar tokenizer is opened in huggingface https://huggingface.co/upstage/solar-pro-tokenizer"""
    default_headers: Union[Mapping[str, str], None] = DEFAULT_HEADERS
    """add trace header."""

    disabled_params: dict[str, Any] = Field(
        default_factory=lambda: {"parallel_tool_calls": None}
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        # Ensure x-upstage-client header is always set to "langchain"
        if self.default_headers is None:
            self.default_headers = DEFAULT_HEADERS
        else:
            # Create a copy to avoid modifying the original
            headers = dict(self.default_headers)
            headers[UPSTAGE_CLIENT_HEADER] = UPSTAGE_CLIENT_VALUE
            self.default_headers = headers

        client_params: dict = {
            "api_key": (
                self.upstage_api_key.get_secret_value()
                if self.upstage_api_key
                else None
            ),
            "base_url": self.upstage_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if not self.root_client:
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)

        if not self.client:
            self.client = self.root_client.chat.completions

        if not self.root_async_client:
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            )

        if not self.async_client:
            self.async_client = self.root_async_client.chat.completions

        return self

    def _get_tokenizer(self) -> Tokenizer:
        self.tokenizer_name = SOLAR_TOKENIZERS.get(self.model_name, self.tokenizer_name)
        return Tokenizer.from_pretrained(self.tokenizer_name)

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text."""
        tokenizer = self._get_tokenizer()
        encode = tokenizer.encode(text, add_special_tokens=False)
        return encode.ids

    def get_num_tokens_from_messages(
        self, messages: Sequence[BaseMessage], tools: Sequence[Any] | None = None
    ) -> int:
        """Calculate num tokens for solar model."""
        tokenizer = self._get_tokenizer()
        tokens_per_message = 5  # <|im_start|>{role}\n{message}<|im_end|>
        tokens_prefix = 1  # <|startoftext|>
        tokens_suffix = 3  # <|im_start|>assistant\n

        num_tokens = 0

        num_tokens += tokens_prefix

        messages_dict = convert_to_openai_messages(messages)
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(
                    tokenizer.encode(str(value), add_special_tokens=False)
                )
        # every reply is primed with <|im_start|>assistant
        num_tokens += tokens_suffix
        return num_tokens

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._using_doc_parsing_model(kwargs):
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(HumanMessage(document_contents))

        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._using_doc_parsing_model(kwargs):
            document_contents = self._parse_documents(kwargs.pop("file_path"))
            messages.append(HumanMessage(document_contents))

        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def _using_doc_parsing_model(self, kwargs: Dict[str, Any]) -> bool:
        if "file_path" in kwargs:
            if self.model_name in DOC_PARSING_MODEL:
                return True
            raise ValueError("file_path is not supported for this model.")
        return False

    def _parse_documents(self, file_path: str) -> str:
        document_contents = "Documents:\n"

        loader = UpstageDocumentParseLoader(
            api_key=(
                self.upstage_api_key.get_secret_value()
                if self.upstage_api_key
                else None
            ),
            file_path=file_path,
            output_format="text",
            coordinates=False,
        )
        docs = loader.load()

        if isinstance(file_path, list):
            file_titles = [os.path.basename(path) for path in file_path]
        else:
            file_titles = [os.path.basename(file_path)]

        for i, doc in enumerate(docs):
            file_title = file_titles[min(i, len(file_titles) - 1)]
            document_contents += f"{file_title}:\n{doc.page_content}\n\n"
        return document_contents
