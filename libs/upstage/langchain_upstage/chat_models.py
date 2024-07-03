import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from langchain_openai.chat_models.base import BaseChatOpenAI, _convert_message_to_dict
from tokenizers import Tokenizer


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

    model_name: str = Field(default="solar-1-mini-chat", alias="model")
    """Model name to use."""
    upstage_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env are `UPSTAGE_API_KEY` if not provided."""
    upstage_api_base: Optional[str] = Field(
        default="https://api.upstage.ai/v1/solar", alias="base_url"
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
    tokenizer_name: Optional[str] = "upstage/solar-1-mini-tokenizer"
    """huggingface tokenizer name. Solar tokenizer is opened in huggingface https://huggingface.co/upstage/solar-1-mini-tokenizer"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        values["upstage_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "upstage_api_key", "UPSTAGE_API_KEY")
        )
        values["upstage_api_base"] = values["upstage_api_base"] or os.getenv(
            "UPSTAGE_API_BASE"
        )

        client_params = {
            "api_key": (
                values["upstage_api_key"].get_secret_value()
                if values["upstage_api_key"]
                else None
            ),
            "base_url": values["upstage_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }

        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return values

    def _get_tokenizer(self) -> Tokenizer:
        if self.tokenizer_name is None:
            raise Exception("tokenizer_name should be given.")
        return Tokenizer.from_pretrained(self.tokenizer_name)

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text."""
        tokenizer = self._get_tokenizer()
        encode = tokenizer.encode(text, add_special_tokens=False)
        return encode.ids

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for solar model."""
        tokenizer = self._get_tokenizer()
        tokens_per_message = 5  # <|im_start|>{role}\n{message}<|im_end|>
        tokens_prefix = 1  # <|startoftext|>
        tokens_suffix = 3  # <|im_start|>assistant\n

        num_tokens = 0

        num_tokens += tokens_prefix

        messages_dict = [_convert_message_to_dict(m) for m in messages]
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
