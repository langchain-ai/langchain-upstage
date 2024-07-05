import os
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    overload,
)

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models.base import (
    BaseChatOpenAI,
    _AllReturnType,
    _convert_message_to_dict,
    _DictOrPydantic,
    _DictOrPydanticClass,
    _is_pydantic_class,
)
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

    # TODO: Fix typing.
    @overload  # type: ignore[override]
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[True] = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _AllReturnType]:
        ...

    @overload
    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: Literal[False] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        ...

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec or be a valid JSON schema
                with top level 'title' and 'description' keys specified.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_upstage import ChatUpstage
                from langchain_core.pydantic_v1 import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatUpstage(model="solar-1-mini-chat", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_upstage import ChatUpstage
                from langchain_core.pydantic_v1 import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatUpstage(model="solar-1-mini-chat", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_upstage import ChatUpstage
                from langchain_core.pydantic_v1 import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatUpstage(model="solar-1-mini-chat", temperature=0)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if schema is None:
            raise ValueError("schema must be specified. Received None.")
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        llm = self.bind_tools([schema], tool_choice=tool_name)
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name, first_tool_only=True
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with Upstage tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Options are:
                name of the tool (str): calls corresponding tool;
                "auto": automatically selects a tool (including no tool);
                "none": does not call a tool;
                True: forces tool call (requires `tools` be length 1);
                False: no effect;
                or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice in ("any", "required", "auto"):
                    tool_choice = "auto"
                elif tool_choice == "none":
                    tool_choice = "none"
                else:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

            elif isinstance(tool_choice, bool):
                tool_choice = "auto"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
