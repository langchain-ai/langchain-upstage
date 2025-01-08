"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_upstage import ChatUpstage


class TestUpstageStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUpstage

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "solar-mini",
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "UPSTAGE_API_KEY": "api_key",
                "UPSTAGE_API_BASE": "https://base.com",
            },
            {},
            {
                "upstage_api_key": "api_key",
                "upstage_api_base": "https://base.com",
            },
        )
