"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_upstage import ChatUpstage


class TestUpstageStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatUpstage

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "solar-pro2",
        }

    @property
    def has_tool_choice(self) -> bool:
        """Upstage API tool_choice support status.
        
        Note: The Upstage API does support tool_choice parameters and actually
        calls tools correctly. However, there is a known bug where the API
        returns an incorrect finish_reason value ('stop' instead of 'tool_calls')
        when tool_choice is set to 'required' or a specific tool name.
        
        The API team is aware of this issue and is currently working on a fix.
        Once fixed, this property should be changed to return True.
        
        Therefore, we skip the test_tool_choice test until the API bug is resolved.
        """
        return False

    @pytest.mark.xfail(reason="Not implemented.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)
