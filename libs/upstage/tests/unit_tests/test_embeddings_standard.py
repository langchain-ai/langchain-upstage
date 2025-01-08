"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests

from langchain_upstage import UpstageEmbeddings


class TestUpstageStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[Embeddings]:
        return UpstageEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model": "solar-embedding-1-large",
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "UPSTAGE_API_KEY": "api_key",
                "UPSTAGE_API_BASE": "https://base.com",
            },
            {
                "model": "solar-embedding-1-large",
            },
            {
                "upstage_api_key": "api_key",
                "upstage_api_base": "https://base.com",
            },
        )
