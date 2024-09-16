from langchain_upstage import ChatUpstage, UpstageEmbeddings


def test_chat_upstage_secrets() -> None:
    o = ChatUpstage(api_key="foo")  # type: ignore[arg-type]
    s = str(o)
    assert "foo" not in s


def test_upstage_embeddings_secrets() -> None:
    o = UpstageEmbeddings(model="solar-embedding-1-large", api_key="foo")  # type: ignore[arg-type]
    s = str(o)
    assert "foo" not in s
