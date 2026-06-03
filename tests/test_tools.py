from __future__ import annotations

import pytest

from gat.tools import ToolRegistry, get_tool_spec, google_search_grounding_tool, tool


@tool
def fetch_token(symbol: str, include_history: bool = False) -> dict[str, object]:
    """Fetch token info from DexScreener.

    Args:
        symbol: Token symbol like PEPE.
        include_history: Include 24h price history.
    """
    return {"symbol": symbol, "history": include_history}


def test_tool_decorator_extracts_schema() -> None:
    spec = get_tool_spec(fetch_token)
    assert spec.name == "fetch_token"
    assert spec.description == "Fetch token info from DexScreener."
    assert spec.parameters["required"] == ["symbol"]
    assert spec.parameters["properties"]["symbol"]["type"] == "string"
    assert spec.parameters["properties"]["symbol"]["description"] == "Token symbol like PEPE."
    assert spec.parameters["properties"]["include_history"]["type"] == "boolean"


def test_registry_execute_and_declarations() -> None:
    registry = ToolRegistry([fetch_token])
    assert registry.execute("fetch_token", {"symbol": "PEPE"}) == {
        "symbol": "PEPE",
        "history": False,
    }
    declarations = registry.function_declarations()
    assert declarations[0]["name"] == "fetch_token"


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError):
        ToolRegistry([fetch_token, fetch_token])


def test_tool_decorator_with_custom_name() -> None:
    @tool(name="lookup")
    def local_lookup(query: str) -> str:
        """Lookup a value."""
        return query

    spec = get_tool_spec(local_lookup)
    assert spec.name == "lookup"
    assert spec(query="x") == "x"


class Usage:
    prompt_token_count = 100
    candidates_token_count = 25
    cached_content_token_count = 0


class Web:
    uri = "https://example.com/source"
    title = "Example Source"


class GroundingChunk:
    web = Web()


class GroundingMetadata:
    web_search_queries = ["test query"]
    grounding_chunks = [GroundingChunk()]


class Candidate:
    grounding_metadata = GroundingMetadata()


class GroundedResponse:
    text = "grounded answer"
    usage_metadata = Usage()
    candidates = [Candidate()]


class FakeGroundedModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> GroundedResponse:
        self.calls.append(kwargs)
        return GroundedResponse()


class FakeGroundedSdk:
    def __init__(self) -> None:
        self.models = FakeGroundedModels()


def test_google_search_grounding_tool_records_cost_and_sources() -> None:
    from gat import GeminiClient

    client = GeminiClient(model="gemini-2.5-flash")
    fake = FakeGroundedSdk()
    client._client = fake
    search = google_search_grounding_tool(client)

    result = search("test query")

    assert get_tool_spec(search).name == "google_search"
    assert result == {
        "answer": "grounded answer",
        "queries": ["test query"],
        "sources": [{"title": "Example Source", "uri": "https://example.com/source"}],
    }
    assert client.cost_tracker.summary()["calls"] == 1
    config = fake.models.calls[0]["config"]
    tools = config["tools"] if isinstance(config, dict) else config.tools
    assert tools
