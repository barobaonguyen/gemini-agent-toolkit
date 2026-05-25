from __future__ import annotations

import pytest

from gat.tools import ToolRegistry, get_tool_spec, tool


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
