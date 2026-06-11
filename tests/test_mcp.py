from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gat.mcp import MissingMCPDependencyError, load_mcp_tools
from gat.tools import ToolRegistry


class FakeParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeStdioClient:
    def __init__(self, params: FakeParams) -> None:
        self.params = params

    async def __aenter__(self) -> tuple[str, str]:
        return "read", "write"

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeSession:
    def __init__(self, read_stream: str, write_stream: str) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="echo",
                    description="Echo text.",
                    inputSchema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                )
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
        assert name == "echo"
        return SimpleNamespace(content=[SimpleNamespace(text=arguments["text"])])


def test_load_mcp_tools_exposes_tool_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> Any:
        if name == "mcp":
            return SimpleNamespace(
                ClientSession=FakeSession,
                StdioServerParameters=FakeParams,
            )
        if name == "mcp.client.stdio":
            return SimpleNamespace(stdio_client=FakeStdioClient)
        raise ImportError(name)

    monkeypatch.setattr("gat.mcp.importlib.import_module", fake_import_module)

    specs = load_mcp_tools("python", ["server.py"], name_prefix="local")

    assert len(specs) == 1
    assert specs[0].name == "local_echo"
    assert specs[0].parameters["required"] == ["text"]
    registry = ToolRegistry(specs)
    assert registry.execute("local_echo", {"text": "hello"}) == "hello"


def test_mcp_missing_dependency_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(_: str) -> Any:
        raise ImportError("missing")

    monkeypatch.setattr("gat.mcp.importlib.import_module", fake_import_module)

    match = r"pip install .*gemini-agent-toolkit\[mcp\]"
    with pytest.raises(MissingMCPDependencyError, match=match):
        load_mcp_tools("python")
