"""MCP stdio adapter for GAT tools."""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import threading
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from gat.tools import ToolSpec


class MissingMCPDependencyError(RuntimeError):
    """Raised when MCP helpers are used without installing the optional extra."""


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for a stdio MCP server process."""

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    name_prefix: str | None = None


class MCPToolAdapter:
    """List MCP tools from a stdio server and expose them as ``gat`` tools."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config

    def list_tools(self) -> list[ToolSpec]:
        """Return GAT ``ToolSpec`` objects for the configured MCP server."""

        return cast(list[ToolSpec], _run_blocking(self.alist_tools()))

    async def alist_tools(self) -> list[ToolSpec]:
        """Async variant of :meth:`list_tools`."""

        return await _alist_mcp_tools(self.config)


def load_mcp_tools(
    command: str,
    args: Sequence[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    name_prefix: str | None = None,
) -> list[ToolSpec]:
    """Load MCP stdio server tools as GAT ``ToolSpec`` objects."""

    config = MCPServerConfig(
        command=command,
        args=tuple(args),
        env=env,
        cwd=cwd,
        name_prefix=name_prefix,
    )
    return MCPToolAdapter(config).list_tools()


async def aload_mcp_tools(
    command: str,
    args: Sequence[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    name_prefix: str | None = None,
) -> list[ToolSpec]:
    """Async variant of :func:`load_mcp_tools`."""

    config = MCPServerConfig(
        command=command,
        args=tuple(args),
        env=env,
        cwd=cwd,
        name_prefix=name_prefix,
    )
    return await MCPToolAdapter(config).alist_tools()


async def _alist_mcp_tools(config: MCPServerConfig) -> list[ToolSpec]:
    mcp = _mcp_imports()
    params = _stdio_params(mcp["StdioServerParameters"], config)
    async with mcp["stdio_client"](params) as streams:
        read_stream, write_stream = streams
        async with mcp["ClientSession"](read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

    tools = _value(result, "tools")
    if tools is None:
        tools = result
    if not isinstance(tools, list):
        return []

    specs: list[ToolSpec] = []
    for item in tools:
        original_name = _string_value(item, "name")
        if not original_name:
            continue
        tool_name = _public_tool_name(original_name, config.name_prefix)
        description = _string_value(item, "description") or f"Call MCP tool {original_name}."
        parameters = _parameters_schema(item)
        specs.append(
            ToolSpec(
                name=tool_name,
                description=description,
                parameters=parameters,
                func=_make_mcp_callable(config, original_name),
            )
        )
    return specs


async def _call_mcp_tool(config: MCPServerConfig, name: str, args: Mapping[str, Any]) -> Any:
    mcp = _mcp_imports()
    params = _stdio_params(mcp["StdioServerParameters"], config)
    async with mcp["stdio_client"](params) as streams:
        read_stream, write_stream = streams
        async with mcp["ClientSession"](read_stream, write_stream) as session:
            await session.initialize()
            try:
                result = await session.call_tool(name, arguments=dict(args))
            except TypeError:
                result = await session.call_tool(name, dict(args))
    return _serialize_mcp_result(result)


def _make_mcp_callable(config: MCPServerConfig, name: str) -> Any:
    def call_mcp_tool(**kwargs: Any) -> Any:
        return _run_blocking(_call_mcp_tool(config, name, kwargs))

    call_mcp_tool.__name__ = _identifier(_public_tool_name(name, config.name_prefix))
    call_mcp_tool.__doc__ = f"Call MCP tool {name}."
    return call_mcp_tool


def _mcp_imports() -> dict[str, Any]:
    try:
        mcp_module = importlib.import_module("mcp")
        stdio_module = importlib.import_module("mcp.client.stdio")
    except ImportError as exc:
        raise MissingMCPDependencyError(
            "MCP support requires the optional dependency. "
            "Install with `pip install gemini-agent-toolkit[mcp]`."
        ) from exc

    session = getattr(mcp_module, "ClientSession", None)
    params = getattr(mcp_module, "StdioServerParameters", None)
    if params is None:
        params = getattr(stdio_module, "StdioServerParameters", None)
    stdio_client = getattr(stdio_module, "stdio_client", None)
    if session is None or params is None or stdio_client is None:
        raise MissingMCPDependencyError("Installed `mcp` package is missing stdio client APIs.")
    return {
        "ClientSession": session,
        "StdioServerParameters": params,
        "stdio_client": stdio_client,
    }


def _stdio_params(params_type: Any, config: MCPServerConfig) -> Any:
    kwargs: dict[str, Any] = {"command": config.command, "args": list(config.args)}
    if config.env is not None:
        kwargs["env"] = dict(config.env)
    if config.cwd is not None:
        kwargs["cwd"] = config.cwd
    try:
        return params_type(**kwargs)
    except TypeError:
        kwargs.pop("cwd", None)
        return params_type(**kwargs)


def _parameters_schema(item: Any) -> dict[str, Any]:
    schema = _value(item, "inputSchema", "input_schema")
    if isinstance(schema, dict) and schema.get("type") == "object":
        return dict(schema)
    if isinstance(schema, dict):
        return {"type": "object", "properties": dict(schema)}
    return {"type": "object", "properties": {}}


def _serialize_mcp_result(result: Any) -> Any:
    structured = _value(result, "structuredContent", "structured_content")
    if structured is not None:
        return structured

    content = _value(result, "content")
    if isinstance(content, list):
        serialized = [_serialize_content(item) for item in content]
        if len(serialized) == 1:
            only = serialized[0]
            if isinstance(only, dict) and set(only) == {"text"}:
                return only["text"]
        return serialized
    return _jsonable(result)


def _serialize_content(item: Any) -> Any:
    text = _value(item, "text")
    if isinstance(text, str):
        return {"text": text}
    return _jsonable(item)


def _jsonable(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _public_tool_name(name: str, prefix: str | None) -> str:
    clean = _identifier(name)
    if prefix:
        return f"{_identifier(prefix)}_{clean}"
    return clean


def _identifier(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not clean:
        return "mcp_tool"
    if clean[0].isdigit():
        return f"mcp_{clean}"
    return clean


def _string_value(obj: Any, *names: str) -> str:
    value = _value(obj, *names)
    return value if isinstance(value, str) else ""


def _value(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _run_blocking(awaitable: Coroutine[Any, Any, Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - defensive handoff
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        error = result["error"]
        if isinstance(error, BaseException):
            raise error
    return result.get("value")
