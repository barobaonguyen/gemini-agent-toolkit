"""Tool decorator and registry for Gemini agents."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, get_type_hints, overload

from gat.retry import retry
from gat.schemas import python_type_to_schema

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        doc = inspect.getdoc(func) or ""
        arg_descriptions = _parse_args_section(doc)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = type_hints.get(param_name, Any)
            schema = python_type_to_schema(annotation)
            if param_name in arg_descriptions:
                schema["description"] = arg_descriptions[param_name]
            properties[param_name] = schema
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required

        return cls(
            name=name or func.__name__,
            description=description or _first_doc_line(doc) or f"Call {func.__name__}.",
            parameters=parameters,
            func=func,
        )

    def to_gemini_function_declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """Name-addressable registry of decorated or plain Python callables."""

    def __init__(self, tools: Iterable[Callable[..., Any] | ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for item in tools:
            self.register(item)

    def register(self, item: Callable[..., Any] | ToolSpec) -> ToolSpec:
        if isinstance(item, ToolSpec):
            spec = item
        else:
            existing = getattr(item, "__gat_tool__", None)
            spec = existing if isinstance(existing, ToolSpec) else ToolSpec.from_callable(item)
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def execute(self, name: str, args: Mapping[str, Any] | None = None) -> Any:
        return self.get(name)(**dict(args or {}))

    async def aexecute(self, name: str, args: Mapping[str, Any] | None = None) -> Any:
        """Execute one tool without blocking the event loop.

        Awaits native coroutine tools directly; runs sync tools in a worker
        thread so several can be gathered concurrently.
        """

        spec = self.get(name)
        kwargs = dict(args or {})
        if inspect.iscoroutinefunction(spec.func):
            return await spec.func(**kwargs)
        return await asyncio.to_thread(spec, **kwargs)

    def list(self) -> builtins.list[ToolSpec]:
        return list(self._tools.values())

    def function_declarations(self) -> builtins.list[dict[str, Any]]:
        return [tool.to_gemini_function_declaration() for tool in self._tools.values()]

    def prompt_block(self) -> str:
        declarations = self.function_declarations()
        if not declarations:
            return "No tools are available."
        lines = ["Available tools:"]
        for declaration in declarations:
            lines.append(
                f"- {declaration['name']}: {declaration['description']} "
                f"parameters={declaration['parameters']}"
            )
        return "\n".join(lines)


@overload
def tool(func: F) -> F: ...


@overload
def tool(
    func: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[F], F]: ...


def tool(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> F | Callable[[F], F]:
    """Decorate a function as a Gemini tool while keeping it directly callable."""

    def decorate(inner: F) -> F:
        spec = ToolSpec.from_callable(inner, name=name, description=description)
        setattr(inner, "__gat_tool__", spec)  # noqa: B010
        return inner

    if func is not None:
        return decorate(func)
    return decorate


def get_tool_spec(func: Callable[..., Any]) -> ToolSpec:
    existing = getattr(func, "__gat_tool__", None)
    if isinstance(existing, ToolSpec):
        return existing
    return ToolSpec.from_callable(func)


def google_search_grounding_tool(
    client: Any,
    *,
    name: str = "google_search",
    system: str | None = None,
    temperature: float = 0.2,
    thinking_budget: int | None = 0,
) -> Callable[[str], dict[str, Any]]:
    """Return an agent-callable Gemini Google Search grounding tool."""

    sdk_tool = _google_search_sdk_tool()

    @tool(
        name=name,
        description="Search Google through Gemini grounding and return an answer with sources.",
    )
    @retry(attempts=3, base_delay_s=1.0)
    def google_search(query: str) -> dict[str, Any]:
        """Search the web with Gemini Google Search grounding.

        Args:
            query: Specific research question to ground with Google Search.
        """

        response = client._ensure_client().models.generate_content(
            model=client.model,
            contents=query,
            config=client._build_config(
                system=system,
                thinking_budget=thinking_budget,
                temperature=temperature,
                tools=[sdk_tool],
            ),
        )
        client._record_usage(response)
        return {
            "answer": client._response_text(response),
            "queries": _grounding_queries(response),
            "sources": _grounding_sources(response),
        }

    return google_search


def _google_search_sdk_tool() -> Any:
    try:
        types = importlib.import_module("google.genai.types")
        return types.Tool(google_search=types.GoogleSearch())
    except Exception:
        return {"google_search": {}}


def _grounding_queries(response: Any) -> list[str]:
    metadata = _grounding_metadata(response)
    queries = _value(metadata, "web_search_queries", "webSearchQueries")
    if isinstance(queries, list):
        return [query for query in queries if isinstance(query, str)]
    return []


def _grounding_sources(response: Any) -> list[dict[str, str]]:
    metadata = _grounding_metadata(response)
    chunks = _value(metadata, "grounding_chunks", "groundingChunks")
    if not isinstance(chunks, list):
        return []

    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        web = _value(chunk, "web")
        uri = _value(web, "uri")
        if not isinstance(uri, str) or uri in seen:
            continue
        title = _value(web, "title")
        sources.append({"title": title if isinstance(title, str) else uri, "uri": uri})
        seen.add(uri)
    return sources


def _grounding_metadata(response: Any) -> Any | None:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    candidate = candidates[0]
    return _value(candidate, "grounding_metadata", "groundingMetadata")


def _value(obj: Any, *names: str) -> Any:
    for field_name in names:
        if isinstance(obj, Mapping) and field_name in obj:
            return obj[field_name]
        value = getattr(obj, field_name, None)
        if value is not None:
            return value
    return None


def _first_doc_line(doc: str) -> str:
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parse_args_section(doc: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    in_args = False
    current_name: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_name, current_parts
        if current_name is not None:
            descriptions[current_name] = " ".join(current_parts).strip()
        current_name = None
        current_parts = []

    for raw_line in doc.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            in_args = True
            continue
        if not in_args:
            continue
        if stripped and not raw_line.startswith((" ", "\t")):
            flush()
            break
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if match:
            flush()
            current_name = match.group(1)
            current_parts = [match.group(2)]
        elif current_name is not None and stripped:
            current_parts.append(stripped)

    flush()
    return descriptions
