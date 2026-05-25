"""Lightweight agent loop for Gemini."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from gat.client import GeminiClient
from gat.memory import InMemoryStore, MemoryStore
from gat.tools import ToolRegistry


class MaxIterationsExceeded(RuntimeError):
    """Raised when the agent cannot reach a final answer within the limit."""


@dataclass(frozen=True)
class AgentEvent:
    type: str
    payload: dict[str, Any]


class Agent:
    def __init__(
        self,
        client: GeminiClient,
        tools: Iterable[Callable[..., Any]] = (),
        memory: MemoryStore | None = None,
        max_iterations: int = 10,
        system: str | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.client = client
        self.tools = ToolRegistry(tools)
        self.memory = memory or InMemoryStore()
        self.max_iterations = max_iterations
        self.system = system

    def run(
        self,
        task: str,
        output_schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        if not self.tools.list():
            if output_schema is not None:
                structured_result = self.client.generate_structured(
                    task, output_schema, system=self.system
                )
                self.memory.add(
                    {"type": "final", "task": task, "output": structured_result.model_dump()}
                )
                return structured_result
            text_result = self.client.generate(task, system=self.system)
            self.memory.add({"type": "final", "task": task, "output": text_result})
            return text_result

        history: list[dict[str, Any]] = []
        self.memory.add({"type": "task", "task": task, "ts": time.time()})

        for iteration in range(1, self.max_iterations + 1):
            prompt = self._render_prompt(task, history, output_schema)
            response = self.client.generate(prompt, system=self.system)
            self.memory.add(
                {
                    "type": "assistant",
                    "iteration": iteration,
                    "content": response,
                    "ts": time.time(),
                }
            )

            parsed = _maybe_json(response)
            tool_call = _extract_tool_call(parsed)
            if tool_call is None:
                final_result = self._coerce_final(response, parsed, output_schema)
                self.memory.add(
                    {
                        "type": "final",
                        "iteration": iteration,
                        "output": _serializable(final_result),
                        "ts": time.time(),
                    }
                )
                return final_result

            name, args = tool_call
            self.memory.add(
                {"type": "tool_call", "iteration": iteration, "name": name, "args": args}
            )
            tool_result = self.tools.execute(name, args)
            self.memory.add(
                {
                    "type": "tool_result",
                    "iteration": iteration,
                    "name": name,
                    "result": _serializable(tool_result),
                    "ts": time.time(),
                }
            )
            history.append(
                {
                    "tool_call": {"name": name, "args": args},
                    "tool_result": tool_result,
                }
            )

        raise MaxIterationsExceeded(f"agent exceeded max_iterations={self.max_iterations}")

    def stream(self, task: str) -> Iterator[AgentEvent]:
        raise NotImplementedError("streaming agent events are planned for v0.2")

    def _render_prompt(
        self,
        task: str,
        history: list[dict[str, Any]],
        output_schema: type[BaseModel] | None,
    ) -> str:
        final_shape = (
            "Return final answer as JSON matching this schema: "
            f"{output_schema.model_json_schema()}"
            if output_schema is not None
            else "Return final answer as plain text or {'final': '...'} JSON."
        )
        return "\n\n".join(
            [
                "You are running a tool-using agent loop.",
                self.tools.prompt_block(),
                "When you need a tool, respond only as JSON: "
                '{"tool_call": {"name": "<tool_name>", "args": {"arg": "value"}}}.',
                f"When the task is complete, {final_shape}",
                f"Task: {task}",
                f"History: {json.dumps(history, ensure_ascii=False, default=str)}",
            ]
        )

    def _coerce_final(
        self,
        response: str,
        parsed: Any,
        output_schema: type[BaseModel] | None,
    ) -> str | BaseModel:
        final_payload = _extract_final_payload(parsed, response)
        if output_schema is None:
            if isinstance(final_payload, str):
                return final_payload
            return json.dumps(final_payload, ensure_ascii=False, default=str)

        for candidate in (final_payload, parsed):
            try:
                return output_schema.model_validate(candidate)
            except ValidationError:
                continue
        return self.client.generate_structured(
            f"Convert this answer to the requested schema:\n{response}",
            output_schema,
            system=self.system,
        )


def _maybe_json(text: str) -> Any:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean
        clean = clean.removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def _extract_tool_call(parsed: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(parsed, dict):
        return None
    call = parsed.get("tool_call") or parsed.get("function_call")
    if not isinstance(call, dict):
        if "name" in parsed and ("args" in parsed or "arguments" in parsed):
            call = parsed
        else:
            return None
    name = call.get("name")
    args = call.get("args", call.get("arguments", {}))
    if not isinstance(name, str):
        return None
    if not isinstance(args, dict):
        raise ValueError(f"tool args for {name} must be an object")
    return name, args


def _extract_final_payload(parsed: Any, fallback: str) -> Any:
    if isinstance(parsed, dict):
        for key in ("final", "answer", "result", "output"):
            if key in parsed:
                return parsed[key]
        return parsed
    return fallback


def _serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value
