"""Lightweight agent loop for Gemini."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from gat.client import GeminiClient
from gat.memory import InMemoryStore, MemoryStore
from gat.tools import ToolRegistry
from gat.trace import TraceWriter, make_span


class MaxIterationsExceeded(RuntimeError):
    """Raised when the agent cannot reach a final answer within the limit."""


@dataclass(frozen=True)
class AgentEvent:
    type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AgentConfig:
    """Optional agent behavior switches."""

    planner: Literal["default", "react"] = "default"
    max_plan_steps: int | None = None


class Agent:
    def __init__(
        self,
        client: GeminiClient,
        tools: Iterable[Callable[..., Any]] = (),
        memory: MemoryStore | None = None,
        max_iterations: int = 10,
        system: str | None = None,
        planner: Literal["default", "react"] = "default",
        config: AgentConfig | dict[str, Any] | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        resolved_config = _resolve_config(config, planner)
        if resolved_config.max_plan_steps is not None:
            max_iterations = resolved_config.max_plan_steps
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.client = client
        self.tools = ToolRegistry(tools)
        self.memory = memory or InMemoryStore()
        self.max_iterations = max_iterations
        self.system = system
        self.config = resolved_config
        self.trace = trace

    def run(
        self,
        task: str,
        output_schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        if not self.tools.list():
            if output_schema is not None:
                started = time.perf_counter()
                wall_started = time.time()
                before = _cost_entry_count(self.client)
                structured_result = self.client.generate_structured(
                    task, output_schema, system=self.system
                )
                self._trace_model_call(
                    "generate_structured",
                    wall_started,
                    started,
                    before,
                    iteration=None,
                )
                self.memory.add(
                    {"type": "final", "task": task, "output": structured_result.model_dump()}
                )
                self._trace_final(iteration=None)
                return structured_result
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            text_result = self.client.generate(task, system=self.system)
            self._trace_model_call("generate", wall_started, started, before, iteration=None)
            self.memory.add({"type": "final", "task": task, "output": text_result})
            self._trace_final(iteration=None)
            return text_result

        history: list[dict[str, Any]] = []
        self.memory.add({"type": "task", "task": task, "ts": time.time()})

        for iteration in range(1, self.max_iterations + 1):
            prompt = self._render_prompt(task, history, output_schema)
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            response = self.client.generate(prompt, system=self.system)
            self._trace_model_call("generate", wall_started, started, before, iteration=iteration)
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
                final_result = self._coerce_final(
                    response, parsed, output_schema, iteration=iteration
                )
                self.memory.add(
                    {
                        "type": "final",
                        "iteration": iteration,
                        "output": _serializable(final_result),
                        "ts": time.time(),
                    }
                )
                self._trace_final(iteration=iteration)
                return final_result

            name, args = tool_call
            self.memory.add(
                {"type": "tool_call", "iteration": iteration, "name": name, "args": args}
            )
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            tool_result = self.tools.execute(name, args)
            self._trace_tool_call(name, wall_started, started, before, iteration=iteration)
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

    def stream(
        self,
        task: str,
        output_schema: type[BaseModel] | None = None,
    ) -> Iterator[AgentEvent]:
        if not self.tools.list():
            if output_schema is not None:
                started = time.perf_counter()
                wall_started = time.time()
                before = _cost_entry_count(self.client)
                structured_result = self.client.generate_structured(
                    task, output_schema, system=self.system
                )
                self._trace_model_call(
                    "generate_structured",
                    wall_started,
                    started,
                    before,
                    iteration=None,
                )
                output = structured_result.model_dump()
                self.memory.add({"type": "final", "task": task, "output": output})
                self._trace_final(iteration=None)
                yield AgentEvent("final", {"output": output})
                return
            self.memory.add({"type": "task", "task": task, "ts": time.time()})
            chunks: list[str] = []
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            for chunk in self.client.stream(task, system=self.system):
                chunks.append(chunk)
                yield AgentEvent("chunk", {"text": chunk})
            self._trace_model_call("stream", wall_started, started, before, iteration=None)
            text_result = "".join(chunks).strip()
            self.memory.add({"type": "final", "task": task, "output": text_result})
            self._trace_final(iteration=None)
            yield AgentEvent("final", {"output": text_result})
            return

        history: list[dict[str, Any]] = []
        self.memory.add({"type": "task", "task": task, "ts": time.time()})

        for iteration in range(1, self.max_iterations + 1):
            prompt = self._render_prompt(task, history, output_schema=output_schema)
            chunks = []
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            for chunk in self.client.stream(prompt, system=self.system):
                chunks.append(chunk)
                yield AgentEvent("chunk", {"iteration": iteration, "text": chunk})
            self._trace_model_call("stream", wall_started, started, before, iteration=iteration)

            response = "".join(chunks).strip()
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
                final_result = self._coerce_final(
                    response, parsed, output_schema=output_schema, iteration=iteration
                )
                self.memory.add(
                    {
                        "type": "final",
                        "iteration": iteration,
                        "output": _serializable(final_result),
                        "ts": time.time(),
                    }
                )
                self._trace_final(iteration=iteration)
                yield AgentEvent(
                    "final",
                    {"iteration": iteration, "output": _serializable(final_result)},
                )
                return

            name, args = tool_call
            self.memory.add(
                {"type": "tool_call", "iteration": iteration, "name": name, "args": args}
            )
            yield AgentEvent("tool_call", {"iteration": iteration, "name": name, "args": args})
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            tool_result = self.tools.execute(name, args)
            self._trace_tool_call(name, wall_started, started, before, iteration=iteration)
            serializable_result = _serializable(tool_result)
            self.memory.add(
                {
                    "type": "tool_result",
                    "iteration": iteration,
                    "name": name,
                    "result": serializable_result,
                    "ts": time.time(),
                }
            )
            yield AgentEvent(
                "tool_result",
                {"iteration": iteration, "name": name, "result": serializable_result},
            )
            history.append(
                {
                    "tool_call": {"name": name, "args": args},
                    "tool_result": tool_result,
                }
            )

        raise MaxIterationsExceeded(f"agent exceeded max_iterations={self.max_iterations}")

    async def arun(
        self,
        task: str,
        output_schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        """Async agent loop with concurrent multi-tool execution.

        Mirrors :meth:`run`, but when the model requests several tool calls in a
        single turn they are dispatched together via :func:`asyncio.gather`
        instead of serially. Cost accounting (delegated to the client) and the
        tool-level retry decorators stay intact because each call still flows
        through the same client/registry code paths.
        """

        if not self.tools.list():
            if output_schema is not None:
                started = time.perf_counter()
                wall_started = time.time()
                before = _cost_entry_count(self.client)
                structured_result = await self.client.agenerate_structured(
                    task, output_schema, system=self.system
                )
                self._trace_model_call(
                    "agenerate_structured",
                    wall_started,
                    started,
                    before,
                    iteration=None,
                )
                self.memory.add(
                    {"type": "final", "task": task, "output": structured_result.model_dump()}
                )
                self._trace_final(iteration=None)
                return structured_result
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            text_result = await self.client.agenerate(task, system=self.system)
            self._trace_model_call("agenerate", wall_started, started, before, iteration=None)
            self.memory.add({"type": "final", "task": task, "output": text_result})
            self._trace_final(iteration=None)
            return text_result

        history: list[dict[str, Any]] = []
        self.memory.add({"type": "task", "task": task, "ts": time.time()})

        for iteration in range(1, self.max_iterations + 1):
            prompt = self._render_prompt(task, history, output_schema)
            started = time.perf_counter()
            wall_started = time.time()
            before = _cost_entry_count(self.client)
            response = await self.client.agenerate(prompt, system=self.system)
            self._trace_model_call("agenerate", wall_started, started, before, iteration=iteration)
            self.memory.add(
                {
                    "type": "assistant",
                    "iteration": iteration,
                    "content": response,
                    "ts": time.time(),
                }
            )

            parsed = _maybe_json(response)
            tool_calls = _extract_tool_calls(parsed)
            if not tool_calls:
                final_result = self._coerce_final(
                    response, parsed, output_schema, iteration=iteration
                )
                self.memory.add(
                    {
                        "type": "final",
                        "iteration": iteration,
                        "output": _serializable(final_result),
                        "ts": time.time(),
                    }
                )
                self._trace_final(iteration=iteration)
                return final_result

            for name, args in tool_calls:
                self.memory.add(
                    {"type": "tool_call", "iteration": iteration, "name": name, "args": args}
                )

            results = await asyncio.gather(
                *(
                    self._atraced_tool_call(name, args, iteration=iteration)
                    for name, args in tool_calls
                )
            )

            for (name, args), tool_result in zip(tool_calls, results, strict=True):
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
        if self.config.planner == "react":
            return "\n\n".join(
                [
                    "You are running a ReAct-style plan-then-act agent loop.",
                    "Decompose the task into a concise plan, then act, observe, and revise.",
                    self.tools.prompt_block(),
                    "When you need a tool, respond only as JSON. On the first action include "
                    '{"plan": ["step"], "tool_call": {"name": "<tool_name>", '
                    '"args": {"arg": "value"}}}.',
                    "On later actions respond as JSON with "
                    '{"thought": "...", "tool_call": {"name": "<tool_name>", '
                    '"args": {"arg": "value"}}}.',
                    f"When the task is complete, {final_shape}",
                    f"Task: {task}",
                    f"Observations: {json.dumps(history, ensure_ascii=False, default=str)}",
                ]
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
        *,
        iteration: int | None = None,
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
        started = time.perf_counter()
        wall_started = time.time()
        before = _cost_entry_count(self.client)
        result = self.client.generate_structured(
            f"Convert this answer to the requested schema:\n{response}",
            output_schema,
            system=self.system,
        )
        self._trace_model_call(
            "generate_structured",
            wall_started,
            started,
            before,
            iteration=iteration,
        )
        return result

    async def _atraced_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        iteration: int,
    ) -> Any:
        started = time.perf_counter()
        wall_started = time.time()
        before = _cost_entry_count(self.client)
        result = await self.tools.aexecute(name, args)
        self._trace_tool_call(name, wall_started, started, before, iteration=iteration)
        return result

    def _trace_model_call(
        self,
        name: str,
        wall_started: float,
        started: float,
        cost_start: int,
        *,
        iteration: int | None,
    ) -> None:
        if self.trace is None:
            return
        tokens, cost_usd = _cost_delta(self.client, cost_start)
        self.trace.write_span(
            make_span(
                span_type="model_call",
                name=name,
                started_at=wall_started,
                latency_ms=(time.perf_counter() - started) * 1000,
                iteration=iteration,
                tokens=tokens,
                cost_usd=cost_usd,
                metadata={"model": getattr(self.client, "model", None)},
            )
        )

    def _trace_tool_call(
        self,
        name: str,
        wall_started: float,
        started: float,
        cost_start: int,
        *,
        iteration: int,
    ) -> None:
        if self.trace is None:
            return
        tokens, cost_usd = _cost_delta(self.client, cost_start)
        self.trace.write_span(
            make_span(
                span_type="tool_call",
                name=name,
                started_at=wall_started,
                latency_ms=(time.perf_counter() - started) * 1000,
                iteration=iteration,
                tokens=tokens,
                cost_usd=cost_usd,
            )
        )

    def _trace_final(self, *, iteration: int | None) -> None:
        if self.trace is None:
            return
        now = time.time()
        self.trace.write_span(
            make_span(
                span_type="final",
                name="final",
                started_at=now,
                latency_ms=0.0,
                iteration=iteration,
            )
        )


def _resolve_config(
    config: AgentConfig | dict[str, Any] | None,
    planner: Literal["default", "react"],
) -> AgentConfig:
    if config is None:
        return AgentConfig(planner=planner)
    if isinstance(config, AgentConfig):
        return config
    resolved_planner = config.get("planner", planner)
    if resolved_planner not in {"default", "react"}:
        raise ValueError("planner must be 'default' or 'react'")
    max_plan_steps = config.get("max_plan_steps")
    if max_plan_steps is not None and not isinstance(max_plan_steps, int):
        raise TypeError("max_plan_steps must be an int or None")
    return AgentConfig(planner=resolved_planner, max_plan_steps=max_plan_steps)


def _cost_entry_count(client: Any) -> int:
    tracker = getattr(client, "cost_tracker", None)
    entries = getattr(tracker, "entries", ())
    return len(entries) if isinstance(entries, tuple) else 0


def _cost_delta(client: Any, start: int) -> tuple[dict[str, int], float]:
    tracker = getattr(client, "cost_tracker", None)
    entries = getattr(tracker, "entries", ())
    if not isinstance(entries, tuple):
        return {}, 0.0
    new_entries = entries[start:]
    input_tokens = sum(_int_attr(entry, "input_tokens") for entry in new_entries)
    output_tokens = sum(_int_attr(entry, "output_tokens") for entry in new_entries)
    cached_tokens = sum(_int_attr(entry, "cached_tokens") for entry in new_entries)
    cost_usd = sum(_float_attr(entry, "usd") for entry in new_entries)
    tokens = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return tokens, cost_usd


def _int_attr(value: Any, name: str) -> int:
    attr = getattr(value, name, 0)
    return attr if isinstance(attr, int) else 0


def _float_attr(value: Any, name: str) -> float:
    attr = getattr(value, name, 0.0)
    return float(attr) if isinstance(attr, int | float) else 0.0


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


def _extract_tool_calls(parsed: Any) -> list[tuple[str, dict[str, Any]]]:
    """Extract one or more tool calls from a parsed model response.

    Supports the single-call shapes handled by :func:`_extract_tool_call` plus a
    ``{"tool_calls": [...]}`` list and a bare top-level list of call objects, so
    a model can request several tools to run concurrently in one turn.
    """

    if isinstance(parsed, dict):
        batch = parsed.get("tool_calls") or parsed.get("function_calls")
        if isinstance(batch, list):
            calls: list[tuple[str, dict[str, Any]]] = []
            for item in batch:
                call = _extract_tool_call(item)
                if call is not None:
                    calls.append(call)
            return calls

    if isinstance(parsed, list):
        calls = []
        for item in parsed:
            call = _extract_tool_call(item)
            if call is not None:
                calls.append(call)
        return calls

    single = _extract_tool_call(parsed)
    return [single] if single is not None else []


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
