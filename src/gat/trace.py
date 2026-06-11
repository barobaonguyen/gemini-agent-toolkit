"""Structured JSONL tracing for agent runs."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO


class TraceWriter(Protocol):
    """Minimal trace sink used by :class:`gat.agent.Agent`."""

    run_id: str

    def write_span(self, span: Mapping[str, Any]) -> None:
        """Persist one trace span."""


@dataclass
class JsonlTraceWriter:
    """Append agent spans as newline-delimited JSON."""

    path: str | Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write_span(self, span: Mapping[str, Any]) -> None:
        payload = {"run_id": self.run_id, **dict(span)}
        self._file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlTraceWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_span(
    *,
    span_type: str,
    name: str,
    started_at: float,
    latency_ms: float,
    iteration: int | None = None,
    tokens: Mapping[str, int] | None = None,
    cost_usd: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized trace span dictionary."""

    span: dict[str, Any] = {
        "span_id": uuid.uuid4().hex,
        "type": span_type,
        "name": name,
        "started_at": started_at,
        "ended_at": started_at + (latency_ms / 1000.0),
        "latency_ms": round(latency_ms, 3),
        "tokens": dict(tokens or {}),
        "cost_usd": round(cost_usd, 8),
    }
    if iteration is not None:
        span["iteration"] = iteration
    if metadata:
        span["metadata"] = dict(metadata)
    return span


def read_spans(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL spans from a trace file."""

    spans: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                spans.append(value)
    return spans


def format_timeline(path: str | Path) -> str:
    """Pretty-print a JSONL trace as a compact timeline."""

    spans = read_spans(path)
    if not spans:
        return "No spans found."

    run_id = spans[0].get("run_id", "unknown")
    total_cost = sum(_float(span.get("cost_usd")) for span in spans)
    lines = [f"Trace run {run_id}", "step  type        name                 latency   cost"]
    for index, span in enumerate(spans, start=1):
        iteration = span.get("iteration")
        step = f"{iteration}" if isinstance(iteration, int) else f"{index}"
        span_type = str(span.get("type", ""))[:10]
        name = str(span.get("name", ""))[:20]
        latency = _float(span.get("latency_ms"))
        cost = _float(span.get("cost_usd"))
        lines.append(f"{step:>4}  {span_type:<10} {name:<20} {latency:>7.1f}ms  ${cost:.6f}")

        tokens = span.get("tokens")
        if isinstance(tokens, dict) and tokens:
            input_tokens = int(tokens.get("input_tokens", 0))
            output_tokens = int(tokens.get("output_tokens", 0))
            cached_tokens = int(tokens.get("cached_tokens", 0))
            lines.append(
                f"      tokens in={input_tokens} out={output_tokens} cached={cached_tokens}"
            )

    lines.append(f"total cost: ${total_cost:.6f}")
    return "\n".join(lines)


def view_trace(path: str | Path, out: TextIO) -> None:
    """Write a formatted trace timeline to a stream."""

    out.write(format_timeline(path) + "\n")


def _float(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
