from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from gat import cli
from gat.agent import Agent
from gat.cost import CostTracker
from gat.tools import tool
from gat.trace import JsonlTraceWriter, format_timeline, read_spans


class TraceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.cost_tracker = CostTracker()

    def generate(self, prompt: str, **_: Any) -> str:
        self.cost_tracker.add("gemini-2.5-flash", 100, 20)
        return self.responses.pop(0)


@tool
def lookup(symbol: str) -> dict[str, str]:
    """Look up a symbol."""

    return {"symbol": symbol, "status": "ok"}


def test_trace_writer_records_valid_jsonl_spans(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    client = TraceClient(
        [
            '{"tool_call": {"name": "lookup", "args": {"symbol": "ETH"}}}',
            '{"final": "done"}',
        ]
    )
    with JsonlTraceWriter(path) as trace:
        result = Agent(client=client, tools=[lookup], trace=trace).run("check ETH")

    assert result == "done"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert all(isinstance(json.loads(line), dict) for line in lines)

    spans = read_spans(path)
    assert [span["type"] for span in spans] == [
        "model_call",
        "tool_call",
        "model_call",
        "final",
    ]
    assert spans[0]["tokens"]["input_tokens"] == 100
    assert spans[0]["cost_usd"] > 0
    assert spans[1]["name"] == "lookup"


def test_trace_view_formats_timeline(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with JsonlTraceWriter(path, run_id="run-test") as trace:
        trace.write_span(
            {
                "type": "model_call",
                "name": "generate",
                "started_at": 1.0,
                "ended_at": 1.1,
                "latency_ms": 100.0,
                "tokens": {"input_tokens": 10, "output_tokens": 2, "cached_tokens": 0},
                "cost_usd": 0.001,
            }
        )

    formatted = format_timeline(path)
    assert "Trace run run-test" in formatted
    assert "generate" in formatted
    assert "total cost: $0.001000" in formatted

    out = io.StringIO()
    code = cli.main(["trace", "view", str(path)], out=out)
    assert code == 0
    assert "tokens in=10 out=2 cached=0" in out.getvalue()
