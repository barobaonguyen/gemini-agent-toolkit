from __future__ import annotations

import importlib
from typing import Any

from gat import Agent, GeminiClient, tool

retry_mod = importlib.import_module("gat.retry")


class Usage:
    prompt_token_count = 1_000
    candidates_token_count = 200
    cached_content_token_count = 100


class FakeChunk:
    def __init__(self, text: str, usage_metadata: Usage | None = None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata


class FakeModels:
    def __init__(self, streams: list[list[FakeChunk] | BaseException]) -> None:
        self.streams = streams
        self.calls: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        stream = self.streams.pop(0)
        if isinstance(stream, BaseException):
            raise stream
        return iter(stream)


class FakeSdkClient:
    def __init__(self, streams: list[list[FakeChunk] | BaseException]) -> None:
        self.models = FakeModels(streams)


class FakeAgentClient:
    def __init__(self, streams: list[list[str]]) -> None:
        self.streams = streams
        self.prompts: list[str] = []

    def stream(self, prompt: str, **_: Any) -> Any:
        self.prompts.append(prompt)
        return iter(self.streams.pop(0))


@tool
def lookup_topic(topic: str) -> dict[str, str]:
    """Look up a topic."""

    return {"topic": topic, "note": "grounded"}


def test_client_stream_yields_chunks_and_records_cost() -> None:
    client = GeminiClient(model="gemini-2.5-flash")
    fake = FakeSdkClient([[FakeChunk("hel"), FakeChunk("lo", Usage())]])
    client._client = fake

    assert list(client.stream("Say hello", system="Be brief")) == ["hel", "lo"]

    assert fake.models.calls[0]["model"] == "gemini-2.5-flash"
    assert fake.models.calls[0]["contents"] == "Say hello"
    summary = client.cost_tracker.summary()
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 1_000
    assert summary["output_tokens"] == 200
    assert summary["cached_tokens"] == 100


def test_client_stream_retries_transient_start_failure(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)
    client = GeminiClient(model="gemini-2.5-flash")
    fake = FakeSdkClient([RuntimeError("503 UNAVAILABLE"), [FakeChunk("ok", Usage())]])
    client._client = fake

    assert "".join(client.stream("retry")) == "ok"

    assert len(fake.models.calls) == 2
    assert sleeps == [1.0]
    assert client.cost_tracker.summary()["calls"] == 1


def test_agent_stream_executes_tool_then_streams_final() -> None:
    client = FakeAgentClient(
        [
            [
                '{"tool_call": {"name": "lookup_topic", ',
                '"args": {"topic": "Gemini"}}}',
            ],
            ["Final ", "answer."],
        ]
    )
    agent = Agent(client=client, tools=[lookup_topic])

    events = list(agent.stream("Research Gemini"))

    assert [event.type for event in events] == [
        "chunk",
        "chunk",
        "tool_call",
        "tool_result",
        "chunk",
        "chunk",
        "final",
    ]
    assert events[2].payload == {
        "iteration": 1,
        "name": "lookup_topic",
        "args": {"topic": "Gemini"},
    }
    assert events[-1].payload == {"iteration": 2, "output": "Final answer."}
    assert "History:" in client.prompts[1]
