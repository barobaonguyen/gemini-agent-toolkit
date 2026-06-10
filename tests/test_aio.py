from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel

from gat.agent import Agent, MaxIterationsExceeded
from gat.cost import CostTracker
from gat.memory import InMemoryStore
from gat.tools import tool


class FakeAsyncClient:
    """Minimal async-capable stand-in for GeminiClient (no live calls)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self.cost_tracker = CostTracker()

    async def agenerate(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        return self.responses.pop(0)

    async def agenerate_structured(
        self, prompt: str, schema: type[BaseModel], **_: Any
    ) -> BaseModel:
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        return schema.model_validate_json(self.responses.pop(0))


class TokenInfo(BaseModel):
    symbol: str
    market_cap_usd: float
    risk_score: int


# A tool that sleeps so we can prove the two calls overlap instead of serializing.
_GATE = {"running": 0, "max": 0}


@tool
def slow_lookup(symbol: str) -> dict[str, str]:
    """Look up a symbol slowly."""

    _GATE["running"] += 1
    _GATE["max"] = max(_GATE["max"], _GATE["running"])
    time.sleep(0.05)
    _GATE["running"] -= 1
    return {"symbol": symbol, "note": "ok"}


@tool
async def async_lookup(symbol: str) -> dict[str, str]:
    """Async tool variant."""

    await asyncio.sleep(0.01)
    return {"symbol": symbol, "note": "async-ok"}


def test_arun_executes_multiple_tools_concurrently() -> None:
    _GATE.update(running=0, max=0)
    client = FakeAsyncClient(
        [
            '{"tool_calls": ['
            '{"name": "slow_lookup", "args": {"symbol": "A"}}, '
            '{"name": "slow_lookup", "args": {"symbol": "B"}}, '
            '{"name": "slow_lookup", "args": {"symbol": "C"}}]}',
            '{"final": "done"}',
        ]
    )
    memory = InMemoryStore()
    agent = Agent(client=client, tools=[slow_lookup], memory=memory)

    start = time.perf_counter()
    result = asyncio.run(agent.arun("look up A, B, C"))
    elapsed = time.perf_counter() - start

    assert result == "done"
    # Three 50ms sleeps serially would take >=150ms; concurrent stays well under.
    assert elapsed < 0.12
    assert _GATE["max"] >= 2  # at least two tool calls were in flight together
    types = [record["type"] for record in memory.replay()]
    assert types.count("tool_call") == 3
    assert types.count("tool_result") == 3


def test_arun_returns_structured_output() -> None:
    client = FakeAsyncClient(
        [
            '{"tool_call": {"name": "async_lookup", "args": {"symbol": "ETH"}}}',
            '{"symbol": "ETH", "market_cap_usd": 1.0, "risk_score": 3}',
        ]
    )
    agent = Agent(client=client, tools=[async_lookup])
    result = asyncio.run(agent.arun("assess ETH", output_schema=TokenInfo))
    assert result == TokenInfo(symbol="ETH", market_cap_usd=1.0, risk_score=3)


def test_arun_without_tools_uses_async_text() -> None:
    client = FakeAsyncClient(["plain answer"])
    agent = Agent(client=client)
    result = asyncio.run(agent.arun("hello"))
    assert result == "plain answer"


def test_arun_respects_max_iterations() -> None:
    client = FakeAsyncClient(['{"tool_call": {"name": "async_lookup", "args": {"symbol": "X"}}}'])
    agent = Agent(client=client, tools=[async_lookup], max_iterations=1)
    with pytest.raises(MaxIterationsExceeded):
        asyncio.run(agent.arun("loop"))


def test_arun_preserves_cost_accounting() -> None:
    from gat import GeminiClient

    class Usage:
        prompt_token_count = 1_000
        candidates_token_count = 200
        cached_content_token_count = 0

    class FakeResponse:
        text = '{"final": "ok"}'
        usage_metadata = Usage()

    class FakeModels:
        def generate_content(self, **_: Any) -> FakeResponse:
            return FakeResponse()

    class FakeSdk:
        models = FakeModels()

    client = GeminiClient(model="gemini-2.5-flash")
    client._client = FakeSdk()

    # No tools -> arun delegates to agenerate, which records usage via generate.
    result = asyncio.run(Agent(client=client).arun("hi"))
    assert result == '{"final": "ok"}'
    summary = client.cost_tracker.summary()
    assert summary["calls"] == 1
    assert summary["input_tokens"] == 1_000
    assert summary["total_usd"] > 0
