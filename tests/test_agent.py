from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from gat.agent import Agent, MaxIterationsExceeded
from gat.memory import InMemoryStore
from gat.tools import tool


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)

    def generate_structured(self, prompt: str, schema: type[BaseModel], **_: Any) -> BaseModel:
        self.prompts.append(prompt)
        return schema.model_validate_json(self.responses.pop(0))


class TokenInfo(BaseModel):
    symbol: str
    market_cap_usd: float
    risk_score: int


@tool
def fetch_token(symbol: str) -> dict[str, object]:
    """Fetch token info."""
    return {"symbol": symbol, "market_cap_usd": 50_000_000, "risk_hint": "medium"}


def test_agent_executes_tool_then_returns_structured_output() -> None:
    client = FakeClient(
        [
            '{"tool_call": {"name": "fetch_token", "args": {"symbol": "PEPE"}}}',
            '{"symbol": "PEPE", "market_cap_usd": 50000000, "risk_score": 6}',
        ]
    )
    memory = InMemoryStore()
    agent = Agent(client=client, tools=[fetch_token], memory=memory)

    result = agent.run("Assess PEPE", output_schema=TokenInfo)

    assert result == TokenInfo(symbol="PEPE", market_cap_usd=50_000_000, risk_score=6)
    records = memory.replay()
    assert [record["type"] for record in records] == [
        "task",
        "assistant",
        "tool_call",
        "tool_result",
        "assistant",
        "final",
    ]
    assert "Available tools:" in client.prompts[0]


def test_agent_raises_on_max_iterations() -> None:
    client = FakeClient(['{"tool_call": {"name": "fetch_token", "args": {"symbol": "A"}}}'])
    agent = Agent(client=client, tools=[fetch_token], max_iterations=1)
    with pytest.raises(MaxIterationsExceeded):
        agent.run("loop forever")


def test_agent_without_tools_uses_structured_call() -> None:
    client = FakeClient(['{"symbol": "ETH", "market_cap_usd": 1, "risk_score": 2}'])
    agent = Agent(client=client)
    result = agent.run("Return ETH", output_schema=TokenInfo)
    assert result == TokenInfo(symbol="ETH", market_cap_usd=1, risk_score=2)

