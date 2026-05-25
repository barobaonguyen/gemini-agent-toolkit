from __future__ import annotations

from typing import Any

import pytest
from conftest import config_value
from pydantic import BaseModel

from gat.client import GeminiClient, MissingApiKeyError


class Usage:
    prompt_token_count = 1_000
    candidates_token_count = 200
    cached_content_token_count = 100


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = Usage()


class FakeModels:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeSdkClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.models = FakeModels(responses)


class RankedPost(BaseModel):
    title: str
    score: int


def test_generate_shapes_request_and_tracks_cost() -> None:
    client = GeminiClient(model="gemini-2.5-flash")
    fake = FakeSdkClient([FakeResponse("hello")])
    client._client = fake

    assert client.generate("Say hi", system="You are terse", thinking_budget=0) == "hello"
    call = fake.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "Say hi"
    assert config_value(call["config"], "temperature") == 0.7
    assert client.cost_tracker.summary()["calls"] == 1
    assert client.cost_tracker.summary()["input_tokens"] == 1_000


def test_generate_structured_parses_pydantic() -> None:
    client = GeminiClient(model="gemini-2.5-flash-lite")
    fake = FakeSdkClient([FakeResponse('{"title": "A", "score": 8}')])
    client._client = fake

    result = client.generate_structured("rank", RankedPost)

    assert result == RankedPost(title="A", score=8)
    config = fake.models.calls[0]["config"]
    assert config_value(config, "response_mime_type") == "application/json"
    schema = config_value(config, "response_schema")
    assert schema["properties"]["title"]["type"] == "string"


def test_batch_preserves_order() -> None:
    client = GeminiClient()
    fake = FakeSdkClient([FakeResponse("one"), FakeResponse("two")])
    client._client = fake
    assert client.batch(["1", "2"], concurrency=1) == ["one", "two"]


def test_missing_api_key_raises_on_real_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient(api_key=None)
    with pytest.raises(MissingApiKeyError):
        client.generate("hello")
