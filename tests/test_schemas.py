from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from gat.schemas import python_type_to_schema, to_gemini_schema


class Risk(BaseModel):
    score: int
    label: Literal["low", "medium", "high"]


class TokenReport(BaseModel):
    symbol: str
    risk: Risk
    tags: list[str]
    note: str | None = None


def test_to_gemini_schema_resolves_refs_and_nullable() -> None:
    schema = to_gemini_schema(TokenReport)
    assert schema["type"] == "object"
    assert schema["properties"]["symbol"]["type"] == "string"
    assert schema["properties"]["risk"]["properties"]["score"]["type"] == "integer"
    assert schema["properties"]["risk"]["properties"]["label"]["enum"] == [
        "low",
        "medium",
        "high",
    ]
    assert schema["properties"]["tags"]["items"]["type"] == "string"
    assert schema["properties"]["note"]["nullable"] is True
    assert "$defs" not in schema


def test_python_type_to_schema_handles_common_annotations() -> None:
    assert python_type_to_schema(str)["type"] == "string"
    assert python_type_to_schema(int)["type"] == "integer"
    assert python_type_to_schema(str | None) == {"type": "string", "nullable": True}
    assert python_type_to_schema(list[int]) == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert python_type_to_schema(dict[str, float]) == {
        "type": "object",
        "additionalProperties": {"type": "number"},
    }
    assert python_type_to_schema(Literal["a", "b"]) == {"type": "string", "enum": ["a", "b"]}

