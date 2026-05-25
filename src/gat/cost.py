"""Token and USD accounting for Gemini calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from gat.pricing import estimate_cost_usd, normalize_model


@dataclass(frozen=True)
class CostEntry:
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    usd: float
    tool_name: str | None = None


class CostTracker:
    """Accumulates Gemini token usage and estimates spend from frozen pricing."""

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []

    @property
    def entries(self) -> tuple[CostEntry, ...]:
        return tuple(self._entries)

    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        tool_name: str | None = None,
    ) -> None:
        normalized = normalize_model(model)
        self._entries.append(
            CostEntry(
                model=normalized,
                input_tokens=max(input_tokens, 0),
                output_tokens=max(output_tokens, 0),
                cached_tokens=max(cached_tokens, 0),
                usd=estimate_cost_usd(normalized, input_tokens, output_tokens, cached_tokens),
                tool_name=tool_name,
            )
        )

    def summary(self) -> dict[str, Any]:
        input_tokens = sum(entry.input_tokens for entry in self._entries)
        output_tokens = sum(entry.output_tokens for entry in self._entries)
        cached_tokens = sum(entry.cached_tokens for entry in self._entries)
        return {
            "total_usd": round(sum(entry.usd for entry in self._entries), 8),
            "calls": len(self._entries),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def by_model(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for entry in self._entries:
            bucket = buckets.setdefault(entry.model, self._empty_bucket())
            self._add_to_bucket(bucket, entry)
        return buckets

    def by_tool(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for entry in self._entries:
            if entry.tool_name is None:
                continue
            bucket = buckets.setdefault(entry.tool_name, self._empty_bucket())
            self._add_to_bucket(bucket, entry)
        return buckets

    def reset(self) -> None:
        self._entries.clear()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._entries]

    @staticmethod
    def _empty_bucket() -> dict[str, Any]:
        return {
            "total_usd": 0.0,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _add_to_bucket(bucket: dict[str, Any], entry: CostEntry) -> None:
        bucket["total_usd"] = round(float(bucket["total_usd"]) + entry.usd, 8)
        bucket["calls"] = int(bucket["calls"]) + 1
        bucket["input_tokens"] = int(bucket["input_tokens"]) + entry.input_tokens
        bucket["output_tokens"] = int(bucket["output_tokens"]) + entry.output_tokens
        bucket["cached_tokens"] = int(bucket["cached_tokens"]) + entry.cached_tokens
        bucket["total_tokens"] = (
            int(bucket["total_tokens"]) + entry.input_tokens + entry.output_tokens
        )
