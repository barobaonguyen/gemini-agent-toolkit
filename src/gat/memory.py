"""Small memory stores for agent trajectories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class MemoryStore(Protocol):
    def add(self, record: Mapping[str, Any]) -> None:
        """Persist one trajectory record."""
        ...

    def replay(self) -> list[dict[str, Any]]:
        """Return records in insertion order."""
        ...

    def clear(self) -> None:
        """Remove all records."""
        ...


class InMemoryStore:
    """In-process append-only memory with optional oldest-item eviction."""

    def __init__(self, max_items: int | None = 1_000) -> None:
        self.max_items = max_items
        self._records: list[dict[str, Any]] = []

    def add(self, record: Mapping[str, Any]) -> None:
        self._records.append(dict(record))
        if self.max_items is not None and len(self._records) > self.max_items:
            overflow = len(self._records) - self.max_items
            del self._records[:overflow]

    def replay(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def clear(self) -> None:
        self._records.clear()


class JsonlStore:
    """Append-only JSONL memory that can replay an agent run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")

    def replay(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                records.append(data)
        return records

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")
