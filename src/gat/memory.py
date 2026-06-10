"""Small memory stores for agent trajectories."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, Literal, Protocol


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


class SqliteStore:
    """Append-only SQLite memory with the same read/write contract as JsonlStore.

    Records are stored as JSON text in insertion order. Useful when a run writes
    many trajectory records or several processes share one durable memory file,
    where the JSONL append-and-rescan pattern becomes awkward.
    """

    def __init__(self, path: str | Path, *, table: str = "memory") -> None:
        if not table.isidentifier():
            raise ValueError(f"invalid table name: {table!r}")
        self.path = Path(path)
        self.table = table
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, record TEXT NOT NULL)"
            )

    def add(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(dict(record), ensure_ascii=False, default=str)
        with closing(self._connect()) as conn, conn:
            conn.execute(f"INSERT INTO {self.table} (record) VALUES (?)", (payload,))

    def replay(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT record FROM {self.table} ORDER BY id").fetchall()
        records: list[dict[str, Any]] = []
        for (raw,) in rows:
            data = json.loads(raw)
            if isinstance(data, dict):
                records.append(data)
        return records

    def clear(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(f"DELETE FROM {self.table}")


MemoryBackend = Literal["memory", "jsonl", "sqlite"]


def build_memory(
    backend: MemoryBackend = "memory",
    *,
    path: str | Path | None = None,
) -> MemoryStore:
    """Construct a memory store by backend name (selectable from config).

    ``"jsonl"`` and ``"sqlite"`` require ``path``; ``"memory"`` is in-process.
    """

    if backend == "memory":
        return InMemoryStore()
    if path is None:
        raise ValueError(f"backend {backend!r} requires a path")
    if backend == "jsonl":
        return JsonlStore(path)
    if backend == "sqlite":
        return SqliteStore(path)
    raise ValueError(f"unknown memory backend: {backend!r}")
