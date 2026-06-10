from __future__ import annotations

from pathlib import Path

import pytest

from gat.memory import InMemoryStore, JsonlStore, SqliteStore, build_memory


def test_in_memory_eviction() -> None:
    store = InMemoryStore(max_items=2)
    store.add({"i": 1})
    store.add({"i": 2})
    store.add({"i": 3})
    assert store.replay() == [{"i": 2}, {"i": 3}]


def test_in_memory_replay_is_copy() -> None:
    store = InMemoryStore()
    store.add({"items": []})
    replay = store.replay()
    replay.append({"items": [1]})
    assert len(store.replay()) == 1


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "run.jsonl"
    store = JsonlStore(path)
    store.add({"type": "task", "task": "hello"})
    store.add({"type": "final", "output": {"ok": True}})

    loaded = JsonlStore(path)
    assert loaded.replay() == [
        {"type": "task", "task": "hello"},
        {"type": "final", "output": {"ok": True}},
    ]

    loaded.clear()
    assert loaded.replay() == []


def test_sqlite_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "run.db"
    store = SqliteStore(path)
    store.add({"type": "task", "task": "hello"})
    store.add({"type": "final", "output": {"ok": True}})

    loaded = SqliteStore(path)
    assert loaded.replay() == [
        {"type": "task", "task": "hello"},
        {"type": "final", "output": {"ok": True}},
    ]

    loaded.clear()
    assert loaded.replay() == []


def test_sqlite_preserves_insertion_order(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "ordered.db")
    for i in range(5):
        store.add({"i": i})
    assert [record["i"] for record in store.replay()] == [0, 1, 2, 3, 4]


def test_sqlite_rejects_bad_table_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        SqliteStore(tmp_path / "x.db", table="bad table; DROP")


def test_build_memory_backends(tmp_path: Path) -> None:
    assert isinstance(build_memory("memory"), InMemoryStore)
    assert isinstance(build_memory("jsonl", path=tmp_path / "m.jsonl"), JsonlStore)
    assert isinstance(build_memory("sqlite", path=tmp_path / "m.db"), SqliteStore)

    with pytest.raises(ValueError):
        build_memory("jsonl")  # path required
    with pytest.raises(ValueError):
        build_memory("redis", path=tmp_path / "x")  # type: ignore[arg-type]
