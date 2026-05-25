from __future__ import annotations

from pathlib import Path

from gat.memory import InMemoryStore, JsonlStore


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
