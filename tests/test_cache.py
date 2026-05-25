from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

from gat.cache import ExplicitPromptCache, prompt_cache_key, should_explicitly_cache

cache_mod = importlib.import_module("gat.cache")


def test_prompt_cache_key_is_stable_and_sensitive() -> None:
    assert prompt_cache_key("a", "b") == prompt_cache_key("a", "b")
    assert prompt_cache_key("ab") != prompt_cache_key("a", "b")


def test_should_explicitly_cache_uses_length_threshold() -> None:
    assert should_explicitly_cache("x" * 10, min_chars=5)
    assert not should_explicitly_cache("x" * 4, min_chars=5)


def test_explicit_prompt_cache_create_and_delete(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeTypes:
        CreateCachedContentConfig = FakeConfig

    class FakeCaches:
        def __init__(self) -> None:
            self.created: dict[str, Any] | None = None
            self.deleted: str | None = None

        def create(self, **kwargs: Any) -> SimpleNamespace:
            self.created = kwargs
            return SimpleNamespace(name="cachedContents/demo")

        def delete(self, *, name: str) -> None:
            self.deleted = name

    class FakeSdk:
        def __init__(self) -> None:
            self.caches = FakeCaches()

    monkeypatch.setattr(cache_mod.importlib, "import_module", lambda _: FakeTypes)
    sdk = FakeSdk()
    cache = ExplicitPromptCache(sdk)

    ref = cache.create(
        model="gemini-2.5-flash",
        contents="long prompt",
        ttl_s=120,
        display_name="demo",
        system_instruction="system",
    )
    cache.delete(ref.name)

    assert ref.name == "cachedContents/demo"
    assert sdk.caches.created is not None
    assert sdk.caches.created["model"] == "gemini-2.5-flash"
    assert sdk.caches.created["config"].kwargs["ttl"] == "120s"
    assert sdk.caches.deleted == "cachedContents/demo"

