"""Prompt caching helpers.

Gemini performs implicit caching automatically for repeated large prompts. This
module adds small utilities for explicit cache calls and local cache keys.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any


def prompt_cache_key(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedContentRef:
    name: str
    model: str
    ttl_s: int


class ExplicitPromptCache:
    """Thin wrapper around the Google GenAI explicit cache API."""

    def __init__(self, sdk_client: Any) -> None:
        self.sdk_client = sdk_client

    def create(
        self,
        *,
        model: str,
        contents: Any,
        ttl_s: int = 3_600,
        display_name: str | None = None,
        system_instruction: str | None = None,
    ) -> CachedContentRef:
        types = importlib.import_module("google.genai.types")
        config_kwargs: dict[str, Any] = {"ttl": f"{ttl_s}s"}
        if display_name is not None:
            config_kwargs["display_name"] = display_name
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        cached = self.sdk_client.caches.create(
            model=model,
            contents=contents,
            config=types.CreateCachedContentConfig(**config_kwargs),
        )
        name = getattr(cached, "name", "")
        return CachedContentRef(name=name, model=model, ttl_s=ttl_s)

    def delete(self, name: str) -> None:
        self.sdk_client.caches.delete(name=name)


def should_explicitly_cache(text: str, *, min_chars: int = 8_000) -> bool:
    """Heuristic for when explicit caching is worth considering."""

    return len(text) >= min_chars

