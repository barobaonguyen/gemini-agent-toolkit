"""Streaming generation helpers for Gemini clients."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from gat.retry import CircuitBreaker, is_transient_error, retry


def stream_generate(
    client: Any,
    prompt: Any,
    *,
    system: str | None = None,
    thinking_budget: int | None = None,
    temperature: float = 0.7,
    attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    retry_if: Callable[[BaseException], bool] = is_transient_error,
    circuit_breaker: CircuitBreaker | None = None,
) -> Iterator[str]:
    """Yield text deltas from ``generate_content_stream`` and record final usage.

    Retry handling wraps stream creation, which is the safe retry boundary for a
    generator. If the SDK raises after bytes have already been yielded, callers
    should handle the exception instead of replaying duplicate text.
    """

    @retry(
        attempts=attempts,
        base_delay_s=base_delay_s,
        max_delay_s=max_delay_s,
        retry_if=retry_if,
        circuit_breaker=circuit_breaker,
    )
    def open_stream() -> Any:
        return client._ensure_client().models.generate_content_stream(
            model=client.model,
            contents=client._contents(prompt),
            config=client._build_config(
                system=system,
                thinking_budget=thinking_budget,
                temperature=temperature,
            ),
        )

    usage_chunk: Any | None = None
    for chunk in open_stream():
        if _has_usage(chunk):
            usage_chunk = chunk
        text = _chunk_text(chunk)
        if text:
            yield text

    if usage_chunk is not None:
        client._record_usage(usage_chunk)


def _has_usage(chunk: Any) -> bool:
    return getattr(chunk, "usage_metadata", None) is not None


def _chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text

    candidates = getattr(chunk, "candidates", None)
    if not candidates:
        return ""

    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts_text.append(part_text)
    return "".join(parts_text)
