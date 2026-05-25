"""Retry and circuit-breaker helpers for Gemini and external tools."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


TRANSIENT_MARKERS = (
    "429",
    "503",
    "deadline",
    "temporarily",
    "timeout",
    "too many requests",
    "rate limit",
    "resource_exhausted",
    "unavailable",
    "overload",
)

HARD_QUOTA_MARKERS = (
    "api key not valid",
    "billing",
    "permission_denied",
    "quota_exceeded",
    "spending cap",
    "spend cap",
    "exceeded its monthly",
)


class RetryError(RuntimeError):
    """Raised when retry attempts are exhausted."""


class CircuitBreakerOpen(RuntimeError):
    """Raised when a circuit breaker is open and fail-fast is active."""


def is_transient_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in TRANSIENT_MARKERS)


def is_hard_quota_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in HARD_QUOTA_MARKERS)


@dataclass
class CircuitBreaker:
    max_failures: int = 5
    reset_timeout_s: float = 60.0
    failures: int = 0
    opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None

    def before_call(self) -> None:
        if self.opened_at is None:
            return
        if time.monotonic() - self.opened_at >= self.reset_timeout_s:
            self.opened_at = None
            self.failures = 0
            return
        raise CircuitBreakerOpen("circuit breaker is open")

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, *, hard: bool = False) -> None:
        self.failures += 1
        if hard or self.failures >= self.max_failures:
            self.opened_at = time.monotonic()


@overload
def retry(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def retry(
    func: None = None,
    *,
    attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    backoff_factor: float = 2.0,
    jitter_s: float = 0.0,
    retry_if: Callable[[BaseException], bool] = is_transient_error,
    circuit_breaker: CircuitBreaker | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def retry(
    func: Callable[P, R] | None = None,
    *,
    attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    backoff_factor: float = 2.0,
    jitter_s: float = 0.0,
    retry_if: Callable[[BaseException], bool] = is_transient_error,
    circuit_breaker: CircuitBreaker | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry transient failures with exponential backoff.

    The decorator can be used as ``@retry`` or ``@retry(attempts=5)``. Hard
    quota/auth/billing errors open the circuit immediately.
    """

    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    breaker = circuit_breaker or CircuitBreaker()

    def decorate(inner: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(inner)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            delay = base_delay_s
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                breaker.before_call()
                try:
                    result = inner(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    hard = is_hard_quota_error(exc)
                    if hard:
                        breaker.record_failure(hard=True)
                        raise
                    if attempt >= attempts or not retry_if(exc):
                        breaker.record_failure()
                        raise
                    sleep_for = min(delay, max_delay_s)
                    if jitter_s:
                        sleep_for += random.uniform(0, jitter_s)
                    time.sleep(sleep_for)
                    delay = min(delay * backoff_factor, max_delay_s)
                    continue
                breaker.record_success()
                return result
            raise RetryError(f"retry exhausted: {last_error}")

        setattr(wrapper, "circuit_breaker", breaker)  # noqa: B010
        return wrapper

    if func is not None:
        return decorate(func)
    return decorate
