from __future__ import annotations

import importlib

import pytest

from gat.retry import CircuitBreaker, CircuitBreakerOpen, is_hard_quota_error, retry

retry_mod = importlib.import_module("gat.retry")


def test_retry_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_mod.time, "sleep", sleeps.append)
    attempts = {"count": 0}

    @retry(attempts=3, base_delay_s=1, backoff_factor=2)
    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    assert flaky() == "ok"
    assert sleeps == [1, 2]
    assert attempts["count"] == 3


def test_non_transient_error_is_not_retried() -> None:
    attempts = {"count": 0}

    @retry(attempts=3)
    def bad() -> None:
        attempts["count"] += 1
        raise ValueError("invalid payload")

    with pytest.raises(ValueError):
        bad()
    assert attempts["count"] == 1


def test_hard_quota_opens_circuit() -> None:
    breaker = CircuitBreaker(max_failures=3, reset_timeout_s=60)

    @retry(attempts=3, circuit_breaker=breaker)
    def quota() -> None:
        raise RuntimeError("PERMISSION_DENIED: API key not valid")

    with pytest.raises(RuntimeError):
        quota()
    assert breaker.is_open
    with pytest.raises(CircuitBreakerOpen):
        quota()


def test_circuit_breaker_resets_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"value": 10.0}
    monkeypatch.setattr(retry_mod.time, "monotonic", lambda: now["value"])
    breaker = CircuitBreaker(max_failures=1, reset_timeout_s=5)
    breaker.record_failure()
    assert breaker.is_open
    now["value"] = 16.0
    breaker.before_call()
    assert not breaker.is_open


def test_hard_quota_classifier() -> None:
    assert is_hard_quota_error(RuntimeError("billing spending cap exceeded"))
