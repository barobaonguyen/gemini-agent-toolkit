from __future__ import annotations

import pytest

from gat.cost import CostTracker
from gat.pricing import estimate_cost_usd


def test_estimate_cost_with_cached_tokens() -> None:
    usd = estimate_cost_usd(
        "models/gemini-2.5-flash",
        input_tokens=10_000,
        output_tokens=1_000,
        cached_tokens=4_000,
    )
    expected = (6_000 * 0.30 + 4_000 * 0.03 + 1_000 * 2.50) / 1_000_000
    assert usd == pytest.approx(expected)


def test_cost_tracker_summary_and_grouping() -> None:
    tracker = CostTracker()
    tracker.add("gemini-2.5-flash", 1_000, 100, tool_name="rank")
    tracker.add("gemini-2.5-flash-lite", 2_000, 200, cached_tokens=1_000, tool_name="rank")

    summary = tracker.summary()
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 3_000
    assert summary["output_tokens"] == 300
    assert summary["cached_tokens"] == 1_000
    assert summary["total_usd"] > 0
    assert tracker.by_model()["gemini-2.5-flash"]["calls"] == 1
    assert tracker.by_tool()["rank"]["calls"] == 2


def test_reset_clears_entries() -> None:
    tracker = CostTracker()
    tracker.add("gemini-2.5-pro", 100, 10)
    tracker.reset()
    assert tracker.summary()["calls"] == 0
    assert tracker.entries == ()

