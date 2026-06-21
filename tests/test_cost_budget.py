"""Tests for the v0.5 cost budget guard + report formatter."""

from __future__ import annotations

import pytest

from gat.cost import BudgetExceededError, CostTracker, format_cost_report


def _tracker() -> CostTracker:
    tracker = CostTracker()
    tracker.add("gemini-2.5-flash", 100_000, 10_000, tool_name="rank")
    tracker.add("gemini-2.5-flash-lite", 50_000, 5_000)
    return tracker


def test_total_and_remaining() -> None:
    tracker = _tracker()
    total = tracker.total_usd()
    assert total > 0
    assert tracker.remaining(total + 1) == pytest.approx(1.0)
    # remaining never goes negative
    assert tracker.remaining(0.0) == 0.0


def test_within_budget_and_assert() -> None:
    tracker = _tracker()
    total = tracker.total_usd()
    assert tracker.within_budget(total)  # boundary is inclusive
    assert not tracker.within_budget(total / 2)
    tracker.assert_within(total)  # no raise at the boundary
    with pytest.raises(BudgetExceededError) as exc:
        tracker.assert_within(total / 2)
    assert exc.value.spent > exc.value.cap


def test_empty_tracker_within_any_budget() -> None:
    tracker = CostTracker()
    assert tracker.total_usd() == 0.0
    assert tracker.within_budget(0.0)


def test_format_cost_report_text_and_markdown() -> None:
    tracker = _tracker()
    text = format_cost_report(tracker)
    assert "Cost report" in text
    assert "gemini-2.5-flash" in text

    md = format_cost_report(tracker, "markdown")
    assert md.startswith("| Model | Calls |")
    assert "**total**" in md
    # one row per model + header + separator + total
    assert md.count("\n") >= 4


def test_format_cost_report_unknown_format() -> None:
    with pytest.raises(ValueError):
        format_cost_report(CostTracker(), "html")
