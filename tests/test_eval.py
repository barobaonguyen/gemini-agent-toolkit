from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from gat.agent import Agent
from gat.cost import CostTracker
from gat.eval import EvalCase, EvalError, evaluate, format_report, load_cases


class ScriptedClient:
    """Returns canned answers keyed by prompt substring; tracks fake cost."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.cost_tracker = CostTracker()

    def generate(self, prompt: str, **_: Any) -> str:
        self.cost_tracker.add("gemini-2.5-flash", 100, 20)
        for needle, answer in self.answers.items():
            if needle in prompt:
                return answer
        return "unknown"

    def generate_structured(self, prompt: str, schema: type[BaseModel], **_: Any) -> BaseModel:
        self.cost_tracker.add("gemini-2.5-flash", 50, 10)
        return schema.model_validate({"passed": True, "reason": "stub judge"})


def _agent(answers: dict[str, str]) -> Agent:
    return Agent(client=ScriptedClient(answers))


def test_load_cases_from_json(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            [
                {"name": "c1", "prompt": "p1", "expected": "Paris", "matcher": "contains"},
                {"prompt": "p2", "expected": "144", "matcher": "exact"},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [case.name for case in cases] == ["c1", "case_1"]
    assert cases[1].matcher == "exact"


def test_load_cases_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"prompt": "only prompt"}]), encoding="utf-8")
    with pytest.raises(EvalError):
        load_cases(path)


def test_evaluate_all_matchers() -> None:
    cases = [
        EvalCase(name="exact", prompt="exact?", expected="144", matcher="exact"),
        EvalCase(name="contains", prompt="cap?", expected="Paris", matcher="contains"),
        EvalCase(name="regex", prompt="prime?", expected=r"1[1379]", matcher="regex"),
    ]
    agent = _agent({"exact?": "144", "cap?": "The capital is Paris.", "prime?": "13"})
    report = evaluate(agent, cases)
    assert report.passed == 3
    assert report.pass_rate == pytest.approx(1.0)
    assert report.total_usd > 0


def test_evaluate_marks_failures() -> None:
    cases = [
        EvalCase(name="exact", prompt="exact?", expected="144", matcher="exact"),
        EvalCase(name="contains", prompt="cap?", expected="Paris", matcher="contains"),
    ]
    agent = _agent({"exact?": "145", "cap?": "London"})
    report = evaluate(agent, cases)
    assert report.passed == 0
    detail = {result.name: result.detail for result in report.results}
    assert detail["exact"] == "exact mismatch"
    assert "substring" in detail["contains"]


def test_evaluate_llm_judge_uses_injected_judge() -> None:
    case = EvalCase(name="judge", prompt="explain", expected="concurrent", matcher="llm_judge")
    agent = _agent({"explain": "runs awaitables concurrently"})
    calls: list[str] = []

    def judge(c: EvalCase, actual: str) -> tuple[bool, str]:
        calls.append(actual)
        return "concurrent" in actual, "looks right"

    report = evaluate(agent, [case], judge=judge)
    assert report.passed == 1
    assert calls == ["runs awaitables concurrently"]


def test_evaluate_llm_judge_default_path() -> None:
    case = EvalCase(name="judge", prompt="explain", expected="x", matcher="llm_judge")
    agent = _agent({"explain": "anything"})
    report = evaluate(agent, [case])  # default judge -> ScriptedClient stub returns passed=True
    assert report.passed == 1


def test_evaluate_records_errors_as_failures() -> None:
    class Boom:
        cost_tracker = CostTracker()

        def generate(self, prompt: str, **_: Any) -> str:
            raise RuntimeError("kaboom")

    agent = Agent(client=Boom())
    case = EvalCase(name="err", prompt="x", expected="y", matcher="contains")
    report = evaluate(agent, [case])
    assert report.passed == 0
    assert "kaboom" in report.results[0].detail


def test_invalid_matcher_rejected() -> None:
    with pytest.raises(EvalError):
        EvalCase(name="bad", prompt="p", expected="e", matcher="nope")


def test_format_report_is_readable() -> None:
    cases = [EvalCase(name="c", prompt="cap?", expected="Paris", matcher="contains")]
    agent = _agent({"cap?": "Paris"})
    report = evaluate(agent, cases)
    text = format_report(report)
    assert "pass rate: 1/1" in text
    assert "[PASS] c" in text
