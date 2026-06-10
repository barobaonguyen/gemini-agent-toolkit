"""Golden-set evaluation harness for Gemini agents.

Load cases from JSON or YAML (``prompt`` + ``expected``), run each through an
:class:`~gat.agent.Agent`, and score the output with a matcher. Built-in
matchers are ``exact``, ``contains``, and ``regex``; an optional ``llm_judge``
matcher asks Gemini to grade the answer. The report carries a pass-rate and the
total USD cost accumulated on the agent's client.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gat.agent import Agent

Matcher = str

_VALID_MATCHERS = frozenset({"exact", "contains", "regex", "llm_judge"})


class EvalError(RuntimeError):
    """Raised when a suite file is malformed."""


@dataclass(frozen=True)
class EvalCase:
    """One golden case: a prompt plus the expected answer and matcher."""

    name: str
    prompt: str
    expected: str
    matcher: Matcher = "contains"
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.matcher not in _VALID_MATCHERS:
            raise EvalError(
                f"case {self.name!r}: unknown matcher {self.matcher!r} "
                f"(expected one of {sorted(_VALID_MATCHERS)})"
            )

    @classmethod
    def from_dict(cls, index: int, data: dict[str, Any]) -> EvalCase:
        if "prompt" not in data or "expected" not in data:
            raise EvalError(f"case #{index}: requires 'prompt' and 'expected' keys")
        return cls(
            name=str(data.get("name", f"case_{index}")),
            prompt=str(data["prompt"]),
            expected=str(data["expected"]),
            matcher=str(data.get("matcher", "contains")),
            case_sensitive=bool(data.get("case_sensitive", False)),
        )


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    matcher: Matcher
    expected: str
    actual: str
    detail: str = ""


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    total_usd: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.total - self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "total_usd": round(self.total_usd, 8),
            "cases": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "matcher": result.matcher,
                    "expected": result.expected,
                    "actual": result.actual,
                    "detail": result.detail,
                }
                for result in self.results
            ],
        }


def load_cases(source: str | Path | Iterable[dict[str, Any]]) -> list[EvalCase]:
    """Load eval cases from a JSON/YAML file path or an iterable of dicts."""

    raw = _load_case_data(Path(source)) if isinstance(source, (str, Path)) else list(source)
    if not isinstance(raw, list):
        raise EvalError("eval suite must be a list of cases")
    return [EvalCase.from_dict(index, _require_dict(index, item)) for index, item in enumerate(raw)]


def evaluate(
    agent: Agent,
    cases: Sequence[EvalCase],
    *,
    judge: Callable[[EvalCase, str], tuple[bool, str]] | None = None,
) -> EvalReport:
    """Run each case through ``agent`` and score it. Returns an :class:`EvalReport`.

    ``judge`` overrides the default Gemini LLM-judge used by ``llm_judge`` cases;
    inject a stub here to keep CI free of live model calls.
    """

    start_usd = _client_cost(agent)
    judge_fn = judge or _make_default_judge(agent)
    results: list[CaseResult] = []

    for case in cases:
        try:
            output = agent.run(case.prompt)
            actual = _stringify(output)
            passed, detail = _score(case, actual, judge_fn)
        except Exception as exc:  # noqa: BLE001 - surface as a failing case, not a crash
            actual = ""
            passed, detail = False, f"error: {exc}"
        results.append(
            CaseResult(
                name=case.name,
                passed=passed,
                matcher=case.matcher,
                expected=case.expected,
                actual=actual,
                detail=detail,
            )
        )

    total_usd = max(_client_cost(agent) - start_usd, 0.0)
    return EvalReport(results=results, total_usd=total_usd)


def format_report(report: EvalReport) -> str:
    """Render a human-readable summary of an :class:`EvalReport`."""

    lines = [
        f"pass rate: {report.passed}/{report.total} ({report.pass_rate:.0%})",
        f"total cost: ${report.total_usd:.6f}",
        "",
    ]
    for result in report.results:
        marker = "PASS" if result.passed else "FAIL"
        line = f"[{marker}] {result.name} ({result.matcher})"
        if not result.passed and result.detail:
            line += f" -- {result.detail}"
        lines.append(line)
    return "\n".join(lines)


def _score(
    case: EvalCase,
    actual: str,
    judge_fn: Callable[[EvalCase, str], tuple[bool, str]],
) -> tuple[bool, str]:
    if case.matcher == "exact":
        passed = _normalize(actual, case) == _normalize(case.expected, case)
        return passed, "" if passed else "exact mismatch"
    if case.matcher == "contains":
        passed = _normalize(case.expected, case) in _normalize(actual, case)
        return passed, "" if passed else "expected substring not found"
    if case.matcher == "regex":
        flags = 0 if case.case_sensitive else re.IGNORECASE
        passed = re.search(case.expected, actual, flags) is not None
        return passed, "" if passed else "pattern did not match"
    if case.matcher == "llm_judge":
        return judge_fn(case, actual)
    # Unreachable: matcher is validated in EvalCase.__post_init__.
    raise EvalError(f"unknown matcher: {case.matcher!r}")  # pragma: no cover


def _normalize(text: str, case: EvalCase) -> str:
    text = text.strip()
    return text if case.case_sensitive else text.casefold()


class _JudgeVerdict(BaseModel):
    passed: bool
    reason: str


def _make_default_judge(agent: Agent) -> Callable[[EvalCase, str], tuple[bool, str]]:
    def judge(case: EvalCase, actual: str) -> tuple[bool, str]:
        prompt = (
            "You are grading an AI answer against an expected answer.\n"
            f"Question: {case.prompt}\n"
            f"Expected: {case.expected}\n"
            f"Answer: {actual}\n"
            "Return JSON {\"passed\": bool, \"reason\": str}. "
            "Pass if the answer is substantively correct even if worded differently."
        )
        verdict = agent.client.generate_structured(prompt, _JudgeVerdict)
        assert isinstance(verdict, _JudgeVerdict)
        return verdict.passed, verdict.reason

    return judge


def _client_cost(agent: Agent) -> float:
    tracker = getattr(agent.client, "cost_tracker", None)
    if tracker is None:
        return 0.0
    summary = tracker.summary()
    return float(summary.get("total_usd", 0.0))


def _stringify(output: str | BaseModel) -> str:
    if isinstance(output, BaseModel):
        return json.dumps(output.model_dump(), ensure_ascii=False, default=str)
    return str(output)


def _load_case_data(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            yaml = __import__("yaml")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EvalError("PyYAML is required to load YAML eval suites") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _require_dict(index: int, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise EvalError(f"case #{index} must be an object")
    return item
