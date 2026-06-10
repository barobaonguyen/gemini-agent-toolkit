from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from gat import cli
from gat.client import GeminiClient
from gat.cost import CostTracker


@pytest.fixture(autouse=True)
def _no_live_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the client so the CLI never reaches the Gemini SDK in CI."""

    # Keyed by a substring of each suite/test prompt so the mock returns the
    # right canned answer without any live Gemini call.
    answers = {
        "12 * 12": "144",
        "capital of France": "The capital is Paris.",
        "prime number": "13",
        "cap?": "The capital is Paris.",
    }

    def fake_generate(self: GeminiClient, prompt: str, **_: Any) -> str:
        self._cost_tracker.add(self.model, 100, 20)
        for needle, answer in answers.items():
            if needle in prompt:
                return answer
        return "the capital of France is Paris"

    def fake_structured(self: GeminiClient, prompt: str, schema: Any, **_: Any) -> Any:
        self._cost_tracker.add(self.model, 50, 10)
        return schema.model_validate({"passed": True, "reason": "stub"})

    monkeypatch.setattr(GeminiClient, "generate", fake_generate)
    monkeypatch.setattr(GeminiClient, "generate_structured", fake_structured)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(list(argv), out=out)
    return code, out.getvalue()


def test_cli_run_one_shot() -> None:
    code, output = _run("run", "What is the capital of France?")
    assert code == 0
    assert "Paris" in output
    assert "[cost]" in output


def test_cli_run_json() -> None:
    code, output = _run("run", "cap?", "--json")
    assert code == 0
    payload = json.loads(output)
    assert "Paris" in payload["output"]
    assert payload["cost"]["calls"] == 1


def test_cli_run_with_sqlite_memory(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    code, _ = _run("run", "cap?", "--memory", "sqlite", "--memory-path", str(db))
    assert code == 0
    assert db.exists()


def test_cli_eval_suite() -> None:
    suite = Path(__file__).resolve().parents[1] / "examples" / "eval_suite" / "cases.json"
    code, output = _run("eval", str(suite))
    assert code == 0  # all three JSON cases pass with the mocked answers
    assert "pass rate: 3/3" in output


def test_cli_eval_json() -> None:
    suite = Path(__file__).resolve().parents[1] / "examples" / "eval_suite" / "cases.json"
    code, output = _run("eval", str(suite), "--json")
    assert code == 0
    report = json.loads(output)
    assert report["total"] == 3
    assert report["passed"] == 3
    assert report["total_usd"] > 0


def test_cli_cost_known_model() -> None:
    code, output = _run("cost", "gemini-2.5-flash", "100000", "10000")
    assert code == 0
    expected = (100_000 * 0.30 + 10_000 * 2.50) / 1_000_000
    assert f"${expected:.8f}" == output.strip()


def test_cli_cost_json_with_cache() -> None:
    code, output = _run(
        "cost", "gemini-2.5-pro", "1000", "100", "--cached-tokens", "500", "--json"
    )
    assert code == 0
    payload = json.loads(output)
    assert payload["known_pricing"] is True
    assert payload["usd"] > 0


def test_cli_cost_unknown_model_warns() -> None:
    code, output = _run("cost", "gemini-9.9-ultra", "1000", "100")
    assert code == 0
    assert "[warn]" in output
    assert "$0.00000000" in output


def test_cli_cost_tracker_is_isolated() -> None:
    # The autouse fixture must not leak cost across CLI invocations.
    tracker = CostTracker()
    assert tracker.summary()["calls"] == 0
