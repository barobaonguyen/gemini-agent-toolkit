"""Command-line interface for gemini-agent-toolkit.

Subcommands::

    gat run "<task>"              one-shot agent, prints the answer + cost
    gat eval <suite>              run a golden-set eval suite and print the report
    gat cost <model> <in> <out>   price a token count from gat.pricing

The CLI reads ``GEMINI_API_KEY`` and ``GEMINI_MODEL`` from the environment; no
keys are ever read from arguments. Memory backend is selectable with
``--memory {memory,jsonl,sqlite}`` (path via ``--memory-path``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import TextIO

from gat.agent import Agent
from gat.client import GeminiClient
from gat.eval import evaluate, format_report, load_cases
from gat.memory import build_memory
from gat.pricing import PRICING, estimate_cost_usd, get_pricing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gat", description="Gemini agent toolkit CLI.")
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model id (default: $GEMINI_MODEL or gemini-2.5-flash).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a one-shot agent task.")
    run_parser.add_argument("task", help="Task prompt for the agent.")
    run_parser.add_argument("--system", default=None, help="Optional system instruction.")
    run_parser.add_argument(
        "--memory",
        choices=("memory", "jsonl", "sqlite"),
        default="memory",
        help="Memory backend (default: memory).",
    )
    run_parser.add_argument(
        "--memory-path",
        default=None,
        help="Path for jsonl/sqlite memory backends.",
    )
    run_parser.add_argument(
        "--json", action="store_true", help="Emit task output plus cost summary as JSON."
    )
    run_parser.set_defaults(func=_cmd_run)

    eval_parser = sub.add_parser("eval", help="Run a golden-set eval suite.")
    eval_parser.add_argument("suite", help="Path to a JSON/YAML eval suite.")
    eval_parser.add_argument("--system", default=None, help="Optional system instruction.")
    eval_parser.add_argument(
        "--json", action="store_true", help="Emit the full report as JSON."
    )
    eval_parser.set_defaults(func=_cmd_eval)

    cost_parser = sub.add_parser("cost", help="Price a token count from gat.pricing.")
    cost_parser.add_argument("price_model", help="Model id to price (e.g. gemini-2.5-flash).")
    cost_parser.add_argument("input_tokens", type=int, help="Number of input tokens.")
    cost_parser.add_argument("output_tokens", type=int, help="Number of output tokens.")
    cost_parser.add_argument(
        "--cached-tokens", type=int, default=0, help="Cached input tokens (default: 0)."
    )
    cost_parser.add_argument(
        "--json", action="store_true", help="Emit the cost breakdown as JSON."
    )
    cost_parser.set_defaults(func=_cmd_cost)

    return parser


def _build_agent(args: argparse.Namespace) -> Agent:
    client = GeminiClient(model=args.model)
    memory = build_memory(
        getattr(args, "memory", "memory"),
        path=getattr(args, "memory_path", None),
    )
    return Agent(client=client, memory=memory, system=args.system)


def _cmd_run(args: argparse.Namespace, out: TextIO) -> int:
    agent = _build_agent(args)
    result = agent.run(args.task)
    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result.model_dump(), ensure_ascii=False)
    summary = agent.client.cost_tracker.summary()
    if args.json:
        out.write(json.dumps({"output": text, "cost": summary}, ensure_ascii=False) + "\n")
    else:
        out.write(text + "\n")
        out.write(f"[cost] ${summary['total_usd']:.6f} over {summary['calls']} call(s)\n")
    return 0


def _cmd_eval(args: argparse.Namespace, out: TextIO) -> int:
    cases = load_cases(args.suite)
    client = GeminiClient(model=args.model)
    agent = Agent(client=client, system=args.system)
    report = evaluate(agent, cases)
    if args.json:
        out.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
    else:
        out.write(format_report(report) + "\n")
    return 0 if report.passed == report.total else 1


def _cmd_cost(args: argparse.Namespace, out: TextIO) -> int:
    usd = estimate_cost_usd(
        args.price_model,
        args.input_tokens,
        args.output_tokens,
        cached_tokens=args.cached_tokens,
    )
    known = get_pricing(args.price_model) is not None
    if args.json:
        out.write(
            json.dumps(
                {
                    "model": args.price_model,
                    "known_pricing": known,
                    "input_tokens": args.input_tokens,
                    "output_tokens": args.output_tokens,
                    "cached_tokens": args.cached_tokens,
                    "usd": usd,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    else:
        if not known:
            out.write(
                f"[warn] no frozen pricing for {args.price_model!r}; "
                f"known models: {', '.join(sorted(PRICING))}\n"
            )
        out.write(f"${usd:.8f}\n")
    return 0


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = out if out is not None else sys.stdout
    func = args.func
    return int(func(args, stream))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
