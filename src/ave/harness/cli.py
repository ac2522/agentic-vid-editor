"""`ave-harness` CLI.

Usage
-----

    ave-harness run --scenario-file path/to/scenario.yaml [--tier plan|execute]
                    [--model mockllm/mock] [--log-dir ./logs]
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


SUPPORTED_TIERS = ("plan", "execute")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ave-harness", description="AVE harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a scenario at a given tier")
    run.add_argument("--scenario-file", required=True, help="Path to the scenario YAML")
    run.add_argument(
        "--tier",
        default="plan",
        choices=list(SUPPORTED_TIERS),
        help="Evaluation tier: plan (tool selection) or execute (real tool execution)",
    )
    run.add_argument(
        "--model",
        default="mockllm/mock",
        help="Inspect AI model spec (e.g., anthropic/claude-opus-4-7, mockllm/mock)",
    )
    run.add_argument(
        "--log-dir",
        default=None,
        help="Directory for Inspect AI eval logs (default: ./logs)",
    )
    return p


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command != "run":
        parser.error(f"unknown command: {ns.command}")

    try:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import get_model

        from ave.harness.task import execute_rung_task, plan_rung_task
    except ImportError as exc:
        print(
            f"error: harness deps missing ({exc}). Install with `pip install ave[harness]`",
            file=sys.stderr,
        )
        return 3

    if ns.tier == "plan":
        harness_task = plan_rung_task(scenario_file=ns.scenario_file)
    else:
        harness_task = execute_rung_task(scenario_file=ns.scenario_file)

    model = get_model(ns.model)
    log_dir = ns.log_dir or "./logs"

    results = inspect_eval(harness_task, model=model, display="plain", log_dir=log_dir)
    if not results:
        print("error: no eval results produced", file=sys.stderr)
        return 4
    return 0


def main() -> None:
    sys.exit(cli_main(sys.argv[1:]))
