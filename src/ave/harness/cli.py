"""`ave-harness` CLI — minimal entry point for Phase 2.

Usage
-----

    ave-harness run --scenario-file path/to/scenario.yaml [--tier plan]
                    [--model mockllm/mock] [--log-dir ./logs]

Only the ``plan`` tier is implemented in Phase 2; the other tiers return a
non-zero exit code with a clear error until Phases 3 and 4 land.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


SUPPORTED_TIERS = ("plan",)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ave-harness", description="AVE harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a scenario at a given tier")
    run.add_argument("--scenario-file", required=True, help="Path to the scenario YAML")
    run.add_argument(
        "--tier",
        default="plan",
        help="Evaluation tier (only 'plan' is available in Phase 2)",
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

    if ns.tier not in SUPPORTED_TIERS:
        print(
            f"error: tier {ns.tier!r} not implemented in Phase 2 "
            f"(supported: {list(SUPPORTED_TIERS)})",
            file=sys.stderr,
        )
        return 2

    try:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import get_model

        from ave.harness.task import plan_rung_task
    except ImportError as exc:
        print(
            f"error: harness deps missing ({exc}). Install with `pip install ave[harness]`",
            file=sys.stderr,
        )
        return 3

    task = plan_rung_task(scenario_file=ns.scenario_file)
    model = get_model(ns.model)
    log_dir = ns.log_dir or "./logs"

    results = inspect_eval(task, model=model, display="plain", log_dir=log_dir)
    if not results:
        print("error: no eval results produced", file=sys.stderr)
        return 4
    return 0


def main() -> None:
    sys.exit(cli_main(sys.argv[1:]))
