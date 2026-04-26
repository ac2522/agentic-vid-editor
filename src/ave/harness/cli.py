"""`ave-harness` CLI.

Subcommands
-----------

    ave-harness run --scenario-file path/to/scenario.yaml [--tier plan|execute|render]
                    [--model mockllm/mock] [--log-dir ./logs]

    ave-harness export-dataset --output dataset.jsonl
                               [--scenarios-dir src/ave/harness/scenarios]
                               [--name harness]

    ave-harness analyze-log --log-file path/to/run.eval [--failures-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SUPPORTED_TIERS = ("plan", "execute", "render")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ave-harness", description="AVE harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a scenario at a given tier")
    run.add_argument("--scenario-file", required=True, help="Path to the scenario YAML")
    run.add_argument(
        "--tier",
        default="plan",
        choices=list(SUPPORTED_TIERS),
        help="Evaluation tier: plan, execute, or render (full rendered MP4 + VLM judge)",
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

    export = sub.add_parser(
        "export-dataset",
        help="Export harness scenarios as an opik-compatible EvalDataset JSONL",
    )
    export.add_argument("--output", required=True, help="Output JSONL path")
    export.add_argument(
        "--scenarios-dir",
        default=None,
        help="Directory of scenario YAMLs (defaults to bundled flagship scenarios)",
    )
    export.add_argument(
        "--name",
        default="harness",
        help="Dataset name (default: harness)",
    )

    analyze = sub.add_parser(
        "analyze-log",
        help="Parse an Inspect AI eval log into structured FeedbackRow records",
    )
    analyze.add_argument("--log-file", required=True, help="Path to a .eval log file")
    analyze.add_argument(
        "--failures-only",
        action="store_true",
        help="Print only failing rows (training-signal mode)",
    )

    return p


def _run_subcommand(ns: argparse.Namespace) -> int:
    try:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import get_model

        from ave.harness.task import (
            execute_rung_task,
            plan_rung_task,
            render_rung_task,
        )
    except ImportError as exc:
        print(
            f"error: harness deps missing ({exc}). Install with `pip install ave[harness]`",
            file=sys.stderr,
        )
        return 3

    if ns.tier == "plan":
        harness_task = plan_rung_task(scenario_file=ns.scenario_file)
    elif ns.tier == "execute":
        harness_task = execute_rung_task(scenario_file=ns.scenario_file)
    else:
        harness_task = render_rung_task(scenario_file=ns.scenario_file)

    model = get_model(ns.model)
    log_dir = ns.log_dir or "./logs"

    results = inspect_eval(harness_task, model=model, display="plain", log_dir=log_dir)
    if not results:
        print("error: no eval results produced", file=sys.stderr)
        return 4
    return 0


def _export_dataset_subcommand(ns: argparse.Namespace) -> int:
    from ave.harness.feedback.scenarios_to_dataset import (
        scenarios_to_dataset,
        write_dataset_to_jsonl,
    )
    from ave.harness.loader import load_scenario_from_yaml
    from ave.harness.pytest_plugin import (
        bundled_scenarios_dir,
        discover_plan_scenarios,
    )

    scenarios_dir = Path(ns.scenarios_dir) if ns.scenarios_dir else bundled_scenarios_dir()
    paths = discover_plan_scenarios(scenarios_dir)
    if not paths:
        print(f"error: no .yaml scenarios in {scenarios_dir}", file=sys.stderr)
        return 5
    scenarios = [load_scenario_from_yaml(Path(p)) for p in paths]
    dataset = scenarios_to_dataset(scenarios, name=ns.name)
    n = write_dataset_to_jsonl(dataset, Path(ns.output))
    print(f"wrote {n} rows to {ns.output}")
    return 0


def _analyze_log_subcommand(ns: argparse.Namespace) -> int:
    try:
        from ave.harness.feedback.eval_log import (
            eval_log_to_feedback_rows,
            summarize_failures,
        )
    except ImportError as exc:
        print(f"error: feedback module missing ({exc})", file=sys.stderr)
        return 3

    log_path = Path(ns.log_file)
    if not log_path.exists():
        print(f"error: log file not found: {log_path}", file=sys.stderr)
        return 6

    rows = eval_log_to_feedback_rows(log_path)
    if ns.failures_only:
        rows = summarize_failures(rows)
    for row in rows:
        verdict = "PASS" if row.passed else "FAIL"
        print(json.dumps({
            "sample_id": row.sample_id,
            "scorer": row.scorer_name,
            "verdict": verdict,
            "rule": row.verdict_rule,
            "reason": row.reason,
            "called_tools": row.called_tools,
            "expected_tools": list(row.expected_tools),
        }))
    print(f"# {len(rows)} row(s)", file=sys.stderr)
    return 0


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "run":
        return _run_subcommand(ns)
    if ns.command == "export-dataset":
        return _export_dataset_subcommand(ns)
    if ns.command == "analyze-log":
        return _analyze_log_subcommand(ns)

    parser.error(f"unknown command: {ns.command}")
    return 2


def main() -> None:
    sys.exit(cli_main(sys.argv[1:]))
