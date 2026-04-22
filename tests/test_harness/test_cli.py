"""Tests for the `ave-harness` CLI."""

from pathlib import Path

import pytest

from tests.conftest import requires_inspect


def test_parse_args_run_subcommand_requires_scenario():
    from ave.harness.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_parse_args_run_with_scenario_file():
    from ave.harness.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["run", "--scenario-file", "x.yaml", "--tier", "plan"])
    assert ns.command == "run"
    assert ns.scenario_file == "x.yaml"
    assert ns.tier == "plan"


def test_parse_args_default_tier_is_plan():
    from ave.harness.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["run", "--scenario-file", "x.yaml"])
    assert ns.tier == "plan"


def test_run_rejects_unsupported_tier():
    from ave.harness.cli import cli_main

    rc = cli_main(["run", "--scenario-file", "x.yaml", "--tier", "execute"])
    assert rc != 0


@requires_inspect
def test_run_plan_tier_on_scenario_returns_zero_on_eval(tmp_path: Path):
    """Running a trivial scenario via the CLI returns 0 (smoke test)."""
    from ave.harness.cli import cli_main

    yaml_text = """
id: cli.smoke
description: ""
tiers: [plan]
prompt: "do it"
scope:
  allowed_agents: [editor]
  forbidden_layers: []
inputs:
  assets: []
expected:
  plan:
    tools_required:
      all_of: []
      any_of: []
    tools_forbidden: []
    irrelevance_allowed: true
"""
    s = tmp_path / "s.yaml"
    s.write_text(yaml_text)

    rc = cli_main(
        [
            "run",
            "--scenario-file",
            str(s),
            "--tier",
            "plan",
            "--model",
            "mockllm/mock",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    assert rc == 0
