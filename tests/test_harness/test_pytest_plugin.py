"""Tests for the pytest helper module."""

from pathlib import Path

from tests.conftest import requires_inspect


def test_discover_plan_scenarios_returns_yaml_paths(tmp_path: Path):
    from ave.harness.pytest_plugin import discover_plan_scenarios

    (tmp_path / "a.yaml").write_text("id: a\ntiers: [plan]\nprompt: p")
    (tmp_path / "b.yaml").write_text("id: b\ntiers: [execute]\nprompt: p")
    (tmp_path / "ignored.txt").write_text("not a scenario")

    scenarios = discover_plan_scenarios(tmp_path)
    names = sorted(Path(p).name for p in scenarios)
    # Only YAML files; both a and b are returned because discover does NOT
    # filter by tier (filtering happens when the scenario is loaded).
    assert names == ["a.yaml", "b.yaml"]


def test_discover_plan_scenarios_sorts_stably(tmp_path: Path):
    from ave.harness.pytest_plugin import discover_plan_scenarios

    (tmp_path / "z.yaml").write_text("id: z\ntiers: [plan]\nprompt: p")
    (tmp_path / "a.yaml").write_text("id: a\ntiers: [plan]\nprompt: p")

    scenarios = discover_plan_scenarios(tmp_path)
    assert [Path(p).name for p in scenarios] == ["a.yaml", "z.yaml"]


def test_bundled_scenarios_discovered():
    from ave.harness.pytest_plugin import discover_plan_scenarios, bundled_scenarios_dir

    scenarios = discover_plan_scenarios(bundled_scenarios_dir())
    names = {Path(p).name for p in scenarios}
    assert "reel.filler-word-trim.yaml" in names
    assert "short.highlight-reel-from-long.yaml" in names
    assert "talking-head.subtitled-vertical.yaml" in names


@requires_inspect
def test_run_plan_scenario_helper_returns_ok_for_irrelevance_scenario(tmp_path: Path):
    """The convenience helper runs one scenario and returns True on pass."""
    from ave.harness.pytest_plugin import run_plan_scenario

    yaml_text = """
id: helper.smoke
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
    p = tmp_path / "s.yaml"
    p.write_text(yaml_text)

    ok = run_plan_scenario(str(p), model="mockllm/mock", log_dir=str(tmp_path / "logs"))
    assert ok is True
