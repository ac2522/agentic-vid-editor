"""Tests for YAML scenario loading."""

import pytest

pytest.importorskip("yaml")

from pathlib import Path

from ave.harness.loader import load_scenario_from_yaml
from ave.harness.schema import Scenario


_VALID = """
id: reel.filler-word-trim
description: "Remove filler words from an interview clip"
tiers: [plan]
prompt: "clean up the ums and ahs in this interview clip"
scope:
  allowed_agents: [transcriptionist, editor]
  forbidden_layers: []
inputs:
  assets:
    - id: clip1
      ref: "fixture://interview-60s.mp4"
expected:
  plan:
    tools_required:
      all_of: [find_fillers]
      any_of: [text_cut, trim]
    tools_forbidden: [apply_blend_mode]
    irrelevance_allowed: false
"""


def test_load_valid_yaml(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text(_VALID)
    scenario = load_scenario_from_yaml(p)
    assert isinstance(scenario, Scenario)
    assert scenario.id == "reel.filler-word-trim"
    assert "find_fillers" in scenario.expected.plan.tools_required.all_of


def test_load_invalid_yaml_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: scenario: schema")
    with pytest.raises(Exception):
        load_scenario_from_yaml(p)


def test_load_missing_required_field(tmp_path: Path):
    p = tmp_path / "missing.yaml"
    p.write_text("id: x\n")  # missing tiers, prompt
    with pytest.raises(Exception):
        load_scenario_from_yaml(p)


def test_load_nonexistent_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_scenario_from_yaml(tmp_path / "nope.yaml")
