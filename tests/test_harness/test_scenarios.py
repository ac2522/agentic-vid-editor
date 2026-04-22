"""Tests that the bundled flagship scenarios load and validate cleanly."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from ave.harness.loader import load_scenario_from_yaml


SCENARIO_DIR = Path(__file__).parent.parent.parent / "src" / "ave" / "harness" / "scenarios"


def test_reel_filler_word_trim_loads():
    s = load_scenario_from_yaml(SCENARIO_DIR / "reel.filler-word-trim.yaml")
    assert s.id == "reel.filler-word-trim"
    assert "plan" in s.tiers
    # Realism: the required tool chain should mention find_fillers for fillers
    assert "find_fillers" in s.expected.plan.tools_required.all_of


def test_highlight_reel_loads():
    s = load_scenario_from_yaml(SCENARIO_DIR / "short.highlight-reel-from-long.yaml")
    assert s.id == "short.highlight-reel-from-long"
    # Requires scene detection to chunk the input
    assert "detect_scenes" in s.expected.plan.tools_required.all_of


def test_subtitled_vertical_loads():
    s = load_scenario_from_yaml(SCENARIO_DIR / "talking-head.subtitled-vertical.yaml")
    assert s.id == "talking-head.subtitled-vertical"
    # Requires text overlay for the captions
    assert "add_text_overlay" in s.expected.plan.tools_required.all_of


def test_all_flagship_scenarios_use_real_tool_names():
    """Every tool named in the required/forbidden lists must exist in the registry."""
    from ave.agent.session import EditingSession

    session = EditingSession()
    known = {t.name for t in session.registry.search_tools()}
    for yaml_name in (
        "reel.filler-word-trim.yaml",
        "short.highlight-reel-from-long.yaml",
        "talking-head.subtitled-vertical.yaml",
    ):
        s = load_scenario_from_yaml(SCENARIO_DIR / yaml_name)
        mentioned = set(s.expected.plan.tools_required.all_of)
        mentioned |= set(s.expected.plan.tools_required.any_of)
        mentioned |= set(s.expected.plan.tools_forbidden)
        unknown = mentioned - known
        assert not unknown, f"{yaml_name} references unknown tools: {unknown}"
