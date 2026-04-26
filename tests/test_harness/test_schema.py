"""Tests for the harness Scenario Pydantic schema."""

import pytest
from pydantic import ValidationError

from ave.harness.schema import (
    InputAsset,
    PlanExpected,
    Scenario,
    ScopeSpec,
)


def _minimal_scenario_dict() -> dict:
    return {
        "id": "reel.test",
        "description": "Minimal test scenario",
        "tiers": ["plan"],
        "prompt": "do a thing",
        "scope": {
            "allowed_agents": ["editor"],
            "forbidden_layers": [],
        },
        "inputs": {
            "assets": [{"id": "clip1", "ref": "fixture://testclip.mp4"}],
        },
        "expected": {
            "plan": {
                "tools_required": {"all_of": ["trim"], "any_of": []},
                "tools_forbidden": [],
                "irrelevance_allowed": False,
            }
        },
    }


def test_scenario_minimal_loads():
    s = Scenario.model_validate(_minimal_scenario_dict())
    assert s.id == "reel.test"
    assert s.tiers == ("plan",)
    assert s.prompt == "do a thing"
    assert s.expected.plan is not None


def test_scenario_rejects_unknown_tier():
    d = _minimal_scenario_dict()
    d["tiers"] = ["plan", "nonsense"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_scope_spec_accepts_empty_lists():
    s = ScopeSpec(allowed_agents=[], forbidden_layers=[])
    assert s.allowed_agents == ()
    assert s.forbidden_layers == ()


def test_input_asset_ref_required():
    with pytest.raises(ValidationError):
        InputAsset.model_validate({"id": "x"})


def test_plan_expected_defaults_empty():
    pe = PlanExpected.model_validate({})
    assert pe.tools_required.all_of == ()
    assert pe.tools_required.any_of == ()
    assert pe.tools_forbidden == ()
    assert pe.irrelevance_allowed is False


def test_scenario_without_plan_expected_is_allowed():
    """Scenarios may declare only a subset of tiers — that's fine."""
    d = _minimal_scenario_dict()
    d["tiers"] = ["execute"]
    d["expected"] = {}
    s = Scenario.model_validate(d)
    assert s.expected.plan is None


def test_populated_lists_coerce_to_tuples():
    s = ScopeSpec(allowed_agents=["editor", "colorist"], forbidden_layers=["video"])
    assert s.allowed_agents == ("editor", "colorist")
    assert s.forbidden_layers == ("video",)
    assert isinstance(s.allowed_agents, tuple)


def test_unknown_plan_field_rejected():
    """extra='forbid' catches typos and premature/stub fields like arg_constraints."""
    with pytest.raises(ValidationError):
        PlanExpected.model_validate(
            {
                "arg_constraints": {"trim": {"in_point": ">0"}},  # Spec mentions this
            }
        )


# --- Phase 3: ExecuteExpected expansion tests ---

def test_execute_expected_defaults_no_constraints():
    from ave.harness.schema import ExecuteExpected, MinMax, TimelineBounds
    e = ExecuteExpected()
    assert e.timeline.clip_count.min is None
    assert e.timeline.clip_count.max is None
    assert e.timeline.duration_seconds.min is None
    assert e.snapshots_created.min is None
    assert e.activity_log_entries.min is None


def test_minmax_accepts_int_and_float():
    from ave.harness.schema import MinMax
    m = MinMax(min=1, max=10.5)
    assert m.min == 1.0
    assert m.max == 10.5


def test_minmax_accepts_none_bounds():
    from ave.harness.schema import MinMax
    m = MinMax()
    assert m.min is None
    assert m.max is None


def test_timeline_bounds_effects_coerce_to_tuples():
    from ave.harness.schema import TimelineBounds
    b = TimelineBounds(effects_applied=["blur"], effects_forbidden=["cdl"])
    assert b.effects_applied == ("blur",)
    assert b.effects_forbidden == ("cdl",)


def test_execute_expected_loads_from_full_dict():
    from ave.harness.schema import ExecuteExpected
    d = {
        "timeline": {
            "clip_count": {"min": 1, "max": 10},
            "duration_seconds": {"min": 25.0, "max": 35.0},
            "effects_applied": [],
            "effects_forbidden": ["cdl"],
        },
        "snapshots_created": {"min": 1},
        "activity_log_entries": {"min": 1},
    }
    e = ExecuteExpected.model_validate(d)
    assert e.timeline.clip_count.min == 1.0
    assert e.timeline.clip_count.max == 10.0
    assert e.timeline.duration_seconds.min == 25.0
    assert e.snapshots_created.min == 1.0
    assert e.activity_log_entries.min == 1.0
    assert e.timeline.effects_forbidden == ("cdl",)


def test_scenario_with_execute_block_loads():
    from ave.harness.schema import Scenario
    d = {
        "id": "test.execute",
        "tiers": ["execute"],
        "prompt": "trim something",
        "expected": {
            "execute": {
                "timeline": {"clip_count": {"min": 1}},
                "snapshots_created": {"min": 1},
                "activity_log_entries": {"min": 1},
            }
        },
    }
    s = Scenario.model_validate(d)
    assert s.expected.execute is not None
    assert s.expected.execute.timeline.clip_count.min == 1.0
