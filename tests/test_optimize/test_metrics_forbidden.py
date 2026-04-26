"""Tests for ForbiddenToolsPenalty metric."""

from ave.optimize.datasets import EvalItem
from ave.optimize.metrics import ForbiddenToolsPenalty, LLMResponse, ToolCall


def _item(forbidden=None) -> EvalItem:
    return EvalItem(
        id="t", task="x",
        expected_tools=["good"],
        expected_output_pattern=None,
        context={"tools_forbidden": list(forbidden or [])},
    )


def _resp(*names) -> LLMResponse:
    return LLMResponse(text="", tool_calls=tuple(ToolCall(name=n, arguments={}) for n in names))


def test_no_forbidden_constraint_returns_one():
    item = EvalItem(id="t", task="x", expected_tools=["a"], expected_output_pattern=None, context={})
    m = ForbiddenToolsPenalty()
    r = m.score(item, _resp("a", "b"))
    assert r.value == 1.0
    assert "no forbidden" in r.reason.lower()


def test_no_violation_returns_one():
    m = ForbiddenToolsPenalty()
    r = m.score(_item(forbidden=["bad"]), _resp("good", "ok"))
    assert r.value == 1.0


def test_single_violation_returns_zero():
    m = ForbiddenToolsPenalty()
    r = m.score(_item(forbidden=["cdl", "lut_parse"]), _resp("good", "cdl"))
    assert r.value == 0.0
    assert "cdl" in r.reason


def test_multiple_violations_listed_in_reason():
    m = ForbiddenToolsPenalty()
    r = m.score(_item(forbidden=["a", "b"]), _resp("good", "a", "b"))
    assert r.value == 0.0
    assert "a" in r.reason
    assert "b" in r.reason


def test_empty_response_passes():
    """No tool calls at all = no forbidden tools called."""
    m = ForbiddenToolsPenalty()
    r = m.score(_item(forbidden=["bad"]), _resp())
    assert r.value == 1.0


def test_metric_name_class_var():
    assert ForbiddenToolsPenalty.name == "forbidden_tools_penalty"


def test_metric_name_unique_from_tool_selection_accuracy():
    from ave.optimize.metrics import ToolSelectionAccuracy
    assert ForbiddenToolsPenalty.name != ToolSelectionAccuracy.name
