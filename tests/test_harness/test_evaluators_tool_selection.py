"""Pure tool-selection evaluator tests (no Inspect AI dependency)."""

from ave.harness.evaluators.tool_selection import Verdict, evaluate_plan
from ave.harness.schema import PlanExpected, ToolsRequired


def _plan(all_of=(), any_of=(), forbidden=(), irrelevance=False) -> PlanExpected:
    return PlanExpected(
        tools_required=ToolsRequired(all_of=tuple(all_of), any_of=tuple(any_of)),
        tools_forbidden=tuple(forbidden),
        irrelevance_allowed=irrelevance,
    )


def test_all_of_satisfied():
    v = evaluate_plan(["find_fillers", "text_cut"], _plan(all_of=["find_fillers", "text_cut"]))
    assert v.passed is True


def test_all_of_missing():
    v = evaluate_plan(["find_fillers"], _plan(all_of=["find_fillers", "text_cut"]))
    assert v.passed is False
    assert "text_cut" in v.reason
    assert "missing" in v.reason.lower()


def test_any_of_satisfied_with_one():
    v = evaluate_plan(["trim"], _plan(any_of=["trim", "text_cut"]))
    assert v.passed is True


def test_any_of_not_satisfied():
    v = evaluate_plan(["apply_blend_mode"], _plan(any_of=["trim", "text_cut"]))
    assert v.passed is False
    assert "any_of" in v.reason.lower()


def test_forbidden_called_fails():
    v = evaluate_plan(
        ["find_fillers", "apply_blend_mode"],
        _plan(
            all_of=["find_fillers"],
            forbidden=["apply_blend_mode"],
        ),
    )
    assert v.passed is False
    assert "forbidden" in v.reason.lower()


def test_all_and_any_both_required():
    v_pass = evaluate_plan(
        ["find_fillers", "trim"],
        _plan(all_of=["find_fillers"], any_of=["trim", "text_cut"]),
    )
    assert v_pass.passed is True

    v_fail = evaluate_plan(
        ["find_fillers"],  # missing any_of member
        _plan(all_of=["find_fillers"], any_of=["trim", "text_cut"]),
    )
    assert v_fail.passed is False


def test_irrelevance_no_tools_called_passes_when_allowed():
    v = evaluate_plan([], _plan(all_of=["find_fillers"], irrelevance=True))
    assert v.passed is True
    assert "irrelevance" in v.reason.lower()


def test_irrelevance_no_tools_called_fails_when_not_allowed():
    v = evaluate_plan([], _plan(all_of=["find_fillers"], irrelevance=False))
    assert v.passed is False


def test_verdict_is_hashable_frozen_dataclass():
    v = Verdict(passed=True, reason="ok")
    assert v.passed is True
    assert v.reason == "ok"
    assert v.rule == "ok"


def test_duplicate_tools_do_not_affect_verdict():
    """Duplicate tool names in called_tools are treated as a set for matching."""
    v = evaluate_plan(
        ["trim", "trim", "find_fillers"],
        _plan(all_of=["trim", "find_fillers"]),
    )
    assert v.passed is True


def test_empty_constraints_and_nonempty_called_pass():
    """A plan with no required/forbidden/any_of and tools called -> pass."""
    v = evaluate_plan(["anything"], _plan())
    assert v.passed is True
    assert v.rule == "ok"


def test_rules_are_machine_readable():
    """Each branch populates a distinct rule tag."""
    assert evaluate_plan([], _plan(irrelevance=True)).rule == "irrelevance_ok"
    assert evaluate_plan([], _plan()).rule == "irrelevance_required"
    assert evaluate_plan(["x"], _plan(forbidden=["x"])).rule == "forbidden_tools"
    assert evaluate_plan(["y"], _plan(all_of=["x"])).rule == "missing_all_of"
    assert evaluate_plan(["x"], _plan(all_of=["x"], any_of=["y"])).rule == "missing_any_of"
