"""Inspect AI scorer wrapper tests for state-diff."""
import pytest
from tests.conftest import requires_inspect


@requires_inspect
@pytest.mark.asyncio
async def test_state_diff_scorer_passes_on_valid_state():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from ave.harness.schema import ExecuteExpected, MinMax, TimelineBounds, Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.scorers.state_diff import state_diff_scorer

    xges = """<ges version="0.3"><project><timeline>
      <layer priority="0">
        <clip id="c1" asset-id="a" type-name="GESUriClip" start="0" duration="60000000000" inpoint="0" rate="0"/>
      </layer></timeline></project></ges>"""

    scenario = Scenario(
        id="t", tiers=("execute",), prompt="",
        scope=ScopeSpec(), inputs=Inputs(),
        expected=Expected(execute=ExecuteExpected(
            timeline=TimelineBounds(clip_count=MinMax(min=1, max=5))
        ))
    )
    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[],
        metadata={"scenario": scenario, "final_xges": xges, "called_tools": []}
    )
    score = await state_diff_scorer()(state, Target("t"))
    assert score.value == 1


@requires_inspect
@pytest.mark.asyncio
async def test_state_diff_scorer_fails_on_clip_count_violation():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from ave.harness.schema import ExecuteExpected, MinMax, TimelineBounds, Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.scorers.state_diff import state_diff_scorer

    empty_xges = """<ges version="0.3"><project><timeline><layer priority="0"/></timeline></project></ges>"""
    scenario = Scenario(
        id="t", tiers=("execute",), prompt="",
        scope=ScopeSpec(), inputs=Inputs(),
        expected=Expected(execute=ExecuteExpected(
            timeline=TimelineBounds(clip_count=MinMax(min=2))
        ))
    )
    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[],
        metadata={"scenario": scenario, "final_xges": empty_xges, "called_tools": []}
    )
    score = await state_diff_scorer()(state, Target("t"))
    assert score.value == 0


@requires_inspect
@pytest.mark.asyncio
async def test_state_diff_scorer_skips_when_no_final_xges():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from ave.harness.schema import Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.scorers.state_diff import state_diff_scorer

    scenario = Scenario(id="t", tiers=("execute",), prompt="", scope=ScopeSpec(), inputs=Inputs(), expected=Expected())
    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[], metadata={"scenario": scenario}
    )
    score = await state_diff_scorer()(state, Target("t"))
    assert score.value == 1
    assert "skipping" in (score.explanation or "").lower()
