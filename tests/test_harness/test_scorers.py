"""Tests for Inspect AI scorer wrappers around the pure evaluators."""

import pytest

from tests.conftest import requires_inspect


@requires_inspect
@pytest.mark.asyncio
async def test_tool_selection_scorer_reads_state_metadata():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from ave.harness.schema import (
        Expected,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.scorers.tool_selection import tool_selection_scorer

    scenario = Scenario(
        id="t",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(allowed_agents=("editor",)),
        inputs=Inputs(),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("trim",)),
            )
        ),
    )
    state = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["trim", "concat"]},
    )

    scorer = tool_selection_scorer()
    score = await scorer(state, Target("t"))
    assert score.value == 1  # passing
    assert "plan satisfies" in (score.explanation or "").lower()


@requires_inspect
@pytest.mark.asyncio
async def test_tool_selection_scorer_fails_on_missing_required():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from ave.harness.schema import (
        Expected,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.scorers.tool_selection import tool_selection_scorer

    scenario = Scenario(
        id="t",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("trim", "text_cut")),
            )
        ),
    )
    state = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["trim"]},
    )

    score = await tool_selection_scorer()(state, Target("t"))
    assert score.value == 0


@requires_inspect
@pytest.mark.asyncio
async def test_scope_scorer_respects_forbidden_layers():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from ave.harness.schema import Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.scorers.scope import scope_scorer

    scenario = Scenario(
        id="t",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(allowed_agents=("sound_designer",), forbidden_layers=("video",)),
        inputs=Inputs(),
        expected=Expected(),
    )
    # "trim" touches TIMELINE_STRUCTURE in the default registry; doesn't violate "video"
    state_ok = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["trim"]},
    )
    score_ok = await scope_scorer()(state_ok, Target("t"))
    assert score_ok.value == 1

    # "apply_blend_mode" touches VIDEO -> violation
    state_bad = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["apply_blend_mode"]},
    )
    score_bad = await scope_scorer()(state_bad, Target("t"))
    assert score_bad.value == 0
