"""Plan-solver tests using Inspect AI's mockllm backend."""

import pytest

from tests.conftest import requires_inspect


@requires_inspect
@pytest.mark.asyncio
async def test_plan_solver_records_tool_calls_from_mockllm():
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ToolCall
    from inspect_ai.solver import TaskState

    from ave.harness.schema import (
        Expected,
        InputAsset,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.solvers.plan import extract_tool_calls

    # Build a state with a synthetic tool-use assistant message.
    scenario = Scenario(
        id="test",
        tiers=("plan",),
        prompt="do it",
        scope=ScopeSpec(allowed_agents=("editor",), forbidden_layers=()),
        inputs=Inputs(),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("trim",)),
            )
        ),
    )
    sample = Sample(id="test", input="do it", target="test", metadata={"scenario": scenario})

    state = TaskState(
        model="mockllm/mock",
        sample_id="test",
        epoch=0,
        input=sample.input,
        messages=[
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id="1", function="trim", arguments={"clip_id": "c1"}),
                    ToolCall(id="2", function="find_fillers", arguments={}),
                ],
            ),
            ChatMessageTool(content="trim ok", tool_call_id="1"),
            ChatMessageTool(content="fillers ok", tool_call_id="2"),
        ],
    )

    called = extract_tool_calls(state)
    assert called == ["trim", "find_fillers"]


@requires_inspect
def test_plan_solver_is_decorated_solver():
    """The plan_solver factory returns something Inspect AI accepts as a Solver."""
    from ave.harness.solvers.plan import plan_solver

    s = plan_solver()
    # Inspect AI solvers are callables (coroutines once awaited); verifying
    # it's callable avoids coupling to the internal Solver type.
    assert callable(s)


@requires_inspect
@pytest.mark.asyncio
async def test_plan_solver_wires_expected_tools():
    """When a scenario declares tool constraints, the solver exposes stubs for them."""
    from inspect_ai.tool import ToolDef

    from ave.harness.schema import (
        Expected,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.solvers.plan import _tools_for_scenario

    scenario = Scenario(
        id="test",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("trim",), any_of=("text_cut",)),
                tools_forbidden=("apply_blend_mode",),
            )
        ),
    )
    tools = _tools_for_scenario(scenario)
    assert len(tools) == 3
    # Tools union the all_of, any_of, and forbidden sets. ToolDef(t).name is
    # the only stable way to recover the declared tool name from the wrapped
    # callable returned by ToolDef.as_tool() in Inspect AI >= 0.3.
    names = {ToolDef(t).name for t in tools}
    assert names == {"trim", "text_cut", "apply_blend_mode"}


@requires_inspect
@pytest.mark.asyncio
async def test_plan_solver_stub_tool_is_non_executing_and_callable():
    """Each stub tool returns a short OK string and accepts arbitrary kwargs."""
    from inspect_ai.tool import ToolDef

    from ave.harness.schema import (
        Expected,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.solvers.plan import _tools_for_scenario

    scenario = Scenario(
        id="test",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(
            plan=PlanExpected(tools_required=ToolsRequired(all_of=("trim",))),
        ),
    )
    tools = _tools_for_scenario(scenario)
    assert len(tools) == 1
    # Build a kwargs dict that matches the real trim parameter set; the stub
    # should accept it and return a predictable confirmation string.
    result = await tools[0](clip_duration_ns=1, in_ns=0, out_ns=1)
    assert "trim" in result and "stub" in result
    assert ToolDef(tools[0]).name == "trim"
