"""Tests for the execute-tier solver."""

import pytest
from tests.conftest import requires_ges, requires_inspect


@requires_inspect
def test_execute_solver_is_callable():
    from ave.harness.solvers.execute import execute_solver
    s = execute_solver()
    assert callable(s)


@requires_inspect
def test_minimal_xges_template_is_valid_xml():
    from ave.harness.solvers.execute import MINIMAL_XGES
    import xml.etree.ElementTree as ET
    root = ET.fromstring(MINIMAL_XGES)
    assert root.tag == "ges"


@requires_inspect
def test_execute_solver_module_importable():
    import ave.harness.solvers.execute  # noqa: F401


@requires_inspect
@requires_ges
@pytest.mark.asyncio
async def test_execute_solver_populates_metadata(tmp_path):
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import TaskState
    from ave.harness.schema import Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.solvers.execute import execute_solver

    scenario = Scenario(
        id="test.execute",
        tiers=("execute",),
        prompt="load the project",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(),
    )

    state = TaskState(
        model="mockllm/mock",
        sample_id="test",
        epoch=0,
        input="load the project",
        messages=[],
        metadata={"scenario": scenario},
    )

    solver = execute_solver()

    async def fake_generate(state):
        return state

    result = await solver(state, fake_generate)
    assert "called_tools" in result.metadata
    assert "final_xges" in result.metadata
    assert "snapshot_count" in result.metadata
    assert "activity_entries" in result.metadata
    assert isinstance(result.metadata["called_tools"], list)
    assert isinstance(result.metadata["final_xges"], str)
    assert isinstance(result.metadata["snapshot_count"], int)
    assert isinstance(result.metadata["activity_entries"], list)


@requires_inspect
def test_execute_solver_handles_no_scenario():
    """Solver should not crash when no scenario in metadata."""
    import asyncio
    from inspect_ai.solver import TaskState
    from ave.harness.solvers.execute import execute_solver

    state = TaskState(
        model="mockllm/mock",
        sample_id="test",
        epoch=0,
        input="do it",
        messages=[],
        metadata={},
    )

    async def fake_generate(s):
        return s

    solver = execute_solver()
    result = asyncio.run(solver(state, fake_generate))
    assert "called_tools" in result.metadata


@requires_inspect
def test_execute_solver_handles_empty_tool_list():
    """Solver should work when scenario has no tool expectations."""
    import asyncio
    from inspect_ai.solver import TaskState
    from ave.harness.schema import Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.solvers.execute import execute_solver

    scenario = Scenario(
        id="test.empty",
        tiers=("execute",),
        prompt="do nothing",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(),
    )

    state = TaskState(
        model="mockllm/mock",
        sample_id="test",
        epoch=0,
        input="do nothing",
        messages=[],
        metadata={"scenario": scenario},
    )

    async def fake_generate(s):
        return s

    solver = execute_solver()
    result = asyncio.run(solver(state, fake_generate))
    assert result.metadata["called_tools"] == []
    assert isinstance(result.metadata["snapshot_count"], int)
    assert isinstance(result.metadata["final_xges"], str)
    assert isinstance(result.metadata["activity_entries"], list)
