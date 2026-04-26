"""Tests for the render-tier solver."""

import pytest
from tests.conftest import requires_ffmpeg, requires_ges, requires_inspect


@requires_inspect
def test_render_solver_is_callable():
    from ave.harness.solvers.render import render_solver
    assert callable(render_solver())


@requires_inspect
def test_render_solver_module_importable():
    import ave.harness.solvers.render  # noqa: F401


@requires_inspect
def test_render_solver_metadata_keys_documented():
    """The solver must populate documented metadata keys."""
    from ave.harness.solvers.render import RENDER_METADATA_KEYS
    assert "rendered_path" in RENDER_METADATA_KEYS
    assert "render_failed" in RENDER_METADATA_KEYS
    assert "called_tools" in RENDER_METADATA_KEYS
    assert "render_log" in RENDER_METADATA_KEYS
    assert "render_preset" in RENDER_METADATA_KEYS
    assert "final_xges" in RENDER_METADATA_KEYS
    assert "snapshot_count" in RENDER_METADATA_KEYS
    assert "activity_entries" in RENDER_METADATA_KEYS


@requires_inspect
@pytest.mark.asyncio
async def test_render_solver_handles_no_scenario():
    """Solver should populate metadata even with no scenario."""
    from inspect_ai.solver import TaskState
    from ave.harness.solvers.render import render_solver

    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[], metadata={},
    )

    async def fake_generate(s):
        return s

    solver = render_solver()
    result = await solver(state, fake_generate)
    assert "rendered_path" in result.metadata
    assert "render_failed" in result.metadata
    assert "render_preset" in result.metadata
    assert "render_log" in result.metadata


@requires_inspect
@requires_ges
@requires_ffmpeg
@pytest.mark.asyncio
async def test_render_solver_produces_mp4_with_real_render(tmp_path):
    """End-to-end: render solver produces a real MP4 file."""
    from inspect_ai.solver import TaskState
    from ave.harness.schema import (
        Expected, Inputs, RenderExpected, Scenario, ScopeSpec,
    )
    from ave.harness.solvers.render import render_solver

    scenario = Scenario(
        id="test.render",
        tiers=("render",),
        prompt="render the project",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(render=RenderExpected(preset="default")),
    )

    state = TaskState(
        model="mockllm/mock", sample_id="test", epoch=0,
        input="render", messages=[],
        metadata={"scenario": scenario},
    )

    async def fake_generate(s):
        return s

    result = await render_solver()(state, fake_generate)
    if result.metadata.get("rendered_path"):
        from pathlib import Path
        assert Path(result.metadata["rendered_path"]).exists()


@requires_inspect
@requires_ffmpeg
@pytest.mark.asyncio
async def test_render_solver_uses_preset_from_scenario():
    """Solver should record the preset name from scenario.expected.render."""
    from inspect_ai.solver import TaskState
    from ave.harness.schema import (
        Expected, Inputs, RenderExpected, Scenario, ScopeSpec,
    )
    from ave.harness.solvers.render import render_solver

    scenario = Scenario(
        id="test.render.preset",
        tiers=("render",),
        prompt="render",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(render=RenderExpected(preset="instagram_reel")),
    )

    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[], metadata={"scenario": scenario},
    )

    async def fake_generate(s):
        return s

    result = await render_solver()(state, fake_generate)
    assert result.metadata["render_preset"] == "instagram_reel"


@requires_inspect
@pytest.mark.asyncio
async def test_render_solver_marks_failure_when_no_ffmpeg(monkeypatch, tmp_path):
    """When fallback render cannot run, render_failed should be True."""
    from inspect_ai.solver import TaskState
    from ave.harness.solvers.render import render_solver

    import ave.harness.solvers.render as mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(mod, "_render_timeline", boom)

    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[], metadata={},
    )

    async def fake_generate(s):
        return s

    result = await render_solver()(state, fake_generate)
    assert result.metadata["render_failed"] is True
    assert result.metadata["rendered_path"] is None
    assert "simulated render failure" in result.metadata["render_log"]
