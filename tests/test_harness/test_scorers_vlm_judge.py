"""Tests for the VLM-judge Inspect AI scorer wrapper."""
import pytest

from tests.conftest import requires_inspect


@requires_inspect
@pytest.mark.asyncio
async def test_vlm_scorer_skips_when_no_rendered_path():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from ave.harness.schema import Expected, Inputs, RenderExpected, Scenario, ScopeSpec
    from ave.harness.scorers.vlm_judge import vlm_judge_scorer

    scenario = Scenario(
        id="t", tiers=("render",), prompt="",
        scope=ScopeSpec(), inputs=Inputs(),
        expected=Expected(render=RenderExpected(preset="x")),
    )
    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[], metadata={"scenario": scenario, "render_failed": True},
    )
    score = await vlm_judge_scorer(judges=[])(state, Target("t"))
    assert score.value == 0
    assert "render" in (score.explanation or "").lower()


@requires_inspect
@pytest.mark.asyncio
async def test_vlm_scorer_no_render_expectations_passes():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from ave.harness.schema import Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.scorers.vlm_judge import vlm_judge_scorer

    scenario = Scenario(
        id="t", tiers=("render",), prompt="",
        scope=ScopeSpec(), inputs=Inputs(),
        expected=Expected(),
    )
    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[], metadata={"scenario": scenario, "rendered_path": "/x.mp4"},
    )
    score = await vlm_judge_scorer(judges=[])(state, Target("t"))
    assert score.value == 1
    assert "no render" in (score.explanation or "").lower()


@requires_inspect
@pytest.mark.asyncio
async def test_vlm_scorer_passes_when_all_dimensions_pass(tmp_path):
    """Use fake judges that return high scores for everything."""
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    from ave.harness.schema import (
        Expected, Inputs, RenderExpected, RubricDimension, Scenario, ScopeSpec,
    )
    from ave.harness.scorers.vlm_judge import vlm_judge_scorer
    from ave.harness.judges._protocol import JudgeVerdict

    class GoodJudge:
        @property
        def name(self): return "good"
        @property
        def supported_dimension_types(self): return ("still", "static")
        def judge_dimension(self, *, rendered_path, dimension, prompt, pass_threshold):
            return JudgeVerdict("good", dimension, 0.9, True, "looks great")

    scenario = Scenario(
        id="t", tiers=("render",), prompt="",
        scope=ScopeSpec(), inputs=Inputs(),
        expected=Expected(render=RenderExpected(
            preset="default",
            rubric=(RubricDimension(dimension="framing", prompt="x", pass_threshold=0.5),)
        )),
    )
    fake_path = tmp_path / "render.mp4"
    fake_path.write_bytes(b"\x00" * 100)
    state = TaskState(
        model="mockllm/mock", sample_id="t", epoch=0, input="",
        messages=[],
        metadata={"scenario": scenario, "rendered_path": str(fake_path), "render_failed": False},
    )
    scorer = vlm_judge_scorer(judges=[GoodJudge()])
    score = await scorer(state, Target("t"))
    assert score.value == 1
