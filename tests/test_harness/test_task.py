"""End-to-end test: Scenario -> Inspect AI Task -> mockllm run -> score."""

from pathlib import Path

from tests.conftest import requires_inspect


_SCENARIO_YAML = """
id: reel.filler-word-trim
description: ""
tiers: [plan]
prompt: "clean up the ums"
scope:
  allowed_agents: [transcriptionist, editor]
  forbidden_layers: []
inputs:
  assets: []
expected:
  plan:
    tools_required:
      all_of: [find_fillers]
      any_of: [text_cut, trim]
    tools_forbidden: [apply_blend_mode]
    irrelevance_allowed: false
"""


@requires_inspect
def test_plan_rung_task_constructs_valid_task(tmp_path: Path):
    from inspect_ai import Task

    from ave.harness.task import plan_rung_task

    p = tmp_path / "s.yaml"
    p.write_text(_SCENARIO_YAML)

    task = plan_rung_task(scenario_file=str(p))
    assert isinstance(task, Task)
    assert task.dataset is not None
    assert task.solver is not None
    assert task.scorer is not None


@requires_inspect
def test_plan_rung_task_runs_with_mockllm_canned_response(tmp_path: Path):
    """Run the task end-to-end with a mock model."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ave.harness.task import plan_rung_task

    p = tmp_path / "s.yaml"
    p.write_text(_SCENARIO_YAML)

    task = plan_rung_task(scenario_file=str(p))
    model = get_model("mockllm/mock")
    results = inspect_eval(task, model=model, display="plain", log_dir=str(tmp_path / "logs"))

    assert len(results) == 1
    eval_log = results[0]
    assert len(eval_log.samples) == 1
    sample = eval_log.samples[0]
    # The scorer produced a score (pass or fail depends on mockllm behaviour).
    assert sample.scores is not None
