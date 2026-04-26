"""Assemble scenarios into Inspect AI Tasks for each harness rung."""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from ave.harness.adapter import scenarios_to_dataset
from ave.harness.judges._protocol import JudgeBackend
from ave.harness.judges.claude_vlm import ClaudeVlmJudge
from ave.harness.judges.deterministic import DeterministicJudge
from ave.harness.loader import load_scenario_from_yaml
from ave.harness.schema import Scenario
from ave.harness.scorers.scope import scope_scorer
from ave.harness.scorers.state_diff import state_diff_scorer
from ave.harness.scorers.safety import safety_scorer
from ave.harness.scorers.tool_selection import tool_selection_scorer
from ave.harness.scorers.vlm_judge import vlm_judge_scorer
from ave.harness.solvers.execute import execute_solver
from ave.harness.solvers.plan import plan_solver
from ave.harness.solvers.render import render_solver


def default_judges() -> list[JudgeBackend]:
    """Default judge ensemble: ffprobe (static) + Claude Haiku (still).

    Claude VLM gracefully degrades to passed=False when ANTHROPIC_API_KEY is
    absent, so this default works in CI without API access (the corresponding
    rubric dimensions just fail).
    """
    return [DeterministicJudge(), ClaudeVlmJudge(model="claude-haiku-4-5-20251001")]


def _build_task(scenario: Scenario, solver, scorers: list) -> Task:
    return Task(
        dataset=scenarios_to_dataset([scenario]),
        solver=solver,
        scorer=scorers,
    )


@task
def plan_rung_task(scenario_file: str) -> Task:
    """Evaluate a scenario at the plan rung (tool selection, no execution)."""
    scenario = load_scenario_from_yaml(Path(scenario_file))
    return _build_task(
        scenario,
        solver=plan_solver(),
        scorers=[tool_selection_scorer(), scope_scorer()],
    )


@task
def execute_rung_task(scenario_file: str) -> Task:
    """Evaluate a scenario at the execute rung (real tool execution + state diff)."""
    scenario = load_scenario_from_yaml(Path(scenario_file))
    return _build_task(
        scenario,
        solver=execute_solver(),
        scorers=[state_diff_scorer(), safety_scorer()],
    )


@task
def render_rung_task(
    scenario_file: str,
    judges: list[JudgeBackend] | None = None,
) -> Task:
    """Evaluate a scenario at the render rung (rendered MP4 + VLM judge)."""
    scenario = load_scenario_from_yaml(Path(scenario_file))
    return _build_task(
        scenario,
        solver=render_solver(),
        scorers=[
            state_diff_scorer(),
            safety_scorer(),
            vlm_judge_scorer(judges=judges or default_judges()),
        ],
    )
