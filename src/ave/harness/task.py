"""Assemble a scenario into an Inspect AI Task at the plan rung."""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from ave.harness.adapter import scenarios_to_dataset
from ave.harness.loader import load_scenario_from_yaml
from ave.harness.scorers.scope import scope_scorer
from ave.harness.scorers.tool_selection import tool_selection_scorer
from ave.harness.solvers.plan import plan_solver


@task
def plan_rung_task(scenario_file: str) -> Task:
    """Build a Task that evaluates a single YAML scenario at the plan rung."""
    scenario = load_scenario_from_yaml(Path(scenario_file))
    return Task(
        dataset=scenarios_to_dataset([scenario]),
        solver=plan_solver(),
        scorer=[tool_selection_scorer(), scope_scorer()],
    )
