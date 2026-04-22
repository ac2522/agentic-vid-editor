"""Pytest helpers for harness scenarios.

Consumers wire these into a test module of their own using
pytest.mark.parametrize, e.g.:

    import pytest
    from ave.harness.pytest_plugin import (
        bundled_scenarios_dir,
        discover_plan_scenarios,
        run_plan_scenario,
    )

    @pytest.mark.parametrize("scenario_file", discover_plan_scenarios(bundled_scenarios_dir()))
    def test_plan_scenario(scenario_file):
        assert run_plan_scenario(scenario_file) is True
"""

from __future__ import annotations

from pathlib import Path


def bundled_scenarios_dir() -> Path:
    """Return the directory of flagship scenarios shipped with ave.harness."""
    return Path(__file__).parent / "scenarios"


def discover_plan_scenarios(scenario_dir: Path) -> list[str]:
    """Return stably sorted YAML file paths under scenario_dir."""
    directory = Path(scenario_dir)
    paths = sorted(p for p in directory.glob("*.yaml") if p.is_file())
    return [str(p) for p in paths]


def run_plan_scenario(
    scenario_file: str,
    *,
    model: str = "mockllm/mock",
    log_dir: str | None = None,
) -> bool:
    """Run a single plan-tier scenario; return True iff all samples passed all scorers."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ave.harness.task import plan_rung_task

    task = plan_rung_task(scenario_file=scenario_file)
    results = inspect_eval(
        task,
        model=get_model(model),
        display="plain",
        log_dir=log_dir or "./logs",
    )
    if not results:
        return False
    for eval_log in results:
        for sample in eval_log.samples or []:
            for score in (sample.scores or {}).values():
                if getattr(score, "value", 0) != 1:
                    return False
    return True
