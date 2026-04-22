"""Inspect AI scorer wrapper for plan-level tool selection."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.harness.evaluators.tool_selection import evaluate_plan
from ave.harness.schema import Scenario


@scorer(metrics=[])
def tool_selection_scorer() -> Scorer:
    """Pass/fail based on tool-invocation constraints in the scenario's plan expectations."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        scenario: Scenario = meta["scenario"]
        called: list[str] = list(meta.get("called_tools", []))
        plan = scenario.expected.plan
        if plan is None:
            return Score(value=1, explanation="scenario has no plan expectations; skipping")
        verdict = evaluate_plan(called, plan)
        return Score(
            value=1 if verdict.passed else 0,
            answer=",".join(called),
            explanation=verdict.reason,
            metadata={"rule": verdict.rule},
        )

    return score
