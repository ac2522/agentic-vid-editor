"""Inspect AI scorer wrapper for execute-tier XGES state-diff."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.harness.evaluators.state_diff import evaluate_execute_state, extract_timeline_metrics
from ave.harness.schema import Scenario


@scorer(metrics=[])
def state_diff_scorer() -> Scorer:
    """Checks timeline structure after execute-tier run against scenario expectations."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        final_xges: str | None = meta.get("final_xges")
        if final_xges is None:
            return Score(value=1, explanation="no final_xges in metadata; skipping")

        scenario: Scenario = meta["scenario"]
        execute_expected = scenario.expected.execute
        if execute_expected is None:
            return Score(value=1, explanation="no execute expectations; skipping")

        actual = extract_timeline_metrics(final_xges)
        verdict = evaluate_execute_state(actual, execute_expected)

        return Score(
            value=1 if verdict.passed else 0,
            explanation=verdict.reason,
            metadata={
                "rule": verdict.rule,
                "clip_count": actual.clip_count,
                "duration_seconds": actual.duration_seconds,
            },
        )

    return score
