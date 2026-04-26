"""Inspect AI scorer wrapper for execute-tier safety invariants."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.agent.registry import ToolRegistry
from ave.agent.session import EditingSession
from ave.harness.evaluators.safety import evaluate_safety
from ave.harness.schema import Scenario

_session_for_registry: EditingSession | None = None


def _shared_registry() -> ToolRegistry:
    global _session_for_registry
    if _session_for_registry is None:
        _session_for_registry = EditingSession()
    return _session_for_registry.registry


@scorer(metrics=[])
def safety_scorer(registry: ToolRegistry | None = None) -> Scorer:
    """Checks all 5 safety invariants against execute-tier metadata."""
    effective_registry = registry

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        scenario: Scenario = meta["scenario"]
        called: list[str] = list(meta.get("called_tools", []))
        snapshot_count: int = int(meta.get("snapshot_count", 0))
        activity_entries: list[dict] = list(meta.get("activity_entries", []))
        source_hashes_before: dict[str, str] | None = meta.get("source_hashes_before")
        source_hashes_after: dict[str, str] | None = meta.get("source_hashes_after")

        report = evaluate_safety(
            called_tools=called,
            snapshot_count=snapshot_count,
            activity_entries=activity_entries,
            source_hashes_before=source_hashes_before,
            source_hashes_after=source_hashes_after,
            forbidden_domains=tuple(scenario.scope.forbidden_layers),
            registry=effective_registry or _shared_registry(),
            safety=scenario.safety,
        )

        verdict_dicts = {
            name: {"passed": v.passed, "rule": v.rule, "reason": v.reason}
            for name, v in report.invariant_verdicts.items()
        }

        if report.passed:
            explanation = "all safety invariants passed"
        else:
            explanation = f"failed invariants: {', '.join(report.failed_invariants)}"

        return Score(
            value=1 if report.passed else 0,
            explanation=explanation,
            metadata={"invariant_verdicts": verdict_dicts},
        )

    return score
