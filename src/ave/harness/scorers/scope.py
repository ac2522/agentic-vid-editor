"""Inspect AI scorer wrapper for plan-level scope compliance."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.agent.registry import ToolRegistry
from ave.agent.session import EditingSession
from ave.harness.evaluators.scope import evaluate_scope
from ave.harness.schema import Scenario

_session_for_registry: EditingSession | None = None


def _shared_registry() -> ToolRegistry:
    """Lazily build a full-registry EditingSession to look up tool domains."""
    global _session_for_registry
    if _session_for_registry is None:
        _session_for_registry = EditingSession()
    return _session_for_registry.registry


@scorer(metrics=[])
def scope_scorer(registry: ToolRegistry | None = None) -> Scorer:
    """Pass/fail based on scope.

    Uses the supplied ``registry`` (for dependency injection in tests), or
    falls back to the default shared session registry.
    """
    effective_registry = registry

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        scenario: Scenario = meta["scenario"]
        called: list[str] = list(meta.get("called_tools", []))
        verdict = evaluate_scope(
            called_tools=called,
            registry=effective_registry or _shared_registry(),
            forbidden_domains=scenario.scope.forbidden_layers,
        )
        return Score(
            value=1 if verdict.passed else 0,
            answer=",".join(called),
            explanation=verdict.reason,
            metadata={"rule": verdict.rule},
        )

    return score
