"""Standalone evaluator — no Opik dependency, CI-friendly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ave.optimize.artifacts import ArtifactKind, ContextArtifact
from ave.optimize.datasets import EvalDataset
from ave.optimize.metrics import LLMResponse


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating artifacts against a dataset."""

    overall_score: float
    per_metric: dict[str, float]
    per_item: list[dict[str, Any]]
    artifact_scores: dict[str, float]


class StandaloneEvaluator:
    """Evaluate artifacts against datasets using direct LLM calls.

    No Opik dependency. Accepts a caller function for testability.

    Args:
        caller: A callable (system_prompt: str, user_message: str) -> LLMResponse.
                If None, uses litellm for real LLM calls (requires API key).
        model: Model string for litellm (only used if caller is None).
    """

    def __init__(
        self,
        caller: Callable[[str, str], LLMResponse] | None = None,
        model: str = "anthropic/claude-haiku-4-5",
    ):
        self._caller = caller
        self._model = model

    def _build_system_prompt(self, artifacts: list[ContextArtifact]) -> str:
        """Build a system prompt from artifacts."""
        parts: list[str] = []
        for artifact in artifacts:
            if artifact.kind == ArtifactKind.SYSTEM_PROMPT:
                parts.append(artifact.content)
            elif artifact.kind == ArtifactKind.ROLE_DESCRIPTION:
                parts.append(f"Role: {artifact.content}")
            elif artifact.kind == ArtifactKind.TOOL_DESCRIPTION:
                parts.append(f"Available tool: {artifact.content}")
            elif artifact.kind == ArtifactKind.ORCHESTRATOR_PROMPT:
                parts.append(artifact.content)
        return "\n\n".join(parts)

    def _get_caller(self) -> Callable[[str, str], LLMResponse]:
        if self._caller is not None:
            return self._caller
        # Lazy import litellm for real calls
        return self._litellm_caller

    def _litellm_caller(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Call LLM via litellm."""
        import litellm

        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content or ""
        # Parse tool calls if present
        tool_calls = ()
        if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            from ave.optimize.metrics import ToolCall
            import json

            tool_calls = tuple(
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                )
                for tc in response.choices[0].message.tool_calls
            )
        return LLMResponse(text=text, tool_calls=tool_calls)

    def evaluate(
        self,
        artifacts: list[ContextArtifact],
        dataset: EvalDataset,
        metrics: Sequence,
    ) -> EvaluationResult:
        """Run each dataset item against the artifacts, score with metrics."""
        caller = self._get_caller()
        system_prompt = self._build_system_prompt(artifacts)

        per_item: list[dict[str, Any]] = []
        metric_totals: dict[str, float] = {}
        metric_counts: dict[str, int] = {}

        for item in dataset.items:
            response = caller(system_prompt, item.task)
            item_scores: dict[str, float] = {}

            for metric in metrics:
                result = metric.score(item, response)
                item_scores[result.name] = result.value
                metric_totals[result.name] = metric_totals.get(result.name, 0.0) + result.value
                metric_counts[result.name] = metric_counts.get(result.name, 0) + 1

            per_item.append({"item_id": item.id, "scores": item_scores})

        # Compute averages
        per_metric = {
            name: metric_totals[name] / metric_counts[name]
            for name in metric_totals
        }
        overall = sum(per_metric.values()) / len(per_metric) if per_metric else 0.0

        return EvaluationResult(
            overall_score=overall,
            per_metric=per_metric,
            per_item=per_item,
            artifact_scores={},
        )
