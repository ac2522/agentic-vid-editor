"""Optimization campaign orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from ave.optimize.artifacts import ArtifactExtractor, ContextArtifact
from ave.optimize.backends._protocol import OptimizationConfig, OptimizationResult
from ave.optimize.datasets import EvalDataset
from ave.optimize.evaluate import EvaluationResult, StandaloneEvaluator
from ave.optimize.metrics import LLMResponse
from ave.optimize.store import ArtifactStore


@dataclass(frozen=True)
class CampaignResult:
    """Result of a full optimization campaign."""

    campaign_id: str
    baseline: EvaluationResult
    optimized: EvaluationResult
    validation: EvaluationResult | None
    artifacts_improved: list[tuple[ContextArtifact, ContextArtifact, float]]
    artifacts_rejected: list[tuple[ContextArtifact, float, float]]
    duration_seconds: float


class OptimizationCampaign:
    """Runs a complete optimization cycle."""

    def __init__(
        self,
        backend,
        store: ArtifactStore,
        extractor: ArtifactExtractor,
        roles: Sequence | None = None,
        registry=None,
    ):
        self._backend = backend
        self._store = store
        self._extractor = extractor
        self._roles = roles
        self._registry = registry

    def run(
        self,
        dataset: EvalDataset,
        metrics: Sequence,
        config: OptimizationConfig,
        artifact_filter: Callable[[ContextArtifact], bool] | None = None,
        validation_dataset: EvalDataset | None = None,
        caller: Callable[[str, str], LLMResponse] | None = None,
    ) -> CampaignResult:
        """Full optimization cycle: extract -> evaluate baseline -> optimize -> store."""
        start = time.monotonic()
        campaign_id = self._generate_campaign_id()

        # Extract artifacts
        artifacts = self._extractor.extract_all(
            roles=self._roles, registry=self._registry
        )
        if artifact_filter:
            artifacts = [a for a in artifacts if artifact_filter(a)]

        # Evaluate baseline
        evaluator = StandaloneEvaluator(caller=caller)
        baseline = evaluator.evaluate(
            artifacts=artifacts, dataset=dataset, metrics=metrics
        )

        # Optimize each artifact
        improved: list[tuple[ContextArtifact, ContextArtifact, float]] = []
        rejected: list[tuple[ContextArtifact, float, float]] = []

        for artifact in artifacts:
            result: OptimizationResult = self._backend.optimize(
                artifact=artifact,
                dataset=dataset,
                metrics=metrics,
                config=config,
            )

            if result.accepted and result.improvement >= config.min_improvement:
                improved.append(
                    (artifact, result.optimized_artifact, result.improvement)
                )
                self._store.save(
                    result.optimized_artifact,
                    score=result.optimized_score,
                    campaign_id=campaign_id,
                )
            else:
                rejected.append(
                    (artifact, result.original_score, result.optimized_score)
                )

        # Re-evaluate with improved artifacts
        all_artifacts_updated = list(artifacts)  # default: unchanged
        if improved:
            improved_map = {orig.id: after for orig, after, _ in improved}
            all_artifacts_updated = [
                improved_map.get(a.id, a) for a in artifacts
            ]
            optimized_eval = evaluator.evaluate(
                artifacts=all_artifacts_updated, dataset=dataset, metrics=metrics
            )
        else:
            optimized_eval = baseline

        # Validation pass
        validation_eval = None
        if validation_dataset and improved:
            validation_eval = evaluator.evaluate(
                artifacts=all_artifacts_updated,
                dataset=validation_dataset,
                metrics=metrics,
            )

        duration = time.monotonic() - start
        return CampaignResult(
            campaign_id=campaign_id,
            baseline=baseline,
            optimized=optimized_eval,
            validation=validation_eval,
            artifacts_improved=improved,
            artifacts_rejected=rejected,
            duration_seconds=duration,
        )

    def evaluate_only(
        self,
        dataset: EvalDataset,
        metrics: Sequence,
        caller: Callable[[str, str], LLMResponse] | None = None,
    ) -> EvaluationResult:
        """Evaluate current artifacts without optimizing. For CI regression."""
        artifacts = self._extractor.extract_all(
            roles=self._roles, registry=self._registry
        )
        evaluator = StandaloneEvaluator(caller=caller)
        return evaluator.evaluate(
            artifacts=artifacts, dataset=dataset, metrics=metrics
        )

    @staticmethod
    def _generate_campaign_id() -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{date_str}_{uuid.uuid4().hex[:8]}"
