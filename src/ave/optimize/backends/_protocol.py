"""Optimizer backend protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ave.optimize.artifacts import ContextArtifact
from ave.optimize.datasets import EvalDataset


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for an optimization run."""

    model: str = "anthropic/claude-sonnet-4-6"
    target_model: str = "anthropic/claude-haiku-4-5"
    max_trials: int = 10
    n_samples: int | None = None
    algorithm: str = "meta_prompt"  # "meta_prompt" | "evolutionary" | "hrpo"
    n_threads: int = 8
    seed: int = 42
    min_improvement: float = 0.01
    on_regression: str = "reject"  # "reject" | "warn" | "store_anyway"


@dataclass(frozen=True)
class OptimizationResult:
    """Result of optimizing a single artifact."""

    original_score: float
    optimized_score: float
    improvement: float
    optimized_artifact: ContextArtifact
    accepted: bool
    trial_history: tuple[dict[str, Any], ...] = ()


class OptimizerBackend(Protocol):
    """Protocol for optimization backends."""

    def optimize(
        self,
        artifact: ContextArtifact,
        dataset: EvalDataset,
        metrics: Sequence,
        config: OptimizationConfig,
    ) -> OptimizationResult: ...
