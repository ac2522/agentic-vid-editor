"""AVE Context Optimizer — systematic optimization of LLM context artifacts.

Provides tools for extracting, evaluating, and optimizing text artifacts
sent to LLMs: system prompts, tool descriptions, and orchestrator instructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ave.optimize.evaluate import EvaluationResult


def evaluate_artifacts(
    *,
    roles: Sequence | None = None,
    registry=None,
    dataset_path: Path,
    caller=None,
) -> EvaluationResult:
    """Evaluate current artifacts against a dataset. CI-friendly, no Opik needed.

    Args:
        roles: AgentRole instances to extract system prompts from.
        registry: ToolRegistry to extract tool descriptions from.
        dataset_path: Path to JSONL eval dataset.
        caller: Optional callable(system_prompt, user_message) -> LLMResponse for testing.
    """
    from ave.optimize.artifacts import ArtifactExtractor
    from ave.optimize.datasets import EvalDataset
    from ave.optimize.evaluate import StandaloneEvaluator
    from ave.optimize.metrics import ToolSelectionAccuracy

    extractor = ArtifactExtractor()
    artifacts = extractor.extract_all(roles=roles, registry=registry)
    dataset = EvalDataset.from_jsonl(dataset_path)
    metrics = [ToolSelectionAccuracy()]
    evaluator = StandaloneEvaluator(caller=caller)
    return evaluator.evaluate(artifacts=artifacts, dataset=dataset, metrics=metrics)


def optimize_artifacts(
    *,
    roles: Sequence | None = None,
    registry=None,
    dataset_path: Path,
    config=None,
    store_dir: Path,
    caller=None,
):
    """Run full optimization campaign. Requires opik-optimizer.

    Args:
        roles: AgentRole instances to extract system prompts from.
        registry: ToolRegistry to extract tool descriptions from.
        dataset_path: Path to JSONL eval dataset.
        config: OptimizationConfig (uses defaults if None).
        store_dir: Base directory for artifact store.
        caller: Optional callable for standalone evaluation baseline.
    """
    from ave.optimize.artifacts import ArtifactExtractor
    from ave.optimize.backends._protocol import OptimizationConfig
    from ave.optimize.campaign import OptimizationCampaign
    from ave.optimize.datasets import EvalDataset
    from ave.optimize.metrics import ToolSelectionAccuracy
    from ave.optimize.store import ArtifactStore

    if config is None:
        config = OptimizationConfig()

    extractor = ArtifactExtractor()
    store = ArtifactStore(store_dir)
    dataset = EvalDataset.from_jsonl(dataset_path)
    metrics = [ToolSelectionAccuracy()]

    # Lazy import Opik backend
    from ave.optimize.backends._opik import OpikOptimizerBackend

    backend = OpikOptimizerBackend()
    campaign = OptimizationCampaign(
        backend=backend, store=store, extractor=extractor, roles=roles, registry=registry
    )
    return campaign.run(dataset=dataset, metrics=metrics, config=config, caller=caller)
