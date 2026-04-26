"""Opik Agent Optimizer backend implementation.

Requires: pip install ave[optimize]
"""

from __future__ import annotations

from typing import Any, Sequence

from ave._compat import import_optional
from ave.optimize.artifacts import ContextArtifact
from ave.optimize.backends._protocol import OptimizationConfig, OptimizationResult
from ave.optimize.datasets import EvalDataset


class OpikOptimizerBackend:
    """Opik Agent Optimizer backend.

    Wraps opik-optimizer to optimize individual artifacts.
    """

    def optimize(
        self,
        artifact: ContextArtifact,
        dataset: EvalDataset,
        metrics: Sequence,
        config: OptimizationConfig,
    ) -> OptimizationResult:
        opik_optimizer = import_optional("opik_optimizer", extra="optimize")

        ChatPrompt = opik_optimizer.ChatPrompt

        # Build ChatPrompt from artifact
        prompt = ChatPrompt(
            messages=[
                {"role": "system", "content": artifact.content},
                {"role": "user", "content": "{task}"},
            ],
            model=config.target_model,
        )

        # Convert dataset to Opik format
        opik_dataset = self._convert_dataset(dataset)

        # Build combined metric function
        opik_metric = self._build_metric(metrics)

        # Create optimizer
        optimizer = self._create_optimizer(config, opik_optimizer)

        # Run optimization
        result = optimizer.optimize_prompt(
            prompt=prompt,
            dataset=opik_dataset,
            metric=opik_metric,
            max_trials=config.max_trials,
            n_samples=config.n_samples,
        )

        # Extract optimized content
        optimized_content = artifact.content
        if hasattr(result, "prompt") and result.prompt:
            for msg in result.prompt.messages:
                if msg.get("role") == "system":
                    optimized_content = msg["content"]
                    break

        original_score = getattr(result, "original_score", 0.0)
        optimized_score = getattr(result, "best_score", original_score)
        improvement = optimized_score - original_score
        accepted = improvement >= config.min_improvement

        optimized_artifact = ContextArtifact(
            id=artifact.id,
            kind=artifact.kind,
            content=optimized_content,
            source_location=artifact.source_location,
            metadata=artifact.metadata,
        )

        return OptimizationResult(
            original_score=original_score,
            optimized_score=optimized_score,
            improvement=improvement,
            optimized_artifact=optimized_artifact,
            accepted=accepted,
            trial_history=[],
        )

    def _convert_dataset(self, dataset: EvalDataset) -> list[dict[str, Any]]:
        """Convert EvalDataset to Opik dataset format (list of dicts)."""
        return [
            {
                "task": item.task,
                "expected_tools": item.expected_tools,
                "expected_output_pattern": item.expected_output_pattern,
                **(item.context or {}),
            }
            for item in dataset.items
        ]

    def _build_metric(self, metrics: Sequence) -> Any:
        """Build Opik-compatible metric function from AVE metrics."""
        from ave.optimize.metrics import LLMResponse

        def combined_metric(dataset_item: dict, llm_output: str) -> float:
            # Parse LLM output into structured response
            response = LLMResponse(text=llm_output, tool_calls=())

            # Build EvalItem from dataset_item
            from ave.optimize.datasets import EvalItem

            item = EvalItem(
                id=dataset_item.get("id", ""),
                task=dataset_item.get("task", ""),
                expected_tools=dataset_item.get("expected_tools", []),
                expected_output_pattern=dataset_item.get("expected_output_pattern"),
                context={},
            )

            if not metrics:
                return 0.0

            total = sum(m.score(item, response).value for m in metrics)
            return total / len(metrics)

        return combined_metric

    def _create_optimizer(self, config: OptimizationConfig, opik_optimizer):
        """Create the appropriate Opik optimizer based on config."""
        if config.algorithm == "evolutionary":
            return opik_optimizer.EvolutionaryOptimizer(
                model=config.model,
                n_threads=config.n_threads,
                seed=config.seed,
            )
        elif config.algorithm == "hrpo":
            if not hasattr(opik_optimizer, "HRPO"):
                raise ValueError(
                    "HRPO algorithm requested but not available in the installed "
                    "version of opik-optimizer."
                )
            return opik_optimizer.HRPO(
                model=config.model,
                n_threads=config.n_threads,
                seed=config.seed,
            )
        # Default: MetaPromptOptimizer
        return opik_optimizer.MetaPromptOptimizer(
            model=config.model,
            n_threads=config.n_threads,
            seed=config.seed,
        )
