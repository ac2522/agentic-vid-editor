"""Tests for optimization campaign orchestrator."""

from __future__ import annotations

from pathlib import Path


from ave.optimize.artifacts import ArtifactExtractor, ContextArtifact
from ave.optimize.campaign import CampaignResult, OptimizationCampaign
from ave.optimize.datasets import EvalDataset, EvalItem
from ave.optimize.evaluate import EvaluationResult
from ave.optimize.metrics import LLMResponse, ToolCall, ToolSelectionAccuracy
from ave.optimize.store import ArtifactStore


def _make_dataset() -> EvalDataset:
    return EvalDataset(
        name="test",
        items=[
            EvalItem(
                id="001",
                task="Trim clip",
                expected_tools=["trim_clip"],
                expected_output_pattern=None,
                context={},
            ),
        ],
    )


class MockOptimizerBackend:
    """Mock optimizer that returns improved artifacts."""

    def __init__(self, improvement: float = 0.1):
        self._improvement = improvement
        self.optimize_calls: list[ContextArtifact] = []

    def optimize(self, artifact, dataset, metrics, config):
        from ave.optimize.backends._protocol import OptimizationResult

        self.optimize_calls.append(artifact)
        new_content = f"OPTIMIZED: {artifact.content}"
        optimized = ContextArtifact(
            id=artifact.id,
            kind=artifact.kind,
            content=new_content,
            source_location=artifact.source_location,
            metadata=artifact.metadata,
        )
        original_score = 0.7
        optimized_score = original_score + self._improvement
        return OptimizationResult(
            original_score=original_score,
            optimized_score=optimized_score,
            improvement=self._improvement,
            optimized_artifact=optimized,
            accepted=self._improvement >= config.min_improvement,
            trial_history=(),
        )


class MockRegressingBackend:
    """Mock optimizer that returns worse artifacts."""

    def optimize(self, artifact, dataset, metrics, config):
        from ave.optimize.backends._protocol import OptimizationResult

        optimized = ContextArtifact(
            id=artifact.id,
            kind=artifact.kind,
            content=f"WORSE: {artifact.content}",
            source_location=artifact.source_location,
            metadata=artifact.metadata,
        )
        return OptimizationResult(
            original_score=0.8,
            optimized_score=0.6,
            improvement=-0.2,
            optimized_artifact=optimized,
            accepted=False,
            trial_history=(),
        )


class TestCampaignResult:
    def test_create(self):
        result = CampaignResult(
            campaign_id="2026-03-16_abc12345",
            baseline=EvaluationResult(0.7, {}, [], {}),
            optimized=EvaluationResult(0.8, {}, [], {}),
            validation=None,
            artifacts_improved=[],
            artifacts_rejected=[],
            duration_seconds=10.0,
        )
        assert result.campaign_id.startswith("2026")


class TestOptimizationCampaign:
    def _make_roles(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockRole:
            name: str
            description: str
            system_prompt: str
            domains: tuple[str, ...] = ()

        return [MockRole("editor", "Video editor", "You edit videos.")]

    def test_run_calls_optimizer_per_artifact(self, tmp_path: Path):
        backend = MockOptimizerBackend()
        store = ArtifactStore(tmp_path)
        extractor = ArtifactExtractor()
        roles = self._make_roles()

        campaign = OptimizationCampaign(
            backend=backend,
            store=store,
            extractor=extractor,
            roles=roles,
        )

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            return LLMResponse(text="ok", tool_calls=(ToolCall(name="trim_clip", arguments={}),))

        from ave.optimize.backends._protocol import OptimizationConfig

        result = campaign.run(
            dataset=_make_dataset(),
            metrics=[ToolSelectionAccuracy()],
            config=OptimizationConfig(),
            caller=mock_caller,
        )
        # 2 artifacts from role (system_prompt + description)
        assert len(backend.optimize_calls) == 2
        assert len(result.artifacts_improved) == 2

    def test_run_rejects_regressed_artifacts(self, tmp_path: Path):
        backend = MockRegressingBackend()
        store = ArtifactStore(tmp_path)
        extractor = ArtifactExtractor()
        roles = self._make_roles()

        campaign = OptimizationCampaign(
            backend=backend,
            store=store,
            extractor=extractor,
            roles=roles,
        )

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            return LLMResponse(text="ok", tool_calls=())

        from ave.optimize.backends._protocol import OptimizationConfig

        result = campaign.run(
            dataset=_make_dataset(),
            metrics=[ToolSelectionAccuracy()],
            config=OptimizationConfig(),
            caller=mock_caller,
        )
        assert len(result.artifacts_improved) == 0
        assert len(result.artifacts_rejected) == 2

    def test_run_stores_only_improved_artifacts(self, tmp_path: Path):
        backend = MockOptimizerBackend(improvement=0.1)
        store = ArtifactStore(tmp_path)
        extractor = ArtifactExtractor()
        roles = self._make_roles()

        campaign = OptimizationCampaign(
            backend=backend,
            store=store,
            extractor=extractor,
            roles=roles,
        )

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            return LLMResponse(text="ok", tool_calls=())

        from ave.optimize.backends._protocol import OptimizationConfig

        campaign.run(
            dataset=_make_dataset(),
            metrics=[ToolSelectionAccuracy()],
            config=OptimizationConfig(),
            caller=mock_caller,
        )
        # Both artifacts should be stored since improvement > min_improvement
        assert store.load_best("role.editor.system_prompt") is not None
        assert store.load_best("role.editor.description") is not None

    def test_evaluate_only_does_not_optimize(self, tmp_path: Path):
        backend = MockOptimizerBackend()
        store = ArtifactStore(tmp_path)
        extractor = ArtifactExtractor()
        roles = self._make_roles()

        campaign = OptimizationCampaign(
            backend=backend,
            store=store,
            extractor=extractor,
            roles=roles,
        )

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            return LLMResponse(text="ok", tool_calls=(ToolCall(name="trim_clip", arguments={}),))

        result = campaign.evaluate_only(
            dataset=_make_dataset(),
            metrics=[ToolSelectionAccuracy()],
            caller=mock_caller,
        )
        assert isinstance(result, EvaluationResult)
        assert len(backend.optimize_calls) == 0  # No optimization happened
