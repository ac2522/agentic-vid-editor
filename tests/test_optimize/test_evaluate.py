"""Tests for standalone evaluator."""

from __future__ import annotations


import pytest

from ave.optimize.artifacts import ArtifactKind, ContextArtifact
from ave.optimize.datasets import EvalDataset, EvalItem
from ave.optimize.evaluate import EvaluationResult, StandaloneEvaluator
from ave.optimize.metrics import (
    LLMResponse,
    ToolSelectionAccuracy,
)


class TestEvaluationResult:
    def test_create_result(self):
        result = EvaluationResult(
            overall_score=0.85,
            per_metric={"tool_selection_accuracy": 0.85},
            per_item=[],
            artifact_scores={},
        )
        assert result.overall_score == 0.85

    def test_result_is_frozen(self):
        result = EvaluationResult(
            overall_score=0.5,
            per_metric={},
            per_item=[],
            artifact_scores={},
        )
        with pytest.raises(AttributeError):
            result.overall_score = 1.0  # type: ignore[misc]


class TestStandaloneEvaluator:
    """Tests for standalone evaluation (no Opik dependency).

    These test the evaluator's metric aggregation logic using a mock LLM caller,
    not real LLM calls.
    """

    def test_evaluate_with_mock_caller(self):
        """Evaluator accepts a custom caller for testing without LLM."""
        artifacts = [
            ContextArtifact(
                id="role.editor.system_prompt",
                kind=ArtifactKind.SYSTEM_PROMPT,
                content="You are a video editor.",
                source_location="roles.py",
                metadata={},
            ),
        ]
        dataset = EvalDataset(
            name="test",
            items=[
                EvalItem(
                    id="001",
                    task="Trim the clip",
                    expected_tools=["trim_clip"],
                    expected_output_pattern=None,
                    context={},
                ),
            ],
        )

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            from ave.optimize.metrics import ToolCall

            return LLMResponse(
                text="I'll trim the clip.",
                tool_calls=(ToolCall(name="trim_clip", arguments={}),),
            )

        evaluator = StandaloneEvaluator(caller=mock_caller)
        result = evaluator.evaluate(
            artifacts=artifacts,
            dataset=dataset,
            metrics=[ToolSelectionAccuracy()],
        )
        assert result.overall_score == 1.0

    def test_evaluate_aggregates_multiple_items(self):
        artifacts = [
            ContextArtifact(
                id="role.editor.system_prompt",
                kind=ArtifactKind.SYSTEM_PROMPT,
                content="You are a video editor.",
                source_location="roles.py",
                metadata={},
            ),
        ]
        dataset = EvalDataset(
            name="test",
            items=[
                EvalItem(
                    id="001",
                    task="Trim clip",
                    expected_tools=["trim_clip"],
                    expected_output_pattern=None,
                    context={},
                ),
                EvalItem(
                    id="002",
                    task="Adjust volume",
                    expected_tools=["adjust_volume"],
                    expected_output_pattern=None,
                    context={},
                ),
            ],
        )

        call_count = 0

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            from ave.optimize.metrics import ToolCall

            # First call matches, second doesn't
            if "Trim" in user_message:
                return LLMResponse(
                    text="trimming",
                    tool_calls=(ToolCall(name="trim_clip", arguments={}),),
                )
            return LLMResponse(
                text="wrong tool",
                tool_calls=(ToolCall(name="trim_clip", arguments={}),),
            )

        evaluator = StandaloneEvaluator(caller=mock_caller)
        result = evaluator.evaluate(
            artifacts=artifacts,
            dataset=dataset,
            metrics=[ToolSelectionAccuracy()],
        )
        assert call_count == 2
        assert result.overall_score == 0.5  # 1 correct + 0 correct / 2

    def test_evaluate_empty_dataset_returns_zero(self):
        artifacts = [
            ContextArtifact(
                id="role.editor.system_prompt",
                kind=ArtifactKind.SYSTEM_PROMPT,
                content="You are a video editor.",
                source_location="roles.py",
                metadata={},
            ),
        ]
        dataset = EvalDataset(name="empty", items=[])

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            raise AssertionError("Should not be called with empty dataset")

        evaluator = StandaloneEvaluator(caller=mock_caller)
        result = evaluator.evaluate(
            artifacts=artifacts,
            dataset=dataset,
            metrics=[ToolSelectionAccuracy()],
        )
        assert result.overall_score == 0.0
        assert result.per_metric == {}

    def test_build_system_prompt_includes_all_kinds(self):
        artifacts = [
            ContextArtifact(
                id="role.editor.system_prompt",
                kind=ArtifactKind.SYSTEM_PROMPT,
                content="SYSTEM_TEXT",
                source_location="roles.py",
                metadata={},
            ),
            ContextArtifact(
                id="role.editor.description",
                kind=ArtifactKind.ROLE_DESCRIPTION,
                content="ROLE_DESC_TEXT",
                source_location="roles.py",
                metadata={},
            ),
            ContextArtifact(
                id="tool.trim.description",
                kind=ArtifactKind.TOOL_DESCRIPTION,
                content="TOOL_DESC_TEXT",
                source_location="tools.py",
                metadata={},
            ),
        ]
        evaluator = StandaloneEvaluator(caller=lambda s, u: LLMResponse(text="", tool_calls=()))
        prompt = evaluator._build_system_prompt(artifacts)
        assert "SYSTEM_TEXT" in prompt
        assert "Role: ROLE_DESC_TEXT" in prompt
        assert "Available tool: TOOL_DESC_TEXT" in prompt

    def test_evaluate_builds_system_prompt_from_artifacts(self):
        """Verify the evaluator passes artifact content as system prompt."""
        captured_prompts: list[str] = []

        artifacts = [
            ContextArtifact(
                id="role.editor.system_prompt",
                kind=ArtifactKind.SYSTEM_PROMPT,
                content="CUSTOM_SYSTEM_PROMPT_TEXT",
                source_location="roles.py",
                metadata={},
            ),
        ]
        dataset = EvalDataset(
            name="test",
            items=[
                EvalItem(
                    id="001",
                    task="test task",
                    expected_tools=[],
                    expected_output_pattern=None,
                    context={},
                ),
            ],
        )

        def capturing_caller(system_prompt: str, user_message: str) -> LLMResponse:
            captured_prompts.append(system_prompt)
            return LLMResponse(text="ok", tool_calls=())

        evaluator = StandaloneEvaluator(caller=capturing_caller)
        evaluator.evaluate(artifacts=artifacts, dataset=dataset, metrics=[])
        assert len(captured_prompts) == 1
        assert "CUSTOM_SYSTEM_PROMPT_TEXT" in captured_prompts[0]
