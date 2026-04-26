"""Tests for the public API in ave.optimize.__init__."""

from __future__ import annotations

import json
from pathlib import Path


from ave.optimize import evaluate_artifacts
from ave.optimize.metrics import LLMResponse, ToolCall


class TestEvaluateArtifacts:
    """Tests for the top-level evaluate_artifacts() function."""

    def test_evaluate_artifacts_with_mock_caller(self, tmp_path: Path):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockRole:
            name: str
            description: str
            system_prompt: str
            domains: tuple[str, ...] = ()

        roles = [MockRole("editor", "Video editor", "You edit videos.")]

        # Write a simple dataset
        dataset_path = tmp_path / "test.jsonl"
        items = [
            {
                "id": "001",
                "task": "Trim the clip",
                "expected_tools": ["trim_clip"],
                "context": {},
            },
        ]
        with open(dataset_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        def mock_caller(system_prompt: str, user_message: str) -> LLMResponse:
            return LLMResponse(
                text="Trimming",
                tool_calls=(ToolCall(name="trim_clip", arguments={}),),
            )

        result = evaluate_artifacts(
            roles=roles,
            dataset_path=dataset_path,
            caller=mock_caller,
        )
        assert result.overall_score == 1.0
        assert "tool_selection_accuracy" in result.per_metric
