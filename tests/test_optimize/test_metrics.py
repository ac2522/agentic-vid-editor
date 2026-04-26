"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from ave.optimize.datasets import EvalItem
from ave.optimize.metrics import (
    ConventionCompliance,
    LLMResponse,
    MetricResult,
    ToolCall,
    ToolSelectionAccuracy,
)


def _make_item(expected_tools: list[str], task: str = "test task") -> EvalItem:
    return EvalItem(
        id="test",
        task=task,
        expected_tools=expected_tools,
        expected_output_pattern=None,
        context={},
    )


def _make_response(
    tool_names: list[str] | None = None,
    text: str = "",
) -> LLMResponse:
    calls = []
    if tool_names:
        calls = [ToolCall(name=n, arguments={}) for n in tool_names]
    return LLMResponse(text=text, tool_calls=tuple(calls))


class TestToolSelectionAccuracy:
    def test_exact_match_scores_one(self):
        metric = ToolSelectionAccuracy()
        item = _make_item(["trim_clip"])
        response = _make_response(["trim_clip"])
        result = metric.score(item, response)
        assert result.value == 1.0

    def test_no_match_scores_zero(self):
        metric = ToolSelectionAccuracy()
        item = _make_item(["trim_clip"])
        response = _make_response(["adjust_volume"])
        result = metric.score(item, response)
        assert result.value == 0.0

    def test_partial_match_jaccard(self):
        metric = ToolSelectionAccuracy()
        item = _make_item(["trim_clip", "concatenate"])
        response = _make_response(["trim_clip", "adjust_volume"])
        result = metric.score(item, response)
        # Jaccard: intersection=1 (trim_clip), union=3 (trim,concat,adjust_volume)
        assert abs(result.value - 1 / 3) < 0.01

    def test_empty_expected_and_empty_response(self):
        metric = ToolSelectionAccuracy()
        item = _make_item([])
        response = _make_response([])
        result = metric.score(item, response)
        assert result.value == 1.0  # Both empty = perfect match

    def test_empty_expected_nonempty_response(self):
        metric = ToolSelectionAccuracy()
        item = _make_item([])
        response = _make_response(["trim_clip"])
        result = metric.score(item, response)
        assert result.value == 0.0

    def test_result_has_reason(self):
        metric = ToolSelectionAccuracy()
        item = _make_item(["trim_clip"])
        response = _make_response(["trim_clip"])
        result = metric.score(item, response)
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_name_is_set(self):
        metric = ToolSelectionAccuracy()
        assert metric.name == "tool_selection_accuracy"

    def test_no_tool_calls_in_response(self):
        metric = ToolSelectionAccuracy()
        item = _make_item(["trim_clip"])
        response = _make_response(text="I would use trim_clip")
        result = metric.score(item, response)
        assert result.value == 0.0


class TestConventionCompliance:
    def test_nanosecond_timestamps_pass(self):
        metric = ConventionCompliance()
        item = _make_item([], task="Set start to 2 seconds")
        response = _make_response(text="Setting start_ns to 2000000000")
        result = metric.score(item, response)
        # Should get credit for using nanoseconds
        assert result.value > 0.0

    def test_agent_prefix_in_metadata(self):
        metric = ConventionCompliance()
        item = _make_item([], task="Label this clip")
        response = _make_response(text='Setting metadata key "agent:label" to "intro"')
        result = metric.score(item, response)
        assert result.value > 0.0

    def test_no_conventions_found_scores_zero(self):
        metric = ConventionCompliance()
        item = _make_item([], task="Do something")
        response = _make_response(text="OK done")
        result = metric.score(item, response)
        assert result.value == 0.0

    def test_result_has_reason(self):
        metric = ConventionCompliance()
        item = _make_item([], task="test")
        response = _make_response(text="agent:label")
        result = metric.score(item, response)
        assert isinstance(result.reason, str)

    def test_all_three_conventions_scores_one(self):
        metric = ConventionCompliance()
        item = _make_item([], task="Set metadata and codec")
        response = _make_response(
            text='Setting agent:label to "intro" at 2000000000 ns, encoding DNxHR HQX'
        )
        result = metric.score(item, response)
        assert result.value == 1.0

    def test_name_is_set(self):
        assert ConventionCompliance.name == "convention_compliance"


class TestMetricResult:
    def test_create_result(self):
        result = MetricResult(name="test", value=0.85, reason="Good match")
        assert result.name == "test"
        assert result.value == 0.85
        assert result.reason == "Good match"

    def test_result_is_frozen(self):
        result = MetricResult(name="test", value=0.5, reason="ok")
        with pytest.raises(AttributeError):
            result.value = 1.0  # type: ignore[misc]


class TestLLMResponse:
    def test_create_response(self):
        response = LLMResponse(text="hello", tool_calls=())
        assert response.text == "hello"
        assert len(response.tool_calls) == 0

    def test_response_with_tool_calls(self):
        calls = (ToolCall(name="trim", arguments={"start": 0}),)
        response = LLMResponse(text="", tool_calls=calls)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "trim"
