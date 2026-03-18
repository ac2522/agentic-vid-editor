"""Tests for tool transition graph."""

from __future__ import annotations

import json

import pytest

from ave.agent.transitions import ToolTransitionGraph


class TestToolTransitionGraph:
    """Tests for ToolTransitionGraph."""

    def test_record_increments_count(self):
        graph = ToolTransitionGraph()
        graph.record("trim", "split")
        assert graph.get_transition_count("trim", "split") == 1
        graph.record("trim", "split")
        assert graph.get_transition_count("trim", "split") == 2

    def test_suggest_next_sorted_by_probability(self):
        graph = ToolTransitionGraph()
        graph.record("trim", "split")
        graph.record("trim", "split")
        graph.record("trim", "split")
        graph.record("trim", "volume")

        suggestions = graph.suggest_next("trim")
        assert len(suggestions) == 2
        # split has 3/4 probability, volume has 1/4
        assert suggestions[0][0] == "split"
        assert suggestions[0][1] == pytest.approx(0.75)
        assert suggestions[1][0] == "volume"
        assert suggestions[1][1] == pytest.approx(0.25)

    def test_suggest_next_unknown_tool_returns_empty(self):
        graph = ToolTransitionGraph()
        graph.record("trim", "split")

        suggestions = graph.suggest_next("nonexistent")
        assert suggestions == []

    def test_suggest_next_limit(self):
        graph = ToolTransitionGraph()
        graph.record("trim", "a")
        graph.record("trim", "b")
        graph.record("trim", "c")
        graph.record("trim", "d")

        suggestions = graph.suggest_next("trim", limit=2)
        assert len(suggestions) == 2

    def test_get_transition_count_no_record(self):
        graph = ToolTransitionGraph()
        assert graph.get_transition_count("trim", "split") == 0

    def test_json_roundtrip(self):
        graph = ToolTransitionGraph()
        graph.record("trim", "split")
        graph.record("trim", "split")
        graph.record("trim", "volume")
        graph.record("split", "concat")

        json_data = graph.to_json()
        restored = ToolTransitionGraph.from_json(json_data)

        assert restored.get_transition_count("trim", "split") == 2
        assert restored.get_transition_count("trim", "volume") == 1
        assert restored.get_transition_count("split", "concat") == 1

        # Suggestions should be identical
        assert graph.suggest_next("trim") == restored.suggest_next("trim")

    def test_multiple_transitions_correct_probabilities(self):
        graph = ToolTransitionGraph()
        # 5 transitions from "ingest": 2 to trim, 2 to split, 1 to volume
        graph.record("ingest", "trim")
        graph.record("ingest", "trim")
        graph.record("ingest", "split")
        graph.record("ingest", "split")
        graph.record("ingest", "volume")

        suggestions = graph.suggest_next("ingest")
        probs = {name: prob for name, prob in suggestions}

        assert probs["trim"] == pytest.approx(0.4)
        assert probs["split"] == pytest.approx(0.4)
        assert probs["volume"] == pytest.approx(0.2)

    def test_to_json_valid_json(self):
        graph = ToolTransitionGraph()
        graph.record("a", "b")
        data = graph.to_json()
        parsed = json.loads(data)
        assert isinstance(parsed, dict)
