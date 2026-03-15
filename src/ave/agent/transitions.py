"""Tool transition graph for tracking usage patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


class ToolTransitionGraph:
    """Tracks tool usage patterns as a directed graph of transition counts.

    Records which tools are called after which, building transition
    probabilities. Used to suggest likely next tools (AutoTool-style).
    """

    def __init__(self) -> None:
        self._transitions: dict[str, dict[str, int]] = {}
        self._total_from: dict[str, int] = {}

    def record(self, from_tool: str, to_tool: str) -> None:
        """Record a transition from one tool to another."""
        if from_tool not in self._transitions:
            self._transitions[from_tool] = {}
        self._transitions[from_tool][to_tool] = (
            self._transitions[from_tool].get(to_tool, 0) + 1
        )
        self._total_from[from_tool] = self._total_from.get(from_tool, 0) + 1

    def suggest_next(
        self, current_tool: str, limit: int = 5
    ) -> list[tuple[str, float]]:
        """Suggest next tools based on transition probabilities.

        Returns list of (tool_name, probability) sorted by probability descending.
        """
        if current_tool not in self._transitions:
            return []

        total = self._total_from[current_tool]
        suggestions = [
            (to_tool, count / total)
            for to_tool, count in self._transitions[current_tool].items()
        ]
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:limit]

    def get_transition_count(self, from_tool: str, to_tool: str) -> int:
        """Get raw transition count between two tools."""
        return self._transitions.get(from_tool, {}).get(to_tool, 0)

    def to_json(self) -> str:
        """Serialize transition graph to JSON."""
        return json.dumps(
            {
                "transitions": self._transitions,
                "total_from": self._total_from,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> ToolTransitionGraph:
        """Deserialize transition graph from JSON."""
        parsed = json.loads(data)
        graph = cls()
        graph._transitions = parsed["transitions"]
        graph._total_from = parsed["total_from"]
        return graph
