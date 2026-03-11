"""Tool dependency graph — prerequisite/provision tracking.

Tools declare what state they require and what state they provide.
The session tracks accumulated state across tool calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


class DependencyError(Exception):
    """Raised when dependency operations fail."""


@dataclass
class ToolDependency:
    """Dependencies for a single tool."""

    tool_name: str
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)


class DependencyGraph:
    """Tracks tool prerequisites and provisions."""

    def __init__(self) -> None:
        self._deps: dict[str, ToolDependency] = {}

    def add_tool(self, name: str, requires: list[str], provides: list[str]) -> None:
        self._deps[name] = ToolDependency(
            tool_name=name,
            requires=list(requires),
            provides=list(provides),
        )

    def check_prerequisites(self, tool_name: str, current_state: set[str] | frozenset[str]) -> list[str]:
        """Returns list of missing prerequisites. Empty = all met."""
        dep = self._deps.get(tool_name)
        if dep is None:
            raise DependencyError(f"Tool not found in dependency graph: {tool_name}")
        return [r for r in dep.requires if r not in current_state]

    def get_provisions(self, tool_name: str) -> list[str]:
        dep = self._deps.get(tool_name)
        if dep is None:
            raise DependencyError(f"Tool not found in dependency graph: {tool_name}")
        return list(dep.provides)

    def to_json(self) -> str:
        data = {
            name: {"requires": dep.requires, "provides": dep.provides}
            for name, dep in self._deps.items()
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> DependencyGraph:
        parsed = json.loads(data)
        graph = cls()
        for name, dep_data in parsed.items():
            graph.add_tool(name, requires=dep_data["requires"], provides=dep_data["provides"])
        return graph


class SessionState:
    """Tracks accumulated state across tool calls in a session."""

    def __init__(self) -> None:
        self._state: set[str] = set()

    def add(self, *provisions: str) -> None:
        self._state.update(provisions)

    def discard(self, *provisions: str) -> None:
        for p in provisions:
            self._state.discard(p)

    def has(self, requirement: str) -> bool:
        return requirement in self._state

    def has_all(self, requirements: list[str]) -> bool:
        return all(r in self._state for r in requirements)

    def reset(self) -> None:
        self._state.clear()

    @property
    def provisions(self) -> frozenset[str]:
        return frozenset(self._state)
