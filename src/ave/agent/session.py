"""Editing session manager — lifecycle for agent-driven editing.

Manages: project loading, tool execution with state tracking,
undo history, and session serialization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ave.agent.registry import ToolRegistry, PrerequisiteError
from ave.agent.dependencies import SessionState


class SessionError(Exception):
    """Raised when session operations fail."""


@dataclass
class ToolCall:
    """Record of a tool invocation."""

    tool_name: str
    params: dict
    result: Any
    timestamp: float
    provisions: list[str]


class EditingSession:
    """Manages an editing session lifecycle.

    Provides:
    - Project loading (sets timeline_loaded state)
    - Tool execution with prerequisite validation and state tracking
    - Call history for undo and audit
    - Session serialization
    """

    def __init__(self) -> None:
        self._registry = ToolRegistry()
        self._state = SessionState()
        self._history: list[ToolCall] = []
        self._project_path: Path | None = None
        self._load_all_tools()

    def _load_all_tools(self) -> None:
        """Register all domain tools."""
        from ave.agent.tools.editing import register_editing_tools
        from ave.agent.tools.audio import register_audio_tools
        from ave.agent.tools.color import register_color_tools
        from ave.agent.tools.transcription import register_transcription_tools
        from ave.agent.tools.render import register_render_tools
        from ave.agent.tools.project import register_project_tools
        from ave.agent.tools.compositing import register_compositing_tools
        from ave.agent.tools.motion_graphics import register_motion_graphics_tools
        from ave.agent.tools.scene import register_scene_tools
        from ave.agent.tools.interchange import register_interchange_tools

        register_editing_tools(self._registry)
        register_audio_tools(self._registry)
        register_color_tools(self._registry)
        register_transcription_tools(self._registry)
        register_render_tools(self._registry)
        register_project_tools(self._registry)
        register_compositing_tools(self._registry)
        register_motion_graphics_tools(self._registry)
        register_scene_tools(self._registry)
        register_interchange_tools(self._registry)

    def load_project(self, xges_path: Path) -> None:
        """Load a project file. Sets timeline_loaded state."""
        path = Path(xges_path)
        if not path.exists():
            raise SessionError(f"Project file not found: {path}")
        self._project_path = path
        self._state.add("timeline_loaded")

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def history(self) -> list[ToolCall]:
        return list(self._history)

    def search_tools(self, query: str = "", domain: str | None = None):
        """Search for tools by keyword and/or domain."""
        return self._registry.search_tools(query, domain)

    def get_tool_schema(self, tool_name: str):
        """Get full parameter schema for a tool."""
        return self._registry.get_tool_schema(tool_name)

    def call_tool(self, tool_name: str, params: dict) -> Any:
        """Execute a tool with state tracking and history recording."""
        provisions = self._registry.get_tool_provisions(tool_name)

        result = self._registry.call_tool(tool_name, params, self._state)

        self._history.append(
            ToolCall(
                tool_name=tool_name,
                params=params,
                result=result,
                timestamp=time.time(),
                provisions=provisions,
            )
        )

        return result

    def undo_last(self) -> ToolCall | None:
        """Remove last tool call from history. Returns the removed call.

        Only removes provisions that are not also provided by earlier history entries.
        """
        if not self._history:
            return None
        call = self._history.pop()
        # Collect provisions still provided by remaining history
        remaining_provisions: set[str] = set()
        for h in self._history:
            remaining_provisions.update(h.provisions)
        # Only discard provisions not covered by remaining history
        for p in call.provisions:
            if p not in remaining_provisions:
                self._state.discard(p)
        return call

    def reset(self) -> None:
        """Clear state and history."""
        self._state.reset()
        self._history.clear()
        self._project_path = None

    def to_dict(self) -> dict:
        """Serializable session summary."""
        return {
            "tool_count": self._registry.tool_count,
            "state": sorted(self._state.provisions),
            "history_length": len(self._history),
            "project_path": str(self._project_path) if self._project_path else None,
        }
