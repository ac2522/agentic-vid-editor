"""Editing session manager — lifecycle for agent-driven editing.

Manages: project loading, tool execution with state tracking,
undo history, snapshot-based rollback, and session serialization.

All tool calls are serialized via a threading lock to ensure
snapshot and provision consistency when subagents run concurrently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ave.agent.registry import ToolRegistry, PrerequisiteError
from ave.agent.dependencies import SessionState
from ave.agent.transitions import ToolTransitionGraph
from ave.project.snapshots import SnapshotManager


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
    - XGES snapshot capture before each tool call (for rollback)
    - Call history for undo and audit
    - Thread-safe tool execution (lock serializes concurrent subagent calls)
    - Session serialization
    """

    def __init__(
        self,
        snapshot_manager: SnapshotManager | None = None,
        transition_graph: ToolTransitionGraph | None = None,
    ) -> None:
        self._registry = ToolRegistry()
        self._state = SessionState()
        self._history: list[ToolCall] = []
        self._project_path: Path | None = None
        self._snapshot_manager = snapshot_manager
        self._transition_graph = transition_graph
        self._lock = threading.Lock()
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
        from ave.agent.tools.download import register_download_tools

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
        register_download_tools(self._registry)

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

    @property
    def snapshot_manager(self) -> SnapshotManager | None:
        return self._snapshot_manager

    def call_tool(self, tool_name: str, params: dict) -> Any:
        """Execute a tool with state tracking, snapshots, and history recording.

        Thread-safe: serialized via lock for concurrent subagent safety.
        If a snapshot manager is configured and a project is loaded, captures
        a snapshot before execution. On failure, auto-restores the latest snapshot.
        """
        with self._lock:
            provisions = self._registry.get_tool_provisions(tool_name)

            # Capture snapshot before execution
            if self._project_path and self._snapshot_manager:
                self._snapshot_manager.capture(
                    self._project_path,
                    label=f"before {tool_name}",
                    provisions=self._state.provisions,
                    tool_name=tool_name,
                )

            try:
                result = self._registry.call_tool(tool_name, params, self._state)
            except Exception:
                # Auto-restore on failure
                if self._project_path and self._snapshot_manager:
                    restored = self._snapshot_manager.restore_latest(self._project_path)
                    if restored:
                        _, snap_provisions = restored
                        self._state.reset()
                        self._state.add(*snap_provisions)
                raise

            # Record transition from previous tool
            if self._transition_graph and self._history:
                prev_tool = self._history[-1].tool_name
                self._transition_graph.record(prev_tool, tool_name)

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
