"""Editing session manager — lifecycle for agent-driven editing.

Manages: project loading, tool execution with state tracking,
undo history, snapshot-based rollback, and session serialization.

All tool calls are serialized via a threading lock to ensure
snapshot and provision consistency when subagents run concurrently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ave.agent.activity import ActivityLog
from ave.agent.dependencies import SessionState
from ave.agent.errors import ScopeViolationError, SourceAssetWriteError
from ave.agent.registry import ToolRegistry
from ave.agent.roles import AgentRole
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
        plugin_dirs: list[Path] | None = None,
        skill_dirs: list[Path] | None = None,
        activity_log: ActivityLog | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._registry = ToolRegistry()
        self._state = SessionState()
        self._history: list[ToolCall] = []
        self._project_path: Path | None = None
        self._snapshot_manager = snapshot_manager
        self._transition_graph = transition_graph
        self._activity_log = activity_log
        self._project_root = Path(project_root).resolve() if project_root else None
        self._lock = threading.Lock()
        self._orchestrator_lock = threading.Lock()
        self._load_all_tools()

        # Plugin and skill systems
        from ave.plugins.loader import PluginLoader
        from ave.skills.loader import SkillLoader

        self._plugin_loader = PluginLoader(self._registry)
        self._skill_loader = SkillLoader()

        if plugin_dirs:
            from ave.plugins.discovery import discover_plugins

            for manifest in discover_plugins(plugin_dirs):
                self._plugin_loader.register_manifest(manifest)

        if skill_dirs:
            from ave.skills.discovery import discover_skills

            for meta in discover_skills(skill_dirs):
                self._skill_loader.register(meta)

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
        from ave.agent.tools.research import register_research_tools
        from ave.agent.tools.vfx import register_vfx_tools

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
        register_research_tools(self._registry)
        register_vfx_tools(self._registry)

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
    def orchestrator_lock(self) -> threading.Lock:
        """Lock for serializing entire orchestration runs (e.g., MCP edit_video).

        Use this to wrap multi-step operations that must not interleave
        with other orchestration runs. The per-call _lock remains for
        fine-grained snapshot integrity within a single tool call.
        """
        return self._orchestrator_lock

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def history(self) -> list[ToolCall]:
        return list(self._history)

    def search_tools(self, query: str = "", domain: str | None = None):
        """Search for tools by keyword and/or domain."""
        return self._registry.search_tools(query, domain)

    def match_skills(self, intent: str) -> list:
        """Match user intent against registered skills."""
        return self._skill_loader.match(intent)

    def load_skill(self, skill_name: str) -> str:
        """Load full skill body by name."""
        meta = self._skill_loader.get(skill_name)
        if meta is None:
            raise SessionError(f"Unknown skill: {skill_name}")
        return self._skill_loader.load_body(meta)

    def get_tool_schema(self, tool_name: str):
        """Get full parameter schema for a tool."""
        return self._registry.get_tool_schema(tool_name)

    @property
    def snapshot_manager(self) -> SnapshotManager | None:
        return self._snapshot_manager

    def call_tool(
        self,
        tool_name: str,
        params: dict,
        *,
        agent_role: AgentRole | None = None,
    ) -> Any:
        """Execute a tool with state tracking, snapshots, and history recording.

        Thread-safe: serialized via lock for concurrent subagent safety.
        If a snapshot manager is configured and a project is loaded, captures
        a snapshot before execution. On failure, auto-restores the latest snapshot.

        When ``agent_role`` is provided, the tool's declared ``domains_touched``
        must all be within the role's ``owned_domains``; otherwise
        ``ScopeViolationError`` is raised before execution.
        """
        if agent_role is not None:
            touched = self._registry.get_tool_domains_touched(tool_name)
            out_of_scope = [d for d in touched if d not in agent_role.owned_domains]
            if out_of_scope:
                raise ScopeViolationError(
                    f"Role {agent_role.name!r} cannot call {tool_name!r}: "
                    f"tool touches {[d.value for d in out_of_scope]}, "
                    f"role owns {[d.value for d in agent_role.owned_domains]}"
                )

        self._check_source_asset_immutability(params)

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

            activity_log = getattr(self, "_activity_log", None)
            if activity_log is not None:
                snap_id = ""
                if self._snapshot_manager is not None:
                    snaps = self._snapshot_manager.list_snapshots()
                    if snaps:
                        snap_id = snaps[-1].snapshot_id
                activity_log.append(
                    agent_id=(agent_role.name if agent_role else "anonymous"),
                    tool_name=tool_name,
                    summary=self._build_log_summary(tool_name, params),
                    snapshot_id=snap_id,
                )

            return result

    def _build_log_summary(self, tool_name: str, params: dict) -> str:
        """Build a terse one-line summary for the activity log.

        Default format: tool_name(key1=val1, key2=val2) with values truncated.
        """
        items = []
        for k, v in params.items():
            s = repr(v)
            if len(s) > 30:
                s = s[:27] + "..."
            items.append(f"{k}={s}")
        return f"{tool_name}({', '.join(items)})"

    def _check_source_asset_immutability(self, params: dict) -> None:
        """Reject tool calls whose params include paths under assets/media/source/.

        Walks the params dict looking for str/Path values. For each, resolves
        the path and checks whether it falls under ``<project_root>/assets/media/source/``.
        Raises ``SourceAssetWriteError`` on hit.

        Skips when no ``project_root`` is configured (backward compat).
        """
        project_root = getattr(self, "_project_root", None)
        if project_root is None:
            return
        source_root = (project_root / "assets" / "media" / "source").resolve()
        for value in params.values():
            if not isinstance(value, (str, Path)):
                continue
            try:
                candidate = Path(value).resolve()
            except (OSError, ValueError):
                continue
            try:
                candidate.relative_to(source_root)
            except ValueError:
                continue  # not under source root
            raise SourceAssetWriteError(
                f"Tool call would touch source asset: {candidate} "
                f"(source assets under {source_root} are write-protected)"
            )

    def begin_turn(self, turn_id: str) -> None:
        """Capture a pre-turn checkpoint. Call once at the start of a user turn."""
        if self._project_path is None or self._snapshot_manager is None:
            return  # silently no-op when project not loaded
        self._snapshot_manager.capture_turn_checkpoint(
            xges_path=self._project_path,
            turn_id=turn_id,
            provisions=self._state.provisions,
        )

    def end_turn(self, turn_id: str) -> None:
        """Capture a post-turn checkpoint for redo. Call once after the agent finishes."""
        if self._project_path is None or self._snapshot_manager is None:
            return
        self._snapshot_manager.capture_post_turn(
            xges_path=self._project_path,
            turn_id=turn_id,
            provisions=self._state.provisions,
        )

    def undo_turn(self, turn_id: str) -> None:
        """Roll back to the pre-turn checkpoint identified by turn_id."""
        if self._project_path is None or self._snapshot_manager is None:
            raise SessionError("Cannot undo: no project loaded or no snapshot manager")
        _, provs = self._snapshot_manager.rollback_to_turn(
            turn_id=turn_id,
            xges_path=self._project_path,
        )
        self._state.reset()
        self._state.add(*provs)

    def redo_turn(self, turn_id: str) -> None:
        """Re-apply the post-turn checkpoint identified by turn_id."""
        if self._project_path is None or self._snapshot_manager is None:
            raise SessionError("Cannot redo: no project loaded or no snapshot manager")
        _, provs = self._snapshot_manager.redo_turn(
            turn_id=turn_id,
            xges_path=self._project_path,
        )
        self._state.reset()
        self._state.add(*provs)

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
