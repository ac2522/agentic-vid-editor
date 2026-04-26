"""Execute-tier solver — drives a real EditingSession and records XGES state.

Unlike the plan solver (which uses non-executing stubs), this solver wires
real tools backed by ``session.call_tool()``. After the generate step, it
harvests session metadata into ``state.metadata`` for downstream scorers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools
from inspect_ai.tool import Tool, ToolDef, ToolParams
from inspect_ai.util import JSONSchema

from ave.agent.activity import ActivityLog
from ave.agent.registry import RegistryError, ToolSchema
from ave.agent.session import EditingSession
from ave.project.snapshots import SnapshotManager
from ave.harness.solvers.plan import (
    _json_type_for,
    _params_for,
    _shared_registry,
    extract_tool_calls,
)


MINIMAL_XGES = """\
<?xml version="1.0" encoding="utf-8"?>
<ges version="0.3">
  <project>
    <encoding-profiles/>
    <timeline>
      <track track-type="2" track-caps="video/x-raw" restriction-caps="video/x-raw,format=(string)I420,width=(int)1920,height=(int)1080,framerate=(fraction)25/1"/>
      <track track-type="1" track-caps="audio/x-raw"/>
      <layer priority="0"/>
    </timeline>
  </project>
</ges>"""


def _make_real_tool(name: str, session: EditingSession) -> Tool:
    """Build an Inspect AI Tool whose execute function calls session.call_tool."""
    registry = session.registry
    try:
        schema: ToolSchema | None = registry.get_tool_schema(name)
    except RegistryError:
        schema = None

    description = (
        schema.description if schema and schema.description else f"{name}"
    ).strip()
    parameters = _params_for(schema)

    async def execute(**kwargs: Any) -> str:
        try:
            result = session.call_tool(name, kwargs)
            return str(result) if result is not None else f"[ok] {name} completed"
        except Exception as exc:
            return f"[error] {name}: {exc}"

    tool_def = ToolDef(
        execute,
        name=name,
        description=description,
        parameters=parameters,
    )
    return tool_def.as_tool()


def _all_registry_tools(session: EditingSession) -> list[Tool]:
    """Build real Inspect AI tools for every tool in the session registry."""
    registry = session.registry
    summaries = registry.search_tools()
    return [_make_real_tool(s.name, session) for s in summaries]


@solver
def execute_solver() -> Solver:
    """Drive the agent with a real EditingSession; tools actually execute."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tmp_dir = tempfile.mkdtemp(prefix="ave_execute_")
        xges_path = Path(tmp_dir) / "project.xges"
        xges_path.write_text(MINIMAL_XGES)

        snapshot_manager = SnapshotManager()
        activity_log = ActivityLog()
        session = EditingSession(
            snapshot_manager=snapshot_manager,
            activity_log=activity_log,
        )
        session.load_project(xges_path)

        state.metadata = dict(state.metadata or {})
        state.metadata["_xges_dir"] = tmp_dir

        tools = _all_registry_tools(session)
        if tools:
            state = await use_tools(tools)(state, generate)
        state = await generate(state)

        state.metadata["called_tools"] = extract_tool_calls(state)
        state.metadata["final_xges"] = xges_path.read_text()
        state.metadata["snapshot_count"] = len(snapshot_manager.list_snapshots())
        state.metadata["activity_entries"] = [
            e.to_dict() for e in activity_log.entries()
        ]
        return state

    return solve
