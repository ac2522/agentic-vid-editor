"""Tests for EditingSession — lifecycle for agent-driven editing."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ave.agent.session import EditingSession, SessionError, ToolCall
from ave.agent.registry import PrerequisiteError, ToolRegistry
from ave.agent.dependencies import SessionState


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def session() -> EditingSession:
    """Create a session with a minimal test tool registered."""
    s = EditingSession.__new__(EditingSession)
    s._registry = ToolRegistry()
    s._state = SessionState()
    s._history = []
    s._project_path = None
    s._snapshot_manager = None
    s._transition_graph = None
    s._lock = __import__("threading").Lock()

    # Register a simple tool with no prerequisites
    @s._registry.tool(domain="test", requires=[], provides=["thing_done"])
    def do_thing(value: int) -> dict:
        """Do a thing with a value."""
        return {"value": value, "status": "done"}

    # Register a tool that requires "thing_done"
    @s._registry.tool(domain="test", requires=["thing_done"], provides=["second_done"])
    def do_second(x: int) -> dict:
        """Do a second thing that requires the first."""
        return {"x": x}

    return s


@pytest.fixture
def full_session() -> EditingSession:
    """Create a fully-loaded session with all domain tools."""
    return EditingSession()


# ── test_session_create ───────────────────────────────────────────────────────


def test_session_create(session: EditingSession):
    """EditingSession() initializes with empty state."""
    assert session.history == []
    assert session.state.provisions == frozenset()
    assert session._project_path is None


# ── test_session_load_project ─────────────────────────────────────────────────


def test_session_load_project(session: EditingSession, tmp_path: Path):
    """load_project(xges_path) sets timeline_loaded in state."""
    xges = tmp_path / "project.xges"
    xges.write_text("<ges/>")

    session.load_project(xges)

    assert session.state.has("timeline_loaded")
    assert session._project_path == xges


# ── test_session_load_project_nonexistent ─────────────────────────────────────


def test_session_load_project_nonexistent(session: EditingSession):
    """Raises SessionError for missing file."""
    with pytest.raises(SessionError, match="not found"):
        session.load_project(Path("/nonexistent/project.xges"))


# ── test_session_has_registry ─────────────────────────────────────────────────


def test_session_has_registry(full_session: EditingSession):
    """Session has a fully-loaded tool registry."""
    assert full_session.registry is not None
    assert isinstance(full_session.registry, ToolRegistry)
    assert full_session.registry.tool_count >= 20


# ── test_session_search_tools ─────────────────────────────────────────────────


def test_session_search_tools(session: EditingSession):
    """Delegates to registry.search_tools."""
    results = session.search_tools("thing")
    assert len(results) >= 1
    assert any(r.name == "do_thing" for r in results)


# ── test_session_get_tool_schema ──────────────────────────────────────────────


def test_session_get_tool_schema(session: EditingSession):
    """Delegates to registry.get_tool_schema."""
    schema = session.get_tool_schema("do_thing")
    assert schema.name == "do_thing"
    assert schema.domain == "test"
    assert any(p.name == "value" for p in schema.params)


# ── test_session_call_tool ────────────────────────────────────────────────────


def test_session_call_tool(session: EditingSession):
    """Executes tool through registry with session state tracking."""
    result = session.call_tool("do_thing", {"value": 42})
    assert result == {"value": 42, "status": "done"}


# ── test_session_call_tool_tracks_state ───────────────────────────────────────


def test_session_call_tool_tracks_state(session: EditingSession):
    """After calling a tool, provisions are tracked."""
    session.call_tool("do_thing", {"value": 1})
    assert session.state.has("thing_done")


# ── test_session_call_tool_prerequisite_check ─────────────────────────────────


def test_session_call_tool_prerequisite_check(session: EditingSession):
    """Calling tool without prerequisites raises PrerequisiteError."""
    with pytest.raises(PrerequisiteError, match="thing_done"):
        session.call_tool("do_second", {"x": 1})


# ── test_session_history ──────────────────────────────────────────────────────


def test_session_history(session: EditingSession):
    """Session tracks tool call history (name, params, result, timestamp)."""
    before = time.time()
    session.call_tool("do_thing", {"value": 99})
    after = time.time()

    assert len(session.history) == 1
    call = session.history[0]
    assert call.tool_name == "do_thing"
    assert call.params == {"value": 99}
    assert call.result == {"value": 99, "status": "done"}
    assert before <= call.timestamp <= after
    assert call.provisions == ["thing_done"]


# ── test_session_undo_last ────────────────────────────────────────────────────


def test_session_undo_last(session: EditingSession):
    """Undo removes last history entry and reverts provisions."""
    session.call_tool("do_thing", {"value": 1})
    assert session.state.has("thing_done")

    undone = session.undo_last()

    assert undone is not None
    assert undone.tool_name == "do_thing"
    assert not session.state.has("thing_done")
    assert len(session.history) == 0


def test_session_undo_last_empty(session: EditingSession):
    """Undo on empty history returns None."""
    assert session.undo_last() is None


# ── test_session_reset ────────────────────────────────────────────────────────


def test_session_reset(session: EditingSession, tmp_path: Path):
    """Reset clears state and history."""
    xges = tmp_path / "project.xges"
    xges.write_text("<ges/>")
    session.load_project(xges)
    session.call_tool("do_thing", {"value": 1})

    session.reset()

    assert session.state.provisions == frozenset()
    assert session.history == []
    assert session._project_path is None


# ── test_session_to_dict ──────────────────────────────────────────────────────


def test_session_to_dict(session: EditingSession):
    """Serializable summary (tool count, state, history length)."""
    session.call_tool("do_thing", {"value": 1})

    d = session.to_dict()

    assert d["tool_count"] == 2  # do_thing + do_second
    assert "thing_done" in d["state"]
    assert d["history_length"] == 1
    assert d["project_path"] is None
