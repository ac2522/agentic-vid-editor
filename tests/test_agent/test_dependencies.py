"""Tests for Tool Dependency Graph — prerequisite/provision tracking."""

import json

import pytest

from ave.agent.dependencies import DependencyGraph, SessionState


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def graph() -> DependencyGraph:
    return DependencyGraph()


@pytest.fixture
def state() -> SessionState:
    return SessionState()


# ── test_dependency_graph_add_tool ───────────────────────────────────────────


def test_dependency_graph_add_tool(graph: DependencyGraph) -> None:
    graph.add_tool("trim", requires=["media_loaded"], provides=["trimmed_clip"])
    provisions = graph.get_provisions("trim")
    assert provisions == ["trimmed_clip"]


# ── test_dependency_graph_check_prerequisites_met ────────────────────────────


def test_dependency_graph_check_prerequisites_met(graph: DependencyGraph) -> None:
    graph.add_tool("trim", requires=["media_loaded"], provides=["trimmed_clip"])
    missing = graph.check_prerequisites("trim", current_state={"media_loaded"})
    assert missing == []


# ── test_dependency_graph_check_prerequisites_not_met ────────────────────────


def test_dependency_graph_check_prerequisites_not_met(graph: DependencyGraph) -> None:
    graph.add_tool("trim", requires=["media_loaded", "project_open"], provides=["trimmed_clip"])
    missing = graph.check_prerequisites("trim", current_state={"project_open"})
    assert missing == ["media_loaded"]


# ── test_dependency_graph_apply_provisions ───────────────────────────────────


def test_dependency_graph_apply_provisions(graph: DependencyGraph, state: SessionState) -> None:
    graph.add_tool("trim", requires=[], provides=["trimmed_clip", "edit_made"])
    provisions = graph.get_provisions("trim")
    state.add(*provisions)
    assert state.has("trimmed_clip")
    assert state.has("edit_made")


# ── test_dependency_graph_chain ──────────────────────────────────────────────


def test_dependency_graph_chain(graph: DependencyGraph, state: SessionState) -> None:
    graph.add_tool("load", requires=[], provides=["media_loaded"])
    graph.add_tool("trim", requires=["media_loaded"], provides=["trimmed_clip"])

    # load has no prerequisites — can always run
    missing = graph.check_prerequisites("load", current_state=state.provisions)
    assert missing == []
    state.add(*graph.get_provisions("load"))

    # now trim's prerequisite is met
    missing = graph.check_prerequisites("trim", current_state=state.provisions)
    assert missing == []


# ── test_dependency_graph_no_requirements ────────────────────────────────────


def test_dependency_graph_no_requirements(graph: DependencyGraph) -> None:
    graph.add_tool("info", requires=[], provides=[])
    missing = graph.check_prerequisites("info", current_state=set())
    assert missing == []


# ── test_dependency_graph_serialize_json ──────────────────────────────────────


def test_dependency_graph_serialize_json(graph: DependencyGraph) -> None:
    graph.add_tool("load", requires=[], provides=["media_loaded"])
    graph.add_tool("trim", requires=["media_loaded"], provides=["trimmed_clip"])

    serialized = graph.to_json()
    data = json.loads(serialized)
    assert isinstance(data, dict)

    restored = DependencyGraph.from_json(serialized)
    assert restored.get_provisions("load") == ["media_loaded"]
    assert restored.check_prerequisites("trim", current_state=set()) == ["media_loaded"]
    assert restored.check_prerequisites("trim", current_state={"media_loaded"}) == []


# ── test_session_state_tracks_provisions ─────────────────────────────────────


def test_session_state_tracks_provisions(state: SessionState) -> None:
    state.add("media_loaded")
    state.add("trimmed_clip", "edit_made")
    assert state.has("media_loaded")
    assert state.has("trimmed_clip")
    assert state.has("edit_made")
    assert state.has_all(["media_loaded", "trimmed_clip", "edit_made"])
    assert not state.has("nonexistent")


# ── test_session_state_reset ─────────────────────────────────────────────────


def test_session_state_reset(state: SessionState) -> None:
    state.add("media_loaded", "trimmed_clip")
    assert state.has("media_loaded")

    state.reset()
    assert not state.has("media_loaded")
    assert not state.has("trimmed_clip")
    assert state.provisions == frozenset()
