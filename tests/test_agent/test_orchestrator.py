"""Tests for Orchestrator — bridges Claude Agent SDK with AVE tool registry."""

from __future__ import annotations

import pytest

from ave.agent.session import EditingSession
from ave.agent.registry import ToolRegistry, RegistryError
from ave.agent.dependencies import SessionState
from ave.agent.orchestrator import Orchestrator, MetaToolDef


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def session() -> EditingSession:
    """Create a session with test tools."""
    s = EditingSession.__new__(EditingSession)
    s._registry = ToolRegistry()
    s._state = SessionState()
    s._history = []
    s._project_path = None
    s._snapshot_manager = None
    s._lock = __import__("threading").Lock()

    @s._registry.tool(domain="math", requires=[], provides=["computed"])
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @s._registry.tool(domain="math", requires=["computed"], provides=[])
    def double(x: int) -> int:
        """Double a number. Requires a previous computation."""
        return x * 2

    @s._registry.tool(domain="text", requires=[], provides=[])
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    return s


@pytest.fixture
def orchestrator(session: EditingSession) -> Orchestrator:
    return Orchestrator(session)


# ── test_orchestrator_creates_with_session ────────────────────────────────────


def test_orchestrator_creates_with_session(orchestrator: Orchestrator):
    """Orchestrator(session) initializes."""
    assert orchestrator.session is not None
    assert orchestrator.turn_count == 0


# ── test_orchestrator_get_system_prompt ───────────────────────────────────────


def test_orchestrator_get_system_prompt(orchestrator: Orchestrator):
    """Returns system prompt with domain summaries."""
    prompt = orchestrator.get_system_prompt()

    assert "math" in prompt
    assert "text" in prompt
    assert "search_tools" in prompt
    assert "call_tool" in prompt


# ── test_orchestrator_meta_tools ──────────────────────────────────────────────


def test_orchestrator_meta_tools(orchestrator: Orchestrator):
    """Has 3 meta-tool definitions (search_tools, get_tool_schema, call_tool)."""
    meta_tools = orchestrator.get_meta_tools()

    assert len(meta_tools) == 3
    names = {mt.name for mt in meta_tools}
    assert names == {"search_tools", "get_tool_schema", "call_tool"}

    for mt in meta_tools:
        assert isinstance(mt, MetaToolDef)
        assert mt.description
        assert mt.parameters


# ── test_orchestrator_handle_search_tools ─────────────────────────────────────


def test_orchestrator_handle_search_tools(orchestrator: Orchestrator):
    """Processes search request, returns summaries."""
    result = orchestrator.handle_tool_call("search_tools", {"query": "add"})

    assert "add" in result
    assert "math" in result
    assert "Found" in result


def test_orchestrator_handle_search_tools_by_domain(orchestrator: Orchestrator):
    """Search by domain returns only matching tools."""
    result = orchestrator.handle_tool_call("search_tools", {"domain": "text"})

    assert "greet" in result
    assert "add" not in result


def test_orchestrator_handle_search_tools_no_results(orchestrator: Orchestrator):
    """Search with no matches returns appropriate message."""
    result = orchestrator.handle_tool_call("search_tools", {"query": "zzzznonexistent"})
    assert "No tools found" in result


# ── test_orchestrator_handle_get_schema ───────────────────────────────────────


def test_orchestrator_handle_get_schema(orchestrator: Orchestrator):
    """Processes schema request, returns full schema."""
    result = orchestrator.handle_tool_call("get_tool_schema", {"tool_name": "add"})

    assert "add" in result
    assert "math" in result
    assert "a" in result  # parameter name
    assert "b" in result  # parameter name


def test_orchestrator_handle_get_schema_not_found(orchestrator: Orchestrator):
    """Schema for unknown tool returns error."""
    result = orchestrator.handle_tool_call("get_tool_schema", {"tool_name": "nonexistent"})
    assert "Error" in result


# ── test_orchestrator_handle_call_tool ────────────────────────────────────────


def test_orchestrator_handle_call_tool(orchestrator: Orchestrator):
    """Processes tool call, returns result."""
    result = orchestrator.handle_tool_call(
        "call_tool", {"tool_name": "add", "params": {"a": 3, "b": 4}}
    )

    assert "7" in result
    assert "successfully" in result


# ── test_orchestrator_handle_call_tool_error ──────────────────────────────────


def test_orchestrator_handle_call_tool_error(orchestrator: Orchestrator):
    """Tool call error returns error message, not exception."""
    # "double" requires "computed" state which hasn't been set
    result = orchestrator.handle_tool_call(
        "call_tool", {"tool_name": "double", "params": {"x": 5}}
    )

    assert "Error" in result
    assert "PrerequisiteError" in result


def test_orchestrator_handle_unknown_meta_tool(orchestrator: Orchestrator):
    """Unknown meta-tool returns error."""
    result = orchestrator.handle_tool_call("nonexistent_meta", {})
    assert "Error" in result
    assert "Unknown" in result


# ── test_orchestrator_format_tool_result ──────────────────────────────────────


def test_orchestrator_format_tool_result(orchestrator: Orchestrator):
    """Formats tool result for LLM consumption (readable text)."""
    result = orchestrator.handle_tool_call(
        "call_tool", {"tool_name": "greet", "params": {"name": "Alice"}}
    )

    assert "Hello, Alice!" in result
    assert "greet" in result


# ── test_orchestrator_conversation_context ────────────────────────────────────


def test_orchestrator_conversation_context(orchestrator: Orchestrator):
    """Tracks conversation turn count."""
    assert orchestrator.turn_count == 0

    orchestrator.handle_tool_call("search_tools", {"query": ""})
    assert orchestrator.turn_count == 1

    orchestrator.handle_tool_call("search_tools", {"query": "add"})
    assert orchestrator.turn_count == 2
