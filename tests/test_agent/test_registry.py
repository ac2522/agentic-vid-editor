"""Tests for Tool Registry — progressive discovery for agent tool access."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from ave.agent.registry import (
    ParamInfo,
    PrerequisiteError,
    RegistryError,
    ToolRegistry,
    ToolSchema,
    ToolSummary,
)
from ave.agent.dependencies import SessionState


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def _register_add(registry: ToolRegistry) -> None:
    """Register a simple add tool."""

    @registry.tool(domain="math")
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b


def _register_trim(registry: ToolRegistry) -> None:
    """Register a trim tool in the editing domain."""

    @registry.tool(domain="editing", requires=["media_loaded"], provides=["trimmed_clip"])
    def trim(start_ns: int, end_ns: int, track: int = 0) -> dict:
        """Trim a clip to the given time range.

        Removes content outside start_ns..end_ns on the specified track.
        All times in nanoseconds.
        """
        return {"start_ns": start_ns, "end_ns": end_ns, "track": track}


def _register_volume(registry: ToolRegistry) -> None:
    """Register a volume tool in the audio domain."""

    @registry.tool(domain="audio", provides=["volume_adjusted"])
    def set_volume(level: float, fade_in_ns: int = 0) -> dict:
        """Set audio volume level.

        Adjusts volume with optional fade-in.
        """
        return {"level": level, "fade_in_ns": fade_in_ns}


# ── test_registry_register_tool ────────────────────────────────────────────


def test_registry_register_tool(registry: ToolRegistry) -> None:
    _register_add(registry)
    assert registry.tool_count == 1


# ── test_registry_register_tool_metadata ───────────────────────────────────


def test_registry_register_tool_metadata(registry: ToolRegistry) -> None:
    _register_trim(registry)
    schema = registry.get_tool_schema("trim")
    assert schema.name == "trim"
    assert schema.domain == "editing"
    assert "Trim a clip" in schema.description


# ── test_registry_register_tool_schema_from_type_hints ─────────────────────


def test_registry_register_tool_schema_from_type_hints(registry: ToolRegistry) -> None:
    @registry.tool(domain="test")
    def multi_type(
        name: str,
        count: int,
        ratio: float,
        flag: bool,
        items: list,
        meta: dict,
        output: Path,
    ) -> str:
        """Tool with many types."""
        return "ok"

    schema = registry.get_tool_schema("multi_type")
    param_types = {p.name: p.type_str for p in schema.params}
    assert param_types["name"] == "str"
    assert param_types["count"] == "int"
    assert param_types["ratio"] == "float"
    assert param_types["flag"] == "bool"
    assert param_types["items"] == "list"
    assert param_types["meta"] == "dict"
    assert param_types["output"] == "Path"
    # All required (no defaults)
    assert all(p.required for p in schema.params)


# ── test_registry_register_tool_pydantic_params ────────────────────────────


def test_registry_register_tool_pydantic_params(registry: ToolRegistry) -> None:
    class TrimParams(BaseModel):
        start_ns: int
        end_ns: int
        track: int = 0

    @registry.tool(domain="editing")
    def trim_pydantic(params: TrimParams) -> dict:
        """Trim using a Pydantic model."""
        return params.model_dump()

    schema = registry.get_tool_schema("trim_pydantic")
    param_names = {p.name for p in schema.params}
    assert "start_ns" in param_names
    assert "end_ns" in param_names
    assert "track" in param_names
    # track has a default
    track_param = next(p for p in schema.params if p.name == "track")
    assert not track_param.required
    assert track_param.default == 0


# ── test_registry_search_by_keyword ────────────────────────────────────────


def test_registry_search_by_keyword(registry: ToolRegistry) -> None:
    _register_trim(registry)
    _register_add(registry)
    _register_volume(registry)

    results = registry.search_tools("trim")
    assert len(results) == 1
    assert results[0].name == "trim"


# ── test_registry_search_by_domain ─────────────────────────────────────────


def test_registry_search_by_domain(registry: ToolRegistry) -> None:
    _register_trim(registry)
    _register_add(registry)
    _register_volume(registry)

    results = registry.search_tools(domain="editing")
    assert len(results) == 1
    assert results[0].domain == "editing"


# ── test_registry_search_by_keyword_and_domain ─────────────────────────────


def test_registry_search_by_keyword_and_domain(registry: ToolRegistry) -> None:
    _register_trim(registry)
    _register_add(registry)
    _register_volume(registry)

    results = registry.search_tools("volume", domain="audio")
    assert len(results) == 1
    assert results[0].name == "set_volume"


# ── test_registry_search_no_results ────────────────────────────────────────


def test_registry_search_no_results(registry: ToolRegistry) -> None:
    _register_trim(registry)
    results = registry.search_tools("nonexistent")
    assert results == []


# ── test_registry_search_case_insensitive ──────────────────────────────────


def test_registry_search_case_insensitive(registry: ToolRegistry) -> None:
    _register_trim(registry)
    results = registry.search_tools("TRIM")
    assert len(results) == 1
    assert results[0].name == "trim"


# ── test_registry_search_returns_summaries ─────────────────────────────────


def test_registry_search_returns_summaries(registry: ToolRegistry) -> None:
    _register_trim(registry)
    results = registry.search_tools("trim")
    assert len(results) == 1
    summary = results[0]
    assert isinstance(summary, ToolSummary)
    assert summary.name == "trim"
    assert summary.domain == "editing"
    # Description is first line of docstring (compact)
    assert summary.description == "Trim a clip to the given time range."


# ── test_registry_get_schema ───────────────────────────────────────────────


def test_registry_get_schema(registry: ToolRegistry) -> None:
    _register_trim(registry)
    schema = registry.get_tool_schema("trim")
    assert isinstance(schema, ToolSchema)
    assert schema.name == "trim"
    assert schema.requires == ["media_loaded"]
    assert schema.provides == ["trimmed_clip"]
    param_names = [p.name for p in schema.params]
    assert "start_ns" in param_names
    assert "end_ns" in param_names
    assert "track" in param_names
    # track has default=0 so not required
    track_param = next(p for p in schema.params if p.name == "track")
    assert not track_param.required
    assert track_param.default == 0


# ── test_registry_get_schema_unknown_tool ──────────────────────────────────


def test_registry_get_schema_unknown_tool(registry: ToolRegistry) -> None:
    with pytest.raises(RegistryError, match="not found"):
        registry.get_tool_schema("nonexistent")


# ── test_registry_call_tool ────────────────────────────────────────────────


def test_registry_call_tool(registry: ToolRegistry) -> None:
    _register_add(registry)
    result = registry.call_tool("add", {"a": 1, "b": 2})
    assert result == 3


# ── test_registry_call_tool_validates_prerequisites ────────────────────────


def test_registry_call_tool_validates_prerequisites(registry: ToolRegistry) -> None:
    _register_trim(registry)
    session = SessionState()
    # trim requires "media_loaded" which isn't in session
    with pytest.raises(PrerequisiteError, match="media_loaded"):
        registry.call_tool("trim", {"start_ns": 0, "end_ns": 1_000_000_000}, session_state=session)


# ── test_registry_call_tool_tracks_provisions ──────────────────────────────


def test_registry_call_tool_tracks_provisions(registry: ToolRegistry) -> None:
    _register_trim(registry)
    session = SessionState()
    session.add("media_loaded")  # satisfy prerequisite

    registry.call_tool("trim", {"start_ns": 0, "end_ns": 1_000_000_000}, session_state=session)
    assert session.has("trimmed_clip")


# ── test_registry_list_domains ─────────────────────────────────────────────


def test_registry_list_domains(registry: ToolRegistry) -> None:
    _register_trim(registry)
    _register_add(registry)
    _register_volume(registry)

    domains = registry.list_domains()
    domain_map = {d["domain"]: d["count"] for d in domains}
    assert domain_map["editing"] == 1
    assert domain_map["math"] == 1
    assert domain_map["audio"] == 1


# ── test_registry_tool_count ───────────────────────────────────────────────


def test_registry_tool_count(registry: ToolRegistry) -> None:
    assert registry.tool_count == 0
    _register_add(registry)
    assert registry.tool_count == 1
    _register_trim(registry)
    assert registry.tool_count == 2


# ── test_registry_duplicate_name_raises ────────────────────────────────────


def test_registry_duplicate_name_raises(registry: ToolRegistry) -> None:
    _register_add(registry)
    with pytest.raises(RegistryError, match="already registered"):

        @registry.tool(domain="math")
        def add(a: int, b: int) -> int:
            """Duplicate add."""
            return a + b
