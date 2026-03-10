"""Tests for domain tool registration across all 6 domains."""

from __future__ import annotations

import pytest

from ave.agent.registry import ToolRegistry


@pytest.fixture
def full_registry():
    """Create a registry with all domain tools registered."""
    from ave.agent.tools.editing import register_editing_tools
    from ave.agent.tools.audio import register_audio_tools
    from ave.agent.tools.color import register_color_tools
    from ave.agent.tools.transcription import register_transcription_tools
    from ave.agent.tools.render import register_render_tools
    from ave.agent.tools.project import register_project_tools

    registry = ToolRegistry()
    register_editing_tools(registry)
    register_audio_tools(registry)
    register_color_tools(registry)
    register_transcription_tools(registry)
    register_render_tools(registry)
    register_project_tools(registry)
    return registry


# ---- 1. Editing domain ----

def test_editing_tools_registered(full_registry):
    """Editing domain has: trim, split, concatenate, speed, transition (5 tools)."""
    results = full_registry.search_tools(domain="editing")
    names = {t.name for t in results}
    assert names == {"trim", "split", "concatenate", "speed", "transition"}


# ---- 2. Audio domain ----

def test_audio_tools_registered(full_registry):
    """Audio domain has: volume, fade, normalize (3 tools)."""
    results = full_registry.search_tools(domain="audio")
    names = {t.name for t in results}
    assert names == {"volume", "fade", "normalize"}


# ---- 3. Color domain ----

def test_color_tools_registered(full_registry):
    """Color domain has: color_grade, cdl, lut_parse (3 minimum)."""
    results = full_registry.search_tools(domain="color")
    names = {t.name for t in results}
    assert {"color_grade", "cdl", "lut_parse"}.issubset(names)
    assert len(names) >= 3


# ---- 4. Transcription domain ----

def test_transcription_tools_registered(full_registry):
    """Transcription domain has: search_transcript, find_fillers, text_cut, text_keep (4+ tools)."""
    results = full_registry.search_tools(domain="transcription")
    names = {t.name for t in results}
    assert {"search_transcript", "find_fillers", "text_cut", "text_keep"}.issubset(names)
    assert len(names) >= 4


# ---- 5. Render domain ----

def test_render_tools_registered(full_registry):
    """Render domain has: render_proxy, render_segment, compute_segments (3 tools)."""
    results = full_registry.search_tools(domain="render")
    names = {t.name for t in results}
    assert names == {"render_proxy", "render_segment", "compute_segments"}


# ---- 6. Project domain ----

def test_project_tools_registered(full_registry):
    """Project domain has: probe_media, ingest_media (2+ tools)."""
    results = full_registry.search_tools(domain="project")
    names = {t.name for t in results}
    assert {"probe_media", "ingest_media"}.issubset(names)
    assert len(names) >= 2


# ---- 7. All domains listed ----

def test_all_domains_listed(full_registry):
    """list_domains() returns all 6 domains."""
    domains = full_registry.list_domains()
    domain_names = {d["domain"] for d in domains}
    assert domain_names == {"editing", "audio", "color", "transcription", "render", "project"}


# ---- 8. Total tool count ----

def test_total_tool_count(full_registry):
    """Total tool count >= 20."""
    assert full_registry.tool_count >= 20


# ---- 9. Search across domains ----

def test_search_across_domains(full_registry):
    """search("trim") returns editing trim tool."""
    results = full_registry.search_tools("trim")
    names = [t.name for t in results]
    assert "trim" in names
    # Verify it's from the editing domain
    trim_result = next(t for t in results if t.name == "trim")
    assert trim_result.domain == "editing"


# ---- 10. Search within domain ----

def test_search_within_domain(full_registry):
    """search(domain="audio") returns only audio tools."""
    results = full_registry.search_tools(domain="audio")
    for t in results:
        assert t.domain == "audio"
    assert len(results) >= 3


# ---- 11. All tools have descriptions ----

def test_all_tools_have_descriptions(full_registry):
    """Every tool has non-empty description."""
    all_tools = full_registry.search_tools()
    assert len(all_tools) > 0
    for tool in all_tools:
        assert tool.description, f"Tool '{tool.name}' has empty description"


# ---- 12. All tools have schemas ----

def test_all_tools_have_schemas(full_registry):
    """Every tool returns valid ToolSchema with params."""
    all_tools = full_registry.search_tools()
    for tool_summary in all_tools:
        schema = full_registry.get_tool_schema(tool_summary.name)
        assert schema.name == tool_summary.name
        assert schema.domain == tool_summary.domain
        assert schema.description
        assert isinstance(schema.params, list)
        assert len(schema.params) > 0, f"Tool '{schema.name}' has no params"


# ---- 13. Editing trim callable ----

def test_editing_trim_callable(full_registry):
    """call_tool("trim", {...}) with valid params returns TrimParams."""
    from ave.tools.edit import TrimParams

    result = full_registry.call_tool("trim", {
        "clip_duration_ns": 10_000_000_000,
        "in_ns": 1_000_000_000,
        "out_ns": 5_000_000_000,
    })
    assert isinstance(result, TrimParams)
    assert result.in_ns == 1_000_000_000
    assert result.out_ns == 5_000_000_000
    assert result.duration_ns == 4_000_000_000


# ---- 14. Audio volume callable ----

def test_audio_volume_callable(full_registry):
    """call_tool("volume", {...}) returns VolumeParams."""
    from ave.tools.audio import VolumeParams

    result = full_registry.call_tool("volume", {"level_db": -6.0})
    assert isinstance(result, VolumeParams)
    assert result.level_db == -6.0


# ---- 15. Tool dependencies set ----

def test_tool_dependencies_set(full_registry):
    """Editing tools require 'timeline_loaded', etc."""
    schema = full_registry.get_tool_schema("trim")
    assert "timeline_loaded" in schema.requires

    schema = full_registry.get_tool_schema("split")
    assert "timeline_loaded" in schema.requires

    schema = full_registry.get_tool_schema("volume")
    assert "timeline_loaded" in schema.requires

    # Project tools should not require timeline_loaded
    schema = full_registry.get_tool_schema("probe_media")
    assert "timeline_loaded" not in schema.requires
