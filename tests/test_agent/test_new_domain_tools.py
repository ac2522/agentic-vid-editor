"""Tests for new Phase 5/6 domain tool registration (compositing, motion_graphics, scene, interchange)."""

from __future__ import annotations

import pytest

from ave.agent.registry import ToolRegistry, PrerequisiteError
from ave.agent.dependencies import SessionState


@pytest.fixture
def full_registry():
    """Create a registry with ALL domain tools registered (old + new)."""
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

    registry = ToolRegistry()
    register_editing_tools(registry)
    register_audio_tools(registry)
    register_color_tools(registry)
    register_transcription_tools(registry)
    register_render_tools(registry)
    register_project_tools(registry)
    register_compositing_tools(registry)
    register_motion_graphics_tools(registry)
    register_scene_tools(registry)
    register_interchange_tools(registry)
    return registry


# ---- Compositing domain ----


def test_compositing_tools_registered(full_registry):
    """Compositing domain has 4 tools."""
    results = full_registry.search_tools(domain="compositing")
    names = {t.name for t in results}
    assert names == {"set_layer_order", "apply_blend_mode", "set_clip_position", "set_clip_alpha"}


def test_search_blend_finds_compositing(full_registry):
    """search_tools('blend') finds compositing tools."""
    results = full_registry.search_tools("blend")
    names = [t.name for t in results]
    assert "apply_blend_mode" in names


def test_compositing_tools_require_timeline_and_clip(full_registry):
    """All compositing tools require timeline_loaded and clip_exists."""
    for tool_name in ["set_layer_order", "apply_blend_mode", "set_clip_position", "set_clip_alpha"]:
        schema = full_registry.get_tool_schema(tool_name)
        assert "timeline_loaded" in schema.requires
        assert "clip_exists" in schema.requires


# ---- Motion graphics domain ----


def test_motion_graphics_tools_registered(full_registry):
    """Motion graphics domain has 3 tools."""
    results = full_registry.search_tools(domain="motion_graphics")
    names = {t.name for t in results}
    assert names == {"add_text_overlay", "add_lower_third", "add_title_card"}


def test_search_text_finds_motion_graphics(full_registry):
    """search_tools('text') finds motion graphics tools."""
    results = full_registry.search_tools("text")
    names = [t.name for t in results]
    assert "add_text_overlay" in names


def test_motion_graphics_tools_require_timeline(full_registry):
    """All motion graphics tools require timeline_loaded."""
    for tool_name in ["add_text_overlay", "add_lower_third", "add_title_card"]:
        schema = full_registry.get_tool_schema(tool_name)
        assert "timeline_loaded" in schema.requires


# ---- Scene domain ----


def test_scene_tools_registered(full_registry):
    """Scene domain has 3 tools."""
    results = full_registry.search_tools(domain="scene")
    names = {t.name for t in results}
    assert names == {"detect_scenes", "classify_shots", "create_rough_cut"}


def test_search_scene_finds_scene_tools(full_registry):
    """search_tools('scene') finds scene detection tools."""
    results = full_registry.search_tools("scene")
    names = [t.name for t in results]
    assert "detect_scenes" in names


def test_detect_scenes_provides_scenes_detected(full_registry):
    """detect_scenes provides 'scenes_detected'."""
    schema = full_registry.get_tool_schema("detect_scenes")
    assert "scenes_detected" in schema.provides
    assert "timeline_loaded" in schema.requires


def test_classify_shots_requires_scenes_detected(full_registry):
    """classify_shots requires 'scenes_detected'."""
    schema = full_registry.get_tool_schema("classify_shots")
    assert "scenes_detected" in schema.requires
    assert "shots_classified" in schema.provides


def test_classify_prerequisite_enforced(full_registry):
    """classify_shots cannot be called without scenes_detected."""
    state = SessionState()
    state.add("timeline_loaded")
    # scenes_detected NOT added — should fail
    with pytest.raises(PrerequisiteError, match="scenes_detected"):
        full_registry.call_tool("classify_shots", {
            "video_path": "/tmp/test.mp4",
            "scenes_json": "[]",
            "output_dir": "/tmp/out",
        }, session_state=state)


def test_create_rough_cut_requires_scenes_detected(full_registry):
    """create_rough_cut requires 'scenes_detected'."""
    schema = full_registry.get_tool_schema("create_rough_cut")
    assert "scenes_detected" in schema.requires


# ---- Interchange domain ----


def test_interchange_tools_registered(full_registry):
    """Interchange domain has 2 tools."""
    results = full_registry.search_tools(domain="interchange")
    names = {t.name for t in results}
    assert names == {"export_otio", "import_otio"}


def test_search_export_finds_interchange(full_registry):
    """search_tools('export') finds interchange tools."""
    results = full_registry.search_tools("export")
    names = [t.name for t in results]
    assert "export_otio" in names


def test_export_otio_requires_timeline(full_registry):
    """export_otio requires timeline_loaded."""
    schema = full_registry.get_tool_schema("export_otio")
    assert "timeline_loaded" in schema.requires


def test_import_otio_provides_timeline(full_registry):
    """import_otio provides timeline_loaded."""
    schema = full_registry.get_tool_schema("import_otio")
    assert "timeline_loaded" in schema.provides


# ---- All domains listed ----


def test_all_domains_listed(full_registry):
    """list_domains() returns all 10 domains."""
    domains = full_registry.list_domains()
    domain_names = {d["domain"] for d in domains}
    assert domain_names == {
        "editing", "audio", "color", "transcription", "render", "project",
        "compositing", "motion_graphics", "scene", "interchange",
    }


# ---- Tool count ----


def test_total_tool_count_with_new_domains(full_registry):
    """Total tool count includes all new tools (12 new)."""
    # Previous: >= 20 tools. New: 4 + 3 + 3 + 2 = 12
    assert full_registry.tool_count >= 32


# ---- All new tools have descriptions ----


def test_all_new_tools_have_descriptions(full_registry):
    """Every new tool has a non-empty description."""
    new_domains = {"compositing", "motion_graphics", "scene", "interchange"}
    for domain in new_domains:
        tools = full_registry.search_tools(domain=domain)
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has empty description"


# ---- All new tools have schemas with params ----


def test_all_new_tools_have_schemas(full_registry):
    """Every new tool returns valid ToolSchema with params."""
    new_domains = {"compositing", "motion_graphics", "scene", "interchange"}
    for domain in new_domains:
        tools = full_registry.search_tools(domain=domain)
        for tool_summary in tools:
            schema = full_registry.get_tool_schema(tool_summary.name)
            assert schema.name == tool_summary.name
            assert schema.domain == tool_summary.domain
            assert schema.description
            assert isinstance(schema.params, list)
            assert len(schema.params) > 0, f"Tool '{schema.name}' has no params"


# ---- Session integration ----


def test_session_includes_new_tools():
    """EditingSession includes all new domain tools."""
    from ave.agent.session import EditingSession

    session = EditingSession()
    d = session.to_dict()
    assert d["tool_count"] >= 32

    # Check new domains are searchable
    assert len(session.search_tools(domain="compositing")) == 4
    assert len(session.search_tools(domain="motion_graphics")) == 3
    assert len(session.search_tools(domain="scene")) == 3
    assert len(session.search_tools(domain="interchange")) == 2
