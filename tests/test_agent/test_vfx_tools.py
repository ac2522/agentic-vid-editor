"""Tests for VFX tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry
from ave.agent.tools.vfx import register_vfx_tools


class TestVfxTools:
    def test_register_vfx_tools(self):
        reg = ToolRegistry()
        register_vfx_tools(reg)
        results = reg.search_tools("segment")
        assert any("segment_video" in r.name for r in results)

    def test_four_vfx_tools_registered(self):
        reg = ToolRegistry()
        register_vfx_tools(reg)
        vfx_tools = reg.search_tools(domain="vfx")
        names = {r.name for r in vfx_tools}
        assert len(names) == 4

    def test_only_apply_mask_modifies_timeline(self):
        reg = ToolRegistry()
        register_vfx_tools(reg)
        assert reg.tool_modifies_timeline("apply_mask")
        assert not reg.tool_modifies_timeline("segment_video")
        assert not reg.tool_modifies_timeline("evaluate_mask")
        assert not reg.tool_modifies_timeline("refine_mask")
