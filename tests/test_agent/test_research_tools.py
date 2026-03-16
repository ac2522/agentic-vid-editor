"""Tests for research tool registration and researcher role."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry
from ave.agent.tools.research import register_research_tools
from ave.agent.roles import RESEARCHER_ROLE, VFX_ARTIST_ROLE, ALL_ROLES


class TestResearchTools:
    def test_register_research_tools(self):
        reg = ToolRegistry()
        register_research_tools(reg)
        results = reg.search_tools("web search")
        assert any("web_search" in r.name for r in results)

    def test_research_technique_registered(self):
        reg = ToolRegistry()
        register_research_tools(reg)
        results = reg.search_tools("research technique")
        assert any("research_technique" in r.name for r in results)


class TestNewRoles:
    def test_researcher_role_exists(self):
        assert RESEARCHER_ROLE.name == "Researcher"
        assert "research" in RESEARCHER_ROLE.domains

    def test_vfx_artist_role_exists(self):
        assert VFX_ARTIST_ROLE.name == "VFX Artist"
        assert "vfx" in VFX_ARTIST_ROLE.domains

    def test_all_roles_includes_new_roles(self):
        role_names = {r.name for r in ALL_ROLES}
        assert "Researcher" in role_names
        assert "VFX Artist" in role_names
        assert len(ALL_ROLES) == 6
