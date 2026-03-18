"""Tests for namespace support in ToolRegistry."""

from __future__ import annotations

import pytest

from ave.agent.registry import ToolRegistry, RegistryError


class TestNamespacedTools:
    def test_register_tool_with_namespace(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def trim(duration_ns: int) -> dict:
            """Trim a clip."""
            return {"trimmed": duration_ns}

        schema = reg.get_tool_schema("ave:editing.trim")
        assert schema.domain == "editing"

    def test_default_namespace_is_ave(self):
        reg = ToolRegistry()

        @reg.tool(domain="color")
        def grade(intensity: float) -> dict:
            """Color grade."""
            return {"graded": intensity}

        # Should be accessible via full namespaced name
        schema = reg.get_tool_schema("ave:color.grade")
        assert schema.domain == "color"

    def test_backward_compat_short_name(self):
        """Existing code calling tools by short name still works."""
        reg = ToolRegistry()

        @reg.tool(domain="editing")
        def trim(duration_ns: int) -> dict:
            """Trim."""
            return {"trimmed": duration_ns}

        # Short name should resolve when unambiguous
        result = reg.call_tool("trim", {"duration_ns": 100})
        assert result == {"trimmed": 100}

    def test_search_across_namespaces(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def trim(duration_ns: int) -> dict:
            """Trim a clip."""
            return {}

        @reg.tool(domain="editing", namespace="user")
        def smart_trim(duration_ns: int) -> dict:
            """AI-powered trim."""
            return {}

        results = reg.search_tools("trim")
        names = [r.name for r in results]
        assert len(names) >= 2

    def test_call_tool_by_namespaced_name(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="user")
        def my_tool(x: int) -> dict:
            """Custom tool."""
            return {"result": x * 2}

        result = reg.call_tool("user:editing.my_tool", {"x": 5})
        assert result == {"result": 10}

    def test_duplicate_full_name_raises(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def trim(x: int) -> dict:
            """Trim A."""
            return {}

        with pytest.raises(RegistryError, match="already registered"):

            @reg.tool(domain="editing", namespace="ave")
            def trim(x: int) -> dict:  # noqa: F811
                """Trim B."""
                return {}

    def test_same_short_name_different_namespace(self):
        """Different namespaces can have same short name."""
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def process(x: int) -> dict:
            """Ave process."""
            return {"source": "ave"}

        @reg.tool(domain="editing", namespace="user")
        def process(x: int) -> dict:  # noqa: F811
            """User process."""
            return {"source": "user"}

        # Full names should work
        assert reg.call_tool("ave:editing.process", {"x": 1}) == {"source": "ave"}
        assert reg.call_tool("user:editing.process", {"x": 1}) == {"source": "user"}

    def test_ambiguous_short_name_raises(self):
        reg = ToolRegistry()

        @reg.tool(domain="editing", namespace="ave")
        def process(x: int) -> dict:
            """Ave process."""
            return {"source": "ave"}

        @reg.tool(domain="editing", namespace="user")
        def process(x: int) -> dict:  # noqa: F811
            """User process."""
            return {"source": "user"}

        with pytest.raises(KeyError, match="ambiguous"):
            reg.call_tool("process", {"x": 1})

    def test_register_stub(self):
        reg = ToolRegistry()
        reg.register_stub(
            name="segment_video",
            domain="vfx",
            summary="Segment objects in video",
            namespace="user",
            plugin_name="vfx-rotoscope",
        )
        results = reg.search_tools("segment")
        assert len(results) == 1
        assert results[0].description == "Segment objects in video"

    def test_call_stub_raises(self):
        reg = ToolRegistry()
        reg.register_stub(
            name="segment_video",
            domain="vfx",
            summary="Segment objects in video",
        )
        with pytest.raises(RegistryError, match="stub"):
            reg.call_tool("segment_video", {})

    def test_namespace_in_summary(self):
        reg = ToolRegistry()

        @reg.tool(domain="color", namespace="community")
        def film_grain(intensity: float) -> dict:
            """Apply film grain."""
            return {}

        results = reg.search_tools("grain")
        assert results[0].namespace == "community"
