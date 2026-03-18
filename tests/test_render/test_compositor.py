"""Tests for compositor strategy selection."""

import pytest

from ave.render.compositor import (
    COMPOSITOR_ELEMENTS,
    CompositorSelection,
    CompositorStrategy,
)


class TestCompositorSelection:
    def test_fields(self):
        sel = CompositorSelection(strategy="cpu", element_name="compositor", reason="default")
        assert sel.strategy == "cpu"
        assert sel.element_name == "compositor"
        assert sel.reason == "default"

    def test_frozen(self):
        sel = CompositorSelection(strategy="cpu", element_name="compositor", reason="x")
        with pytest.raises(AttributeError):
            sel.strategy = "gl"  # type: ignore[misc]


class TestCompositorStrategy:
    def test_detect_available_returns_list(self):
        result = CompositorStrategy.detect_available()
        assert isinstance(result, list)
        assert "cpu" in result  # cpu is always available as fallback

    def test_select_auto_all_available_picks_skia(self):
        sel = CompositorStrategy.select(preference="auto", available=["skia", "cpu", "gl"])
        assert sel.strategy == "skia"
        assert sel.element_name == COMPOSITOR_ELEMENTS["skia"]

    def test_select_auto_only_cpu(self):
        sel = CompositorStrategy.select(preference="auto", available=["cpu"])
        assert sel.strategy == "cpu"
        assert sel.element_name == COMPOSITOR_ELEMENTS["cpu"]

    def test_select_explicit_preference(self):
        sel = CompositorStrategy.select(preference="gl", available=["skia", "cpu", "gl"])
        assert sel.strategy == "gl"
        assert sel.element_name == COMPOSITOR_ELEMENTS["gl"]

    def test_select_unavailable_preference_falls_back(self):
        sel = CompositorStrategy.select(preference="skia", available=["cpu", "gl"])
        # Should fall back to best available (cpu preferred over gl)
        assert sel.strategy == "cpu"

    def test_get_element_name(self):
        assert CompositorStrategy.get_element_name("skia") == "skiacompositor"
        assert CompositorStrategy.get_element_name("cpu") == "compositor"
        assert CompositorStrategy.get_element_name("gl") == "glvideomixer"

    def test_get_element_name_unknown_raises(self):
        with pytest.raises(ValueError):
            CompositorStrategy.get_element_name("vulkan")
