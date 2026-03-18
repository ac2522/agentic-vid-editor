"""Tests for compositing GES operations layer."""

import pytest
from unittest.mock import MagicMock


class TestApplyBlendMode:
    """Test blend mode application via GES."""

    def test_multiply_adds_compositor_effect(self):
        """apply_blend_mode with MULTIPLY should set blend properties on compositor pad."""
        from ave.tools.compositing_ops import apply_blend_mode
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        effect_id = apply_blend_mode(timeline, "clip_0000", BlendMode.MULTIPLY)

        # Should call set_child_property on the clip for blend params
        assert clip.set_child_property.called
        assert isinstance(effect_id, str)

    def test_over_sets_blend_properties(self):
        """apply_blend_mode with OVER should set GL blend function properties."""
        from ave.tools.compositing_ops import apply_blend_mode
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        effect_id = apply_blend_mode(timeline, "clip_0000", BlendMode.OVER)

        assert isinstance(effect_id, str)
        assert clip.set_child_property.called

    def test_overlay_requires_shader(self):
        """apply_blend_mode with OVERLAY should add glshader effect."""
        from ave.tools.compositing_ops import apply_blend_mode
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip
        timeline.add_effect.return_value = "clip_0000_fx_0"

        effect_id = apply_blend_mode(timeline, "clip_0000", BlendMode.OVERLAY)

        # OVERLAY requires shader — should add glshader effect
        timeline.add_effect.assert_called_once_with("clip_0000", "glshader")
        assert effect_id == "clip_0000_fx_0"

    def test_soft_light_requires_shader(self):
        """apply_blend_mode with SOFT_LIGHT should add glshader effect."""
        from ave.tools.compositing_ops import apply_blend_mode
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip
        timeline.add_effect.return_value = "clip_0000_fx_0"

        effect_id = apply_blend_mode(timeline, "clip_0000", BlendMode.SOFT_LIGHT)

        timeline.add_effect.assert_called_once_with("clip_0000", "glshader")
        assert effect_id == "clip_0000_fx_0"

    def test_add_sets_blend_properties(self):
        """apply_blend_mode with ADD should set additive blend properties."""
        from ave.tools.compositing_ops import apply_blend_mode
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        effect_id = apply_blend_mode(timeline, "clip_0000", BlendMode.ADD)

        assert isinstance(effect_id, str)
        assert clip.set_child_property.called


class TestSetClipPosition:
    """Test clip position setting via GES."""

    def test_sets_position_properties(self):
        """Should set xpos and ypos on the clip's compositor pad."""
        from ave.tools.compositing_ops import set_clip_position

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        set_clip_position(timeline, "clip_0000", x=100, y=200)

        timeline.get_clip.assert_called_once_with("clip_0000")
        # Should set xpos and ypos via set_child_property
        calls = clip.set_child_property.call_args_list
        prop_names = [c[0][0] for c in calls]
        assert "xpos" in prop_names
        assert "ypos" in prop_names

    def test_sets_position_and_size(self):
        """Should set xpos, ypos, width, height when all provided."""
        from ave.tools.compositing_ops import set_clip_position

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        set_clip_position(timeline, "clip_0000", x=50, y=75, width=1920, height=1080)

        calls = clip.set_child_property.call_args_list
        prop_names = [c[0][0] for c in calls]
        assert "xpos" in prop_names
        assert "ypos" in prop_names
        assert "width" in prop_names
        assert "height" in prop_names
        # Verify values
        prop_map = {c[0][0]: c[0][1] for c in calls}
        assert prop_map["xpos"] == 50
        assert prop_map["ypos"] == 75
        assert prop_map["width"] == 1920
        assert prop_map["height"] == 1080

    def test_omits_size_when_not_provided(self):
        """Should only set xpos/ypos when width/height are None."""
        from ave.tools.compositing_ops import set_clip_position

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        set_clip_position(timeline, "clip_0000", x=10, y=20)

        calls = clip.set_child_property.call_args_list
        prop_names = [c[0][0] for c in calls]
        assert "xpos" in prop_names
        assert "ypos" in prop_names
        assert "width" not in prop_names
        assert "height" not in prop_names


class TestSetClipAlpha:
    """Test clip alpha setting via GES."""

    def test_sets_alpha_property(self):
        """Should set alpha on the clip's compositor pad."""
        from ave.tools.compositing_ops import set_clip_alpha

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        set_clip_alpha(timeline, "clip_0000", 0.5)

        timeline.get_clip.assert_called_once_with("clip_0000")
        clip.set_child_property.assert_called_once_with("alpha", 0.5)

    def test_alpha_at_boundaries(self):
        """Should accept alpha at 0.0 and 1.0."""
        from ave.tools.compositing_ops import set_clip_alpha

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        set_clip_alpha(timeline, "clip_0000", 0.0)
        clip.set_child_property.assert_called_with("alpha", 0.0)

        set_clip_alpha(timeline, "clip_0000", 1.0)
        clip.set_child_property.assert_called_with("alpha", 1.0)

    def test_invalid_alpha_too_high(self):
        """Should raise CompositingError for alpha > 1.0."""
        from ave.tools.compositing_ops import set_clip_alpha
        from ave.tools.compositing import CompositingError

        timeline = MagicMock()
        with pytest.raises(CompositingError, match="Alpha"):
            set_clip_alpha(timeline, "clip_0000", 1.5)

    def test_invalid_alpha_too_low(self):
        """Should raise CompositingError for alpha < 0.0."""
        from ave.tools.compositing_ops import set_clip_alpha
        from ave.tools.compositing import CompositingError

        timeline = MagicMock()
        with pytest.raises(CompositingError, match="Alpha"):
            set_clip_alpha(timeline, "clip_0000", -0.1)


class TestApplyLayerCompositing:
    """Test layer compositing application via GES."""

    def test_validates_via_compute_layer_params(self):
        """Should raise CompositingError for empty layers list."""
        from ave.tools.compositing_ops import apply_layer_compositing
        from ave.tools.compositing import CompositingError

        timeline = MagicMock()
        with pytest.raises(CompositingError):
            apply_layer_compositing(timeline, [])

    def test_validates_invalid_alpha(self):
        """Should raise CompositingError for invalid alpha."""
        from ave.tools.compositing_ops import apply_layer_compositing
        from ave.tools.compositing import CompositingError, BlendMode

        timeline = MagicMock()
        layers = [
            {
                "clip_id": "clip_0000",
                "layer_index": 0,
                "alpha": 2.0,  # invalid
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
            }
        ]
        with pytest.raises(CompositingError):
            apply_layer_compositing(timeline, layers)

    def test_applies_layers_correctly(self):
        """Should apply layer, alpha, and position for each layer."""
        from ave.tools.compositing_ops import apply_layer_compositing
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip0 = MagicMock()
        clip1 = MagicMock()
        timeline.get_clip.side_effect = lambda cid: {
            "clip_0000": clip0,
            "clip_0001": clip1,
        }[cid]

        layers = [
            {
                "clip_id": "clip_0000",
                "layer_index": 0,
                "alpha": 1.0,
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
            },
            {
                "clip_id": "clip_0001",
                "layer_index": 1,
                "alpha": 0.5,
                "blend_mode": BlendMode.MULTIPLY,
                "position_x": 100,
                "position_y": 200,
            },
        ]

        result = apply_layer_compositing(timeline, layers)

        assert isinstance(result, list)
        assert len(result) == 2
        # Should have set properties on clips
        assert clip0.set_child_property.called
        assert clip1.set_child_property.called

    def test_returns_operation_descriptions(self):
        """Should return list of operation ID strings."""
        from ave.tools.compositing_ops import apply_layer_compositing
        from ave.tools.compositing import BlendMode

        timeline = MagicMock()
        clip = MagicMock()
        timeline.get_clip.return_value = clip

        layers = [
            {
                "clip_id": "clip_0000",
                "layer_index": 0,
                "alpha": 0.8,
                "blend_mode": BlendMode.SOURCE,
                "position_x": 0,
                "position_y": 0,
            },
        ]

        result = apply_layer_compositing(timeline, layers)

        assert len(result) == 1
        assert "clip_0000" in result[0]


class TestPatternConsistency:
    """Test that compositing_ops follows the same patterns as color_ops."""

    def test_module_uses_type_checking_import(self):
        """Timeline should be imported under TYPE_CHECKING only."""
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "src" / "ave" / "tools" / "compositing_ops.py"
        ).read_text()
        tree = ast.parse(source)

        # Check for TYPE_CHECKING import pattern
        has_type_checking = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                    has_type_checking = True
                elif isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                    has_type_checking = True
        assert has_type_checking, "Should use TYPE_CHECKING for Timeline import"

    def test_functions_accept_timeline_as_first_arg(self):
        """All public functions should take timeline as first argument."""
        from ave.tools import compositing_ops
        import inspect

        public_funcs = [
            name
            for name in dir(compositing_ops)
            if not name.startswith("_") and callable(getattr(compositing_ops, name))
        ]

        for name in public_funcs:
            func = getattr(compositing_ops, name)
            if not inspect.isfunction(func):
                continue
            # Only check functions defined in this module
            if func.__module__ != compositing_ops.__name__:
                continue
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            assert params[0] == "timeline", (
                f"{name} should take 'timeline' as first parameter, got '{params[0]}'"
            )
