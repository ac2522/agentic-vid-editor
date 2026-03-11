"""Tests for motion graphics GES operations layer."""

import pytest
from unittest.mock import MagicMock


class TestRgbaToArgbUint32:
    """Test RGBA to ARGB uint32 color conversion."""

    def test_red_half_alpha(self):
        """rgba_to_argb_uint32((255, 0, 0, 128)) returns correct value."""
        from ave.tools.motion_graphics_ops import rgba_to_argb_uint32

        result = rgba_to_argb_uint32((255, 0, 0, 128))
        expected = (128 << 24) | (255 << 16) | (0 << 8) | 0
        assert result == expected

    def test_black_full_alpha(self):
        """rgba_to_argb_uint32((0, 0, 0, 255)) returns 0xFF000000."""
        from ave.tools.motion_graphics_ops import rgba_to_argb_uint32

        assert rgba_to_argb_uint32((0, 0, 0, 255)) == 0xFF000000

    def test_white_full_alpha(self):
        """rgba_to_argb_uint32((255, 255, 255, 255)) returns 0xFFFFFFFF."""
        from ave.tools.motion_graphics_ops import rgba_to_argb_uint32

        assert rgba_to_argb_uint32((255, 255, 255, 255)) == 0xFFFFFFFF

    def test_transparent_black(self):
        """rgba_to_argb_uint32((0, 0, 0, 0)) returns 0."""
        from ave.tools.motion_graphics_ops import rgba_to_argb_uint32

        assert rgba_to_argb_uint32((0, 0, 0, 0)) == 0

    def test_green_channel_only(self):
        """Green channel placed correctly in uint32."""
        from ave.tools.motion_graphics_ops import rgba_to_argb_uint32

        result = rgba_to_argb_uint32((0, 255, 0, 0))
        assert result == 0x0000FF00


class TestPositionMappings:
    """Test halign/valign mappings cover all TextPosition values."""

    def test_halign_map_covers_all_positions(self):
        from ave.tools.motion_graphics import TextPosition
        from ave.tools.motion_graphics_ops import _HALIGN_MAP

        for pos in TextPosition:
            assert pos in _HALIGN_MAP, f"Missing halign mapping for {pos}"

    def test_valign_map_covers_all_positions(self):
        from ave.tools.motion_graphics import TextPosition
        from ave.tools.motion_graphics_ops import _VALIGN_MAP

        for pos in TextPosition:
            assert pos in _VALIGN_MAP, f"Missing valign mapping for {pos}"

    def test_halign_values(self):
        from ave.tools.motion_graphics import TextPosition
        from ave.tools.motion_graphics_ops import _HALIGN_MAP

        assert _HALIGN_MAP[TextPosition.TOP_LEFT] == "left"
        assert _HALIGN_MAP[TextPosition.TOP_CENTER] == "center"
        assert _HALIGN_MAP[TextPosition.TOP_RIGHT] == "right"
        assert _HALIGN_MAP[TextPosition.CENTER] == "center"
        assert _HALIGN_MAP[TextPosition.BOTTOM_LEFT] == "left"
        assert _HALIGN_MAP[TextPosition.BOTTOM_CENTER] == "center"
        assert _HALIGN_MAP[TextPosition.BOTTOM_RIGHT] == "right"

    def test_valign_values(self):
        from ave.tools.motion_graphics import TextPosition
        from ave.tools.motion_graphics_ops import _VALIGN_MAP

        assert _VALIGN_MAP[TextPosition.TOP_LEFT] == "top"
        assert _VALIGN_MAP[TextPosition.TOP_CENTER] == "top"
        assert _VALIGN_MAP[TextPosition.TOP_RIGHT] == "top"
        assert _VALIGN_MAP[TextPosition.CENTER] == "center"
        assert _VALIGN_MAP[TextPosition.BOTTOM_LEFT] == "bottom"
        assert _VALIGN_MAP[TextPosition.BOTTOM_CENTER] == "bottom"
        assert _VALIGN_MAP[TextPosition.BOTTOM_RIGHT] == "bottom"


class TestApplyTextOverlay:
    """Test text overlay application via GES."""

    def _make_params(self, **overrides):
        from ave.tools.motion_graphics import TextOverlayParams, TextPosition

        defaults = dict(
            text="Hello World",
            font_family="Arial",
            font_size=36,
            position=TextPosition.BOTTOM_CENTER,
            color=(255, 255, 255, 255),
            duration_ns=5_000_000_000,
            bg_color=None,
            padding=0,
        )
        defaults.update(overrides)
        return TextOverlayParams(**defaults)

    def test_adds_textoverlay_effect(self):
        """Should add textoverlay effect to the clip."""
        from ave.tools.motion_graphics_ops import apply_text_overlay

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params()

        result = apply_text_overlay(timeline, "clip_0000", params)

        timeline.add_effect.assert_called_once_with("clip_0000", "textoverlay")
        assert result == "clip_0000_fx_0"

    def test_sets_text_property(self):
        """Should set text property on the effect."""
        from ave.tools.motion_graphics_ops import apply_text_overlay

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params(text="My Text")

        apply_text_overlay(timeline, "clip_0000", params)

        prop_calls = timeline.set_effect_property.call_args_list
        text_calls = [c for c in prop_calls if c[0][2] == "text"]
        assert len(text_calls) == 1
        assert text_calls[0][0][3] == "My Text"

    def test_sets_font_desc_pango_format(self):
        """Should set font-desc in Pango format 'FontFamily FontSize'."""
        from ave.tools.motion_graphics_ops import apply_text_overlay

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params(font_family="Helvetica", font_size=48)

        apply_text_overlay(timeline, "clip_0000", params)

        prop_calls = timeline.set_effect_property.call_args_list
        font_calls = [c for c in prop_calls if c[0][2] == "font-desc"]
        assert len(font_calls) == 1
        assert font_calls[0][0][3] == "Helvetica 48"

    def test_sets_alignment_properties(self):
        """Should set halignment and valignment based on position."""
        from ave.tools.motion_graphics import TextPosition
        from ave.tools.motion_graphics_ops import apply_text_overlay

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params(position=TextPosition.TOP_RIGHT)

        apply_text_overlay(timeline, "clip_0000", params)

        prop_calls = timeline.set_effect_property.call_args_list
        halign_calls = [c for c in prop_calls if c[0][2] == "halignment"]
        valign_calls = [c for c in prop_calls if c[0][2] == "valignment"]
        assert len(halign_calls) == 1
        assert halign_calls[0][0][3] == "right"
        assert len(valign_calls) == 1
        assert valign_calls[0][0][3] == "top"

    def test_sets_color_as_argb_uint32(self):
        """Should convert RGBA color to ARGB uint32."""
        from ave.tools.motion_graphics_ops import apply_text_overlay, rgba_to_argb_uint32

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params(color=(255, 0, 0, 128))

        apply_text_overlay(timeline, "clip_0000", params)

        prop_calls = timeline.set_effect_property.call_args_list
        color_calls = [c for c in prop_calls if c[0][2] == "color"]
        assert len(color_calls) == 1
        assert color_calls[0][0][3] == rgba_to_argb_uint32((255, 0, 0, 128))

    def test_sets_shaded_background_when_bg_color(self):
        """Should set shaded-background when bg_color is provided."""
        from ave.tools.motion_graphics_ops import apply_text_overlay

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params(bg_color=(0, 0, 0, 128))

        apply_text_overlay(timeline, "clip_0000", params)

        prop_calls = timeline.set_effect_property.call_args_list
        bg_calls = [c for c in prop_calls if c[0][2] == "shaded-background"]
        assert len(bg_calls) == 1
        assert bg_calls[0][0][3] is True

    def test_no_shaded_background_when_no_bg_color(self):
        """Should not set shaded-background when bg_color is None."""
        from ave.tools.motion_graphics_ops import apply_text_overlay

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        params = self._make_params(bg_color=None)

        apply_text_overlay(timeline, "clip_0000", params)

        prop_calls = timeline.set_effect_property.call_args_list
        bg_calls = [c for c in prop_calls if c[0][2] == "shaded-background"]
        assert len(bg_calls) == 0


class TestApplyLowerThird:
    """Test lower third application via GES."""

    def _make_lower_third_params(self):
        from ave.tools.motion_graphics import compute_lower_third

        return compute_lower_third(
            name="John Doe",
            title="Senior Editor",
            frame_width=1920,
            frame_height=1080,
            duration_ns=5_000_000_000,
        )

    def test_creates_two_effects(self):
        """Should apply two textoverlay effects (name + title)."""
        from ave.tools.motion_graphics_ops import apply_lower_third

        timeline = MagicMock()
        timeline.add_effect.side_effect = ["clip_0000_fx_0", "clip_0000_fx_1"]
        params = self._make_lower_third_params()

        result = apply_lower_third(timeline, "clip_0000", params)

        assert len(result) == 2
        assert result == ["clip_0000_fx_0", "clip_0000_fx_1"]
        assert timeline.add_effect.call_count == 2
        timeline.add_effect.assert_any_call("clip_0000", "textoverlay")

    def test_returns_list_of_effect_ids(self):
        """Should return a list of effect IDs."""
        from ave.tools.motion_graphics_ops import apply_lower_third

        timeline = MagicMock()
        timeline.add_effect.side_effect = ["clip_0000_fx_0", "clip_0000_fx_1"]
        params = self._make_lower_third_params()

        result = apply_lower_third(timeline, "clip_0000", params)

        assert isinstance(result, list)
        assert all(isinstance(eid, str) for eid in result)


class TestApplyTitleCard:
    """Test title card creation via GES."""

    def _make_title_params(self):
        from ave.tools.motion_graphics import compute_title_card

        return compute_title_card(
            text="My Movie",
            frame_width=1920,
            frame_height=1080,
            duration_ns=3_000_000_000,
        )

    def _make_mock_ges(self):
        """Create mock GES module with TitleClip."""
        mock_title_clip = MagicMock()
        mock_title_clip.set_child_property.return_value = True

        mock_ges = MagicMock()
        mock_ges.TitleClip.new.return_value = mock_title_clip

        mock_gst = MagicMock()

        return mock_ges, mock_gst, mock_title_clip

    def test_creates_title_clip(self):
        """Should create a title clip on the timeline and return clip_id."""
        mock_ges, mock_gst, mock_title_clip = self._make_mock_ges()

        timeline = MagicMock()
        timeline.register_clip.return_value = "clip_0001"
        mock_layer = MagicMock()
        mock_layer.add_clip.return_value = True
        timeline._timeline.get_layers.return_value = [mock_layer]

        params = self._make_title_params()

        from ave.tools.motion_graphics_ops import _apply_title_card_impl

        result = _apply_title_card_impl(
            timeline,
            start_ns=0,
            duration_ns=3_000_000_000,
            params=params,
            title_clip=mock_title_clip,
        )

        assert result == "clip_0001"
        timeline.register_clip.assert_called_once_with(mock_title_clip)

    def test_sets_title_text_on_clip(self):
        """Should set text and font on the title clip via child property."""
        mock_ges, mock_gst, mock_title_clip = self._make_mock_ges()

        timeline = MagicMock()
        timeline.register_clip.return_value = "clip_0001"
        mock_layer = MagicMock()
        mock_layer.add_clip.return_value = True
        timeline._timeline.get_layers.return_value = [mock_layer]

        params = self._make_title_params()

        from ave.tools.motion_graphics_ops import _apply_title_card_impl

        _apply_title_card_impl(
            timeline,
            start_ns=1_000_000_000,
            duration_ns=3_000_000_000,
            params=params,
            title_clip=mock_title_clip,
        )

        # Check that set_child_property was called with text
        child_prop_calls = mock_title_clip.set_child_property.call_args_list
        prop_names = [c[0][0] for c in child_prop_calls]
        assert "text" in prop_names
        assert "font-desc" in prop_names

    def test_sets_start_and_duration(self):
        """Should set start and duration on the title clip."""
        mock_ges, mock_gst, mock_title_clip = self._make_mock_ges()

        timeline = MagicMock()
        timeline.register_clip.return_value = "clip_0001"
        mock_layer = MagicMock()
        mock_layer.add_clip.return_value = True
        timeline._timeline.get_layers.return_value = [mock_layer]

        params = self._make_title_params()

        from ave.tools.motion_graphics_ops import _apply_title_card_impl

        _apply_title_card_impl(
            timeline,
            start_ns=2_000_000_000,
            duration_ns=3_000_000_000,
            params=params,
            title_clip=mock_title_clip,
        )

        mock_title_clip.set_start.assert_called_once_with(2_000_000_000)
        mock_title_clip.set_duration.assert_called_once_with(3_000_000_000)

    def test_adds_clip_to_layer(self):
        """Should add the title clip to a timeline layer."""
        mock_ges, mock_gst, mock_title_clip = self._make_mock_ges()

        timeline = MagicMock()
        timeline.register_clip.return_value = "clip_0001"
        mock_layer = MagicMock()
        mock_layer.add_clip.return_value = True
        timeline._timeline.get_layers.return_value = [mock_layer]

        params = self._make_title_params()

        from ave.tools.motion_graphics_ops import _apply_title_card_impl

        _apply_title_card_impl(
            timeline,
            start_ns=0,
            duration_ns=3_000_000_000,
            params=params,
            title_clip=mock_title_clip,
        )

        mock_layer.add_clip.assert_called_once_with(mock_title_clip)

    def test_raises_on_failed_layer_add(self):
        """Should raise MotionGraphicsOpsError if layer.add_clip fails."""
        from ave.tools.motion_graphics_ops import (
            MotionGraphicsOpsError,
            _apply_title_card_impl,
        )

        mock_title_clip = MagicMock()
        mock_title_clip.set_child_property.return_value = True

        timeline = MagicMock()
        mock_layer = MagicMock()
        mock_layer.add_clip.return_value = False
        timeline._timeline.get_layers.return_value = [mock_layer]

        params = self._make_title_params()

        with pytest.raises(MotionGraphicsOpsError, match="Failed to add"):
            _apply_title_card_impl(
                timeline,
                start_ns=0,
                duration_ns=3_000_000_000,
                params=params,
                title_clip=mock_title_clip,
            )


class TestMotionGraphicsOpsError:
    """Test that the exception class exists."""

    def test_exception_class_exists(self):
        from ave.tools.motion_graphics_ops import MotionGraphicsOpsError

        assert issubclass(MotionGraphicsOpsError, Exception)

    def test_exception_is_raisable(self):
        from ave.tools.motion_graphics_ops import MotionGraphicsOpsError

        with pytest.raises(MotionGraphicsOpsError):
            raise MotionGraphicsOpsError("test error")
