"""Unit tests for motion graphics tools — pure logic, no GES required."""

import pytest


class TestTextPosition:
    """Test TextPosition enum members."""

    def test_all_positions_exist(self):
        from ave.tools.motion_graphics import TextPosition

        expected = {
            "TOP_LEFT",
            "TOP_CENTER",
            "TOP_RIGHT",
            "CENTER",
            "BOTTOM_LEFT",
            "BOTTOM_CENTER",
            "BOTTOM_RIGHT",
        }
        actual = {p.name for p in TextPosition}
        assert actual == expected


class TestComputeTextOverlay:
    """Test text overlay parameter validation and computation."""

    def test_valid_text_overlay(self):
        from ave.tools.motion_graphics import TextPosition, compute_text_overlay

        result = compute_text_overlay(
            text="Hello World",
            font_family="Arial",
            font_size=36,
            position=TextPosition.BOTTOM_CENTER,
            color=(255, 255, 255, 255),
            duration_ns=5_000_000_000,
        )
        assert result.text == "Hello World"
        assert result.font_family == "Arial"
        assert result.font_size == 36
        assert result.position == TextPosition.BOTTOM_CENTER
        assert result.color == (255, 255, 255, 255)
        assert result.duration_ns == 5_000_000_000

    def test_empty_text_raises(self):
        from ave.tools.motion_graphics import (
            MotionGraphicsError,
            TextPosition,
            compute_text_overlay,
        )

        with pytest.raises(MotionGraphicsError, match="[Tt]ext"):
            compute_text_overlay(
                text="",
                font_family="Arial",
                font_size=36,
                position=TextPosition.CENTER,
                color=(255, 255, 255, 255),
                duration_ns=5_000_000_000,
            )

    def test_font_size_too_small_raises(self):
        from ave.tools.motion_graphics import (
            MotionGraphicsError,
            TextPosition,
            compute_text_overlay,
        )

        with pytest.raises(MotionGraphicsError, match="[Ff]ont.*size"):
            compute_text_overlay(
                text="Hello",
                font_family="Arial",
                font_size=4,
                position=TextPosition.CENTER,
                color=(255, 255, 255, 255),
                duration_ns=5_000_000_000,
            )

    def test_font_size_too_large_raises(self):
        from ave.tools.motion_graphics import (
            MotionGraphicsError,
            TextPosition,
            compute_text_overlay,
        )

        with pytest.raises(MotionGraphicsError, match="[Ff]ont.*size"):
            compute_text_overlay(
                text="Hello",
                font_family="Arial",
                font_size=501,
                position=TextPosition.CENTER,
                color=(255, 255, 255, 255),
                duration_ns=5_000_000_000,
            )

    def test_negative_duration_raises(self):
        from ave.tools.motion_graphics import (
            MotionGraphicsError,
            TextPosition,
            compute_text_overlay,
        )

        with pytest.raises(MotionGraphicsError, match="[Dd]uration"):
            compute_text_overlay(
                text="Hello",
                font_family="Arial",
                font_size=36,
                position=TextPosition.CENTER,
                color=(255, 255, 255, 255),
                duration_ns=-1,
            )

    def test_invalid_color_component_raises(self):
        from ave.tools.motion_graphics import (
            MotionGraphicsError,
            TextPosition,
            compute_text_overlay,
        )

        with pytest.raises(MotionGraphicsError, match="[Cc]olor"):
            compute_text_overlay(
                text="Hello",
                font_family="Arial",
                font_size=36,
                position=TextPosition.CENTER,
                color=(255, 255, 256, 255),
                duration_ns=5_000_000_000,
            )

    def test_default_bg_color_and_padding(self):
        from ave.tools.motion_graphics import TextPosition, compute_text_overlay

        result = compute_text_overlay(
            text="Hello",
            font_family="Arial",
            font_size=36,
            position=TextPosition.CENTER,
            color=(255, 255, 255, 255),
            duration_ns=5_000_000_000,
        )
        assert result.bg_color is None
        assert result.padding == 0

    def test_with_background_color(self):
        from ave.tools.motion_graphics import TextPosition, compute_text_overlay

        result = compute_text_overlay(
            text="Hello",
            font_family="Arial",
            font_size=36,
            position=TextPosition.CENTER,
            color=(255, 255, 255, 255),
            duration_ns=5_000_000_000,
            bg_color=(0, 0, 0, 128),
            padding=10,
        )
        assert result.bg_color == (0, 0, 0, 128)
        assert result.padding == 10


class TestComputePositionCoords:
    """Test position coordinate computation."""

    def test_bottom_center_1920x1080(self):
        from ave.tools.motion_graphics import TextPosition, compute_position_coords

        x, y = compute_position_coords(
            position=TextPosition.BOTTOM_CENTER,
            frame_width=1920,
            frame_height=1080,
            text_width=200,
            text_height=40,
            padding=20,
        )
        # Centered horizontally
        assert x == (1920 - 200) // 2
        # Near bottom with padding
        assert y == 1080 - 40 - 20

    def test_top_left_with_padding(self):
        from ave.tools.motion_graphics import TextPosition, compute_position_coords

        x, y = compute_position_coords(
            position=TextPosition.TOP_LEFT,
            frame_width=1920,
            frame_height=1080,
            text_width=200,
            text_height=40,
            padding=30,
        )
        assert x == 30
        assert y == 30

    def test_center_1920x1080(self):
        from ave.tools.motion_graphics import TextPosition, compute_position_coords

        x, y = compute_position_coords(
            position=TextPosition.CENTER,
            frame_width=1920,
            frame_height=1080,
            text_width=200,
            text_height=40,
            padding=0,
        )
        assert x == (1920 - 200) // 2
        assert y == (1080 - 40) // 2

    def test_all_positions_return_valid_coords(self):
        from ave.tools.motion_graphics import TextPosition, compute_position_coords

        for pos in TextPosition:
            x, y = compute_position_coords(
                position=pos,
                frame_width=1920,
                frame_height=1080,
                text_width=200,
                text_height=40,
                padding=10,
            )
            assert 0 <= x <= 1920 - 200
            assert 0 <= y <= 1080 - 40


class TestComputeLowerThird:
    """Test lower third template computation."""

    def test_valid_lower_third(self):
        from ave.tools.motion_graphics import compute_lower_third

        result = compute_lower_third(
            name="John Doe",
            title="Senior Editor",
            frame_width=1920,
            frame_height=1080,
            duration_ns=5_000_000_000,
        )
        assert result.name_params.text == "John Doe"
        assert result.title_params.text == "Senior Editor"
        assert result.duration_ns == 5_000_000_000

    def test_empty_name_raises(self):
        from ave.tools.motion_graphics import MotionGraphicsError, compute_lower_third

        with pytest.raises(MotionGraphicsError, match="[Nn]ame"):
            compute_lower_third(
                name="",
                title="Senior Editor",
                frame_width=1920,
                frame_height=1080,
                duration_ns=5_000_000_000,
            )

    def test_empty_title_raises(self):
        from ave.tools.motion_graphics import MotionGraphicsError, compute_lower_third

        with pytest.raises(MotionGraphicsError, match="[Tt]itle"):
            compute_lower_third(
                name="John Doe",
                title="",
                frame_width=1920,
                frame_height=1080,
                duration_ns=5_000_000_000,
            )

    def test_bg_rect_spans_bottom(self):
        from ave.tools.motion_graphics import compute_lower_third

        result = compute_lower_third(
            name="John Doe",
            title="Senior Editor",
            frame_width=1920,
            frame_height=1080,
            duration_ns=5_000_000_000,
        )
        bg_x, bg_y, bg_w, bg_h = result.bg_rect
        # Background should be at bottom portion of frame
        assert bg_y + bg_h == 1080
        # Should span the full width
        assert bg_w == 1920
        assert bg_x == 0

    def test_name_font_larger_than_title_font(self):
        from ave.tools.motion_graphics import compute_lower_third

        result = compute_lower_third(
            name="John Doe",
            title="Senior Editor",
            frame_width=1920,
            frame_height=1080,
            duration_ns=5_000_000_000,
        )
        assert result.name_params.font_size > result.title_params.font_size


class TestComputeTitleCard:
    """Test title card template computation."""

    def test_valid_title_card(self):
        from ave.tools.motion_graphics import compute_title_card

        result = compute_title_card(
            text="My Movie",
            frame_width=1920,
            frame_height=1080,
            duration_ns=3_000_000_000,
        )
        assert result.text == "My Movie"
        assert result.duration_ns == 3_000_000_000

    def test_centered_position(self):
        from ave.tools.motion_graphics import TextPosition, compute_title_card

        result = compute_title_card(
            text="My Movie",
            frame_width=1920,
            frame_height=1080,
            duration_ns=3_000_000_000,
        )
        assert result.position == TextPosition.CENTER
