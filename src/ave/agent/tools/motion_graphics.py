"""Motion graphics domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_motion_graphics_tools(registry: ToolRegistry) -> None:
    """Register motion graphics domain tools."""

    @registry.tool(
        domain="motion_graphics",
        requires=["timeline_loaded"],
        provides=["text_overlay_added"],
    )
    def add_text_overlay(
        text: str,
        font_family: str,
        font_size: int,
        position: str,
        color_r: int,
        color_g: int,
        color_b: int,
        color_a: int,
        duration_ns: int,
    ):
        """Add a text overlay to the timeline at a specified position."""
        from ave.tools.motion_graphics import TextPosition, compute_text_overlay

        pos = TextPosition(position)
        return compute_text_overlay(
            text=text,
            font_family=font_family,
            font_size=font_size,
            position=pos,
            color=(color_r, color_g, color_b, color_a),
            duration_ns=duration_ns,
        )

    @registry.tool(
        domain="motion_graphics",
        requires=["timeline_loaded"],
        provides=["lower_third_added"],
    )
    def add_lower_third(
        name: str,
        title: str,
        frame_width: int,
        frame_height: int,
        duration_ns: int,
        font_family: str = "Arial",
    ):
        """Add a lower third graphic with name and title text."""
        from ave.tools.motion_graphics import compute_lower_third

        return compute_lower_third(
            name=name,
            title=title,
            frame_width=frame_width,
            frame_height=frame_height,
            duration_ns=duration_ns,
            font_family=font_family,
        )

    @registry.tool(
        domain="motion_graphics",
        requires=["timeline_loaded"],
        provides=["title_card_added"],
    )
    def add_title_card(
        text: str,
        frame_width: int,
        frame_height: int,
        duration_ns: int,
        font_family: str = "Arial",
        font_size: int = 72,
    ):
        """Add a centered title card to the timeline."""
        from ave.tools.motion_graphics import compute_title_card

        return compute_title_card(
            text=text,
            frame_width=frame_width,
            frame_height=frame_height,
            duration_ns=duration_ns,
            font_family=font_family,
            font_size=font_size,
        )
