"""Motion graphics GES operations — bridges pure logic layer to GES effects.

Applies computed motion graphics parameters from ave.tools.motion_graphics
to GES Timeline objects. All GES access goes through Timeline's public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ave.tools.motion_graphics import TextPosition, TextOverlayParams, LowerThirdParams

if TYPE_CHECKING:
    from ave.project.timeline import Timeline


class MotionGraphicsOpsError(Exception):
    """Raised when motion graphics GES operations fail."""


def rgba_to_argb_uint32(rgba: tuple[int, int, int, int]) -> int:
    """Convert (R, G, B, A) to GStreamer ARGB uint32.

    Args:
        rgba: Tuple of (red, green, blue, alpha) each 0-255.

    Returns:
        ARGB packed as a 32-bit unsigned integer.
    """
    r, g, b, a = rgba
    return (a << 24) | (r << 16) | (g << 8) | b


# Position mapping: TextPosition -> GStreamer textoverlay halignment string
_HALIGN_MAP: dict[TextPosition, str] = {
    TextPosition.TOP_LEFT: "left",
    TextPosition.TOP_CENTER: "center",
    TextPosition.TOP_RIGHT: "right",
    TextPosition.CENTER: "center",
    TextPosition.BOTTOM_LEFT: "left",
    TextPosition.BOTTOM_CENTER: "center",
    TextPosition.BOTTOM_RIGHT: "right",
}

# Position mapping: TextPosition -> GStreamer textoverlay valignment string
_VALIGN_MAP: dict[TextPosition, str] = {
    TextPosition.TOP_LEFT: "top",
    TextPosition.TOP_CENTER: "top",
    TextPosition.TOP_RIGHT: "top",
    TextPosition.CENTER: "center",
    TextPosition.BOTTOM_LEFT: "bottom",
    TextPosition.BOTTOM_CENTER: "bottom",
    TextPosition.BOTTOM_RIGHT: "bottom",
}


def apply_text_overlay(
    timeline: Timeline,
    clip_id: str,
    params: TextOverlayParams,
) -> str:
    """Apply a text overlay effect to a clip via GES.

    Adds a GES textoverlay effect to the specified clip and configures
    its properties based on the computed parameters.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to apply the text overlay to.
        params: Computed text overlay parameters.

    Returns:
        Effect ID for the added textoverlay effect.

    Raises:
        MotionGraphicsOpsError: If the effect cannot be applied.
    """
    effect_id = timeline.add_effect(clip_id, "textoverlay")

    timeline.set_effect_property(clip_id, effect_id, "text", params.text)
    timeline.set_effect_property(
        clip_id, effect_id, "font-desc", f"{params.font_family} {params.font_size}"
    )
    timeline.set_effect_property(clip_id, effect_id, "halignment", _HALIGN_MAP[params.position])
    timeline.set_effect_property(clip_id, effect_id, "valignment", _VALIGN_MAP[params.position])
    timeline.set_effect_property(clip_id, effect_id, "color", rgba_to_argb_uint32(params.color))

    if params.bg_color is not None:
        timeline.set_effect_property(clip_id, effect_id, "shaded-background", True)

    return effect_id


def apply_lower_third(
    timeline: Timeline,
    clip_id: str,
    params: LowerThirdParams,
) -> list[str]:
    """Apply a lower third template to a clip via GES.

    Applies two textoverlay effects: one for the name (larger font) and
    one for the title (smaller font).

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to apply the lower third to.
        params: Computed lower third parameters.

    Returns:
        List of effect IDs for the two textoverlay effects [name_id, title_id].

    Raises:
        MotionGraphicsOpsError: If the effects cannot be applied.
    """
    name_effect_id = apply_text_overlay(timeline, clip_id, params.name_params)
    title_effect_id = apply_text_overlay(timeline, clip_id, params.title_params)

    return [name_effect_id, title_effect_id]


def _apply_title_card_impl(
    timeline: Timeline,
    start_ns: int,
    duration_ns: int,
    params: TextOverlayParams,
    title_clip: object,
) -> str:
    """Internal implementation for title card creation.

    Separated from apply_title_card to allow unit testing without GES.

    Args:
        timeline: Target timeline.
        start_ns: Start position in nanoseconds.
        duration_ns: Duration in nanoseconds.
        params: Computed text overlay parameters for title styling.
        title_clip: A GES.TitleClip instance (or mock for testing).

    Returns:
        Clip ID for the created title clip.

    Raises:
        MotionGraphicsOpsError: If the title clip cannot be added to the layer.
    """
    title_clip.set_start(start_ns)
    title_clip.set_duration(duration_ns)

    # Set title text and font via child properties
    title_clip.set_child_property("text", params.text)
    title_clip.set_child_property("font-desc", f"{params.font_family} {params.font_size}")
    title_clip.set_child_property("halignment", _HALIGN_MAP[params.position])
    title_clip.set_child_property("valignment", _VALIGN_MAP[params.position])
    title_clip.set_child_property("color", rgba_to_argb_uint32(params.color))

    # Add to timeline's first layer and register
    ges_timeline = timeline._timeline
    layers = ges_timeline.get_layers()
    if not layers:
        ges_timeline.append_layer()
        layers = ges_timeline.get_layers()

    layer = layers[0]
    if not layer.add_clip(title_clip):
        raise MotionGraphicsOpsError("Failed to add TitleClip to timeline layer")

    clip_id = timeline.register_clip(title_clip)
    return clip_id


def apply_title_card(
    timeline: Timeline,
    start_ns: int,
    duration_ns: int,
    params: TextOverlayParams,
) -> str:
    """Create a GES TitleClip on the timeline.

    Creates a standalone title clip (not an overlay on an existing clip)
    with the specified text and styling.

    Args:
        timeline: Target timeline.
        start_ns: Start position in nanoseconds.
        duration_ns: Duration in nanoseconds.
        params: Computed text overlay parameters for title styling.

    Returns:
        Clip ID for the created title clip.

    Raises:
        MotionGraphicsOpsError: If the title clip cannot be created.
    """
    # Lazy import GES to avoid import errors when GES is not available
    import gi

    gi.require_version("GES", "1.0")
    gi.require_version("Gst", "1.0")
    from gi.repository import GES  # noqa: F811

    title_clip = GES.TitleClip.new()
    if title_clip is None:
        raise MotionGraphicsOpsError("Failed to create GES TitleClip")

    return _apply_title_card_impl(timeline, start_ns, duration_ns, params, title_clip)
