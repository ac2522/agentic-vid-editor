"""Compositing GES operations — bridges pure logic layer to GES effects.

Applies computed compositing parameters from ave.tools.compositing to GES
Timeline objects. All GES access goes through Timeline's public API.

Architecture note: Uses ``compositor`` element (CPU), NOT ``glvideomixer``
(crash bugs #728, #786).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ave.tools.compositing import (
    BlendMode,
    CompositingError,
    compute_blend_info,
    compute_layer_params,
)

if TYPE_CHECKING:
    from ave.project.timeline import Timeline


def apply_layer_compositing(
    timeline: Timeline,
    layers: list[dict],
) -> list[str]:
    """Apply compositing parameters to multiple layers.

    Validates via ``compute_layer_params()`` from the pure logic layer, then
    for each layer adjusts the clip's GES layer (track) and alpha.

    Args:
        timeline: Target timeline.
        layers: List of dicts with keys: clip_id, layer_index, alpha,
                blend_mode, position_x, position_y, and optionally
                width, height.

    Returns:
        List of operation ID/description strings.

    Raises:
        CompositingError: If parameter validation fails.
    """
    # Validate through the pure logic layer
    validated = compute_layer_params(layers)

    # Build clip_id lookup from input layers (compute_layer_params doesn't
    # carry clip_id, so we map by layer_index).
    index_to_clip: dict[int, str] = {
        layer["layer_index"]: layer["clip_id"] for layer in layers
    }

    ops: list[str] = []
    for params in validated:
        clip_id = index_to_clip[params.layer_index]
        clip = timeline.get_clip(clip_id)

        # Set compositor pad properties
        clip.set_child_property("alpha", params.alpha)
        clip.set_child_property("xpos", params.position_x)
        clip.set_child_property("ypos", params.position_y)
        if params.width is not None:
            clip.set_child_property("width", params.width)
        if params.height is not None:
            clip.set_child_property("height", params.height)

        ops.append(
            f"{clip_id}:layer={params.layer_index},"
            f"alpha={params.alpha},"
            f"pos=({params.position_x},{params.position_y})"
        )

    return ops


def apply_blend_mode(
    timeline: Timeline,
    clip_id: str,
    blend_mode: BlendMode,
) -> str:
    """Apply a blend mode to a clip via GES.

    Gets blend params via ``compute_blend_params()``. If the mode requires
    a shader (OVERLAY, SOFT_LIGHT), adds a ``glshader`` effect with a GLSL
    blend fragment. Otherwise sets blend properties on the compositor pad.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to blend.
        blend_mode: The blend mode to apply.

    Returns:
        Effect ID (for shader modes) or description string (for GL blend
        function modes).
    """
    blend_info = compute_blend_info(blend_mode)
    clip = timeline.get_clip(clip_id)

    if blend_info.requires_shader:
        # Shader-based blend: add glshader effect
        effect_id = timeline.add_effect(clip_id, "glshader")
        timeline.set_effect_property(clip_id, effect_id, "fragment", blend_info.glsl_source)
        return effect_id

    # GL blend function mode: set properties on compositor pad
    clip.set_child_property("operator", blend_mode.value)

    return f"{clip_id}:blend={blend_mode.value}"


def set_clip_position(
    timeline: Timeline,
    clip_id: str,
    x: int,
    y: int,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Set position and optional scale on a clip's compositor pad.

    Uses GES pad properties: xpos, ypos, width, height.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to position.
        x: Horizontal position in pixels.
        y: Vertical position in pixels.
        width: Optional width override in pixels.
        height: Optional height override in pixels.
    """
    clip = timeline.get_clip(clip_id)
    clip.set_child_property("xpos", x)
    clip.set_child_property("ypos", y)
    if width is not None:
        clip.set_child_property("width", width)
    if height is not None:
        clip.set_child_property("height", height)


def set_clip_alpha(
    timeline: Timeline,
    clip_id: str,
    alpha: float,
) -> None:
    """Set alpha on a clip's compositor pad.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip.
        alpha: Opacity value, 0.0 (transparent) to 1.0 (opaque).

    Raises:
        CompositingError: If alpha is outside [0.0, 1.0].
    """
    if alpha < 0.0 or alpha > 1.0:
        raise CompositingError(f"Alpha must be 0.0-1.0, got {alpha}")

    clip = timeline.get_clip(clip_id)
    clip.set_child_property("alpha", alpha)
