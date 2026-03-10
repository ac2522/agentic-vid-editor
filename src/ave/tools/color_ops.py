"""Color GES operations — bridges pure logic layer to GES effects.

Applies computed color parameters from ave.tools.color to GES Timeline objects.
All GES access goes through Timeline's public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ave.tools.color import (
    ColorError,
    compute_lut_application,
    compute_color_grade,
    compute_cdl,
    compute_color_transform,
    generate_grade_glsl,
    generate_cdl_glsl,
)

if TYPE_CHECKING:
    from ave.project.timeline import Timeline


def apply_lut(
    timeline: Timeline,
    clip_id: str,
    lut_path: str,
    intensity: float = 1.0,
) -> str:
    """Apply a .cube LUT to a clip via GES.

    Validates parameters through the pure logic layer, then adds a
    placebofilter effect to the timeline clip.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to apply the LUT to.
        lut_path: Path to a .cube LUT file.
        intensity: Blend factor 0.0 (no effect) to 1.0 (full effect).

    Returns:
        Effect ID for the added LUT effect.

    Raises:
        ColorError: If validation fails.
    """
    params = compute_lut_application(lut_path, intensity)

    effect_id = timeline.add_effect(clip_id, "placebofilter")
    timeline.set_effect_property(clip_id, effect_id, "lut-path", params.path)
    timeline.set_effect_property(clip_id, effect_id, "intensity", params.intensity)

    return effect_id


def apply_color_grade(
    timeline: Timeline,
    clip_id: str,
    lift: tuple[float, float, float],
    gamma: tuple[float, float, float],
    gain: tuple[float, float, float],
    saturation: float = 1.0,
    offset: tuple[float, float, float] = (0, 0, 0),
) -> str:
    """Apply lift/gamma/gain colour grade to a clip via GES.

    Validates parameters and generates a GLSL fragment shader, then adds
    a glshader effect to the timeline clip.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to grade.
        lift: Shadow adjustment per channel, each -1.0 to 1.0.
        gamma: Midtone adjustment per channel, each 0.01 to 4.0.
        gain: Highlight adjustment per channel, each 0.0 to 4.0.
        saturation: Global saturation, 0.0 to 4.0.
        offset: Offset per channel, each -1.0 to 1.0.

    Returns:
        Effect ID for the added grade effect.

    Raises:
        ColorError: If validation fails.
    """
    params = compute_color_grade(lift, gamma, gain, saturation, offset)
    glsl_source = generate_grade_glsl(params)

    effect_id = timeline.add_effect(clip_id, "glshader")
    timeline.set_effect_property(clip_id, effect_id, "fragment", glsl_source)

    return effect_id


def apply_cdl(
    timeline: Timeline,
    clip_id: str,
    slope: tuple[float, float, float],
    offset: tuple[float, float, float],
    power: tuple[float, float, float],
    saturation: float = 1.0,
) -> str:
    """Apply ASC CDL to a clip via GES.

    Validates parameters and generates a GLSL fragment shader, then adds
    a glshader effect to the timeline clip.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to apply CDL to.
        slope: Per-channel slope (>= 0.0).
        offset: Per-channel offset (-1.0 to 1.0).
        power: Per-channel power (> 0.0).
        saturation: Global saturation (>= 0.0).

    Returns:
        Effect ID for the added CDL effect.

    Raises:
        ColorError: If validation fails.
    """
    params = compute_cdl(slope, offset, power, saturation)
    glsl_source = generate_cdl_glsl(params)

    effect_id = timeline.add_effect(clip_id, "glshader")
    timeline.set_effect_property(clip_id, effect_id, "fragment", glsl_source)

    return effect_id


def apply_color_transform(
    timeline: Timeline,
    clip_id: str,
    src_colorspace: str,
    dst_colorspace: str,
    config_path: str | None = None,
) -> str:
    """Apply an OCIO colour-space transform to a clip via GES.

    Validates parameters through the pure logic layer, then adds an
    ociofilter effect to the timeline clip.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to transform.
        src_colorspace: Source colour-space name.
        dst_colorspace: Destination colour-space name.
        config_path: Optional path to an OCIO config file.

    Returns:
        Effect ID for the added transform effect.

    Raises:
        ColorError: If validation fails.
    """
    params = compute_color_transform(src_colorspace, dst_colorspace, config_path)

    effect_id = timeline.add_effect(clip_id, "ociofilter")
    if params.config_path is not None:
        timeline.set_effect_property(clip_id, effect_id, "config-path", params.config_path)
    timeline.set_effect_property(clip_id, effect_id, "src-colorspace", params.src_colorspace)
    timeline.set_effect_property(clip_id, effect_id, "dst-colorspace", params.dst_colorspace)

    return effect_id


def apply_idt(
    timeline: Timeline,
    clip_id: str,
    ocio_config_path: str,
) -> str:
    """Apply an Input Device Transform (IDT) to a clip via GES.

    Reads the clip's camera colour-space metadata and applies a transform
    from that space to ACES - ACEScg.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip to apply the IDT to.
        ocio_config_path: Path to an OCIO config file.

    Returns:
        Effect ID for the added IDT effect.

    Raises:
        ColorError: If the clip has no camera colour-space metadata.
    """
    camera_cs = timeline.get_clip_metadata(clip_id, "agent:camera-color-space")
    if not camera_cs:
        raise ColorError(
            f"Clip {clip_id} has no camera color space metadata "
            f"(agent:camera-color-space). Set it before applying an IDT."
        )

    return apply_color_transform(
        timeline,
        clip_id,
        src_colorspace=camera_cs,
        dst_colorspace="ACES - ACEScg",
        config_path=ocio_config_path,
    )


def remove_color_effect(
    timeline: Timeline,
    clip_id: str,
    effect_id: str,
) -> None:
    """Remove a color effect from a clip.

    Args:
        timeline: Target timeline.
        clip_id: ID of the clip the effect belongs to.
        effect_id: ID of the effect to remove.
    """
    timeline.remove_effect(clip_id, effect_id)
