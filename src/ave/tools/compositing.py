"""Compositing tools — layer and blend mode parameter computation.

Pure logic layer: no GES dependency. Computes parameters that the GES
execution layer applies to the timeline.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class CompositingError(Exception):
    """Raised when compositing parameter validation fails."""


class BlendMode(Enum):
    """Supported blend modes for layer compositing."""

    SOURCE = "source"
    OVER = "over"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    ADD = "add"


# GL constants
GL_ONE = 1
GL_ZERO = 0
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_DST_COLOR = 0x0306
GL_ONE_MINUS_DST_COLOR = 0x0307
GL_FUNC_ADD = 0x8006


@dataclass(frozen=True)
class LayerParams:
    """Computed layer parameters."""

    layer_index: int
    alpha: float
    blend_mode: BlendMode
    position_x: int
    position_y: int
    width: int | None
    height: int | None


@dataclass(frozen=True)
class BlendFuncParams:
    """GL blend function parameters for a blend mode."""

    src_rgb: int
    dst_rgb: int
    src_alpha: int
    dst_alpha: int
    equation_rgb: int
    equation_alpha: int
    requires_shader: bool = False


@dataclass(frozen=True)
class BlendShaderInfo:
    """Complete blend information including shader source if needed."""

    blend_mode: BlendMode
    requires_shader: bool
    glsl_source: str | None  # None if GL blend functions suffice
    blend_params: BlendFuncParams | None  # None if shader required


def compute_layer_params(layers: list[dict]) -> list[LayerParams]:
    """Validate and compute layer compositing parameters.

    Args:
        layers: List of dicts with keys: layer_index, alpha, blend_mode,
                position_x, position_y, width, height.

    Returns:
        List of LayerParams frozen dataclasses, sorted by layer_index.

    Raises:
        CompositingError: If parameters are invalid.
    """
    if not layers:
        raise CompositingError("Cannot composite an empty list of layers")

    seen_indices: set[int] = set()
    results: list[LayerParams] = []

    for layer in layers:
        idx = layer["layer_index"]
        alpha = layer["alpha"]

        if idx < 0:
            raise CompositingError(f"Layer index must be >= 0, got {idx}")

        if idx in seen_indices:
            raise CompositingError(f"Duplicate layer index: {idx}")
        seen_indices.add(idx)

        if alpha < 0.0 or alpha > 1.0:
            raise CompositingError(f"Alpha must be 0.0-1.0, got {alpha}")

        results.append(
            LayerParams(
                layer_index=idx,
                alpha=alpha,
                blend_mode=layer["blend_mode"],
                position_x=layer["position_x"],
                position_y=layer["position_y"],
                width=layer.get("width"),
                height=layer.get("height"),
            )
        )

    results.sort(key=lambda p: p.layer_index)
    return results


def compute_blend_params(blend_mode: BlendMode) -> BlendFuncParams:
    """Map a BlendMode enum to GL blend function constants.

    Args:
        blend_mode: The blend mode to map.

    Returns:
        BlendFuncParams with GL blend function constants.
    """
    _BLEND_MAP: dict[BlendMode, BlendFuncParams] = {
        BlendMode.SOURCE: BlendFuncParams(
            src_rgb=GL_ONE,
            dst_rgb=GL_ZERO,
            src_alpha=GL_ONE,
            dst_alpha=GL_ZERO,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
        ),
        BlendMode.OVER: BlendFuncParams(
            src_rgb=GL_SRC_ALPHA,
            dst_rgb=GL_ONE_MINUS_SRC_ALPHA,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
        ),
        # result = src * dst_color + dst * 0 = src * dst
        BlendMode.MULTIPLY: BlendFuncParams(
            src_rgb=GL_DST_COLOR,
            dst_rgb=GL_ZERO,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
        ),
        BlendMode.SCREEN: BlendFuncParams(
            src_rgb=GL_ONE,
            dst_rgb=GL_ONE_MINUS_DST_COLOR,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
        ),
        # Overlay and soft light require per-pixel shader math.
        # GL blend params are fallback alpha compositing; requires_shader
        # signals the execution layer to use the GLSL shader instead.
        BlendMode.OVERLAY: BlendFuncParams(
            src_rgb=GL_SRC_ALPHA,
            dst_rgb=GL_ONE_MINUS_SRC_ALPHA,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
            requires_shader=True,
        ),
        BlendMode.SOFT_LIGHT: BlendFuncParams(
            src_rgb=GL_SRC_ALPHA,
            dst_rgb=GL_ONE_MINUS_SRC_ALPHA,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
            requires_shader=True,
        ),
        BlendMode.ADD: BlendFuncParams(
            src_rgb=GL_ONE,
            dst_rgb=GL_ONE,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
        ),
    }

    return _BLEND_MAP[blend_mode]


# ---------------------------------------------------------------------------
# GLSL blend mode shaders
# ---------------------------------------------------------------------------


def generate_overlay_glsl() -> str:
    """Generate a GLSL fragment shader for overlay blending.

    Overlay formula per channel:
      if dst < 0.5: result = 2.0 * src * dst
      else:         result = 1.0 - 2.0 * (1.0 - src) * (1.0 - dst)

    Returns:
        GLSL source string (#version 120) for a two-input blend shader.
    """
    return """\
#version 120
uniform sampler2D tex;
uniform sampler2D tex_dst;
varying vec2 v_texcoord;

void main() {
    vec4 src = texture2D(tex, v_texcoord);
    vec4 dst = texture2D(tex_dst, v_texcoord);

    vec3 result;

    // Overlay: multiply when dst < 0.5, screen otherwise
    result.r = (dst.r < 0.5) ? 2.0 * src.r * dst.r : 1.0 - 2.0 * (1.0 - src.r) * (1.0 - dst.r);
    result.g = (dst.g < 0.5) ? 2.0 * src.g * dst.g : 1.0 - 2.0 * (1.0 - src.g) * (1.0 - dst.g);
    result.b = (dst.b < 0.5) ? 2.0 * src.b * dst.b : 1.0 - 2.0 * (1.0 - src.b) * (1.0 - dst.b);

    // Alpha compositing: src over dst
    float out_a = src.a + dst.a * (1.0 - src.a);
    vec3 out_rgb = mix(dst.rgb, result, src.a);

    gl_FragColor = vec4(out_rgb, out_a);
}
"""


def generate_soft_light_glsl() -> str:
    """Generate a GLSL fragment shader for soft light blending (Photoshop formula).

    Soft light formula per channel:
      if src < 0.5: result = dst - (1.0 - 2.0*src) * dst * (1.0 - dst)
      else:         result = dst + (2.0*src - 1.0) * (sqrt(dst) - dst)

    Returns:
        GLSL source string (#version 120) for a two-input blend shader.
    """
    return """\
#version 120
uniform sampler2D tex;
uniform sampler2D tex_dst;
varying vec2 v_texcoord;

void main() {
    vec4 src = texture2D(tex, v_texcoord);
    vec4 dst = texture2D(tex_dst, v_texcoord);

    vec3 result;

    // Soft light (Photoshop formula)
    result.r = (src.r < 0.5) ? dst.r - (1.0 - 2.0 * src.r) * dst.r * (1.0 - dst.r)
                              : dst.r + (2.0 * src.r - 1.0) * (sqrt(dst.r) - dst.r);
    result.g = (src.g < 0.5) ? dst.g - (1.0 - 2.0 * src.g) * dst.g * (1.0 - dst.g)
                              : dst.g + (2.0 * src.g - 1.0) * (sqrt(dst.g) - dst.g);
    result.b = (src.b < 0.5) ? dst.b - (1.0 - 2.0 * src.b) * dst.b * (1.0 - dst.b)
                              : dst.b + (2.0 * src.b - 1.0) * (sqrt(dst.b) - dst.b);

    // Alpha compositing: src over dst
    float out_a = src.a + dst.a * (1.0 - src.a);
    vec3 out_rgb = mix(dst.rgb, result, src.a);

    gl_FragColor = vec4(out_rgb, out_a);
}
"""


_SHADER_GENERATORS: dict[BlendMode, Callable[[], str]] = {
    BlendMode.OVERLAY: generate_overlay_glsl,
    BlendMode.SOFT_LIGHT: generate_soft_light_glsl,
}


def compute_blend_info(blend_mode: BlendMode) -> BlendShaderInfo:
    """Get complete blend information including shader if needed.

    For blend modes that require a shader (OVERLAY, SOFT_LIGHT), returns
    a BlendShaderInfo with glsl_source populated and blend_params set to None.
    For modes that can use GL blend functions, returns blend_params with
    no shader.

    Args:
        blend_mode: The blend mode to get info for.

    Returns:
        BlendShaderInfo with either glsl_source or blend_params.
    """
    params = compute_blend_params(blend_mode)

    if params.requires_shader:
        generator = _SHADER_GENERATORS[blend_mode]
        return BlendShaderInfo(
            blend_mode=blend_mode,
            requires_shader=True,
            glsl_source=generator(),
            blend_params=None,
        )

    return BlendShaderInfo(
        blend_mode=blend_mode,
        requires_shader=False,
        glsl_source=None,
        blend_params=params,
    )
