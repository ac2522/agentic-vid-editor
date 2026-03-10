"""Compositing tools — layer and blend mode parameter computation.

Pure logic layer: no GES dependency. Computes parameters that the GES
execution layer applies to the timeline.
"""

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
        # Approximation: true overlay requires per-pixel conditional
        # (multiply when dst < 0.5, screen otherwise). Falls back to
        # standard alpha compositing until shader-based impl is added.
        BlendMode.OVERLAY: BlendFuncParams(
            src_rgb=GL_SRC_ALPHA,
            dst_rgb=GL_ONE_MINUS_SRC_ALPHA,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
        ),
        # Approximation: true soft light requires shader math.
        # Falls back to standard alpha compositing.
        BlendMode.SOFT_LIGHT: BlendFuncParams(
            src_rgb=GL_SRC_ALPHA,
            dst_rgb=GL_ONE_MINUS_SRC_ALPHA,
            src_alpha=GL_ONE,
            dst_alpha=GL_ONE_MINUS_SRC_ALPHA,
            equation_rgb=GL_FUNC_ADD,
            equation_alpha=GL_FUNC_ADD,
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
