"""Compositing domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_compositing_tools(registry: ToolRegistry) -> None:
    """Register compositing domain tools."""

    @registry.tool(
        domain="compositing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["layer_order_set"],
        tags=["z-order", "stack order", "layer stack", "bring to front",
              "send to back", "arrange layers", "compositing order"],
    )
    def set_layer_order(layers: list):
        """Set layer ordering and compositing parameters for multiple clips."""
        from ave.tools.compositing import compute_layer_params

        return compute_layer_params(layers)

    @registry.tool(
        domain="compositing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["blend_mode_applied"],
        tags=["blend", "multiply", "screen", "overlay", "mix layers",
              "composite mode", "layer blend", "blending"],
    )
    def apply_blend_mode(blend_mode: str):
        """Apply a blend mode to a clip (e.g. over, multiply, screen, overlay)."""
        from ave.tools.compositing import BlendMode, compute_blend_params

        mode = BlendMode(blend_mode)
        return compute_blend_params(mode)

    @registry.tool(
        domain="compositing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["clip_position_set"],
        tags=["move clip", "position", "picture in picture", "PiP", "offset",
              "place on canvas", "x y position", "reposition"],
    )
    def set_clip_position(x: int, y: int, width: int = 0, height: int = 0):
        """Set the position and optional size of a clip on the canvas."""
        return {
            "x": x,
            "y": y,
            "width": width if width > 0 else None,
            "height": height if height > 0 else None,
        }

    @registry.tool(
        domain="compositing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["clip_alpha_set"],
        tags=["opacity", "transparency", "see through", "transparent",
              "fade layer", "ghost", "semi-transparent"],
    )
    def set_clip_alpha(alpha: float):
        """Set the opacity (alpha) of a clip, from 0.0 (transparent) to 1.0 (opaque)."""
        from ave.tools.compositing import CompositingError

        if alpha < 0.0 or alpha > 1.0:
            raise CompositingError(f"Alpha must be 0.0-1.0, got {alpha}")
        return {"alpha": alpha}
