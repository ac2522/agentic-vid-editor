"""Color domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_color_tools(registry: ToolRegistry) -> None:
    """Register color domain tools."""

    @registry.tool(
        domain="color",
        requires=["timeline_loaded", "clip_exists"],
        provides=["color_graded"],
        tags=["colour", "lift gamma gain", "color correction", "tint", "warm",
              "cool", "look", "mood", "cinematic", "brighten", "darken",
              "brighter", "darker", "warm up", "cool down", "color balance"],
        modifies_timeline=True,
    )
    def color_grade(
        lift_r: float,
        lift_g: float,
        lift_b: float,
        gamma_r: float,
        gamma_g: float,
        gamma_b: float,
        gain_r: float,
        gain_g: float,
        gain_b: float,
        saturation: float = 1.0,
        offset_r: float = 0.0,
        offset_g: float = 0.0,
        offset_b: float = 0.0,
    ):
        """Apply lift/gamma/gain colour grading to a clip."""
        from ave.tools.color import compute_color_grade

        return compute_color_grade(
            lift=(lift_r, lift_g, lift_b),
            gamma=(gamma_r, gamma_g, gamma_b),
            gain=(gain_r, gain_g, gain_b),
            saturation=saturation,
            offset=(offset_r, offset_g, offset_b),
        )

    @registry.tool(
        domain="color",
        requires=["timeline_loaded", "clip_exists"],
        provides=["cdl_applied"],
        tags=["ASC CDL", "slope offset power", "color decision list",
              "primary correction", "printer lights", "colour correction"],
        modifies_timeline=True,
    )
    def cdl(
        slope_r: float,
        slope_g: float,
        slope_b: float,
        offset_r: float,
        offset_g: float,
        offset_b: float,
        power_r: float,
        power_g: float,
        power_b: float,
        saturation: float = 1.0,
    ):
        """Apply ASC CDL (slope/offset/power) colour correction."""
        from ave.tools.color import compute_cdl

        return compute_cdl(
            slope=(slope_r, slope_g, slope_b),
            offset=(offset_r, offset_g, offset_b),
            power=(power_r, power_g, power_b),
            saturation=saturation,
        )

    @registry.tool(
        domain="color",
        requires=["media_ingested"],
        provides=["lut_parsed"],
        tags=["LUT", "lookup table", "cube file", "color transform",
              "film emulation", "log to rec709"],
    )
    def lut_parse(path: str):
        """Parse a .cube LUT file and return its data."""
        from ave.tools.color import parse_cube_lut

        return parse_cube_lut(path)
