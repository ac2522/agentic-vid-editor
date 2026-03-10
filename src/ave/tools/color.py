"""Color tools — grading, CDL, LUT, and color transform parameter computation.

Pure logic layer: no GES dependency. Computes parameters that the GES
execution layer applies to the pipeline.
"""

import os
from dataclasses import dataclass


class ColorError(Exception):
    """Raised when color parameter validation fails."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CubeLUT:
    """Parsed .cube LUT data."""

    title: str
    size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    table: list[tuple[float, float, float]]


@dataclass(frozen=True)
class ColorGradeParams:
    """Computed colour-grade parameters (lift/gamma/gain)."""

    lift: tuple[float, float, float]
    gamma: tuple[float, float, float]
    gain: tuple[float, float, float]
    saturation: float
    offset: tuple[float, float, float]


@dataclass(frozen=True)
class CDLParams:
    """ASC CDL parameters."""

    slope: tuple[float, float, float]
    offset: tuple[float, float, float]
    power: tuple[float, float, float]
    saturation: float


@dataclass(frozen=True)
class LUTParams:
    """Parameters for LUT application."""

    path: str
    intensity: float


@dataclass(frozen=True)
class ColorTransformParams:
    """Parameters for an OCIO colour-space transform."""

    src_colorspace: str
    dst_colorspace: str
    config_path: str | None


# ---------------------------------------------------------------------------
# parse_cube_lut
# ---------------------------------------------------------------------------


def parse_cube_lut(path: str) -> CubeLUT:
    """Parse a .cube LUT file.

    Supports both LUT_1D_SIZE and LUT_3D_SIZE formats.

    Args:
        path: Filesystem path to the .cube file.

    Returns:
        CubeLUT with parsed data.

    Raises:
        ColorError: If the file is missing or malformed.
    """
    if not os.path.isfile(path):
        raise ColorError(f"LUT file not found: {path}")

    title = ""
    size: int | None = None
    is_3d: bool | None = None
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    table: list[tuple[float, float, float]] = []

    try:
        with open(path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("TITLE"):
                    # TITLE "Some Name"
                    title = line.split('"')[1] if '"' in line else line[6:].strip()
                    continue

                if line.startswith("LUT_3D_SIZE"):
                    size = int(line.split()[1])
                    is_3d = True
                    continue

                if line.startswith("LUT_1D_SIZE"):
                    size = int(line.split()[1])
                    is_3d = False
                    continue

                if line.startswith("DOMAIN_MIN"):
                    parts = line.split()[1:]
                    domain_min = (float(parts[0]), float(parts[1]), float(parts[2]))
                    continue

                if line.startswith("DOMAIN_MAX"):
                    parts = line.split()[1:]
                    domain_max = (float(parts[0]), float(parts[1]), float(parts[2]))
                    continue

                # Try to parse as a data row (three floats)
                parts = line.split()
                if len(parts) == 3:
                    try:
                        table.append(
                            (float(parts[0]), float(parts[1]), float(parts[2]))
                        )
                    except ValueError:
                        raise ColorError(
                            f"LUT file '{path}': cannot parse data row "
                            f"{len(table) + 1}: '{line}'"
                        ) from None

    except OSError as exc:
        raise ColorError(f"Cannot read LUT file: {exc}") from exc

    if size is None:
        raise ColorError(f"LUT file missing LUT_3D_SIZE or LUT_1D_SIZE: {path}")

    expected = size ** 3 if is_3d else size
    if len(table) != expected:
        raise ColorError(
            f"LUT table size mismatch: expected {expected} entries, got {len(table)}"
        )

    return CubeLUT(
        title=title,
        size=size,
        domain_min=domain_min,
        domain_max=domain_max,
        table=table,
    )


# ---------------------------------------------------------------------------
# compute_color_grade
# ---------------------------------------------------------------------------


def _validate_rgb_range(
    value: tuple[float, float, float],
    name: str,
    lo: float,
    hi: float,
) -> None:
    for i, v in enumerate(value):
        if v < lo or v > hi:
            channel = ("R", "G", "B")[i]
            raise ColorError(
                f"{name} channel {channel} value {v} is outside "
                f"allowed range [{lo}, {hi}]"
            )


def compute_color_grade(
    lift: tuple[float, float, float],
    gamma: tuple[float, float, float],
    gain: tuple[float, float, float],
    saturation: float = 1.0,
    offset: tuple[float, float, float] = (0, 0, 0),
) -> ColorGradeParams:
    """Validate and compute colour-grade parameters.

    Args:
        lift: Shadow adjustment per channel, each -1.0 to 1.0.
        gamma: Midtone adjustment per channel, each 0.01 to 4.0.
        gain: Highlight adjustment per channel, each 0.0 to 4.0.
        saturation: Global saturation, 0.0 to 4.0.
        offset: Offset per channel, each -1.0 to 1.0.

    Returns:
        ColorGradeParams with validated values.

    Raises:
        ColorError: If any parameter is out of range.
    """
    _validate_rgb_range(lift, "lift", -1.0, 1.0)
    _validate_rgb_range(gamma, "gamma", 0.01, 4.0)
    _validate_rgb_range(gain, "gain", 0.0, 4.0)
    _validate_rgb_range(offset, "offset", -1.0, 1.0)

    if saturation < 0.0 or saturation > 4.0:
        raise ColorError(
            f"saturation {saturation} is outside allowed range [0.0, 4.0]"
        )

    return ColorGradeParams(
        lift=lift,
        gamma=gamma,
        gain=gain,
        saturation=saturation,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# compute_cdl
# ---------------------------------------------------------------------------


def compute_cdl(
    slope: tuple[float, float, float],
    offset: tuple[float, float, float],
    power: tuple[float, float, float],
    saturation: float = 1.0,
) -> CDLParams:
    """Validate and compute ASC CDL parameters.

    Args:
        slope: Per-channel slope (>= 0.0).
        offset: Per-channel offset (-1.0 to 1.0).
        power: Per-channel power (> 0.0).
        saturation: Global saturation (>= 0.0).

    Returns:
        CDLParams with validated values.

    Raises:
        ColorError: If any parameter is invalid.
    """
    for i, v in enumerate(slope):
        if v < 0.0:
            channel = ("R", "G", "B")[i]
            raise ColorError(
                f"slope channel {channel} value {v} is negative (must be >= 0.0)"
            )

    _validate_rgb_range(offset, "offset", -1.0, 1.0)

    for i, v in enumerate(power):
        if v <= 0.0:
            channel = ("R", "G", "B")[i]
            raise ColorError(
                f"power channel {channel} value {v} must be > 0.0"
            )

    if saturation < 0.0:
        raise ColorError(f"saturation {saturation} must be >= 0.0")

    return CDLParams(
        slope=slope,
        offset=offset,
        power=power,
        saturation=saturation,
    )


# ---------------------------------------------------------------------------
# GLSL generation
# ---------------------------------------------------------------------------


def generate_grade_glsl(grade_params: ColorGradeParams) -> str:
    """Generate a GLSL fragment shader for lift/gamma/gain colour grading.

    The shader applies:
      1. lift (added to shadows)
      2. gain (multiplied into highlights)
      3. gamma (power curve on midtones)
      4. offset
      5. saturation

    Args:
        grade_params: Validated ColorGradeParams.

    Returns:
        GLSL source string suitable for a GStreamer glshader element.
    """
    p = grade_params
    return f"""\
#version 120
uniform sampler2D tex;
varying vec2 v_texcoord;

void main() {{
    vec4 color = texture2D(tex, v_texcoord);

    // Lift (shadows)
    vec3 lifted = color.rgb + vec3({p.lift[0]}, {p.lift[1]}, {p.lift[2]});

    // Gain (highlights)
    vec3 gained = lifted * vec3({p.gain[0]}, {p.gain[1]}, {p.gain[2]});

    // Gamma (midtones)
    vec3 inv_gamma = vec3(
        1.0 / {p.gamma[0]},
        1.0 / {p.gamma[1]},
        1.0 / {p.gamma[2]}
    );
    vec3 graded = pow(max(gained, vec3(0.0)), inv_gamma);

    // Offset
    graded += vec3({p.offset[0]}, {p.offset[1]}, {p.offset[2]});

    // Saturation
    float luma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
    vec3 saturated = mix(vec3(luma), graded, {p.saturation});

    gl_FragColor = vec4(saturated, color.a);
}}
"""


def generate_cdl_glsl(cdl_params: CDLParams) -> str:
    """Generate a GLSL fragment shader for ASC CDL.

    ASC CDL formula: out = pow(clamp(in * slope + offset, 0, 1), power)
    Then saturation is applied.

    Args:
        cdl_params: Validated CDLParams.

    Returns:
        GLSL source string.
    """
    p = cdl_params
    return f"""\
#version 120
uniform sampler2D tex;
varying vec2 v_texcoord;

void main() {{
    vec4 color = texture2D(tex, v_texcoord);

    // ASC CDL: slope, offset, power
    vec3 slope = vec3({p.slope[0]}, {p.slope[1]}, {p.slope[2]});
    vec3 offset = vec3({p.offset[0]}, {p.offset[1]}, {p.offset[2]});
    vec3 power = vec3({p.power[0]}, {p.power[1]}, {p.power[2]});

    vec3 result = pow(clamp(color.rgb * slope + offset, 0.0, 1.0), power);

    // Saturation
    float luma = dot(result, vec3(0.2126, 0.7152, 0.0722));
    vec3 saturated = mix(vec3(luma), result, {p.saturation});

    gl_FragColor = vec4(saturated, color.a);
}}
"""


# ---------------------------------------------------------------------------
# compute_lut_application
# ---------------------------------------------------------------------------


def compute_lut_application(
    lut_path: str,
    intensity: float = 1.0,
) -> LUTParams:
    """Validate parameters for applying a .cube LUT.

    Args:
        lut_path: Path to a .cube LUT file.
        intensity: Blend factor 0.0 (no effect) to 1.0 (full effect).

    Returns:
        LUTParams with validated path and intensity.

    Raises:
        ColorError: If the file is missing, wrong extension, or intensity OOB.
    """
    if not os.path.isfile(lut_path):
        raise ColorError(f"LUT file not found: {lut_path}")

    if not lut_path.lower().endswith(".cube"):
        raise ColorError(
            f"LUT file must have .cube extension, got: {lut_path}"
        )

    if intensity < 0.0 or intensity > 1.0:
        raise ColorError(
            f"intensity {intensity} is outside allowed range [0.0, 1.0]"
        )

    return LUTParams(path=lut_path, intensity=intensity)


# ---------------------------------------------------------------------------
# compute_color_transform
# ---------------------------------------------------------------------------


def compute_color_transform(
    src_colorspace: str,
    dst_colorspace: str,
    config_path: str | None = None,
) -> ColorTransformParams:
    """Validate parameters for an OCIO colour-space transform.

    Args:
        src_colorspace: Source colour-space name (non-empty).
        dst_colorspace: Destination colour-space name (non-empty).
        config_path: Optional path to an OCIO config file.

    Returns:
        ColorTransformParams with validated values.

    Raises:
        ColorError: If names are empty or config_path doesn't exist.
    """
    if not src_colorspace:
        raise ColorError("src colorspace must be a non-empty string")

    if not dst_colorspace:
        raise ColorError("dst colorspace must be a non-empty string")

    if config_path is not None and not os.path.isfile(config_path):
        raise ColorError(f"OCIO config file not found: {config_path}")

    return ColorTransformParams(
        src_colorspace=src_colorspace,
        dst_colorspace=dst_colorspace,
        config_path=config_path,
    )
