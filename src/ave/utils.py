"""Shared utilities for the ave package."""

import math
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote


def path_to_uri(path: Path) -> str:
    """Convert a Path to a file URI."""
    abs_path = str(path.resolve())
    return "file://" + quote(abs_path, safe="/")


def fps_close(a: float, b: float) -> bool:
    """Check if two FPS values are effectively equal.

    Uses relative tolerance to handle float imprecision in
    common broadcast framerates (e.g. 23.976 vs 24000/1001).
    """
    if a == 0.0 and b == 0.0:
        return True
    return math.isclose(a, b, rel_tol=1e-3)


def fps_to_fraction(fps: float) -> tuple[int, int]:
    """Convert fps float to integer numerator/denominator.

    Handles common NTSC fractional frame rates exactly,
    and uses Fraction.limit_denominator for the general case.
    """
    # Handle common fractional frame rates exactly
    common = {
        23.976: (24000, 1001),
        23.98: (24000, 1001),
        29.97: (30000, 1001),
        47.952: (48000, 1001),
        59.94: (60000, 1001),
    }
    for known_fps, frac in common.items():
        if abs(fps - known_fps) < 0.01:
            return frac
    # For integer frame rates
    if fps == int(fps):
        return (int(fps), 1)
    # General case: use Fraction for accurate representation
    f = Fraction(fps).limit_denominator(1001)
    return (f.numerator, f.denominator)
