"""Shared utilities for the ave package."""

import math
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
