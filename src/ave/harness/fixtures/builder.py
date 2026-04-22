"""Deterministic video fixture generation via FFmpeg lavfi.

Produces small, reproducible test videos (color patterns with timestamps,
tone bursts, etc.) for harness scenarios that don't need real footage.

The `lavfi` input device is part of the FFmpeg distribution; see
``https://ffmpeg.org/ffmpeg-filters.html#testsrc`` for available sources.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def build_lavfi_clip(
    expression: str,
    duration_seconds: float,
    output_path: Path,
    *,
    framerate: int = 24,
) -> Path:
    """Build a test video from a lavfi filter expression.

    Examples
    --------
    >>> build_lavfi_clip(
    ...     "testsrc=size=1280x720:rate=24",
    ...     duration_seconds=5.0,
    ...     output_path=Path("/tmp/test.mp4"),
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found on PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        expression,
        "-t",
        str(duration_seconds),
        "-r",
        str(framerate),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg lavfi build failed (exit {result.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    return output_path
