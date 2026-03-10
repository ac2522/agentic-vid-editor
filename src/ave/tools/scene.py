"""Scene detection tools — shot boundary detection with swappable backends.

Pure logic layer for data models and boundary computation.
Scene detection engine integration is conditional.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class SceneError(Exception):
    """Raised when scene detection fails."""


# GES metadata key constants
AGENT_META_SCENE_ID = "agent:scene-id"
AGENT_META_SCENE_START = "agent:scene-start-ns"
AGENT_META_SCENE_END = "agent:scene-end-ns"


class SceneBoundary(BaseModel):
    """A detected scene/shot boundary with timestamps.

    Nanosecond timestamps are authoritative. Frame numbers are derived.
    """

    start_ns: int
    end_ns: int
    fps: float

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def start_frame(self) -> int:
        return round(self.start_ns * self.fps / 1_000_000_000)

    @property
    def end_frame(self) -> int:
        return round(self.end_ns * self.fps / 1_000_000_000)

    @property
    def mid_frame(self) -> int:
        mid_ns = (self.start_ns + self.end_ns) // 2
        return round(mid_ns * self.fps / 1_000_000_000)


class SceneBackend(Protocol):
    """Protocol for scene detection backends. Type-annotation only."""

    def detect_scenes(
        self,
        video_path: Path,
        threshold: float,
        detector: str = "content",
    ) -> list[SceneBoundary]: ...


def extract_keyframes(
    video_path: Path,
    boundaries: list[SceneBoundary],
    output_dir: Path,
    strategy: str = "middle",
) -> list[Path]:
    """Extract one keyframe per scene boundary using FFmpeg.

    This is an I/O utility, not pure logic.

    Args:
        video_path: Source video file.
        boundaries: Scene boundaries to extract from.
        output_dir: Directory to write keyframe images.
        strategy: "middle" (middle of scene) or "first" (first frame).

    Returns:
        List of paths to extracted keyframe images.
    """
    if strategy not in ("middle", "first"):
        raise SceneError(f"Unknown strategy: {strategy}. Use 'middle' or 'first'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i, boundary in enumerate(boundaries):
        if strategy == "middle":
            seek_ns = (boundary.start_ns + boundary.end_ns) // 2
        else:
            seek_ns = boundary.start_ns

        seek_seconds = seek_ns / 1_000_000_000
        output_path = output_dir / f"keyframe_{i:04d}.jpg"

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{seek_seconds:.6f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output_path),
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise SceneError(
                f"Keyframe extraction failed at {seek_seconds}s: {e.stderr.decode()}"
            ) from e

        paths.append(output_path)

    return paths
