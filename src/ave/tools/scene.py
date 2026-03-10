"""Scene detection tools — shot boundary detection with swappable backends.

Pure logic layer for data models and boundary computation.
Scene detection engine integration is conditional.
"""

from __future__ import annotations

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
