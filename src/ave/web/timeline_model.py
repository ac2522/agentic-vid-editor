"""TimelineModel — pure Python timeline state with XGES parser.

No GES/GStreamer dependency. Parses XGES XML directly using stdlib
xml.etree.ElementTree. Provides the data model for the web UI timeline
visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClipState:
    """A single clip on the timeline."""

    clip_id: str
    asset_uri: str
    name: str
    layer_index: int
    start_ns: int
    duration_ns: int
    inpoint_ns: int
    track_types: int  # 4=video, 2=audio, 6=both
    effects: list[str] = field(default_factory=list)

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns

    @property
    def has_video(self) -> bool:
        return bool(self.track_types & 4)

    @property
    def has_audio(self) -> bool:
        return bool(self.track_types & 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "asset_uri": self.asset_uri,
            "name": self.name,
            "layer_index": self.layer_index,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "inpoint_ns": self.inpoint_ns,
            "track_types": self.track_types,
            "effects": list(self.effects),
        }


@dataclass
class LayerState:
    """A layer containing clips."""

    layer_index: int
    clips: list[ClipState] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "clips": [c.to_dict() for c in self.clips],
        }


class TimelineModel:
    """Pure Python timeline state, independent of GES."""

    def __init__(self, fps: float = 24.0) -> None:
        self.fps = fps
        self.layers: list[LayerState] = []
        self._clip_index: dict[str, ClipState] = {}

    # -- Properties ----------------------------------------------------------

    @property
    def duration_ns(self) -> int:
        if not self._clip_index:
            return 0
        return max(c.end_ns for c in self._clip_index.values())

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "duration_ns": self.duration_ns,
            "fps": self.fps,
        }

    # -- CRUD ----------------------------------------------------------------

    def add_clip(self, clip: ClipState) -> None:
        """Add a clip to the timeline, creating layers as needed."""
        if clip.clip_id in self._clip_index:
            raise ValueError(f"Duplicate clip_id: {clip.clip_id!r}")

        # Ensure enough layers exist
        while len(self.layers) <= clip.layer_index:
            self.layers.append(LayerState(layer_index=len(self.layers)))

        self.layers[clip.layer_index].clips.append(clip)
        self._clip_index[clip.clip_id] = clip

    def remove_clip(self, clip_id: str) -> None:
        """Remove a clip by ID. Raises KeyError if not found."""
        clip = self._clip_index.pop(clip_id)  # raises KeyError
        self.layers[clip.layer_index].clips.remove(clip)

    def update_clip(self, clip_id: str, **kwargs: Any) -> None:
        """Update clip attributes. Raises KeyError if not found."""
        clip = self.get_clip(clip_id)
        for key, value in kwargs.items():
            if not hasattr(clip, key):
                raise AttributeError(f"ClipState has no attribute {key!r}")
            setattr(clip, key, value)

    def get_clip(self, clip_id: str) -> ClipState:
        """Get a clip by ID. Raises KeyError if not found."""
        try:
            return self._clip_index[clip_id]
        except KeyError:
            raise KeyError(f"Clip not found: {clip_id!r}") from None
