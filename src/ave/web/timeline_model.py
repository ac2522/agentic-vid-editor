"""TimelineModel — pure Python timeline state with XGES parser.

No GES/GStreamer dependency. Parses XGES XML directly using stdlib
xml.etree.ElementTree. Provides the data model for the web UI timeline
visualization.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


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
        self._xges_path: Path | None = None

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

    # -- XGES parsing --------------------------------------------------------

    @classmethod
    def load_from_xges(cls, path: Path | str) -> TimelineModel:
        """Parse an XGES file from disk."""
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        model = cls.load_from_xges_string(content)
        model._xges_path = path
        return model

    @classmethod
    def load_from_xges_string(cls, xml_str: str) -> TimelineModel:
        """Parse an XGES XML string into a TimelineModel."""
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as exc:
            raise ValueError(f"Malformed XGES XML: {exc}") from exc

        timeline_el = root.find(".//timeline")
        if timeline_el is None:
            raise ValueError("Malformed XGES XML: no <timeline> element found")

        fps = cls._parse_framerate(timeline_el.get("properties", ""))
        model = cls(fps=fps)

        for layer_el in timeline_el.findall("layer"):
            priority = int(layer_el.get("priority", len(model.layers)))
            # Ensure layers up to this priority exist
            while len(model.layers) <= priority:
                model.layers.append(LayerState(layer_index=len(model.layers)))

            for clip_el in layer_el.findall("clip"):
                clip = cls._parse_clip(clip_el, priority)
                model._clip_index[clip.clip_id] = clip
                model.layers[priority].clips.append(clip)

        return model

    def reload_from_xges(self) -> None:
        """Re-parse stored XGES path. No-op if no path set."""
        if self._xges_path is None:
            return
        reloaded = self.load_from_xges(self._xges_path)
        self.fps = reloaded.fps
        self.layers = reloaded.layers
        self._clip_index = reloaded._clip_index

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _parse_framerate(props: str) -> float:
        """Extract framerate from GES properties string."""
        m = re.search(r"framerate=\(fraction\)(\d+)/(\d+)", props)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den != 0:
                return num / den
        return 24.0

    @staticmethod
    def _parse_clip(clip_el: ET.Element, layer_index: int) -> ClipState:
        """Parse a single <clip> element into a ClipState."""
        asset_id = clip_el.get("asset-id", "")
        xml_id = clip_el.get("id", "0")

        # Extract clip ID from metadatas
        clip_id = _extract_agent_clip_id(clip_el.get("metadatas", ""))
        if clip_id is None:
            clip_id = f"clip_{xml_id}"

        # Derive name from asset URI
        name = _name_from_uri(asset_id)

        # Parse effects
        effects = [
            eff.get("asset-id", "") for eff in clip_el.findall("effect") if eff.get("asset-id")
        ]

        return ClipState(
            clip_id=clip_id,
            asset_uri=asset_id,
            name=name,
            layer_index=layer_index,
            start_ns=int(clip_el.get("start", "0")),
            duration_ns=int(clip_el.get("duration", "0")),
            inpoint_ns=int(clip_el.get("inpoint", "0")),
            track_types=int(clip_el.get("track-types", "6")),
            effects=effects,
        )


def _extract_agent_clip_id(metadatas: str) -> str | None:
    """Extract agent:clip-id value from GES metadatas string."""
    m = re.search(r"agent:clip-id=\(string\)([^;]+)", metadatas)
    return m.group(1).strip() if m else None


def _name_from_uri(uri: str) -> str:
    """Extract filename from a file URI or path."""
    if not uri:
        return ""
    parsed = urlparse(uri)
    path = unquote(parsed.path)
    return Path(path).name
