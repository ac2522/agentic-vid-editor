"""GES timeline interface for agent-friendly timeline manipulation."""

from __future__ import annotations

import math
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GES", "1.0")

from gi.repository import GES, Gst  # noqa: E402

from ave.utils import path_to_uri  # noqa: E402

# Initialize GStreamer and GES once
Gst.init(None)
GES.init()

# Tolerance for floating-point FPS comparison (e.g. 23.976 vs 24000/1001)
_FPS_TOLERANCE = 0.01


class TimelineError(Exception):
    """Raised when timeline operations fail."""


class Timeline:
    """High-level wrapper around a GES timeline."""

    def __init__(self, ges_timeline: GES.Timeline, path: Path, fps: float):
        self._timeline = ges_timeline
        self._path = path
        self._fps = fps
        self._clips: dict[str, GES.Clip] = {}
        self._next_clip_id = 0
        self._next_effect_seq = 0

    @classmethod
    def create(cls, path: Path, fps: float = 24.0) -> Timeline:
        """Create a new empty timeline with audio and video tracks."""
        timeline = GES.Timeline.new_audio_video()
        if timeline is None:
            raise TimelineError("Failed to create GES timeline")

        # Set FPS on video track restriction caps so it persists on save/load
        _set_video_track_fps(timeline, fps)

        # Add initial layer
        timeline.append_layer()

        tl = cls(timeline, path, fps)
        return tl

    @classmethod
    def load(cls, path: Path) -> Timeline:
        """Load a timeline from an XGES file."""
        if not path.exists():
            raise TimelineError(f"XGES file not found: {path}")

        timeline = GES.Timeline.new()
        uri = path_to_uri(path)

        project = GES.Project.new(uri)
        timeline = project.extract()

        if timeline is None:
            raise TimelineError(f"Failed to load timeline from {path}")

        # Detect fps from video track restriction caps
        fps = 24.0  # default
        for track in timeline.get_tracks():
            if track.get_property("track-type") == GES.TrackType.VIDEO:
                caps = track.get_restriction_caps()
                if caps and caps.get_size() > 0:
                    structure = caps.get_structure(0)
                    ok, num, den = structure.get_fraction("framerate")
                    if ok and den > 0:
                        fps = num / den
                break

        tl = cls(timeline, path, fps)

        # Restore clip IDs from metadata, or generate new ones
        for layer in timeline.get_layers():
            for clip in layer.get_clips():
                stored_id = clip.get_meta("agent:clip-id")
                if stored_id:
                    clip_id = stored_id
                    # Parse sequence number to keep _next_clip_id correct
                    try:
                        seq = int(stored_id.split("_")[1])
                        if seq >= tl._next_clip_id:
                            tl._next_clip_id = seq + 1
                    except (IndexError, ValueError):
                        pass
                else:
                    clip_id = f"clip_{tl._next_clip_id:04d}"
                    tl._next_clip_id += 1
                tl._clips[clip_id] = clip

        return tl

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration_ns(self) -> int:
        return self._timeline.get_duration()

    @property
    def clip_count(self) -> int:
        count = 0
        for layer in self._timeline.get_layers():
            count += len(layer.get_clips())
        return count

    def add_clip(
        self,
        media_path: Path,
        layer: int = 0,
        start_ns: int = 0,
        duration_ns: int | None = None,
        inpoint_ns: int = 0,
    ) -> str:
        """Add a media clip to the timeline. Returns clip ID."""
        uri = path_to_uri(media_path)

        asset = GES.UriClipAsset.request_sync(uri)
        if asset is None:
            raise TimelineError(f"Failed to load asset: {media_path}")

        if duration_ns is None:
            duration_ns = asset.get_duration()

        # Ensure layer exists
        layers = self._timeline.get_layers()
        while len(layers) <= layer:
            self._timeline.append_layer()
            layers = self._timeline.get_layers()

        target_layer = layers[layer]
        clip = target_layer.add_asset(
            asset,
            start_ns,
            inpoint_ns,
            duration_ns,
            GES.TrackType.UNKNOWN,
        )

        if clip is None:
            raise TimelineError(f"Failed to add clip at start={start_ns}")

        clip_id = f"clip_{self._next_clip_id:04d}"
        self._clips[clip_id] = clip
        self._next_clip_id += 1

        # Store clip ID as metadata for stable IDs across save/load
        clip.set_meta("agent:clip-id", clip_id)

        return clip_id

    def remove_clip(self, clip_id: str) -> None:
        """Remove a clip from the timeline."""
        clip = self._get_clip(clip_id)
        layer = clip.get_layer()
        if layer is None:
            raise TimelineError(f"Clip {clip_id} has no layer")
        layer.remove_clip(clip)
        del self._clips[clip_id]

    def add_effect(self, clip_id: str, element_description: str) -> str:
        """Add a GStreamer effect to a clip. Returns a unique effect ID."""
        clip = self._get_clip(clip_id)
        effect = GES.Effect.new(element_description)
        if effect is None:
            raise TimelineError(f"Failed to create effect: {element_description}")

        if not clip.add(effect):
            raise TimelineError(f"Failed to add effect to {clip_id}")

        # Use a monotonic sequence to avoid collisions when adding
        # multiple effects with the same element name
        effect_id = f"{clip_id}_fx{self._next_effect_seq}_{element_description.split()[0]}"
        self._next_effect_seq += 1

        # Store effect ID on the GES effect for lookup
        effect.set_meta("agent:effect-id", effect_id)

        return effect_id

    def remove_effect(self, clip_id: str, effect_id: str) -> None:
        """Remove an effect from a clip."""
        clip = self._get_clip(clip_id)

        for child in clip.get_children(False):
            if isinstance(child, GES.Effect):
                stored_id = child.get_meta("agent:effect-id")
                if stored_id == effect_id:
                    clip.remove(child)
                    return

        raise TimelineError(f"Effect {effect_id} not found on {clip_id}")

    def set_effect_property(
        self, clip_id: str, effect_id: str, prop_name: str, value: object
    ) -> None:
        """Set a property on an effect."""
        effect = self._get_effect(clip_id, effect_id)
        effect.set_child_property(prop_name, value)

    def get_effect_property(self, clip_id: str, effect_id: str, prop_name: str) -> object:
        """Get a property from an effect."""
        effect = self._get_effect(clip_id, effect_id)
        ok, value = effect.get_child_property(prop_name)
        if ok:
            return value
        raise TimelineError(f"Property {prop_name} not found")

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata on the timeline."""
        self._timeline.set_meta(key, value)

    def get_metadata(self, key: str) -> str | None:
        """Get metadata from the timeline."""
        return self._timeline.get_meta(key)

    def set_clip_metadata(self, clip_id: str, key: str, value: str) -> None:
        """Set metadata on a clip."""
        clip = self._get_clip(clip_id)
        clip.set_meta(key, value)

    def get_clip_metadata(self, clip_id: str, key: str) -> str | None:
        """Get metadata from a clip."""
        clip = self._get_clip(clip_id)
        return clip.get_meta(key)

    def save(self) -> None:
        """Save timeline to XGES file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        uri = path_to_uri(self._path)
        if not self._timeline.save_to_uri(uri, None, True):
            raise TimelineError(f"Failed to save timeline to {self._path}")

    def _get_clip(self, clip_id: str) -> GES.Clip:
        if clip_id not in self._clips:
            raise TimelineError(f"Clip not found: {clip_id}")
        return self._clips[clip_id]

    def _get_effect(self, clip_id: str, effect_id: str) -> GES.Effect:
        clip = self._get_clip(clip_id)
        for child in clip.get_children(False):
            if isinstance(child, GES.Effect):
                stored_id = child.get_meta("agent:effect-id")
                if stored_id == effect_id:
                    return child
        raise TimelineError(f"Effect {effect_id} not found on {clip_id}")


def _set_video_track_fps(timeline: GES.Timeline, fps: float) -> None:
    """Set framerate on the video track's restriction caps."""
    for track in timeline.get_tracks():
        if track.get_property("track-type") == GES.TrackType.VIDEO:
            # Convert float fps to fraction (e.g. 23.976 -> 24000/1001)
            num, den = _fps_to_fraction(fps)
            caps = Gst.Caps.from_string(
                f"video/x-raw,framerate={num}/{den}"
            )
            track.set_restriction_caps(caps)
            break


def _fps_to_fraction(fps: float) -> tuple[int, int]:
    """Convert a float FPS to an integer fraction.

    Handles common broadcast framerates precisely.
    """
    # Common broadcast framerates
    known = {
        23.976: (24000, 1001),
        24.0: (24, 1),
        25.0: (25, 1),
        29.97: (30000, 1001),
        30.0: (30, 1),
        48.0: (48, 1),
        50.0: (50, 1),
        59.94: (60000, 1001),
        60.0: (60, 1),
    }
    for known_fps, fraction in known.items():
        if math.isclose(fps, known_fps, rel_tol=1e-3):
            return fraction
    # Fallback: multiply to get integer
    den = 1001 if not math.isclose(fps, round(fps), rel_tol=1e-3) else 1
    num = round(fps * den)
    return (num, den)


def fps_close(a: float, b: float) -> bool:
    """Check if two FPS values are effectively equal."""
    return math.isclose(a, b, rel_tol=1e-3)
