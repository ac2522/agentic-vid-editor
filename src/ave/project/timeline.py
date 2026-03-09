"""GES timeline interface for agent-friendly timeline manipulation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from ave.utils import fps_to_fraction

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GES", "1.0")

from gi.repository import GES, Gst  # noqa: E402

# Initialize GStreamer and GES once
Gst.init(None)
GES.init()

# Metadata key for storing stable clip IDs (P0-3)
_CLIP_ID_META_KEY = "agent:clip-id"


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
        # P0-2: Index-based effect tracking — per-clip ordered list
        self._effects: dict[str, list[GES.Effect]] = {}

    @classmethod
    def create(cls, path: Path, fps: float = 24.0) -> Timeline:
        """Create a new empty timeline with audio and video tracks."""
        timeline = GES.Timeline.new_audio_video()
        if timeline is None:
            raise TimelineError("Failed to create GES timeline")

        # Add initial layer
        timeline.append_layer()

        # P0-3: Set restriction caps on video track so fps persists in XGES
        for track in timeline.get_tracks():
            if track.get_property("track-type") == GES.TrackType.VIDEO:
                # Convert fps to fraction
                num, den = fps_to_fraction(fps)
                caps = Gst.Caps.from_string(f"video/x-raw,framerate={num}/{den}")
                track.set_restriction_caps(caps)
                break

        tl = cls(timeline, path, fps)
        return tl

    @classmethod
    def load(cls, path: Path) -> Timeline:
        """Load a timeline from an XGES file."""
        if not path.exists():
            raise TimelineError(f"XGES file not found: {path}")

        uri = _path_to_uri(path)

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

        # P0-3: Re-index clips using stable agent:clip-id metadata
        for layer in timeline.get_layers():
            for clip in layer.get_clips():
                stored_id = clip.get_string(_CLIP_ID_META_KEY)
                if stored_id:
                    clip_id = stored_id
                    # Parse the numeric suffix to keep _next_clip_id consistent
                    try:
                        num_part = int(clip_id.split("_")[1])
                        if num_part >= tl._next_clip_id:
                            tl._next_clip_id = num_part + 1
                    except (IndexError, ValueError):
                        pass
                else:
                    # Fallback for clips without metadata (legacy files)
                    clip_id = f"clip_{tl._next_clip_id:04d}"
                    clip.set_string(_CLIP_ID_META_KEY, clip_id)
                    tl._next_clip_id += 1
                tl._clips[clip_id] = clip

                # Reconstruct effect tracking for this clip
                top_effects = clip.get_top_effects()
                if top_effects:
                    tl._effects[clip_id] = list(top_effects)

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

    # --- P0-1: Public clip access methods ---

    def get_clip(self, clip_id: str) -> GES.Clip:
        """Get a GES clip by its ID. Raises TimelineError if not found."""
        if clip_id not in self._clips:
            raise TimelineError(f"Clip not found: {clip_id}")
        return self._clips[clip_id]

    def register_clip(self, ges_clip: GES.Clip) -> str:
        """Register an externally-created GES clip and return its new ID."""
        clip_id = f"clip_{self._next_clip_id:04d}"
        self._clips[clip_id] = ges_clip
        self._next_clip_id += 1
        # P0-3: Store stable ID as metadata
        ges_clip.set_string(_CLIP_ID_META_KEY, clip_id)
        return clip_id

    def add_clip(
        self,
        media_path: Path,
        layer: int = 0,
        start_ns: int = 0,
        duration_ns: int | None = None,
        inpoint_ns: int = 0,
    ) -> str:
        """Add a media clip to the timeline. Returns clip ID."""
        uri = _path_to_uri(media_path)

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

        # Use register_clip to assign ID and store metadata
        return self.register_clip(clip)

    def remove_clip(self, clip_id: str) -> None:
        """Remove a clip from the timeline."""
        clip = self.get_clip(clip_id)
        layer = clip.get_layer()
        if layer is None:
            raise TimelineError(f"Clip {clip_id} has no layer")
        # P0-5: Check return value
        if not layer.remove_clip(clip):
            raise TimelineError(f"GES refused to remove clip {clip_id} from layer")
        del self._clips[clip_id]
        # Clean up effect tracking
        self._effects.pop(clip_id, None)

    def add_clip_from_source(
        self,
        source_clip_id: str,
        start_ns: int,
        inpoint_ns: int,
        duration_ns: int,
    ) -> str:
        """Add a new clip using the same asset as an existing clip.

        Used by split_clip to create the right portion without bypassing
        the Timeline API.
        """
        source_clip = self.get_clip(source_clip_id)
        asset = source_clip.get_asset()
        if asset is None:
            raise TimelineError(f"Clip {source_clip_id} has no asset")

        layer = source_clip.get_layer()
        if layer is None:
            raise TimelineError(f"Clip {source_clip_id} has no layer")

        new_clip = layer.add_asset(
            asset,
            start_ns,
            inpoint_ns,
            duration_ns,
            GES.TrackType.UNKNOWN,
        )
        if new_clip is None:
            raise TimelineError(f"Failed to add clip from {source_clip_id} at start={start_ns}")

        return self.register_clip(new_clip)

    # --- P0-2: Index-based effect management ---

    def add_effect(self, clip_id: str, element_description: str) -> str:
        """Add a GStreamer effect to a clip. Returns index-based effect ID."""
        clip = self.get_clip(clip_id)
        effect = GES.Effect.new(element_description)
        if effect is None:
            raise TimelineError(f"Failed to create effect: {element_description}")

        if not clip.add(effect):
            raise TimelineError(f"Failed to add effect to {clip_id}")

        # Store in per-clip effect list
        if clip_id not in self._effects:
            self._effects[clip_id] = []
        index = len(self._effects[clip_id])
        self._effects[clip_id].append(effect)

        effect_id = f"{clip_id}_fx_{index}"
        return effect_id

    def remove_effect(self, clip_id: str, effect_id: str) -> None:
        """Remove an effect from a clip by index-based ID."""
        clip = self.get_clip(clip_id)
        effect = self._get_effect(clip_id, effect_id)
        if not clip.remove(effect):
            raise TimelineError(f"GES refused to remove effect {effect_id} from {clip_id}")

        # Remove from tracking and mark slot as None to preserve indices
        index = self._parse_effect_index(effect_id)
        self._effects[clip_id][index] = None

    def set_effect_property(
        self, clip_id: str, effect_id: str, prop_name: str, value: object
    ) -> None:
        """Set a property on an effect."""
        effect = self._get_effect(clip_id, effect_id)
        ok = effect.set_child_property(prop_name, value)
        # P0-5: Verify the property was set
        if not ok:
            raise TimelineError(
                f"Failed to set property '{prop_name}' on effect {effect_id}. "
                f"Check that the property name and value type are correct."
            )

    def get_effect_property(self, clip_id: str, effect_id: str, prop_name: str) -> object:
        """Get a property from an effect."""
        effect = self._get_effect(clip_id, effect_id)
        ok, value = effect.get_child_property(prop_name)
        if ok:
            return value
        raise TimelineError(f"Property {prop_name} not found on {effect_id}")

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata on the timeline."""
        self._timeline.set_meta(key, value)

    def get_metadata(self, key: str) -> str | None:
        """Get metadata from the timeline."""
        return self._timeline.get_meta(key)

    def set_clip_metadata(self, clip_id: str, key: str, value: str) -> None:
        """Set metadata on a clip."""
        clip = self.get_clip(clip_id)
        clip.set_meta(key, value)

    def get_clip_metadata(self, clip_id: str, key: str) -> str | None:
        """Get metadata from a clip."""
        clip = self.get_clip(clip_id)
        return clip.get_meta(key)

    def save(self) -> None:
        """Save timeline to XGES file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        uri = _path_to_uri(self._path)
        if not self._timeline.save_to_uri(uri, None, True):
            raise TimelineError(f"Failed to save timeline to {self._path}")

    # --- Internal helpers ---

    def _get_clip(self, clip_id: str) -> GES.Clip:
        """Internal clip lookup. Prefer get_clip() for external callers."""
        return self.get_clip(clip_id)

    def _get_effect(self, clip_id: str, effect_id: str) -> GES.Effect:
        """Look up an effect by its index-based ID."""
        self.get_clip(clip_id)  # validate clip exists
        index = self._parse_effect_index(effect_id)

        effects = self._effects.get(clip_id, [])
        if index < 0 or index >= len(effects) or effects[index] is None:
            raise TimelineError(f"Effect {effect_id} not found on {clip_id}")
        return effects[index]

    @staticmethod
    def _parse_effect_index(effect_id: str) -> int:
        """Extract the integer index from an effect ID like 'clip_0000_fx_2'."""
        try:
            return int(effect_id.split("_fx_")[-1])
        except (ValueError, IndexError):
            raise TimelineError(f"Invalid effect ID format: {effect_id}")



def _path_to_uri(path: Path) -> str:
    """Convert a Path to a file URI."""
    abs_path = str(path.resolve())
    return "file://" + quote(abs_path, safe="/")
