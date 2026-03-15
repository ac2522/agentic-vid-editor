"""Timeline info — clip identity and timeline state queries.

Pure functions that parse XGES XML to expose clip identities, positions,
durations, and track types to the agent. No GES/GStreamer dependency.

Reuses the XGES parsing logic from ave.web.timeline_model.
"""

from __future__ import annotations

from pathlib import Path


class TimelineInfoError(Exception):
    """Raised when timeline info operations fail."""


def list_timeline_clips(xges_path: str) -> list[dict]:
    """Parse an XGES file and return all clips with identity and position info.

    Returns a list of dicts sorted by (start_ns, layer), each containing:
    - clip_id: agent-assigned clip identifier
    - name: filename derived from asset URI
    - start_ns: timeline start position in nanoseconds
    - duration_ns: clip duration in nanoseconds
    - end_ns: start_ns + duration_ns
    - inpoint_ns: media inpoint offset in nanoseconds
    - layer: layer index (0 = bottom)
    - has_video: True if clip has a video track
    - has_audio: True if clip has an audio track
    - effects: list of effect asset-id strings

    Raises TimelineInfoError if the file doesn't exist or is malformed.
    """
    model = _load_model(xges_path)

    clips = []
    for clip in model._clip_index.values():
        clips.append({
            "clip_id": clip.clip_id,
            "name": clip.name,
            "start_ns": clip.start_ns,
            "duration_ns": clip.duration_ns,
            "end_ns": clip.end_ns,
            "inpoint_ns": clip.inpoint_ns,
            "layer": clip.layer_index,
            "has_video": clip.has_video,
            "has_audio": clip.has_audio,
            "effects": list(clip.effects),
        })

    clips.sort(key=lambda c: (c["start_ns"], c["layer"]))
    return clips


def get_timeline_info(xges_path: str) -> dict:
    """Parse an XGES file and return a summary of the timeline.

    Returns a dict with:
    - fps: timeline framerate
    - duration_ns: total timeline duration in nanoseconds
    - duration_seconds: total duration in seconds
    - layer_count: number of layers
    - clip_count: total number of clips

    Raises TimelineInfoError if the file doesn't exist or is malformed.
    """
    model = _load_model(xges_path)

    return {
        "fps": model.fps,
        "duration_ns": model.duration_ns,
        "duration_seconds": model.duration_ns / 1_000_000_000,
        "layer_count": len(model.layers),
        "clip_count": len(model._clip_index),
    }


def _load_model(xges_path: str):
    """Load a TimelineModel from an XGES path. Raises TimelineInfoError on failure."""
    from ave.web.timeline_model import TimelineModel

    path = Path(xges_path)
    if not path.exists():
        raise TimelineInfoError(f"XGES file not found: {xges_path}")
    try:
        return TimelineModel.load_from_xges(path)
    except (ValueError, OSError) as e:
        raise TimelineInfoError(f"Failed to parse XGES: {e}") from e
