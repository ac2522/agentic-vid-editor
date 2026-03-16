"""OTIO export — convert AVE timeline data to OpenTimelineIO format.

All times internally are nanoseconds. This module converts to OTIO
RationalTime/TimeRange at the specified fps.

OpenTimelineIO is an optional dependency; imports are lazy so the module
can be loaded even when ``opentimelineio`` is not installed.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NS_PER_SECOND: float = 1_000_000_000.0

SUPPORTED_EXPORT_FORMATS: set[str] = {".otio", ".fcpxml", ".edl"}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class OTIOExportError(Exception):
    """Raised when OTIO export operations fail."""


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------


def ns_to_rational_time(ns: int, fps: float = 24.0) -> "otio.opentime.RationalTime":  # type: ignore[name-defined]  # noqa: F821
    """Convert nanoseconds to OTIO RationalTime.

    The value is expressed as a frame count at the given *fps* rate so that
    OTIO can round-trip accurately.
    """
    import opentimelineio as otio  # lazy import

    seconds = ns / _NS_PER_SECOND
    value = seconds * fps
    return otio.opentime.RationalTime(value=value, rate=fps)


def ns_range_to_time_range(
    start_ns: int,
    duration_ns: int,
    fps: float = 24.0,
) -> "otio.opentime.TimeRange":  # type: ignore[name-defined]  # noqa: F821
    """Convert nanosecond start + duration to an OTIO TimeRange."""
    import opentimelineio as otio  # lazy import

    start_rt = ns_to_rational_time(start_ns, fps)
    duration_rt = ns_to_rational_time(duration_ns, fps)
    return otio.opentime.TimeRange(start_time=start_rt, duration=duration_rt)


# ---------------------------------------------------------------------------
# Clip conversion
# ---------------------------------------------------------------------------


def clip_to_otio(clip_data: dict, fps: float = 24.0) -> "otio.schema.Clip":  # type: ignore[name-defined]  # noqa: F821
    """Convert an AVE clip dict to an OTIO Clip.

    Expected *clip_data* keys:
        name, source_path, start_ns, duration_ns, in_point_ns
    """
    import opentimelineio as otio  # lazy import

    name = clip_data.get("name", "unnamed")
    source_path = clip_data.get("source_path", "")
    clip_data["start_ns"]
    duration_ns: int = clip_data["duration_ns"]
    in_point_ns: int = clip_data.get("in_point_ns", 0)

    # Source range: where in the media file to read from
    source_range = ns_range_to_time_range(in_point_ns, duration_ns, fps)

    media_ref = otio.schema.ExternalReference(
        target_url=str(source_path),
    )

    clip = otio.schema.Clip(
        name=name,
        media_reference=media_ref,
        source_range=source_range,
    )
    return clip


# ---------------------------------------------------------------------------
# Track / layer conversion
# ---------------------------------------------------------------------------


def layer_to_otio_track(
    layer_index: int,
    clips: list[dict],
    fps: float = 24.0,
) -> "otio.schema.Track":  # type: ignore[name-defined]  # noqa: F821
    """Convert a GES layer's clips to an OTIO Track.

    Clips are sorted by ``start_ns`` and gaps between them are filled with
    OTIO Gap objects so that timeline positions are preserved.
    """
    import opentimelineio as otio  # lazy import

    track = otio.schema.Track(name=f"Layer {layer_index}")

    sorted_clips = sorted(clips, key=lambda c: c["start_ns"])
    current_ns: int = 0

    for clip_data in sorted_clips:
        clip_start: int = clip_data["start_ns"]

        # Insert a gap if there is space before this clip
        if clip_start > current_ns:
            gap_duration = clip_start - current_ns
            gap_range = ns_range_to_time_range(0, gap_duration, fps)
            track.append(otio.schema.Gap(source_range=gap_range))

        otio_clip = clip_to_otio(clip_data, fps)
        track.append(otio_clip)
        current_ns = clip_start + clip_data["duration_ns"]

    return track


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_timeline(
    timeline_data: dict,
    output_path: Path,
    fps: float = 24.0,
) -> Path:
    """Export AVE timeline data to an OTIO file.

    Args:
        timeline_data: dict with keys ``name``, ``duration_ns``, ``layers``.
            Each layer has ``layer_index`` and ``clips`` list.
        output_path: Destination file (typically ``.otio``).
        fps: Frame rate for time conversion.

    Returns:
        The *output_path* after writing.

    Raises:
        OTIOExportError: On any export failure.
    """
    try:
        import opentimelineio as otio  # lazy import
    except ImportError as exc:
        raise OTIOExportError(
            "opentimelineio is required for OTIO export but is not installed"
        ) from exc

    try:
        name = timeline_data.get("name", "Untitled")
        layers = timeline_data.get("layers", [])

        otio_timeline = otio.schema.Timeline(name=name)

        for layer_data in layers:
            layer_index: int = layer_data["layer_index"]
            clips: list[dict] = layer_data.get("clips", [])
            track = layer_to_otio_track(layer_index, clips, fps)
            otio_timeline.tracks.append(track)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(otio_timeline, str(output_path))

        return output_path

    except OTIOExportError:
        raise
    except Exception as exc:
        raise OTIOExportError(f"Failed to export timeline: {exc}") from exc


# ---------------------------------------------------------------------------
# Multi-format dispatcher
# ---------------------------------------------------------------------------


def export_to_format(
    timeline_data: dict,
    output_path: Path,
    fps: float = 24.0,
) -> Path:
    """Export to any supported format (determined by file extension).

    Raises:
        OTIOExportError: If the extension is not in ``SUPPORTED_EXPORT_FORMATS``.
    """
    output_path = Path(output_path)
    ext = output_path.suffix.lower()

    if ext not in SUPPORTED_EXPORT_FORMATS:
        raise OTIOExportError(
            f"Unsupported export format: {ext!r}. Supported: {sorted(SUPPORTED_EXPORT_FORMATS)}"
        )

    return export_timeline(timeline_data, output_path, fps)
