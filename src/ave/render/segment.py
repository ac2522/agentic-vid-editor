"""Segment rendering — render sub-ranges of a timeline to fragmented MP4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ave.utils import path_to_uri


class SegmentError(Exception):
    """Raised when segment computation or rendering fails."""


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentBoundary:
    """A segment boundary within a timeline."""

    index: int
    start_ns: int
    end_ns: int


def compute_segment_boundaries(
    duration_ns: int,
    segment_duration_ns: int = 5_000_000_000,
) -> list[SegmentBoundary]:
    """Compute segment boundaries for a timeline.

    Args:
        duration_ns: Total timeline duration in nanoseconds.
        segment_duration_ns: Target segment length in nanoseconds (default 5s).

    Returns:
        Ordered list of SegmentBoundary covering the full duration.

    Raises:
        SegmentError: If duration_ns is zero or negative.
    """
    if duration_ns <= 0:
        raise SegmentError(
            f"Timeline duration must be positive, got {duration_ns} ns"
        )

    segments: list[SegmentBoundary] = []
    index = 0
    start = 0
    while start < duration_ns:
        end = min(start + segment_duration_ns, duration_ns)
        segments.append(SegmentBoundary(index=index, start_ns=start, end_ns=end))
        start = end
        index += 1

    return segments


def segment_filename(timeline_id: str, start_ns: int, end_ns: int) -> str:
    """Generate a filename for a rendered segment.

    Format: ``{timeline_id}_{start_ns}_{end_ns}.mp4``
    """
    return f"{timeline_id}_{start_ns}_{end_ns}.mp4"


# ---------------------------------------------------------------------------
# GES execution
# ---------------------------------------------------------------------------


def render_segment(
    xges_path: Path,
    output_path: Path,
    start_ns: int,
    end_ns: int,
    height: int = 480,
) -> None:
    """Render a segment of a timeline to fragmented MP4.

    Uses GES pipeline with seek to *start_ns*/*end_ns* positions.
    Output format: fragmented MP4 (``movflags=frag_keyframe+empty_moov``)
    for MSE compatibility.

    Raises:
        SegmentError: On invalid range or render failure.
    """
    # Validate range
    if start_ns >= end_ns:
        raise SegmentError(
            f"Invalid segment range: start_ns ({start_ns}) must be less than end_ns ({end_ns})"
        )

    # Late imports — keep module importable without GES installed
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GES", "1.0")
    gi.require_version("GstPbutils", "1.0")

    from gi.repository import GES, Gst, GstPbutils  # noqa: E402

    Gst.init(None)
    GES.init()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load timeline
    uri = path_to_uri(xges_path)
    project = GES.Project.new(uri)
    timeline = project.extract()

    if timeline is None:
        raise SegmentError(f"Failed to load timeline from {xges_path}")

    # Validate range against timeline duration
    timeline_duration = timeline.get_duration()
    if end_ns > timeline_duration:
        raise SegmentError(
            f"Segment end ({end_ns} ns) exceeds timeline duration ({timeline_duration} ns)"
        )

    # Build pipeline
    pipeline = GES.Pipeline()
    pipeline.set_timeline(timeline)

    # Encoding profile: H.264 video in fMP4 container
    # Use streamheader variant for fragmented MP4
    container_caps = Gst.Caps.from_string(
        "video/quicktime,variant=iso,streamheader=frag"
    )
    container_profile = GstPbutils.EncodingContainerProfile.new(
        "fmp4",
        None,
        container_caps,
        None,
    )

    video_caps = Gst.Caps.from_string(f"video/x-h264,height={height}")
    video_profile = GstPbutils.EncodingVideoProfile.new(
        video_caps,
        None,
        Gst.Caps.from_string(f"video/x-raw,height={height}"),
        0,
    )
    container_profile.add_profile(video_profile)

    audio_caps = Gst.Caps.from_string("audio/mpeg,mpegversion=4")
    audio_profile = GstPbutils.EncodingAudioProfile.new(
        audio_caps,
        None,
        None,
        0,
    )
    container_profile.add_profile(audio_profile)

    output_uri = path_to_uri(output_path)
    pipeline.set_render_settings(output_uri, container_profile)
    pipeline.set_mode(GES.PipelineFlags.RENDER)

    # Move to PAUSED first so we can seek before playing
    pipeline.set_state(Gst.State.PAUSED)
    bus = pipeline.get_bus()

    # Wait for PAUSED to complete (async state change)
    pipeline.get_state(Gst.CLOCK_TIME_NONE)

    # Seek to the desired segment range
    seek_flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE
    pipeline.seek(
        1.0,
        Gst.Format.TIME,
        seek_flags,
        Gst.SeekType.SET,
        start_ns,
        Gst.SeekType.SET,
        end_ns,
    )

    # Now play
    pipeline.set_state(Gst.State.PLAYING)

    while True:
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )
        if msg.type == Gst.MessageType.EOS:
            break
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            pipeline.set_state(Gst.State.NULL)
            raise SegmentError(f"Segment render failed: {err.message}\n{debug}")

    pipeline.set_state(Gst.State.NULL)

    if not output_path.exists():
        raise SegmentError(
            f"Segment render completed but output not found: {output_path}"
        )
