"""Proxy rendering via GES pipeline."""

from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GES", "1.0")
gi.require_version("GstPbutils", "1.0")

from gi.repository import GES, Gst, GstPbutils  # noqa: E402

from ave.project.timeline import _path_to_uri  # noqa: E402

Gst.init(None)
GES.init()


class RenderError(Exception):
    """Raised when rendering fails."""


def render_proxy(
    xges_path: Path,
    output_path: Path,
    height: int = 480,
    video_bitrate: int = 2_000_000,
) -> None:
    """Render an XGES timeline to an H.264 proxy MP4.

    Uses GES.Pipeline for native GStreamer rendering.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    uri = _path_to_uri(xges_path)
    project = GES.Project.new(uri)
    timeline = project.extract()

    if timeline is None:
        raise RenderError(f"Failed to load timeline from {xges_path}")

    pipeline = GES.Pipeline()
    pipeline.set_timeline(timeline)

    # Build encoding profile: H.264 video + AAC audio in MP4
    container_profile = GstPbutils.EncodingContainerProfile.new(
        "mp4",
        None,
        Gst.Caps.from_string("video/quicktime,variant=iso"),
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

    output_uri = _path_to_uri(output_path)
    pipeline.set_render_settings(output_uri, container_profile)
    pipeline.set_mode(GES.PipelineFlags.RENDER)

    # Run the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    bus = pipeline.get_bus()
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
            raise RenderError(f"Render failed: {err.message}\n{debug}")

    pipeline.set_state(Gst.State.NULL)

    if not output_path.exists():
        raise RenderError(f"Render completed but output not found: {output_path}")
