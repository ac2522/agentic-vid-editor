"""Media probe utility wrapping ffprobe."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class ProbeError(Exception):
    """Raised when media probing fails."""


@dataclass(frozen=True)
class VideoStream:
    width: int
    height: int
    codec: str
    pix_fmt: str
    fps: float
    bit_depth: int
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    duration_seconds: float


@dataclass(frozen=True)
class AudioStream:
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str | None
    duration_seconds: float


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    format_name: str
    duration_seconds: float
    size_bytes: int
    video: VideoStream | None
    audio: AudioStream | None

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def probe_media(path: Path) -> MediaInfo:
    """Probe a media file and return structured metadata.

    Raises ProbeError if the file doesn't exist or can't be probed.
    """
    if not path.exists():
        raise ProbeError(f"File does not exist: {path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ProbeError(f"ffprobe failed for {path}: {e.stderr}") from e

    data = json.loads(result.stdout)

    if "format" not in data:
        raise ProbeError(f"ffprobe returned no format info for {path}")

    fmt = data["format"]
    streams = data.get("streams", [])

    video = _parse_video_stream(streams)
    audio = _parse_audio_stream(streams)

    return MediaInfo(
        path=path,
        format_name=fmt.get("format_name", "unknown"),
        duration_seconds=float(fmt.get("duration", 0)),
        size_bytes=int(fmt.get("size", 0)),
        video=video,
        audio=audio,
    )


def _parse_fps(stream: dict) -> float:
    """Parse frame rate from stream, trying r_frame_rate then avg_frame_rate."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = stream.get(key, "0/1")
        if "/" in raw:
            frac = Fraction(raw)
            if frac > 0:
                return float(frac)
    return 0.0


def _parse_video_stream(streams: list[dict]) -> VideoStream | None:
    for s in streams:
        if s.get("codec_type") == "video":
            return VideoStream(
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                codec=s.get("codec_name", "unknown"),
                pix_fmt=s.get("pix_fmt", "unknown"),
                fps=_parse_fps(s),
                bit_depth=int(s.get("bits_per_raw_sample", 8) or 8),
                color_space=s.get("color_space"),
                color_transfer=s.get("color_transfer"),
                color_primaries=s.get("color_primaries"),
                duration_seconds=float(s.get("duration", 0) or 0),
            )
    return None


def _parse_audio_stream(streams: list[dict]) -> AudioStream | None:
    for s in streams:
        if s.get("codec_type") == "audio":
            return AudioStream(
                codec=s.get("codec_name", "unknown"),
                sample_rate=int(s.get("sample_rate", 0)),
                channels=int(s.get("channels", 0)),
                channel_layout=s.get("channel_layout"),
                duration_seconds=float(s.get("duration", 0) or 0),
            )
    return None
