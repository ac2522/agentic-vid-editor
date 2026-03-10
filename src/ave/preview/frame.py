"""Frame extraction — decode single video frames at timecodes.

Uses FFmpeg subprocess. No GES dependency.
"""

import subprocess
import tempfile
from pathlib import Path


class FrameError(Exception):
    """Raised when frame extraction fails."""


def compute_frame_timecode(timestamp_ns: int) -> str:
    """Convert nanosecond timestamp to FFmpeg timecode string (HH:MM:SS.mmm).

    Args:
        timestamp_ns: Timestamp in nanoseconds (must be >= 0).

    Returns:
        Timecode string formatted as HH:MM:SS.mmm.

    Raises:
        FrameError: If timestamp is negative.
    """
    if timestamp_ns < 0:
        raise FrameError(f"Timestamp cannot be negative: {timestamp_ns}")
    total_ms = timestamp_ns // 1_000_000
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1_000
    ms = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def extract_frame(
    video_path: Path,
    timestamp_ns: int,
    output_path: Path | None = None,
    width: int | None = None,
    format: str = "jpeg",
) -> bytes:
    """Extract a single frame from a video at the given timestamp.

    Args:
        video_path: Path to video file.
        timestamp_ns: Timestamp in nanoseconds.
        output_path: If provided, save frame to this path. Otherwise use temp file.
        width: Optional output width (height scales proportionally).
        format: Output format — "jpeg" or "webp".

    Returns:
        Frame image as bytes.

    Raises:
        FrameError: If extraction fails.
    """
    timecode = compute_frame_timecode(timestamp_ns)
    video_path = Path(video_path)

    if not video_path.exists():
        raise FrameError(f"Video file not found: {video_path}")

    # Determine output format settings
    if format == "webp":
        suffix = ".webp"
        fmt_args = ["-f", "webp", "-c:v", "libwebp", "-quality", "80"]
    else:
        suffix = ".jpg"
        fmt_args = ["-f", "image2", "-c:v", "mjpeg", "-q:v", "2"]

    # Build filter chain
    vf_filters: list[str] = []
    if width is not None:
        vf_filters.append(f"scale={width}:-2")

    # Determine output destination
    use_temp = output_path is None
    if use_temp:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        dest = Path(tmp.name)
        tmp.close()
    else:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", timecode,
            "-i", str(video_path),
            "-frames:v", "1",
        ]
        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
        cmd.extend(fmt_args)
        cmd.append(str(dest))

        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )

        data = dest.read_bytes()
        if not data:
            raise FrameError(f"FFmpeg produced empty output for {video_path} at {timecode}")
        return data

    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise FrameError(
            f"FFmpeg failed extracting frame from {video_path} at {timecode}: {stderr}"
        ) from exc
    finally:
        if use_temp:
            dest.unlink(missing_ok=True)
