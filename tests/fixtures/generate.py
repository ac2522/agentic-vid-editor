"""Generate test media fixtures using FFmpeg.

All fixtures are synthetic (color bars, test tones) so tests don't depend
on real media files. Generated files are git-ignored.
"""

import subprocess
from pathlib import Path


def generate_color_bars(
    output: Path,
    duration: int = 5,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
) -> None:
    """Generate SMPTE color bars as H.264 MP4."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        capture_output=True,
        check=True,
    )


def generate_test_tone(
    output: Path,
    frequency: int = 1000,
    duration: int = 5,
    sample_rate: int = 48000,
) -> None:
    """Generate sine wave test tone as WAV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={duration}",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        check=True,
    )


def generate_av_clip(
    output: Path,
    duration: int = 5,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
    audio_freq: int = 440,
) -> None:
    """Generate clip with SMPTE color bars + sine tone."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={audio_freq}:sample_rate=48000:duration={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output),
        ],
        capture_output=True,
        check=True,
    )
