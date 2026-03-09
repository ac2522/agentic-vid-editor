"""Tests that verify test fixtures can be generated and are valid."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestFixtureGeneration:
    def test_generate_color_bars_h264(self, fixtures_dir: Path):
        """Generate 5-second 1080p24 color bars in H.264."""
        output = fixtures_dir / "color_bars_1080p24.mp4"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_color_bars

        generate_color_bars(output, duration=5, width=1920, height=1080, fps=24)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["width"] == 1920
        assert video["height"] == 1080
        assert video["codec_name"] == "h264"
        assert float(probe["format"]["duration"]) == pytest.approx(5.0, abs=0.5)

    def test_generate_color_bars_30fps(self, fixtures_dir: Path):
        """Generate 3-second 720p30 color bars for mixed-fps testing."""
        output = fixtures_dir / "color_bars_720p30.mp4"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_color_bars

        generate_color_bars(output, duration=3, width=1280, height=720, fps=30)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["width"] == 1280
        assert video["height"] == 720

    def test_generate_test_tone(self, fixtures_dir: Path):
        """Generate 5-second 1kHz test tone."""
        output = fixtures_dir / "test_tone_1khz.wav"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_test_tone

        generate_test_tone(output, frequency=1000, duration=5)

        assert output.exists()
        probe = _probe(output)
        audio = _audio_stream(probe)
        assert audio["codec_name"] == "pcm_s16le"
        assert int(audio["sample_rate"]) == 48000

    def test_generate_av_clip(self, fixtures_dir: Path):
        """Generate clip with both video and audio."""
        output = fixtures_dir / "av_clip_1080p24.mp4"
        if output.exists():
            output.unlink()

        from tests.fixtures.generate import generate_av_clip

        generate_av_clip(output, duration=5, width=1920, height=1080, fps=24)

        assert output.exists()
        probe = _probe(output)
        streams = probe["streams"]
        codec_types = {s["codec_type"] for s in streams}
        assert "video" in codec_types
        assert "audio" in codec_types


def _probe(path: Path) -> dict:
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
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


def _audio_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "audio")
