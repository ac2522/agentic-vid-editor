"""Tests for frame extraction — decode single video frames at timecodes."""

from pathlib import Path

import pytest

from ave.preview.frame import FrameError, compute_frame_timecode, extract_frame
from tests.conftest import FIXTURES_DIR, requires_ffmpeg


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip_path() -> Path:
    """Path to 5-second 1080p24 test clip."""
    path = FIXTURES_DIR / "av_clip_1080p24.mp4"
    if not path.exists():
        from tests.fixtures.generate import generate_av_clip

        generate_av_clip(path)
    return path


# ---------------------------------------------------------------------------
# Pure logic tests — compute_frame_timecode
# ---------------------------------------------------------------------------

class TestComputeFrameTimecode:
    def test_compute_frame_timecode(self):
        # 1 second = 1_000_000_000 ns -> "00:00:01.000"
        assert compute_frame_timecode(1_000_000_000) == "00:00:01.000"

    def test_compute_frame_timecode_zero(self):
        assert compute_frame_timecode(0) == "00:00:00.000"

    def test_compute_frame_timecode_subsecond(self):
        assert compute_frame_timecode(500_000_000) == "00:00:00.500"

    def test_compute_frame_timecode_minutes(self):
        assert compute_frame_timecode(90_000_000_000) == "00:01:30.000"

    def test_compute_frame_timecode_negative_raises(self):
        with pytest.raises(FrameError):
            compute_frame_timecode(-1)


# ---------------------------------------------------------------------------
# FFmpeg integration tests — extract_frame
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestExtractFrame:
    def test_extract_frame_returns_bytes(self, clip_path: Path):
        data = extract_frame(clip_path, timestamp_ns=1_000_000_000)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_extract_frame_is_valid_jpeg(self, clip_path: Path):
        data = extract_frame(clip_path, timestamp_ns=1_000_000_000)
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"

    def test_extract_frame_at_start(self, clip_path: Path):
        data = extract_frame(clip_path, timestamp_ns=0)
        assert data[:2] == b"\xff\xd8"

    def test_extract_frame_at_end(self, clip_path: Path):
        # 5s clip at 24fps: last frame at 5s - 1 frame = 5_000_000_000 - 41_666_667 ns
        last_frame_ns = 5_000_000_000 - 41_666_667
        data = extract_frame(clip_path, timestamp_ns=last_frame_ns)
        assert data[:2] == b"\xff\xd8"

    def test_extract_frame_to_file(self, clip_path: Path, tmp_path: Path):
        output = tmp_path / "frame.jpg"
        data = extract_frame(clip_path, timestamp_ns=1_000_000_000, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0
        file_bytes = output.read_bytes()
        assert file_bytes[:2] == b"\xff\xd8"
        assert data == file_bytes

    def test_extract_frame_custom_size(self, clip_path: Path):
        full = extract_frame(clip_path, timestamp_ns=1_000_000_000)
        small = extract_frame(clip_path, timestamp_ns=1_000_000_000, width=320)
        assert len(small) < len(full)
        assert small[:2] == b"\xff\xd8"

    def test_extract_frame_invalid_file(self, tmp_path: Path):
        bogus = tmp_path / "nonexistent.mp4"
        with pytest.raises(FrameError):
            extract_frame(bogus, timestamp_ns=0)

    def test_extract_frame_webp_format(self, clip_path: Path):
        data = extract_frame(clip_path, timestamp_ns=1_000_000_000, format="webp")
        # WebP files start with RIFF....WEBP
        assert data[:4] == b"RIFF"
        assert data[8:12] == b"WEBP"
