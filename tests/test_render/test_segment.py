"""Tests for segment rendering."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_ges

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GST_SECOND = 1_000_000_000  # 1 second in nanoseconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ===================================================================
# Pure logic tests (no GES needed)
# ===================================================================


class TestComputeSegmentBoundaries:
    """Tests for compute_segment_boundaries pure logic."""

    def test_compute_segment_boundaries_default_5s(self):
        from ave.render.segment import compute_segment_boundaries

        segments = compute_segment_boundaries(20 * GST_SECOND)
        assert len(segments) == 4
        for i, seg in enumerate(segments):
            assert seg.index == i
            assert seg.start_ns == i * 5 * GST_SECOND
            assert seg.end_ns == (i + 1) * 5 * GST_SECOND

    def test_compute_segment_boundaries_remainder(self):
        from ave.render.segment import compute_segment_boundaries

        segments = compute_segment_boundaries(12 * GST_SECOND)
        assert len(segments) == 3
        assert segments[0].start_ns == 0
        assert segments[0].end_ns == 5 * GST_SECOND
        assert segments[1].start_ns == 5 * GST_SECOND
        assert segments[1].end_ns == 10 * GST_SECOND
        assert segments[2].start_ns == 10 * GST_SECOND
        assert segments[2].end_ns == 12 * GST_SECOND

    def test_compute_segment_boundaries_shorter_than_segment(self):
        from ave.render.segment import compute_segment_boundaries

        segments = compute_segment_boundaries(3 * GST_SECOND)
        assert len(segments) == 1
        assert segments[0].index == 0
        assert segments[0].start_ns == 0
        assert segments[0].end_ns == 3 * GST_SECOND

    def test_compute_segment_boundaries_custom_duration(self):
        from ave.render.segment import compute_segment_boundaries

        segments = compute_segment_boundaries(
            10 * GST_SECOND, segment_duration_ns=2 * GST_SECOND
        )
        assert len(segments) == 5
        for i, seg in enumerate(segments):
            assert seg.index == i
            assert seg.start_ns == i * 2 * GST_SECOND
            assert seg.end_ns == (i + 1) * 2 * GST_SECOND

    def test_compute_segment_boundaries_empty_duration(self):
        from ave.render.segment import SegmentError, compute_segment_boundaries

        with pytest.raises(SegmentError):
            compute_segment_boundaries(0)

    def test_compute_segment_boundaries_negative_duration(self):
        from ave.render.segment import SegmentError, compute_segment_boundaries

        with pytest.raises(SegmentError):
            compute_segment_boundaries(-1 * GST_SECOND)


class TestSegmentFilename:
    """Tests for segment_filename pure logic."""

    def test_segment_filename(self):
        from ave.render.segment import segment_filename

        name = segment_filename("timeline_abc", 0, 5 * GST_SECOND)
        assert name == "timeline_abc_0_5000000000.mp4"


# ===================================================================
# GES integration tests
# ===================================================================


@requires_ges
@requires_ffmpeg
class TestRenderSegment:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def _create_timeline(self, duration_ns: int) -> Path:
        """Create a simple timeline with a single clip of the given duration."""
        from ave.project.timeline import Timeline

        xges_path = self.project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=duration_ns,
        )
        tl.save()
        return xges_path

    def test_render_segment_produces_file(self, tmp_path: Path):
        from ave.render.segment import render_segment

        xges_path = self._create_timeline(5 * GST_SECOND)
        output = tmp_path / "seg.mp4"

        render_segment(xges_path, output, start_ns=0, end_ns=2 * GST_SECOND)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_render_segment_duration_matches(self, tmp_path: Path):
        from ave.render.segment import render_segment

        xges_path = self._create_timeline(5 * GST_SECOND)
        output = tmp_path / "seg_dur.mp4"
        requested_duration = 2 * GST_SECOND

        render_segment(xges_path, output, start_ns=0, end_ns=requested_duration)

        probe = _probe(output)
        actual_duration = float(probe["format"]["duration"])
        expected_seconds = requested_duration / GST_SECOND
        # Allow 1 frame tolerance at 24fps (~0.042s)
        assert abs(actual_duration - expected_seconds) < 0.05

    def test_render_segment_invalid_range(self, tmp_path: Path):
        from ave.render.segment import SegmentError, render_segment

        xges_path = self._create_timeline(5 * GST_SECOND)
        output = tmp_path / "seg_invalid.mp4"

        with pytest.raises(SegmentError):
            render_segment(
                xges_path, output, start_ns=3 * GST_SECOND, end_ns=3 * GST_SECOND
            )

    def test_render_segment_range_exceeds_timeline(self, tmp_path: Path):
        from ave.render.segment import SegmentError, render_segment

        xges_path = self._create_timeline(5 * GST_SECOND)
        output = tmp_path / "seg_exceed.mp4"

        with pytest.raises(SegmentError):
            render_segment(
                xges_path, output, start_ns=0, end_ns=10 * GST_SECOND
            )
