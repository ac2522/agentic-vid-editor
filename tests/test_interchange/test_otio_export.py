"""Tests for OTIO export module."""

from __future__ import annotations

from pathlib import Path

import pytest

from ave.interchange.otio_export import (
    SUPPORTED_EXPORT_FORMATS,
    OTIOExportError,
    export_to_format,
    export_timeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS = 1_000_000_000  # 1 second in nanoseconds

try:
    import opentimelineio as _otio

    _HAS_OTIO = True
except ImportError:
    _otio = None  # type: ignore[assignment]
    _HAS_OTIO = False

requires_otio = pytest.mark.skipif(not _HAS_OTIO, reason="opentimelineio not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_timeline_data() -> dict:
    """Minimal AVE timeline data dict for testing."""
    return {
        "name": "Test Project",
        "duration_ns": 15 * _NS,
        "layers": [
            {
                "layer_index": 0,
                "clips": [
                    {
                        "name": "clip1",
                        "source_path": "/tmp/video1.mp4",
                        "start_ns": 0,
                        "duration_ns": 10 * _NS,
                        "in_point_ns": 0,
                    },
                    {
                        "name": "clip2",
                        "source_path": "/tmp/video2.mp4",
                        "start_ns": 10 * _NS,
                        "duration_ns": 5 * _NS,
                        "in_point_ns": 5 * _NS,
                    },
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests that do NOT require opentimelineio
# ---------------------------------------------------------------------------


class TestSupportedFormats:
    """Tests for SUPPORTED_EXPORT_FORMATS constant."""

    def test_contains_otio(self) -> None:
        assert ".otio" in SUPPORTED_EXPORT_FORMATS

    def test_contains_fcpxml(self) -> None:
        assert ".fcpxml" in SUPPORTED_EXPORT_FORMATS

    def test_contains_edl(self) -> None:
        assert ".edl" in SUPPORTED_EXPORT_FORMATS


class TestOTIOExportError:
    """Tests for the exception class."""

    def test_is_exception(self) -> None:
        assert issubclass(OTIOExportError, Exception)

    def test_message(self) -> None:
        err = OTIOExportError("boom")
        assert str(err) == "boom"


class TestExportToFormatValidation:
    """Tests for export_to_format validation (no OTIO needed for error path)."""

    def test_unsupported_format_raises(self, sample_timeline_data: dict) -> None:
        with pytest.raises(OTIOExportError, match="Unsupported export format"):
            export_to_format(sample_timeline_data, Path("/tmp/out.avi"))

    def test_unsupported_format_lists_supported(
        self, sample_timeline_data: dict
    ) -> None:
        with pytest.raises(OTIOExportError, match=r"\.otio"):
            export_to_format(sample_timeline_data, Path("/tmp/out.xyz"))


class TestExportTimelineWithoutOTIO:
    """Test that export_timeline raises a clear error when OTIO is missing."""

    @pytest.mark.skipif(_HAS_OTIO, reason="only tests the missing-OTIO path")
    def test_raises_on_missing_otio(
        self, sample_timeline_data: dict, tmp_path: Path
    ) -> None:
        with pytest.raises(OTIOExportError, match="not installed"):
            export_timeline(sample_timeline_data, tmp_path / "out.otio")


# ---------------------------------------------------------------------------
# Tests that REQUIRE opentimelineio
# ---------------------------------------------------------------------------


@requires_otio
class TestNsToRationalTime:
    """Tests for ns_to_rational_time conversion."""

    def test_zero(self) -> None:
        from ave.interchange.otio_export import ns_to_rational_time

        rt = ns_to_rational_time(0, fps=24.0)
        assert rt.value == 0.0
        assert rt.rate == 24.0

    def test_one_second(self) -> None:
        from ave.interchange.otio_export import ns_to_rational_time

        rt = ns_to_rational_time(_NS, fps=24.0)
        assert rt.value == pytest.approx(24.0)
        assert rt.rate == 24.0

    def test_half_second(self) -> None:
        from ave.interchange.otio_export import ns_to_rational_time

        rt = ns_to_rational_time(_NS // 2, fps=30.0)
        assert rt.value == pytest.approx(15.0)
        assert rt.rate == 30.0

    def test_different_fps(self) -> None:
        from ave.interchange.otio_export import ns_to_rational_time

        rt = ns_to_rational_time(2 * _NS, fps=25.0)
        assert rt.value == pytest.approx(50.0)
        assert rt.rate == 25.0


@requires_otio
class TestNsRangeToTimeRange:
    """Tests for ns_range_to_time_range conversion."""

    def test_basic_range(self) -> None:
        from ave.interchange.otio_export import ns_range_to_time_range

        tr = ns_range_to_time_range(0, 2 * _NS, fps=24.0)
        assert tr.start_time.value == pytest.approx(0.0)
        assert tr.duration.value == pytest.approx(48.0)

    def test_offset_range(self) -> None:
        from ave.interchange.otio_export import ns_range_to_time_range

        tr = ns_range_to_time_range(_NS, 3 * _NS, fps=24.0)
        assert tr.start_time.value == pytest.approx(24.0)
        assert tr.duration.value == pytest.approx(72.0)


@requires_otio
class TestClipToOtio:
    """Tests for clip_to_otio conversion."""

    def test_basic_clip(self) -> None:
        from ave.interchange.otio_export import clip_to_otio

        clip_data = {
            "name": "test_clip",
            "source_path": "/tmp/video.mp4",
            "start_ns": 0,
            "duration_ns": 5 * _NS,
            "in_point_ns": 0,
        }
        clip = clip_to_otio(clip_data, fps=24.0)

        assert clip.name == "test_clip"
        assert clip.source_range.duration.value == pytest.approx(120.0)
        assert clip.media_reference.target_url == "/tmp/video.mp4"

    def test_clip_with_in_point(self) -> None:
        from ave.interchange.otio_export import clip_to_otio

        clip_data = {
            "name": "trimmed",
            "source_path": "/tmp/video.mp4",
            "start_ns": 2 * _NS,
            "duration_ns": 3 * _NS,
            "in_point_ns": 1 * _NS,
        }
        clip = clip_to_otio(clip_data, fps=24.0)

        # source_range start should reflect in_point_ns
        assert clip.source_range.start_time.value == pytest.approx(24.0)
        assert clip.source_range.duration.value == pytest.approx(72.0)


@requires_otio
class TestLayerToOtioTrack:
    """Tests for layer_to_otio_track conversion."""

    def test_single_clip(self) -> None:
        from ave.interchange.otio_export import layer_to_otio_track

        clips = [
            {
                "name": "only",
                "source_path": "/tmp/v.mp4",
                "start_ns": 0,
                "duration_ns": 5 * _NS,
                "in_point_ns": 0,
            }
        ]
        track = layer_to_otio_track(0, clips, fps=24.0)
        assert track.name == "Layer 0"
        assert len(track) == 1

    def test_gap_inserted(self) -> None:
        from ave.interchange.otio_export import layer_to_otio_track

        clips = [
            {
                "name": "c1",
                "source_path": "/tmp/v.mp4",
                "start_ns": 0,
                "duration_ns": 2 * _NS,
                "in_point_ns": 0,
            },
            {
                "name": "c2",
                "source_path": "/tmp/v2.mp4",
                "start_ns": 5 * _NS,
                "duration_ns": 3 * _NS,
                "in_point_ns": 0,
            },
        ]
        track = layer_to_otio_track(0, clips, fps=24.0)

        import opentimelineio as otio

        # Should be: clip, gap, clip
        assert len(track) == 3
        assert isinstance(track[0], otio.schema.Clip)
        assert isinstance(track[1], otio.schema.Gap)
        assert isinstance(track[2], otio.schema.Clip)

    def test_contiguous_no_gap(self) -> None:
        from ave.interchange.otio_export import layer_to_otio_track

        clips = [
            {
                "name": "c1",
                "source_path": "/tmp/v.mp4",
                "start_ns": 0,
                "duration_ns": 3 * _NS,
                "in_point_ns": 0,
            },
            {
                "name": "c2",
                "source_path": "/tmp/v2.mp4",
                "start_ns": 3 * _NS,
                "duration_ns": 2 * _NS,
                "in_point_ns": 0,
            },
        ]
        track = layer_to_otio_track(0, clips, fps=24.0)
        # No gap between contiguous clips
        assert len(track) == 2


@requires_otio
class TestExportTimeline:
    """Tests for the main export_timeline function."""

    def test_writes_file(
        self, sample_timeline_data: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "test.otio"
        result = export_timeline(sample_timeline_data, out, fps=24.0)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_round_trip(
        self, sample_timeline_data: dict, tmp_path: Path
    ) -> None:
        import opentimelineio as otio

        out = tmp_path / "roundtrip.otio"
        export_timeline(sample_timeline_data, out, fps=24.0)

        # Read back with OTIO
        tl = otio.adapters.read_from_file(str(out))

        assert tl.name == "Test Project"
        assert len(tl.tracks) == 1

        track = tl.tracks[0]
        # Should have 2 clips (contiguous, no gap)
        clips = [c for c in track if isinstance(c, otio.schema.Clip)]
        assert len(clips) == 2

        # First clip: 10s at 24fps = 240 frames
        assert clips[0].name == "clip1"
        assert clips[0].source_range.duration.value == pytest.approx(240.0)

        # Second clip: 5s at 24fps = 120 frames
        assert clips[1].name == "clip2"
        assert clips[1].source_range.duration.value == pytest.approx(120.0)

    def test_creates_parent_dirs(
        self, sample_timeline_data: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "sub" / "dir" / "out.otio"
        result = export_timeline(sample_timeline_data, out, fps=24.0)
        assert result.exists()

    def test_empty_layers(self, tmp_path: Path) -> None:
        import opentimelineio as otio

        data = {"name": "Empty", "duration_ns": 0, "layers": []}
        out = tmp_path / "empty.otio"
        result = export_timeline(data, out, fps=24.0)
        assert result.exists()

        tl = otio.adapters.read_from_file(str(out))
        assert len(tl.tracks) == 0


@requires_otio
class TestExportToFormat:
    """Tests for multi-format dispatcher."""

    def test_otio_extension(
        self, sample_timeline_data: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "test.otio"
        result = export_to_format(sample_timeline_data, out, fps=24.0)
        assert result.exists()
