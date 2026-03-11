"""Tests for OTIO import functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ave.interchange.otio_import import (
    SUPPORTED_IMPORT_FORMATS,
    OTIOImportError,
    import_timeline,
)


# ---------------------------------------------------------------------------
# Tests that do NOT require opentimelineio
# ---------------------------------------------------------------------------


class TestSupportedFormats:
    def test_contains_otio(self):
        assert ".otio" in SUPPORTED_IMPORT_FORMATS

    def test_contains_fcpxml(self):
        assert ".fcpxml" in SUPPORTED_IMPORT_FORMATS

    def test_contains_edl(self):
        assert ".edl" in SUPPORTED_IMPORT_FORMATS

    def test_contains_aaf(self):
        assert ".aaf" in SUPPORTED_IMPORT_FORMATS


class TestImportErrors:
    def test_nonexistent_file_raises(self):
        with pytest.raises(OTIOImportError, match="File not found"):
            import_timeline(Path("/nonexistent/file.otio"))

    def test_unsupported_format_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            p = Path(f.name)
        try:
            with pytest.raises(OTIOImportError, match="Unsupported format"):
                import_timeline(p)
        finally:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests that REQUIRE opentimelineio
# ---------------------------------------------------------------------------

_needs_otio = pytest.mark.skipif(
    not pytest.importorskip.__module__,  # always True, just need a hook below
    reason="opentimelineio not installed",
)

try:
    import opentimelineio as otio

    _has_otio = True
except ImportError:
    _has_otio = False

needs_otio = pytest.mark.skipif(not _has_otio, reason="opentimelineio not installed")


def _make_simple_otio(tmp_path: Path, name: str = "TestTimeline") -> Path:
    """Create a minimal .otio file with one track and one clip."""
    tl = otio.schema.Timeline(name=name)
    track = otio.schema.Track(name="V1")
    clip = otio.schema.Clip(
        name="clip1",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 24),
            duration=otio.opentime.RationalTime(48, 24),
        ),
        media_reference=otio.schema.ExternalReference(
            target_url="file:///path/to/video.mp4"
        ),
    )
    track.append(clip)
    tl.tracks.append(track)
    out_path = tmp_path / "test.otio"
    otio.adapters.write_to_file(tl, str(out_path))
    return out_path


@needs_otio
class TestRationalTimeToNs:
    def test_one_second(self):
        from ave.interchange.otio_import import rational_time_to_ns

        rt = otio.opentime.RationalTime(24, 24)
        assert rational_time_to_ns(rt) == 1_000_000_000

    def test_half_second(self):
        from ave.interchange.otio_import import rational_time_to_ns

        rt = otio.opentime.RationalTime(12, 24)
        assert rational_time_to_ns(rt) == 500_000_000

    def test_two_frames_at_30fps(self):
        from ave.interchange.otio_import import rational_time_to_ns

        rt = otio.opentime.RationalTime(2, 30)
        expected = round(2 / 30 * 1_000_000_000)
        assert rational_time_to_ns(rt) == expected

    def test_zero(self):
        from ave.interchange.otio_import import rational_time_to_ns

        rt = otio.opentime.RationalTime(0, 24)
        assert rational_time_to_ns(rt) == 0


@needs_otio
class TestTimeRangeToNs:
    def test_basic_range(self):
        from ave.interchange.otio_import import time_range_to_ns

        tr = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(24, 24),
            duration=otio.opentime.RationalTime(48, 24),
        )
        start_ns, duration_ns = time_range_to_ns(tr)
        assert start_ns == 1_000_000_000
        assert duration_ns == 2_000_000_000


@needs_otio
class TestImportTimeline:
    def test_simple_import(self, tmp_path):
        otio_path = _make_simple_otio(tmp_path)
        result = import_timeline(otio_path)

        assert result["name"] == "TestTimeline"
        assert result["duration_ns"] == 2_000_000_000
        assert len(result["layers"]) == 1
        assert len(result["layers"][0]["clips"]) == 1

        clip = result["layers"][0]["clips"][0]
        assert clip["name"] == "clip1"
        assert clip["source_path"] == "/path/to/video.mp4"
        assert clip["duration_ns"] == 2_000_000_000
        assert clip["in_point_ns"] == 0

    def test_warnings_list_present(self, tmp_path):
        otio_path = _make_simple_otio(tmp_path)
        result = import_timeline(otio_path)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_generator_clip_skipped_with_warning(self, tmp_path):
        """Generator clips (color bars, etc.) should be skipped."""
        tl = otio.schema.Timeline(name="GenTest")
        track = otio.schema.Track(name="V1")
        gen_clip = otio.schema.Clip(
            name="color_bars",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 24),
                duration=otio.opentime.RationalTime(24, 24),
            ),
            media_reference=otio.schema.GeneratorReference(
                name="SMPTEBars",
                generator_kind="SMPTEBars",
            ),
        )
        track.append(gen_clip)
        tl.tracks.append(track)

        out_path = tmp_path / "gen_test.otio"
        otio.adapters.write_to_file(tl, str(out_path))

        result = import_timeline(out_path)
        assert len(result["layers"][0]["clips"]) == 0
        assert any("generator" in w.lower() for w in result["warnings"])

    def test_effects_produce_warnings(self, tmp_path):
        """Effects on clips should produce warnings."""
        tl = otio.schema.Timeline(name="FXTest")
        track = otio.schema.Track(name="V1")
        clip = otio.schema.Clip(
            name="clip_with_fx",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 24),
                duration=otio.opentime.RationalTime(24, 24),
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="file:///video.mp4"
            ),
        )
        clip.effects.append(
            otio.schema.Effect(name="Blur", effect_name="blur")
        )
        track.append(clip)
        tl.tracks.append(track)

        out_path = tmp_path / "fx_test.otio"
        otio.adapters.write_to_file(tl, str(out_path))

        result = import_timeline(out_path)
        # Clip is still imported, but effect generates warning
        assert len(result["layers"][0]["clips"]) == 1
        assert any("effect" in w.lower() for w in result["warnings"])

    def test_multi_track(self, tmp_path):
        """Multiple tracks become multiple layers."""
        tl = otio.schema.Timeline(name="MultiTrack")
        for i in range(3):
            track = otio.schema.Track(name=f"V{i+1}")
            clip = otio.schema.Clip(
                name=f"clip_{i}",
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, 24),
                    duration=otio.opentime.RationalTime(24, 24),
                ),
                media_reference=otio.schema.ExternalReference(
                    target_url=f"file:///video_{i}.mp4"
                ),
            )
            track.append(clip)
            tl.tracks.append(track)

        out_path = tmp_path / "multi.otio"
        otio.adapters.write_to_file(tl, str(out_path))

        result = import_timeline(out_path)
        assert len(result["layers"]) == 3
        for i, layer in enumerate(result["layers"]):
            assert layer["layer_index"] == i
            assert len(layer["clips"]) == 1

    def test_clip_with_inpoint(self, tmp_path):
        """Clip with non-zero source range start gets correct in_point_ns."""
        tl = otio.schema.Timeline(name="InPointTest")
        track = otio.schema.Track(name="V1")
        clip = otio.schema.Clip(
            name="trimmed",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(48, 24),  # 2 seconds in
                duration=otio.opentime.RationalTime(24, 24),
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="file:///video.mp4"
            ),
        )
        track.append(clip)
        tl.tracks.append(track)

        out_path = tmp_path / "inpoint.otio"
        otio.adapters.write_to_file(tl, str(out_path))

        result = import_timeline(out_path)
        clip_data = result["layers"][0]["clips"][0]
        assert clip_data["in_point_ns"] == 2_000_000_000
        assert clip_data["duration_ns"] == 1_000_000_000
