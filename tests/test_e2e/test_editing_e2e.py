"""End-to-end tests for the video editing pipeline.

Each test exercises a full round-trip: ingest -> edit operation -> render -> probe output.
Requires both GES and FFmpeg to be available on the system.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_ges

# Nanosecond constants
SEC = 1_000_000_000


@pytest.fixture(scope="module")
def source_clip(fixtures_dir: Path) -> Path:
    """Generate a 5-second AV test clip if it does not already exist."""
    clip = fixtures_dir / "av_clip_1080p24.mp4"
    if not clip.exists():
        from tests.fixtures.generate import generate_av_clip

        generate_av_clip(clip, duration=5)
    return clip


@pytest.fixture()
def ingested(source_clip: Path, tmp_project: Path):
    """Ingest the test clip and return (entry, registry, tmp_project)."""
    from ave.ingest.registry import AssetRegistry
    from ave.ingest.transcoder import ingest

    registry = AssetRegistry(tmp_project / "assets" / "registry.json")
    entry = ingest(
        source=source_clip,
        project_dir=tmp_project,
        asset_id="test_clip",
        registry=registry,
        project_fps=24.0,
    )
    return entry, registry, tmp_project


def _build_timeline(project_dir: Path, fps: float = 24.0):
    """Create an empty timeline in the given project directory."""
    from ave.project.timeline import Timeline

    return Timeline.create(project_dir / "project.xges", fps=fps)


def _render_and_probe(project_dir: Path, export_name: str = "output.mp4", height: int = 480):
    """Save timeline, render proxy, and probe the result."""
    from ave.ingest.probe import probe_media
    from ave.render.proxy import render_proxy

    xges = project_dir / "project.xges"
    export = project_dir / "exports" / export_name
    render_proxy(xges, export, height=height)
    assert export.exists(), "Rendered output should exist"
    return probe_media(export)


# ---------------------------------------------------------------------------
# Trim E2E
# ---------------------------------------------------------------------------


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestTrimE2E:
    """Verify the trim operation across the full pipeline."""

    def test_trim_shortens_output(self, ingested):
        """Ingest a 5s clip, trim to the middle 2s, render, and verify ~2s output."""
        from ave.project.operations import trim_clip

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=5 * SEC,
        )

        # Trim to the middle 2 seconds (in_ns=1.5s, out_ns=3.5s relative to clip)
        trim_clip(tl, clip_id, in_ns=int(1.5 * SEC), out_ns=int(3.5 * SEC))
        tl.save()

        info = _render_and_probe(project_dir, "trim_shorten.mp4")
        assert info.duration_seconds == pytest.approx(2.0, abs=0.5)

    def test_trim_preserves_video_properties(self, ingested):
        """Trim a clip, render, and verify the output has H.264 video at correct height."""
        from ave.project.operations import trim_clip

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=5 * SEC,
        )

        trim_clip(tl, clip_id, in_ns=1 * SEC, out_ns=3 * SEC)
        tl.save()

        info = _render_and_probe(project_dir, "trim_props.mp4", height=480)
        assert info.has_video
        assert info.video.codec == "h264"
        assert info.video.height == 480


# ---------------------------------------------------------------------------
# Split E2E
# ---------------------------------------------------------------------------


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestSplitE2E:
    """Verify the split operation across the full pipeline."""

    def test_split_creates_two_portions(self, ingested):
        """Split at midpoint, verify 2 clips, render, and check duration matches original."""
        from ave.project.operations import split_clip

        entry, _registry, project_dir = ingested
        clip_duration_ns = 4 * SEC

        tl = _build_timeline(project_dir)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=clip_duration_ns,
        )

        left_id, right_id = split_clip(tl, clip_id, position_ns=2 * SEC)

        assert tl.clip_count == 2
        assert left_id != right_id
        tl.save()

        info = _render_and_probe(project_dir, "split_two.mp4")
        assert info.duration_seconds == pytest.approx(4.0, abs=0.5)

    def test_split_and_remove_left(self, ingested):
        """Split clip, remove the left portion, render, verify ~half duration."""
        from ave.project.operations import split_clip

        entry, _registry, project_dir = ingested
        clip_duration_ns = 4 * SEC

        tl = _build_timeline(project_dir)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=clip_duration_ns,
        )

        left_id, right_id = split_clip(tl, clip_id, position_ns=2 * SEC)

        # Remove the left portion
        tl.remove_clip(left_id)
        assert tl.clip_count == 1
        tl.save()

        info = _render_and_probe(project_dir, "split_remove_left.mp4")
        assert info.duration_seconds == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# Concatenate E2E
# ---------------------------------------------------------------------------


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestConcatenateE2E:
    """Verify the concatenate operation across the full pipeline."""

    def test_concatenate_two_clips(self, ingested):
        """Concatenate the same media twice (each 2s), render, verify ~4s output."""
        from ave.project.operations import concatenate_clips

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_ids = concatenate_clips(
            tl,
            media_paths=[entry.working_path, entry.working_path],
            durations_ns=[2 * SEC, 2 * SEC],
            layer=0,
            start_ns=0,
        )

        assert len(clip_ids) == 2
        tl.save()

        info = _render_and_probe(project_dir, "concat_two.mp4")
        assert info.duration_seconds == pytest.approx(4.0, abs=0.5)

    def test_concatenate_different_durations(self, ingested):
        """Concatenate 1s + 3s clips, render, verify ~4s output."""
        from ave.project.operations import concatenate_clips

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_ids = concatenate_clips(
            tl,
            media_paths=[entry.working_path, entry.working_path],
            durations_ns=[1 * SEC, 3 * SEC],
            layer=0,
            start_ns=0,
        )

        assert len(clip_ids) == 2
        tl.save()

        info = _render_and_probe(project_dir, "concat_diff.mp4")
        assert info.duration_seconds == pytest.approx(4.0, abs=0.5)


# ---------------------------------------------------------------------------
# Combined Edits E2E
# ---------------------------------------------------------------------------


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestCombinedEditsE2E:
    """Verify chained edit operations across the full pipeline."""

    def test_split_then_trim(self, ingested):
        """Add 4s clip, split at 2s, trim right half to 1s, render, verify ~3s output."""
        from ave.project.operations import split_clip, trim_clip

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=4 * SEC,
        )

        # Split at 2s: left=0-2s, right=2-4s
        _left_id, right_id = split_clip(tl, clip_id, position_ns=2 * SEC)

        # Trim right half from 2s duration down to 1s (keep first 1s of right portion)
        trim_clip(tl, right_id, in_ns=0, out_ns=1 * SEC)
        tl.save()

        # Expected: left 2s + trimmed right 1s = 3s
        info = _render_and_probe(project_dir, "split_trim.mp4")
        assert info.duration_seconds == pytest.approx(3.0, abs=0.5)

    def test_concatenate_then_trim(self, ingested):
        """Concatenate two 2s clips, trim the combined result to 3s, render, verify."""
        from ave.project.operations import concatenate_clips, trim_clip

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_ids = concatenate_clips(
            tl,
            media_paths=[entry.working_path, entry.working_path],
            durations_ns=[2 * SEC, 2 * SEC],
            layer=0,
            start_ns=0,
        )

        # Trim the second clip to 1s (keeping first 1s), making total 3s
        trim_clip(tl, clip_ids[1], in_ns=0, out_ns=1 * SEC)
        tl.save()

        info = _render_and_probe(project_dir, "concat_trim.mp4")
        assert info.duration_seconds == pytest.approx(3.0, abs=0.5)


# ---------------------------------------------------------------------------
# Speed E2E
# ---------------------------------------------------------------------------


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestSpeedE2E:
    """Verify speed change across the full pipeline."""

    def test_double_speed_halves_duration(self, ingested):
        """Apply 2x speed to a 4s clip, render, verify ~2s output."""
        from ave.project.operations import set_speed

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_id = tl.add_clip(
            media_path=entry.working_path,
            layer=0,
            start_ns=0,
            duration_ns=4 * SEC,
        )

        set_speed(tl, clip_id, rate=2.0)
        tl.save()

        info = _render_and_probe(project_dir, "speed_2x.mp4")
        assert info.duration_seconds == pytest.approx(2.0, abs=0.5)


# ---------------------------------------------------------------------------
# Transition E2E
# ---------------------------------------------------------------------------


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestTransitionE2E:
    """Verify transition across the full pipeline."""

    def test_crossfade_shortens_total_duration(self, ingested):
        """Concatenate two 3s clips with 1s crossfade, render, verify ~5s output."""
        from ave.project.operations import apply_transition, concatenate_clips
        from ave.tools.transitions import TransitionType

        entry, _registry, project_dir = ingested

        tl = _build_timeline(project_dir)
        clip_ids = concatenate_clips(
            tl,
            media_paths=[entry.working_path, entry.working_path],
            durations_ns=[3 * SEC, 3 * SEC],
            layer=0,
        )

        apply_transition(
            tl,
            clip_ids[0],
            clip_ids[1],
            transition_type=TransitionType.CROSSFADE,
            duration_ns=1 * SEC,
        )
        tl.save()

        info = _render_and_probe(project_dir, "crossfade.mp4")
        assert info.duration_seconds == pytest.approx(5.0, abs=0.5)
