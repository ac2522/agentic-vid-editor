"""Tests for the deterministic ffprobe-backed judge."""
from pathlib import Path
import pytest
from tests.conftest import requires_ffmpeg


def test_deterministic_judge_implements_protocol():
    from ave.harness.judges.deterministic import DeterministicJudge
    j = DeterministicJudge()
    assert "static" in j.supported_dimension_types
    assert j.name == "ffprobe"


def test_deterministic_judge_known_dimensions():
    from ave.harness.judges.deterministic import DeterministicJudge, DETERMINISTIC_DIMENSIONS
    assert "duration" in DETERMINISTIC_DIMENSIONS
    assert "resolution" in DETERMINISTIC_DIMENSIONS
    assert "audio_rms" in DETERMINISTIC_DIMENSIONS


@requires_ffmpeg
def test_judge_duration_passes_within_tolerance(tmp_path):
    from ave.harness.fixtures.builder import build_lavfi_clip
    from ave.harness.judges.deterministic import DeterministicJudge

    clip = build_lavfi_clip("testsrc=size=320x240:rate=24", duration_seconds=2.0,
                             output_path=tmp_path / "c.mp4")
    j = DeterministicJudge()
    v = j.judge_dimension(
        rendered_path=clip,
        dimension="duration",
        prompt="duration target=2.0,tolerance=0.5",
        pass_threshold=0.5,
    )
    assert v.passed is True
    assert "duration" in v.dimension


@requires_ffmpeg
def test_judge_resolution_extracts_dimensions(tmp_path):
    from ave.harness.fixtures.builder import build_lavfi_clip
    from ave.harness.judges.deterministic import DeterministicJudge

    clip = build_lavfi_clip("testsrc=size=640x480:rate=24", duration_seconds=1.0,
                             output_path=tmp_path / "c.mp4")
    j = DeterministicJudge()
    v = j.judge_dimension(
        rendered_path=clip,
        dimension="resolution",
        prompt="expect=640x480",
        pass_threshold=0.5,
    )
    assert v.passed is True


def test_judge_unknown_dimension_returns_metadata():
    """Unknown dimensions return passed=False with explanation."""
    from ave.harness.judges.deterministic import DeterministicJudge
    j = DeterministicJudge()
    v = j.judge_dimension(
        rendered_path=Path("/nonexistent.mp4"),
        dimension="totally_unknown",
        prompt="x",
        pass_threshold=0.5,
    )
    assert v.judge_name == "ffprobe"
    # Either passed=False or score=0 indicating unsupported
    assert v.score == 0.0 or "unsupported" in v.explanation.lower()
