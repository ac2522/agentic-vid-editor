"""Tests for the lavfi-based deterministic fixture builder."""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_build_testsrc_clip_creates_mp4(tmp_path: Path):
    from ave.harness.fixtures.builder import build_lavfi_clip

    out = tmp_path / "testsrc.mp4"
    built = build_lavfi_clip(
        expression="testsrc=size=320x240:rate=24",
        duration_seconds=1.0,
        output_path=out,
    )
    assert built == out
    assert out.exists()
    assert out.stat().st_size > 0


@requires_ffmpeg
def test_build_raises_on_bogus_expression(tmp_path: Path):
    from ave.harness.fixtures.builder import build_lavfi_clip

    with pytest.raises(RuntimeError, match="ffmpeg"):
        build_lavfi_clip(
            expression="lavfi_DEFINITELY_INVALID",
            duration_seconds=0.5,
            output_path=tmp_path / "x.mp4",
        )


def test_module_importable_without_ffmpeg():
    """The module should import without ffmpeg present (only callers fail)."""
    import ave.harness.fixtures.builder  # noqa: F401
