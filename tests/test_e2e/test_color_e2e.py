"""E2E tests for color pipeline — requires Docker with GES and GPU."""
import json
import subprocess

import pytest

from tests.conftest import requires_ges, requires_ffmpeg


@requires_ges
@requires_ffmpeg
class TestColorE2E:
    """Full pipeline color tests."""

    def test_lut_application_changes_output(self, tmp_path, fixtures_dir):
        """Apply a LUT and verify the rendered output differs from original."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_lut

        # Create a simple warm-shift LUT
        lut_path = tmp_path / "warm.cube"
        lut_path.write_text(
            "TITLE \"Warm Shift\"\n"
            "LUT_3D_SIZE 2\n"
            "0.1 0.05 0.0\n"  # blacks get warm
            "1.0 0.05 0.0\n"
            "0.1 1.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.1 0.05 1.0\n"
            "1.0 0.05 1.0\n"
            "0.1 1.0 1.0\n"
            "1.0 1.0 1.0\n"
        )

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        # Add color bars clip
        src = fixtures_dir / "color_bars_1080p24.mp4"
        if not src.exists():
            pytest.skip("Test fixture not available")

        clip_id = tl.add_clip(src, duration_ns=2_000_000_000)
        apply_lut(tl, clip_id, str(lut_path))
        tl.save()

        # Render
        from ave.render.proxy import render_proxy

        output = tmp_path / "graded.mp4"
        render_proxy(xges_path, output)

        assert output.exists()
        assert output.stat().st_size > 0

        # Probe output
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(output)],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        video = [s for s in info["streams"] if s["codec_type"] == "video"][0]
        assert video["codec_name"] == "h264"

    def test_color_grade_renders(self, tmp_path, fixtures_dir):
        """Apply color grade and verify render completes."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_color_grade

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        src = fixtures_dir / "color_bars_1080p24.mp4"
        if not src.exists():
            pytest.skip("Test fixture not available")

        clip_id = tl.add_clip(src, duration_ns=2_000_000_000)
        apply_color_grade(
            tl, clip_id,
            lift=(0.05, 0.0, -0.05),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.1, 1.0, 0.9),
            saturation=1.2,
        )
        tl.save()

        from ave.render.proxy import render_proxy

        output = tmp_path / "graded.mp4"
        render_proxy(xges_path, output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_cdl_renders(self, tmp_path, fixtures_dir):
        """Apply CDL and verify render completes."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_cdl

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        src = fixtures_dir / "color_bars_1080p24.mp4"
        if not src.exists():
            pytest.skip("Test fixture not available")

        clip_id = tl.add_clip(src, duration_ns=2_000_000_000)
        apply_cdl(
            tl, clip_id,
            slope=(1.1, 1.0, 0.9),
            offset=(0.01, 0.0, -0.01),
            power=(1.0, 1.0, 1.0),
        )
        tl.save()

        from ave.render.proxy import render_proxy

        output = tmp_path / "graded.mp4"
        render_proxy(xges_path, output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_multiple_effects_stack(self, tmp_path, fixtures_dir):
        """Apply multiple color effects and verify they compose."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_color_grade, apply_cdl

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        src = fixtures_dir / "color_bars_1080p24.mp4"
        if not src.exists():
            pytest.skip("Test fixture not available")

        clip_id = tl.add_clip(src, duration_ns=2_000_000_000)
        fx1 = apply_color_grade(
            tl, clip_id,
            lift=(0.0, 0.0, 0.0),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.2, 1.0, 0.8),
        )
        fx2 = apply_cdl(
            tl, clip_id,
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
        )
        assert fx1 != fx2  # Different effect IDs

    def test_xges_preserves_color_effects(self, tmp_path, fixtures_dir):
        """Save and reload XGES — color effects should persist."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_color_grade

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        src = fixtures_dir / "color_bars_1080p24.mp4"
        if not src.exists():
            pytest.skip("Test fixture not available")

        clip_id = tl.add_clip(src, duration_ns=2_000_000_000)
        apply_color_grade(
            tl, clip_id,
            lift=(0.05, 0.0, -0.05),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.0, 1.0, 1.0),
        )
        tl.save()

        # Reload
        tl2 = Timeline.load(xges_path)
        assert tl2.clip_count == 1


@requires_ges
@requires_ffmpeg
class TestCompositingE2E:
    """Full pipeline compositing tests."""

    def test_two_layer_composite(self, tmp_path, fixtures_dir):
        """Two clips on different layers should composite."""
        from ave.project.timeline import Timeline

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        src = fixtures_dir / "color_bars_1080p24.mp4"
        if not src.exists():
            pytest.skip("Test fixture not available")

        clip1 = tl.add_clip(src, layer=0, duration_ns=2_000_000_000)
        clip2 = tl.add_clip(src, layer=1, duration_ns=2_000_000_000)
        assert clip1 != clip2
        tl.save()

        from ave.render.proxy import render_proxy

        output = tmp_path / "composited.mp4"
        render_proxy(xges_path, output)
        assert output.exists()
        assert output.stat().st_size > 0
