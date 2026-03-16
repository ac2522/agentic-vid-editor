"""E2E tests for 10-bit GPU color pipeline — requires Docker with GES and GPU.

Verifies that 10-bit source material passes through the GPU color pipeline
(placebofilter, ociofilter, glshader) without bit-depth truncation to 8-bit.

All tests are marked @pytest.mark.gpu and @requires_ges so they skip
automatically outside the Docker environment with GPU access.
"""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ges, requires_ffmpeg, requires_gpu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TENBIT_FORMAT = "I420_10LE"

# A minimal 2x2x2 .cube LUT that applies a slight warm shift.
# Keeps values close to identity so bit-depth verification is not confused
# by heavy colour shifts that could clip.
WARM_CUBE_LUT = (
    'TITLE "Warm Test LUT"\n'
    "LUT_3D_SIZE 2\n"
    "0.02 0.01 0.00\n"
    "1.00 0.01 0.00\n"
    "0.02 1.00 0.00\n"
    "1.00 1.00 0.00\n"
    "0.02 0.01 1.00\n"
    "1.00 0.01 1.00\n"
    "0.02 1.00 1.00\n"
    "1.00 1.00 1.00\n"
)


def create_10bit_test_source(output_path: Path, duration_ns: int = 1_000_000_000) -> Path:
    """Generate a short 10-bit test video using GStreamer's videotestsrc.

    Creates a 1920x1080 video with I420_10LE pixel format, encoded to FFV1
    (lossless) so bit depth is preserved exactly.

    Args:
        output_path: Where to write the output file.
        duration_ns: Duration in nanoseconds (default 1 second).

    Returns:
        The output path for chaining.
    """
    duration_sec = duration_ns / 1_000_000_000
    # Use GStreamer to produce a 10-bit source, encode lossless with FFV1
    cmd = [
        "gst-launch-1.0",
        "-e",
        "videotestsrc",
        f"num-buffers={int(duration_sec * 24)}",
        "pattern=smpte",
        "!",
        f"video/x-raw,format={TENBIT_FORMAT},width=1920,height=1080,framerate=24/1",
        "!",
        "videoconvert",
        "!",
        "avenc_ffv1",
        "!",
        "matroskamux",
        "!",
        "filesink",
        f"location={output_path}",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
    assert output_path.exists(), f"Failed to create 10-bit test source at {output_path}"
    return output_path


def probe_bit_depth(media_path: Path) -> int:
    """Probe a media file and return the video stream bit depth.

    Uses ffprobe to extract bits_per_raw_sample from the first video stream.

    Returns:
        Bit depth as integer (e.g. 8, 10, 12).
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {media_path}")

    stream = streams[0]
    # bits_per_raw_sample is the authoritative field; fall back to pix_fmt heuristic
    raw_bits = stream.get("bits_per_raw_sample")
    if raw_bits and int(raw_bits) > 0:
        return int(raw_bits)

    # Fallback: infer from pixel format name
    pix_fmt = stream.get("pix_fmt", "")
    if "10" in pix_fmt:
        return 10
    if "12" in pix_fmt:
        return 12
    return 8


def write_test_cube_lut(directory: Path, name: str = "warm_test.cube") -> Path:
    """Write a minimal .cube LUT file for testing and return its path."""
    lut_path = directory / name
    lut_path.write_text(WARM_CUBE_LUT)
    return lut_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_ges
@requires_ffmpeg
@requires_gpu
class TestTenBitPipeline:
    """Verify 10-bit passthrough in the GPU color pipeline.

    These tests create a 10-bit test source, push it through various color
    pipeline stages, render the output, and verify via ffprobe that the
    output retains >= 10-bit depth.
    """

    def test_placebofilter_preserves_10bit(self, tmp_path):
        """Render 10-bit source through placebofilter, verify output bit depth."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_lut

        # Create 10-bit source
        src = create_10bit_test_source(tmp_path / "src_10bit.mkv")

        # Build timeline
        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        clip_id = tl.add_clip(src, duration_ns=1_000_000_000)

        # Apply a LUT through placebofilter
        lut_path = write_test_cube_lut(tmp_path)
        apply_lut(tl, clip_id, str(lut_path))
        tl.save()

        # Render — use a lossless output to preserve bit depth for probing
        from ave.render.proxy import render_proxy

        output = tmp_path / "graded_10bit.mkv"
        render_proxy(xges_path, output)

        assert output.exists(), "Render output was not created"
        assert output.stat().st_size > 0, "Render output is empty"

        bit_depth = probe_bit_depth(output)
        assert bit_depth >= 10, (
            f"Expected >= 10-bit output from placebofilter, got {bit_depth}-bit. "
            f"The GPU pipeline may be truncating to 8-bit."
        )

    def test_ociofilter_preserves_10bit(self, tmp_path):
        """Render 10-bit source through ociofilter, verify output bit depth."""
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import apply_color_transform

        src = create_10bit_test_source(tmp_path / "src_10bit.mkv")

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        clip_id = tl.add_clip(src, duration_ns=1_000_000_000)

        # Apply a color-space transform through ociofilter.
        # Using ACES config spaces that should be available in the container.
        apply_color_transform(
            tl,
            clip_id,
            src_colorspace="ACES - ACEScg",
            dst_colorspace="Output - sRGB",
        )
        tl.save()

        from ave.render.proxy import render_proxy

        output = tmp_path / "ocio_10bit.mkv"
        render_proxy(xges_path, output)

        assert output.exists(), "Render output was not created"
        assert output.stat().st_size > 0, "Render output is empty"

        bit_depth = probe_bit_depth(output)
        assert bit_depth >= 10, (
            f"Expected >= 10-bit output from ociofilter, got {bit_depth}-bit. "
            f"The OCIO GPU pipeline may be truncating to 8-bit."
        )

    def test_full_color_chain_10bit(self, tmp_path):
        """IDT -> grade -> LUT -> ODT chain on 10-bit source, verify no truncation.

        This is the most realistic test: a full color pipeline with multiple
        stacked effects, ensuring the entire chain preserves high bit depth.
        """
        from ave.project.timeline import Timeline
        from ave.tools.color_ops import (
            apply_color_grade,
            apply_color_transform,
            apply_lut,
        )

        src = create_10bit_test_source(tmp_path / "src_10bit.mkv")

        xges_path = tmp_path / "test.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        clip_id = tl.add_clip(src, duration_ns=1_000_000_000)

        # Simulate a full color chain:
        # 1. IDT: camera space -> working space (via ociofilter)
        tl.set_clip_metadata(clip_id, "agent:camera-color-space", "ACES - ACEScg")
        apply_color_transform(
            tl,
            clip_id,
            src_colorspace="ACES - ACEScg",
            dst_colorspace="ACES - ACEScg",
        )

        # 2. Creative grade (via glshader)
        apply_color_grade(
            tl,
            clip_id,
            lift=(0.02, 0.0, -0.02),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.05, 1.0, 0.95),
            saturation=1.1,
        )

        # 3. LUT (via placebofilter)
        lut_path = write_test_cube_lut(tmp_path)
        apply_lut(tl, clip_id, str(lut_path))

        # 4. ODT: working space -> display space (via ociofilter)
        apply_color_transform(
            tl,
            clip_id,
            src_colorspace="ACES - ACEScg",
            dst_colorspace="Output - sRGB",
        )
        tl.save()

        from ave.render.proxy import render_proxy

        output = tmp_path / "full_chain_10bit.mkv"
        render_proxy(xges_path, output)

        assert output.exists(), "Render output was not created"
        assert output.stat().st_size > 0, "Render output is empty"

        bit_depth = probe_bit_depth(output)
        assert bit_depth >= 10, (
            f"Expected >= 10-bit output from full color chain, got {bit_depth}-bit. "
            f"One or more stages in the IDT->grade->LUT->ODT chain is truncating "
            f"to 8-bit."
        )

    def test_gl_texture_format_negotiation(self, tmp_path):
        """Verify GstGLFilter negotiates 10-bit capable caps.

        Checks that when a 10-bit source is connected to a GL filter element,
        the negotiated caps include a format that supports >= 10-bit precision
        (e.g. RGBA16F, RGB10_A2, RGBA16_LE, etc.).
        """
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError):
            pytest.skip("GStreamer Python bindings not available")

        Gst.init(None)

        # Build a minimal pipeline: videotestsrc ! glupload ! glcolorconvert ! fakesink
        # and inspect the negotiated caps on the glcolorconvert src pad.
        pipeline_str = (
            f"videotestsrc num-buffers=1 pattern=smpte "
            f"! video/x-raw,format={TENBIT_FORMAT},width=320,height=240,framerate=1/1 "
            f"! glupload "
            f"! glcolorconvert name=cc "
            f"! fakesink"
        )

        pipeline = Gst.parse_launch(pipeline_str)
        assert pipeline is not None, "Failed to create GStreamer pipeline"

        # Set to PAUSED to trigger caps negotiation
        ret = pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            pytest.skip("Could not set pipeline to PAUSED — GL context may not be available")

        # Wait for state change to complete
        pipeline.get_state(Gst.CLOCK_TIME_NONE)

        # Inspect negotiated caps on the glcolorconvert src pad
        cc = pipeline.get_by_name("cc")
        assert cc is not None, "Could not find glcolorconvert element"

        src_pad = cc.get_static_pad("src")
        caps = src_pad.get_current_caps()

        pipeline.set_state(Gst.State.NULL)

        if caps is None or caps.is_empty():
            pytest.skip("No caps negotiated — GL pipeline may not be functional")

        caps_str = caps.to_string()

        # These formats support >= 10-bit precision in GL memory
        tenbit_formats = [
            "RGBA16F",
            "RGBx16F",
            "RGB16F",
            "RGB10_A2",
            "RGB10A2_LE",
            "RGBA16_LE",
            "RGBA16_BE",
            "Y410",
            "Y412_LE",
            "P010_10LE",
            "P016_LE",
        ]

        has_tenbit = any(fmt in caps_str for fmt in tenbit_formats)
        assert has_tenbit, (
            f"GL caps do not include any 10-bit capable format. "
            f"Negotiated caps: {caps_str}. "
            f"Expected one of: {tenbit_formats}"
        )
