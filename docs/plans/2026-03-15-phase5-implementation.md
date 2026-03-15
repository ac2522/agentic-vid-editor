# Phase 5: GPU Color Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Full GPU color pipeline — libplacebo tone mapping/gamut mapping, OCIO 2.5.1 multi-texture + ACES 2.0 configs, and compound GES.Effect for zero-copy float-precision IDT→grade→ODT chain.

**Architecture:** Compound `GES.Effect` bin description chains `ociofilter[IDT] ! placebofilter ! ociofilter[ODT]` with `glupload`/`gldownload` bookends. All properties baked into bin string at construction time. Internal FBOs use `GL_RGBA16F` for float precision. Pure logic layer validates parameters; operations layer builds bin descriptions and manages GES effects.

**Tech Stack:** C (GstGLFilter, libplacebo v7.360.0), C++17 (OCIO 2.5.1), Python 3.12+ (PyGObject/GES), meson build system.

**Design doc:** `docs/plans/2026-03-15-phase5-gpu-color-pipeline-design.md`

---

## Task 0: Validate compound GL bin in GES [BLOCKING]

> This task validates that the compound `glupload ! ... ! gldownload` approach works inside GES/NLE. If it fails, the architecture must be redesigned before proceeding.

**Files:**
- Create: `tests/test_project/test_gl_compound_bin.py`

**Step 1: Write the validation test**

```python
"""Validate that compound GL bin descriptions work as GES.Effect in NLE pipeline."""

import pytest
from tests.conftest import requires_ges, requires_ffmpeg, requires_gpu


@requires_ges
@requires_ffmpeg
@requires_gpu
class TestCompoundGLBin:
    """Verify GL context propagation in compound GES.Effect bins.

    This is a blocking validation for the Phase 5 compound effect architecture.
    If these tests fail, the compound approach must be abandoned in favor of
    individual effects (with 8-bit boundary precision loss).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir, tmp_project):
        from pathlib import Path

        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_single_glupload_gldownload_effect(self):
        """Verify a simple glupload ! identity ! gldownload bin works as GES.Effect."""
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0, duration_ns=1_000_000_000,
        )

        # Simplest possible GL roundtrip: upload to GPU, convert, download
        effect_id = tl.add_effect(clip_id, "glupload ! glcolorconvert ! gldownload")
        assert effect_id is not None

    def test_compound_gl_bin_with_identity(self):
        """Verify compound GL bin with glcolorconvert (identity transform) renders."""
        from ave.project.timeline import Timeline
        from ave.render.segment import render_segment

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0, duration_ns=1_000_000_000,
        )

        # Chain two GL elements — validates GL context sharing in NLE
        tl.add_effect(
            clip_id,
            "glupload ! glcolorconvert ! glcolorconvert ! gldownload"
        )

        # Render a short segment — if GL context propagation fails, this will error
        out_path = self.project / "exports" / "gl_test.mp4"
        render_segment(
            tl, start_ns=0, end_ns=500_000_000, output_path=out_path,
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
```

**Step 2: Run test to verify it fails (requires Docker+GPU)**

Run: `python -m pytest tests/test_project/test_gl_compound_bin.py -v`
Expected: SKIP (requires GES+GPU, only runs in Docker)

In Docker: `pytest tests/test_project/test_gl_compound_bin.py -v`
Expected: PASS if compound GL bins work in GES, FAIL if GL context propagation is broken.

**Step 3: Commit**

```bash
git add tests/test_project/test_gl_compound_bin.py
git commit -m "test: Task 0 — validate compound GL bin in GES (blocking)"
```

**Decision gate:** If tests pass in Docker → proceed with compound effect architecture. If tests fail → redesign to use individual effects per element.

---

## Task 1: Pure logic — new dataclasses and validation (`tools/color.py`)

**Files:**
- Modify: `src/ave/tools/color.py`
- Modify: `tests/test_tools/test_color.py`

**Step 1: Write the failing tests**

Add to `tests/test_tools/test_color.py`:

```python
class TestToneMappingParams:
    """Test tone mapping parameter validation."""

    def test_valid_defaults(self):
        from ave.tools.color import compute_tone_mapping

        result = compute_tone_mapping()
        assert result.algorithm == "spline"
        assert result.gamut_mapping == "perceptual"
        assert result.peak_detect is False

    def test_valid_bt2390(self):
        from ave.tools.color import compute_tone_mapping

        result = compute_tone_mapping(algorithm="bt2390", gamut_mapping="softclip")
        assert result.algorithm == "bt2390"
        assert result.gamut_mapping == "softclip"

    def test_invalid_algorithm(self):
        from ave.tools.color import compute_tone_mapping, ColorError

        with pytest.raises(ColorError, match="Unknown tone mapping algorithm"):
            compute_tone_mapping(algorithm="invalid_algo")

    def test_invalid_gamut_mapping(self):
        from ave.tools.color import compute_tone_mapping, ColorError

        with pytest.raises(ColorError, match="Unknown gamut mapping"):
            compute_tone_mapping(gamut_mapping="invalid_gm")

    def test_all_valid_algorithms(self):
        from ave.tools.color import compute_tone_mapping, VALID_TONE_MAPPING

        for algo in VALID_TONE_MAPPING:
            result = compute_tone_mapping(algorithm=algo)
            assert result.algorithm == algo


class TestColorPipelineParams:
    """Test color pipeline parameter validation."""

    def test_valid_defaults(self):
        from ave.tools.color import compute_color_pipeline

        result = compute_color_pipeline(idt_src="ARRI LogC4")
        assert result.idt_src == "ARRI LogC4"
        assert result.working_space == "ACEScg"
        assert result.odt_display == "sRGB"
        assert result.odt_view == "ACES 2.0 - SDR Video"
        assert result.config_path == "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5"

    def test_empty_idt_src_raises(self):
        from ave.tools.color import compute_color_pipeline, ColorError

        with pytest.raises(ColorError, match="idt_src must be a non-empty string"):
            compute_color_pipeline(idt_src="")

    def test_empty_working_space_raises(self):
        from ave.tools.color import compute_color_pipeline, ColorError

        with pytest.raises(ColorError, match="working_space must be a non-empty string"):
            compute_color_pipeline(idt_src="LogC4", working_space="")

    def test_custom_config_path(self):
        from ave.tools.color import compute_color_pipeline

        result = compute_color_pipeline(
            idt_src="LogC4", config_path="ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
        )
        assert result.config_path == "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"

    def test_file_config_path_missing_raises(self):
        from ave.tools.color import compute_color_pipeline, ColorError

        with pytest.raises(ColorError, match="OCIO config file not found"):
            compute_color_pipeline(idt_src="LogC4", config_path="/nonexistent/config.ocio")


class TestColorTransformOcioUri:
    """Test that compute_color_transform accepts ocio:// URIs."""

    def test_ocio_uri_accepted(self):
        from ave.tools.color import compute_color_transform

        result = compute_color_transform(
            src_colorspace="ACEScg",
            dst_colorspace="sRGB",
            config_path="ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
        )
        assert result.config_path == "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5"

    def test_file_path_still_validated(self):
        from ave.tools.color import compute_color_transform, ColorError

        with pytest.raises(ColorError, match="OCIO config file not found"):
            compute_color_transform("ACEScg", "sRGB", config_path="/nonexistent.ocio")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tools/test_color.py::TestToneMappingParams -v`
Expected: FAIL (classes/functions don't exist yet)

Run: `python -m pytest tests/test_tools/test_color.py::TestColorPipelineParams -v`
Expected: FAIL

Run: `python -m pytest tests/test_tools/test_color.py::TestColorTransformOcioUri -v`
Expected: FAIL

**Step 3: Implement the dataclasses and validation**

Add to `src/ave/tools/color.py` after existing constants/imports:

```python
# ---------------------------------------------------------------------------
# Validation sets (for libplacebo enum properties)
# ---------------------------------------------------------------------------

VALID_TONE_MAPPING = frozenset({
    "spline", "bt2390", "bt2446a", "reinhard", "mobius",
    "hable", "gamma", "linear", "clip",
})

VALID_GAMUT_MAPPING = frozenset({
    "perceptual", "softclip", "relative", "absolute",
    "desaturate", "darken", "highlight", "linear",
})

VALID_PRIMARIES = frozenset({
    "bt709", "bt2020", "display-p3", "ap0", "ap1",
    "bt601-525", "bt601-625",
})

VALID_TRANSFERS = frozenset({
    "srgb", "bt1886", "pq", "hlg", "vlog", "slog2",
    "linear", "gamma22", "gamma28",
})


# ---------------------------------------------------------------------------
# New dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToneMappingParams:
    """Parameters for libplacebo tone/gamut mapping."""

    algorithm: str
    gamut_mapping: str
    peak_detect: bool


@dataclass(frozen=True)
class ColorPipelineParams:
    """Parameters for the full IDT→grade→ODT color pipeline."""

    idt_src: str
    working_space: str
    odt_display: str
    odt_view: str
    config_path: str


# ---------------------------------------------------------------------------
# compute_tone_mapping
# ---------------------------------------------------------------------------

def compute_tone_mapping(
    algorithm: str = "spline",
    gamut_mapping: str = "perceptual",
    peak_detect: bool = False,
) -> ToneMappingParams:
    """Validate and compute tone mapping parameters.

    Args:
        algorithm: Tone mapping algorithm name (must be in VALID_TONE_MAPPING).
        gamut_mapping: Gamut mapping mode (must be in VALID_GAMUT_MAPPING).
        peak_detect: Enable dynamic HDR peak detection.

    Returns:
        ToneMappingParams with validated values.

    Raises:
        ColorError: If algorithm or gamut_mapping is unknown.
    """
    if algorithm not in VALID_TONE_MAPPING:
        raise ColorError(
            f"Unknown tone mapping algorithm '{algorithm}'. "
            f"Valid: {sorted(VALID_TONE_MAPPING)}"
        )
    if gamut_mapping not in VALID_GAMUT_MAPPING:
        raise ColorError(
            f"Unknown gamut mapping '{gamut_mapping}'. "
            f"Valid: {sorted(VALID_GAMUT_MAPPING)}"
        )
    return ToneMappingParams(
        algorithm=algorithm,
        gamut_mapping=gamut_mapping,
        peak_detect=peak_detect,
    )


# ---------------------------------------------------------------------------
# compute_color_pipeline
# ---------------------------------------------------------------------------

def compute_color_pipeline(
    idt_src: str,
    working_space: str = "ACEScg",
    odt_display: str = "sRGB",
    odt_view: str = "ACES 2.0 - SDR Video",
    config_path: str = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
) -> ColorPipelineParams:
    """Validate and compute full color pipeline parameters.

    Args:
        idt_src: Source camera color space (e.g., "ARRI LogC4").
        working_space: Working color space (default ACEScg).
        odt_display: Output display name (e.g., "sRGB").
        odt_view: Output view transform name.
        config_path: OCIO config file path or ocio:// built-in URI.

    Returns:
        ColorPipelineParams with validated values.

    Raises:
        ColorError: If required strings are empty or config file missing.
    """
    if not idt_src:
        raise ColorError("idt_src must be a non-empty string")
    if not working_space:
        raise ColorError("working_space must be a non-empty string")
    if not odt_display:
        raise ColorError("odt_display must be a non-empty string")
    if not odt_view:
        raise ColorError("odt_view must be a non-empty string")
    if not config_path:
        raise ColorError("config_path must be a non-empty string")

    # Validate config path: ocio:// URIs skip file check
    if not config_path.startswith("ocio://") and not os.path.isfile(config_path):
        raise ColorError(f"OCIO config file not found: {config_path}")

    return ColorPipelineParams(
        idt_src=idt_src,
        working_space=working_space,
        odt_display=odt_display,
        odt_view=odt_view,
        config_path=config_path,
    )
```

**Step 4: Fix existing `compute_color_transform` to accept `ocio://` URIs**

In `compute_color_transform()`, change the config_path validation from:

```python
    if config_path is not None and not os.path.isfile(config_path):
        raise ColorError(f"OCIO config file not found: {config_path}")
```

to:

```python
    if config_path is not None and not config_path.startswith("ocio://") and not os.path.isfile(config_path):
        raise ColorError(f"OCIO config file not found: {config_path}")
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools/test_color.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/ave/tools/color.py tests/test_tools/test_color.py
git commit -m "feat: add tone mapping and color pipeline validation (Phase 5 Task 1)"
```

---

## Task 2: Pure logic — bin description builder (`tools/color.py`)

**Files:**
- Modify: `src/ave/tools/color.py`
- Modify: `tests/test_tools/test_color.py`

**Step 1: Write the failing tests**

```python
class TestBuildGstBinDescription:
    """Test GStreamer bin description string construction for color effects."""

    def test_simple_lut(self):
        from ave.tools.color import build_lut_bin_description

        result = build_lut_bin_description("/path/to/lut.cube", intensity=0.8)
        assert "glupload" in result
        assert "gldownload" in result
        assert "placebofilter" in result
        assert "lut-path=/path/to/lut.cube" in result
        assert "intensity=0.8" in result

    def test_simple_color_transform(self):
        from ave.tools.color import build_color_transform_bin_description

        result = build_color_transform_bin_description("ACEScg", "sRGB")
        assert "ociofilter" in result
        assert "src-colorspace=ACEScg" in result
        assert "dst-colorspace=sRGB" in result

    def test_color_transform_with_config(self):
        from ave.tools.color import build_color_transform_bin_description

        result = build_color_transform_bin_description(
            "ACEScg", "sRGB",
            config_path="ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
        )
        assert "config-path=ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5" in result

    def test_compound_pipeline(self):
        from ave.tools.color import build_color_pipeline_bin_description

        result = build_color_pipeline_bin_description(
            idt_src="ACEScg",
            working_space="ACEScg",
            odt_display="sRGB",
            odt_view="ACES_2.0_SDR",
            config_path="ocio://default",
        )
        # Should have two ociofilter instances and one placebofilter
        assert result.count("ociofilter") == 2
        assert "placebofilter" in result
        assert "glupload" in result
        assert "gldownload" in result

    def test_compound_pipeline_with_lut(self):
        from ave.tools.color import build_color_pipeline_bin_description

        result = build_color_pipeline_bin_description(
            idt_src="LogC4",
            working_space="ACEScg",
            odt_display="sRGB",
            odt_view="ACES_2.0_SDR",
            config_path="ocio://default",
            lut_path="/path/to/creative.cube",
            lut_intensity=0.7,
        )
        assert "lut-path=/path/to/creative.cube" in result
        assert "intensity=0.7" in result

    def test_escape_spaces_in_colorspace_names(self):
        from ave.tools.color import _escape_gst_value

        assert _escape_gst_value("ARRI LogC4") == r"ARRI\ LogC4"
        assert _escape_gst_value("no_spaces") == "no_spaces"
        assert _escape_gst_value("ACES 2.0 - SDR Video") == r"ACES\ 2.0\ -\ SDR\ Video"

    def test_compound_pipeline_escapes_spaces(self):
        from ave.tools.color import build_color_pipeline_bin_description

        result = build_color_pipeline_bin_description(
            idt_src="ARRI LogC4",
            working_space="ACEScg",
            odt_display="sRGB",
            odt_view="ACES 2.0 - SDR Video",
            config_path="ocio://default",
        )
        assert r"ARRI\ LogC4" in result
        assert r"ACES\ 2.0\ -\ SDR\ Video" in result

    def test_compound_pipeline_with_tone_mapping(self):
        from ave.tools.color import build_color_pipeline_bin_description

        result = build_color_pipeline_bin_description(
            idt_src="LogC4",
            working_space="ACEScg",
            odt_display="sRGB",
            odt_view="SDR",
            config_path="ocio://default",
            tone_mapping="bt2390",
            gamut_mapping="softclip",
        )
        assert "tone-mapping=3" in result or "tone-mapping=" in result  # enum value or name
        # The exact format depends on whether GES parses enum names or values
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tools/test_color.py::TestBuildGstBinDescription -v`
Expected: FAIL (functions don't exist)

**Step 3: Implement the bin description builders**

Add to `src/ave/tools/color.py`:

```python
# ---------------------------------------------------------------------------
# GStreamer bin description helpers
# ---------------------------------------------------------------------------


def _escape_gst_value(value: str) -> str:
    """Escape special characters in a GStreamer bin description value.

    Spaces are escaped with backslash so gst-launch-style parsing treats
    the entire string as a single property value.
    """
    return value.replace(" ", r"\ ")


def build_lut_bin_description(lut_path: str, intensity: float = 1.0) -> str:
    """Build a GStreamer bin description for LUT application via placebofilter.

    Returns a string suitable for GES.Effect.new().
    """
    parts = [
        "glupload",
        "glcolorconvert",
        f"placebofilter lut-path={_escape_gst_value(lut_path)} intensity={intensity}",
        "glcolorconvert",
        "gldownload",
    ]
    return " ! ".join(parts)


def build_color_transform_bin_description(
    src_colorspace: str,
    dst_colorspace: str,
    config_path: str | None = None,
) -> str:
    """Build a GStreamer bin description for OCIO color space transform.

    Returns a string suitable for GES.Effect.new().
    """
    ocio_props = (
        f"src-colorspace={_escape_gst_value(src_colorspace)} "
        f"dst-colorspace={_escape_gst_value(dst_colorspace)}"
    )
    if config_path:
        ocio_props += f" config-path={_escape_gst_value(config_path)}"

    parts = [
        "glupload",
        "glcolorconvert",
        f"ociofilter {ocio_props}",
        "glcolorconvert",
        "gldownload",
    ]
    return " ! ".join(parts)


def build_color_pipeline_bin_description(
    idt_src: str,
    working_space: str,
    odt_display: str,
    odt_view: str,
    config_path: str,
    lut_path: str | None = None,
    lut_intensity: float = 1.0,
    tone_mapping: str = "spline",
    gamut_mapping: str = "perceptual",
) -> str:
    """Build a compound GStreamer bin description for full IDT→grade→ODT pipeline.

    Chains: ociofilter[IDT] ! placebofilter[grade] ! ociofilter[ODT]
    All properties baked into the string. To modify, remove and recreate.

    Returns a string suitable for GES.Effect.new().
    """
    esc = _escape_gst_value
    cfg = f"config-path={esc(config_path)}"

    # IDT: camera log → working space via ColorSpaceTransform
    idt = f"ociofilter {cfg} src-colorspace={esc(idt_src)} dst-colorspace={esc(working_space)}"

    # Grade/LUT/tonemap via placebofilter
    placebo_props = f"tone-mapping={tone_mapping} gamut-mapping={gamut_mapping}"
    if lut_path:
        placebo_props += f" lut-path={esc(lut_path)} intensity={lut_intensity}"
    grade = f"placebofilter {placebo_props}"

    # ODT: working space → display via DisplayViewTransform
    odt = (
        f"ociofilter {cfg} src-colorspace={esc(working_space)} "
        f"display={esc(odt_display)} view={esc(odt_view)}"
    )

    parts = ["glupload", "glcolorconvert", idt, grade, odt, "glcolorconvert", "gldownload"]
    return " ! ".join(parts)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools/test_color.py::TestBuildGstBinDescription -v`
Expected: ALL PASS

**Step 5: Run all existing color tests for regression**

Run: `python -m pytest tests/test_tools/test_color.py -v`
Expected: ALL PASS (existing tests unaffected)

**Step 6: Commit**

```bash
git add src/ave/tools/color.py tests/test_tools/test_color.py
git commit -m "feat: add GStreamer bin description builders for color pipeline (Phase 5 Task 2)"
```

---

## Task 3: Fix `pl_color_repr` bug in `gstplacebofilter` (C)

> Critical bug: existing code uses `pl_color_repr_sdtv` (YCbCr) on RGB GL texture data, corrupting colors.

**Files:**
- Modify: `plugins/gst-libplacebo/gstplacebofilter.c:233-305`

**Step 1: Fix the repr in the passthrough path**

In `gst_placebo_filter_filter_texture()`, find the passthrough path (lines ~233-258) and change both `pl_frame` structs:

Replace:
```c
      .repr = pl_color_repr_sdtv,
```

With:
```c
      .repr = { .sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_FULL },
```

This appears in 4 places: img.repr and target.repr in both the passthrough path and the LUT path.

**Step 2: Fix the repr in the LUT rendering path**

Same change in the LUT rendering path (lines ~288-305).

**Step 3: Verify the fix compiles**

Run (in Docker): `cd /build/gst-libplacebo && meson compile -C build`
Expected: Compiles without errors.

**Step 4: Commit**

```bash
git add plugins/gst-libplacebo/gstplacebofilter.c
git commit -m "fix: use RGB repr instead of SDTV YCbCr in placebofilter (Phase 5 Task 3)"
```

---

## Task 4: Add tone mapping + color space properties to `gstplacebofilter` (C)

**Files:**
- Modify: `plugins/gst-libplacebo/gstplacebofilter.h`
- Modify: `plugins/gst-libplacebo/gstplacebofilter.c`

**Step 1: Add GType enum registrations and new struct fields to header**

In `gstplacebofilter.h`, add before the struct definition:

```c
/* Tone mapping algorithm enum */
typedef enum {
  GST_PLACEBO_TONE_MAP_SPLINE = 0,
  GST_PLACEBO_TONE_MAP_BT2390,
  GST_PLACEBO_TONE_MAP_BT2446A,
  GST_PLACEBO_TONE_MAP_REINHARD,
  GST_PLACEBO_TONE_MAP_MOBIUS,
  GST_PLACEBO_TONE_MAP_HABLE,
  GST_PLACEBO_TONE_MAP_GAMMA,
  GST_PLACEBO_TONE_MAP_LINEAR,
  GST_PLACEBO_TONE_MAP_CLIP,
} GstPlaceboToneMapping;

/* Gamut mapping mode enum */
typedef enum {
  GST_PLACEBO_GAMUT_MAP_PERCEPTUAL = 0,
  GST_PLACEBO_GAMUT_MAP_SOFTCLIP,
  GST_PLACEBO_GAMUT_MAP_RELATIVE,
  GST_PLACEBO_GAMUT_MAP_ABSOLUTE,
  GST_PLACEBO_GAMUT_MAP_DESATURATE,
  GST_PLACEBO_GAMUT_MAP_DARKEN,
  GST_PLACEBO_GAMUT_MAP_HIGHLIGHT,
  GST_PLACEBO_GAMUT_MAP_LINEAR,
} GstPlaceboGamutMapping;

/* Color primaries enum */
typedef enum {
  GST_PLACEBO_PRIM_BT709 = 0,
  GST_PLACEBO_PRIM_BT2020,
  GST_PLACEBO_PRIM_DISPLAY_P3,
  GST_PLACEBO_PRIM_AP0,
  GST_PLACEBO_PRIM_AP1,
  GST_PLACEBO_PRIM_BT601_525,
  GST_PLACEBO_PRIM_BT601_625,
} GstPlaceboPrimaries;

/* Transfer function enum */
typedef enum {
  GST_PLACEBO_TRC_SRGB = 0,
  GST_PLACEBO_TRC_BT1886,
  GST_PLACEBO_TRC_PQ,
  GST_PLACEBO_TRC_HLG,
  GST_PLACEBO_TRC_VLOG,
  GST_PLACEBO_TRC_SLOG2,
  GST_PLACEBO_TRC_LINEAR,
  GST_PLACEBO_TRC_GAMMA22,
  GST_PLACEBO_TRC_GAMMA28,
} GstPlaceboTransfer;
```

Add new fields to `struct _GstPlaceboFilter`:

```c
  /* Color pipeline properties */
  GstPlaceboToneMapping tone_mapping;
  GstPlaceboGamutMapping gamut_mapping;
  gboolean peak_detect;
  GstPlaceboPrimaries src_primaries;
  GstPlaceboTransfer src_transfer;
  GstPlaceboPrimaries dst_primaries;
  GstPlaceboTransfer dst_transfer;
```

Remove the old `src_colorspace` and `dst_colorspace` `gchar *` fields.

**Step 2: Register GType enums and new properties in class_init**

In `gstplacebofilter.c`, add static GType registration functions for each enum, add the new `g_param_spec_enum` properties in `class_init`, and remove old `PROP_SRC_COLORSPACE` / `PROP_DST_COLORSPACE`.

Add static helper functions to map enums to libplacebo constants:

```c
static enum pl_color_primaries
map_primaries(GstPlaceboPrimaries p) {
  switch (p) {
    case GST_PLACEBO_PRIM_BT709: return PL_COLOR_PRIM_BT_709;
    case GST_PLACEBO_PRIM_BT2020: return PL_COLOR_PRIM_BT_2020;
    case GST_PLACEBO_PRIM_DISPLAY_P3: return PL_COLOR_PRIM_DISPLAY_P3;
    case GST_PLACEBO_PRIM_AP0: return PL_COLOR_PRIM_ACES_AP0;
    case GST_PLACEBO_PRIM_AP1: return PL_COLOR_PRIM_ACES_AP1;
    case GST_PLACEBO_PRIM_BT601_525: return PL_COLOR_PRIM_BT_601_525;
    case GST_PLACEBO_PRIM_BT601_625: return PL_COLOR_PRIM_BT_601_625;
    default: return PL_COLOR_PRIM_BT_709;
  }
}

static enum pl_color_transfer
map_transfer(GstPlaceboTransfer t) {
  switch (t) {
    case GST_PLACEBO_TRC_SRGB: return PL_COLOR_TRC_SRGB;
    case GST_PLACEBO_TRC_BT1886: return PL_COLOR_TRC_BT_1886;
    case GST_PLACEBO_TRC_PQ: return PL_COLOR_TRC_PQ;
    case GST_PLACEBO_TRC_HLG: return PL_COLOR_TRC_HLG;
    case GST_PLACEBO_TRC_VLOG: return PL_COLOR_TRC_V_LOG;
    case GST_PLACEBO_TRC_SLOG2: return PL_COLOR_TRC_S_LOG2;
    case GST_PLACEBO_TRC_LINEAR: return PL_COLOR_TRC_LINEAR;
    case GST_PLACEBO_TRC_GAMMA22: return PL_COLOR_TRC_GAMMA22;
    case GST_PLACEBO_TRC_GAMMA28: return PL_COLOR_TRC_GAMMA28;
    default: return PL_COLOR_TRC_SRGB;
  }
}

static const struct pl_tone_map_function *
map_tone_mapping(GstPlaceboToneMapping tm) {
  switch (tm) {
    case GST_PLACEBO_TONE_MAP_SPLINE: return &pl_tone_map_spline;
    case GST_PLACEBO_TONE_MAP_BT2390: return &pl_tone_map_bt2390;
    case GST_PLACEBO_TONE_MAP_BT2446A: return &pl_tone_map_bt2446a;
    case GST_PLACEBO_TONE_MAP_REINHARD: return &pl_tone_map_reinhard;
    case GST_PLACEBO_TONE_MAP_MOBIUS: return &pl_tone_map_mobius;
    case GST_PLACEBO_TONE_MAP_HABLE: return &pl_tone_map_hable;
    case GST_PLACEBO_TONE_MAP_GAMMA: return &pl_tone_map_gamma;
    case GST_PLACEBO_TONE_MAP_LINEAR: return &pl_tone_map_linear;
    case GST_PLACEBO_TONE_MAP_CLIP: return &pl_tone_map_clip;
    default: return &pl_tone_map_spline;
  }
}

static const struct pl_gamut_map_function *
map_gamut_mapping(GstPlaceboGamutMapping gm) {
  switch (gm) {
    case GST_PLACEBO_GAMUT_MAP_PERCEPTUAL: return &pl_gamut_map_perceptual;
    case GST_PLACEBO_GAMUT_MAP_SOFTCLIP: return &pl_gamut_map_softclip;
    case GST_PLACEBO_GAMUT_MAP_RELATIVE: return &pl_gamut_map_relative;
    case GST_PLACEBO_GAMUT_MAP_ABSOLUTE: return &pl_gamut_map_absolute;
    case GST_PLACEBO_GAMUT_MAP_DESATURATE: return &pl_gamut_map_desaturate;
    case GST_PLACEBO_GAMUT_MAP_DARKEN: return &pl_gamut_map_darken;
    case GST_PLACEBO_GAMUT_MAP_HIGHLIGHT: return &pl_gamut_map_highlight;
    case GST_PLACEBO_GAMUT_MAP_LINEAR: return &pl_gamut_map_linear;
    default: return &pl_gamut_map_perceptual;
  }
}
```

**Step 3: Update `filter_texture` to use color space properties and tone mapping**

Replace the hardcoded `pl_color_space_srgb` with constructed color spaces from properties:

```c
  struct pl_color_space src_color = {
      .primaries = map_primaries(self->src_primaries),
      .transfer = map_transfer(self->src_transfer),
  };
  struct pl_color_space dst_color = {
      .primaries = map_primaries(self->dst_primaries),
      .transfer = map_transfer(self->dst_transfer),
  };
```

Use these in the `pl_frame` structs:
```c
      .color = src_color,
```
and
```c
      .color = dst_color,
```

Replace `pl_render_default_params` with `pl_render_high_quality_params` as base, then configure tone mapping:

```c
  struct pl_render_params params = pl_render_high_quality_params;
  struct pl_color_map_params cmap = pl_color_map_default_params;
  cmap.tone_mapping_function = map_tone_mapping(self->tone_mapping);
  cmap.gamut_mapping = map_gamut_mapping(self->gamut_mapping);
  params.color_map_params = &cmap;

  if (self->peak_detect) {
    struct pl_peak_detect_params pdet = pl_peak_detect_default_params;
    params.peak_detect_params = &pdet;
  }
```

**Step 4: Upgrade texture wrapping to GL_RGBA16F**

Change `iformat` in both `pl_opengl_wrap_params`:

```c
      .iformat = GL_RGBA16F,
```

**Step 5: Update `init` and `finalize` for new fields, remove old string fields**

Remove `g_free(self->src_colorspace)` and `g_free(self->dst_colorspace)` from finalize.
Set new enum defaults in init.

**Step 6: Verify compilation**

Run (in Docker): `cd /build/gst-libplacebo && meson compile -C build`
Expected: Compiles without errors.

**Step 7: Commit**

```bash
git add plugins/gst-libplacebo/gstplacebofilter.c plugins/gst-libplacebo/gstplacebofilter.h
git commit -m "feat: add tone mapping, gamut mapping, and color space properties to placebofilter (Phase 5 Task 4)"
```

---

## Task 5: Upgrade `ocio_processor.cpp` — multi-texture + `ocio://` URIs + DisplayViewTransform

**Files:**
- Modify: `plugins/gst-ocio/ocio_processor.h`
- Modify: `plugins/gst-ocio/ocio_processor.cpp`

**Step 1: Add new types and functions to the C header**

Add to `ocio_processor.h`:

```c
/* Texture info for 1D/2D textures */
typedef struct {
    const char *sampler_name;
    unsigned width;
    unsigned height;  /* 1 for 1D textures */
    int channel_count;  /* 1 = R, 3 = RGB */
} OcioTextureInfo;

/* Create processor using DisplayViewTransform (for ODT) */
OcioProcessor* ocio_processor_new_display_view(const char* config_path,
                                                const char* src_cs,
                                                const char* display,
                                                const char* view,
                                                const char** error_out);

/* Number of 3D LUT textures */
int ocio_processor_get_num_3d_textures(OcioProcessor* proc);

/* Get info for 3D texture at index (edge_len, sampler_name) */
int ocio_processor_get_3d_texture_edge_len(OcioProcessor* proc, int index);
const char* ocio_processor_get_3d_texture_sampler(OcioProcessor* proc, int index);
void ocio_processor_get_3d_texture_data(OcioProcessor* proc, int index, float* data);

/* 1D/2D textures */
int ocio_processor_get_num_textures(OcioProcessor* proc);
int ocio_processor_get_texture_width(OcioProcessor* proc, int index);
int ocio_processor_get_texture_height(OcioProcessor* proc, int index);
int ocio_processor_get_texture_channels(OcioProcessor* proc, int index);
const char* ocio_processor_get_texture_sampler(OcioProcessor* proc, int index);
void ocio_processor_get_texture_data(OcioProcessor* proc, int index, float* data);
```

**Step 2: Update the OcioProcessor struct in .cpp to store all textures**

Add vectors for multiple 3D textures and 1D/2D textures:

```cpp
struct Texture3D {
    std::string sampler_name;
    int edge_len;
    std::vector<float> data;
};

struct Texture1D2D {
    std::string sampler_name;
    unsigned width;
    unsigned height;
    int channels;  // 1 or 3
    std::vector<float> data;
};

// In OcioProcessor:
std::vector<Texture3D> textures_3d;
std::vector<Texture1D2D> textures;
```

**Step 3: Add `ocio://` URI support in processor creation**

In `ocio_processor_new()`, detect URI prefix:

```cpp
if (config_path && strncmp(config_path, "ocio://", 7) == 0) {
    proc->config = OCIO::Config::CreateFromBuiltinConfig(config_path);
} else if (config_path && config_path[0] != '\0') {
    proc->config = OCIO::Config::CreateFromFile(config_path);
} else {
    proc->config = OCIO::Config::CreateFromEnv();
}
```

**Step 4: Implement `ocio_processor_new_display_view()`**

Creates a `DisplayViewTransform` instead of `ColorSpaceTransform`:

```cpp
auto dvt = OCIO::DisplayViewTransform::Create();
dvt->setSrc(src_cs);
dvt->setDisplay(display);
dvt->setView(view);
proc->processor = proc->config->getProcessor(dvt);
```

**Step 5: Extract all 3D and 1D/2D textures**

After `extractGpuShaderInfo`, iterate all textures:

```cpp
// 3D textures
for (unsigned i = 0; i < proc->shader_desc->getNum3DTextures(); ++i) {
    const char *t_name = nullptr, *s_name = nullptr;
    unsigned el = 0;
    OCIO::Interpolation ip;
    proc->shader_desc->get3DTexture(i, t_name, s_name, el, ip);

    const float* values = nullptr;
    proc->shader_desc->get3DTextureValues(i, values);

    Texture3D tex;
    tex.sampler_name = s_name ? s_name : "";
    tex.edge_len = (int)el;
    if (values && el > 0) {
        size_t n = (size_t)el * el * el * 3;
        tex.data.assign(values, values + n);
    }
    proc->textures_3d.push_back(std::move(tex));
}

// 1D/2D textures
for (unsigned i = 0; i < proc->shader_desc->getNumTextures(); ++i) {
    const char *t_name = nullptr, *s_name = nullptr;
    unsigned w = 0, h = 0;
    OCIO::GpuShaderDesc::TextureType channel;
    OCIO::GpuShaderDesc::TextureDimensions dims;
    OCIO::Interpolation ip;
    proc->shader_desc->getTexture(i, t_name, s_name, w, h, channel, dims, ip);

    const float* values = nullptr;
    proc->shader_desc->getTextureValues(i, values);

    Texture1D2D tex;
    tex.sampler_name = s_name ? s_name : "";
    tex.width = w;
    tex.height = h;
    tex.channels = (channel == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL) ? 1 : 3;
    if (values && w > 0) {
        size_t n = (size_t)w * (h > 0 ? h : 1) * tex.channels;
        tex.data.assign(values, values + n);
    }
    proc->textures.push_back(std::move(tex));
}
```

**Step 6: Implement all new C API getter functions**

Each function accesses the stored vectors by index, returning the appropriate data.

**Step 7: Remove old single-LUT storage** (`lut3d_edge_len`, `lut3d_data`, `lut3d_sampler_names`) — replaced by `textures_3d` vector. Update existing getters to delegate to new storage for backwards compatibility, or remove old getters if no longer used.

**Step 8: Verify compilation**

Run (in Docker): `cd /build/gst-ocio && meson compile -C build`
Expected: Compiles without errors.

**Step 9: Commit**

```bash
git add plugins/gst-ocio/ocio_processor.h plugins/gst-ocio/ocio_processor.cpp
git commit -m "feat: multi-texture, ocio:// URIs, and DisplayViewTransform in OCIO processor (Phase 5 Task 5)"
```

---

## Task 6: Upgrade `gstociofilter` — multi-texture upload + alpha preservation + FBO caching

**Files:**
- Modify: `plugins/gst-ocio/gstociofilter.h`
- Modify: `plugins/gst-ocio/gstociofilter.c`

**Step 1: Add new struct fields to header**

Add to `struct _GstOCIOFilter`:

```c
  /* New properties */
  gchar *display;
  gchar *view;

  /* Cached GL resources (created once in gl_start) */
  guint fbo;
  guint vao;
  guint vbo;

  /* Multiple texture support */
  guint *tex_3d_ids;     /* array of GL texture IDs for 3D LUTs */
  int num_tex_3d;
  guint *tex_ids;        /* array of GL texture IDs for 1D/2D textures */
  int num_tex;
  int first_tex_unit;    /* first texture unit used (1, after input tex on unit 0) */
```

Remove old `lut3d_tex`, `lut3d_size`, `lut3d_sampler_name` fields.

**Step 2: Add `display` and `view` properties**

Register `PROP_DISPLAY` and `PROP_VIEW` as `g_param_spec_string` in `class_init`. Add get/set handlers.

**Step 3: Update `gl_start` for mode selection and multi-texture upload**

```c
  if (self->display && self->view && self->display[0] && self->view[0]) {
    self->ocio_processor = ocio_processor_new_display_view(
        self->config_path, self->src_colorspace, self->display, self->view, &ocio_err);
  } else if (self->src_colorspace && self->dst_colorspace) {
    self->ocio_processor = ocio_processor_new(
        self->config_path, self->src_colorspace, self->dst_colorspace, &ocio_err);
  } else {
    GST_ERROR_OBJECT(self, "Must set either (src+dst colorspace) or (src+display+view)");
    return FALSE;
  }
```

Upload all 3D and 1D/2D textures to consecutive GL texture units:

```c
  int unit = 1;  /* unit 0 is reserved for the input texture */

  /* Upload 3D LUT textures */
  self->num_tex_3d = ocio_processor_get_num_3d_textures(proc);
  if (self->num_tex_3d > 0) {
    self->tex_3d_ids = g_new0(guint, self->num_tex_3d);
    gl->GenTextures(self->num_tex_3d, self->tex_3d_ids);
    for (int i = 0; i < self->num_tex_3d; i++) {
      int el = ocio_processor_get_3d_texture_edge_len(proc, i);
      gl->ActiveTexture(GL_TEXTURE0 + unit);
      gl->BindTexture(GL_TEXTURE_3D, self->tex_3d_ids[i]);
      /* ... set params, upload data ... */
      unit++;
    }
  }

  /* Upload 1D/2D textures */
  self->num_tex = ocio_processor_get_num_textures(proc);
  if (self->num_tex > 0) {
    self->tex_ids = g_new0(guint, self->num_tex);
    gl->GenTextures(self->num_tex, self->tex_ids);
    for (int i = 0; i < self->num_tex; i++) {
      int w = ocio_processor_get_texture_width(proc, i);
      int h = ocio_processor_get_texture_height(proc, i);
      gl->ActiveTexture(GL_TEXTURE0 + unit);
      if (h <= 1) {
        gl->BindTexture(GL_TEXTURE_1D, self->tex_ids[i]);
        /* ... upload 1D ... */
      } else {
        gl->BindTexture(GL_TEXTURE_2D, self->tex_ids[i]);
        /* ... upload 2D ... */
      }
      unit++;
    }
  }
```

**Step 4: Cache FBO, VAO, VBO in `gl_start`**

Move the FBO, VAO, VBO creation from `filter_texture` to `gl_start`. In `filter_texture`, only bind them; don't create/destroy.

```c
  /* Create persistent FBO */
  gl->GenFramebuffers(1, &self->fbo);

  /* Create persistent VAO/VBO */
  static const GLfloat vertices[] = { /* ... same quad vertices ... */ };
  gl->GenVertexArrays(1, &self->vao);
  gl->BindVertexArray(self->vao);
  gl->GenBuffers(1, &self->vbo);
  gl->BindBuffer(GL_ARRAY_BUFFER, self->vbo);
  gl->BufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  /* ... setup attrib pointers ... */
```

**Step 5: Fix alpha preservation in fragment shader**

Change the fragment shader template:

```c
static const gchar *frag_shader_template =
    "#version 130\n"
    "uniform sampler2D tex;\n"
    "%s\n"  /* OCIO-generated uniforms, textures, and function */
    "in vec2 v_texcoord;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "  vec4 col = texture(tex, v_texcoord);\n"
    "  float saved_alpha = col.a;\n"
    "  %s(col);\n"  /* OCIO function call */
    "  col.a = saved_alpha;\n"
    "  fragColor = col;\n"
    "}\n";
```

**Step 6: Bind all sampler uniforms after program link**

After linking, iterate all textures and bind samplers:

```c
  int unit = 1;
  for (int i = 0; i < self->num_tex_3d; i++) {
    const char *name = ocio_processor_get_3d_texture_sampler(proc, i);
    GLint loc = gl->GetUniformLocation(self->gl_program, name);
    if (loc >= 0) gl->Uniform1i(loc, unit);
    unit++;
  }
  for (int i = 0; i < self->num_tex; i++) {
    const char *name = ocio_processor_get_texture_sampler(proc, i);
    GLint loc = gl->GetUniformLocation(self->gl_program, name);
    if (loc >= 0) gl->Uniform1i(loc, unit);
    unit++;
  }
```

**Step 7: Update `filter_texture` to use cached resources**

Simplify to: bind FBO → attach output texture → bind program → bind input texture → bind VAO → draw → unbind.

**Step 8: Update `gl_stop` to clean up all resources**

Free all texture arrays, FBO, VAO, VBO.

**Step 9: Verify compilation**

Run (in Docker): `cd /build/gst-ocio && meson compile -C build`
Expected: Compiles without errors.

**Step 10: Commit**

```bash
git add plugins/gst-ocio/gstociofilter.c plugins/gst-ocio/gstociofilter.h
git commit -m "feat: multi-texture upload, alpha preservation, FBO caching in ociofilter (Phase 5 Task 6)"
```

---

## Task 7: Python GES operations for color pipeline

**Files:**
- Modify: `src/ave/project/operations.py`
- Modify: `tests/test_project/test_operations.py`

**Step 1: Write the failing tests**

Add to `tests/test_project/test_operations.py`:

```python
@requires_ges
@requires_ffmpeg
class TestColorOperations:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir, tmp_project):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_apply_lut_creates_effect(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_lut

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0, duration_ns=2_000_000_000,
        )

        # Create a dummy .cube LUT file
        lut_path = self.project / "luts" / "test.cube"
        lut_path.write_text(
            "LUT_3D_SIZE 2\n"
            + "\n".join(f"{r} {g} {b}" for r in (0,1) for g in (0,1) for b in (0,1))
        )

        effect_id = apply_lut(tl, clip_id, str(lut_path), intensity=0.5)
        assert effect_id is not None
        assert "fx" in effect_id

    def test_apply_color_transform_creates_effect(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_color_transform

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0, duration_ns=2_000_000_000,
        )

        effect_id = apply_color_transform(tl, clip_id, "sRGB", "sRGB")
        assert effect_id is not None

    def test_apply_color_pipeline_creates_compound_effect(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_color_pipeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0, duration_ns=2_000_000_000,
        )

        effect_id = apply_color_pipeline(
            tl, clip_id,
            idt_src="ACEScg",
            working_space="ACEScg",
            odt_display="sRGB",
            odt_view="ACES 2.0 - SDR Video",
        )
        assert effect_id is not None

    def test_apply_lut_invalid_path_raises(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_lut, OperationError

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0, duration_ns=2_000_000_000,
        )

        with pytest.raises((OperationError, Exception)):
            apply_lut(tl, clip_id, "/nonexistent/lut.cube")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_project/test_operations.py::TestColorOperations -v`
Expected: FAIL (functions don't exist)

**Step 3: Implement the operations**

Add to `src/ave/project/operations.py`:

```python
from ave.tools.color import (
    compute_color_pipeline,
    compute_color_transform,
    compute_lut_application,
    compute_tone_mapping,
    build_lut_bin_description,
    build_color_transform_bin_description,
    build_color_pipeline_bin_description,
)


def apply_lut(
    timeline: Timeline,
    clip_id: str,
    lut_path: str,
    intensity: float = 1.0,
) -> str:
    """Apply a .cube LUT via placebofilter as a GES.Effect.

    Returns effect_id.
    """
    params = compute_lut_application(lut_path, intensity)
    bin_desc = build_lut_bin_description(params.path, params.intensity)
    return timeline.add_effect(clip_id, bin_desc)


def apply_color_transform(
    timeline: Timeline,
    clip_id: str,
    src_colorspace: str,
    dst_colorspace: str,
    config_path: str | None = None,
) -> str:
    """Apply an OCIO color space transform as a GES.Effect.

    Returns effect_id.
    """
    params = compute_color_transform(src_colorspace, dst_colorspace, config_path)
    bin_desc = build_color_transform_bin_description(
        params.src_colorspace, params.dst_colorspace, params.config_path,
    )
    return timeline.add_effect(clip_id, bin_desc)


def apply_color_pipeline(
    timeline: Timeline,
    clip_id: str,
    idt_src: str,
    working_space: str = "ACEScg",
    odt_display: str = "sRGB",
    odt_view: str = "ACES 2.0 - SDR Video",
    config_path: str = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
    lut_path: str | None = None,
    lut_intensity: float = 1.0,
    tone_mapping: str = "spline",
    gamut_mapping: str = "perceptual",
) -> str:
    """Apply full IDT→grade→ODT compound effect. Returns effect_id.

    All properties baked into bin description string. To modify parameters,
    remove effect by effect_id and recreate with new values.
    """
    pipeline_params = compute_color_pipeline(
        idt_src=idt_src,
        working_space=working_space,
        odt_display=odt_display,
        odt_view=odt_view,
        config_path=config_path,
    )
    tm_params = compute_tone_mapping(algorithm=tone_mapping, gamut_mapping=gamut_mapping)

    bin_desc = build_color_pipeline_bin_description(
        idt_src=pipeline_params.idt_src,
        working_space=pipeline_params.working_space,
        odt_display=pipeline_params.odt_display,
        odt_view=pipeline_params.odt_view,
        config_path=pipeline_params.config_path,
        lut_path=lut_path,
        lut_intensity=lut_intensity,
        tone_mapping=tm_params.algorithm,
        gamut_mapping=tm_params.gamut_mapping,
    )
    return timeline.add_effect(clip_id, bin_desc)
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_project/test_operations.py::TestColorOperations -v`
Expected: SKIP locally (requires GES). In Docker: PASS.

**Step 5: Run all operations tests for regression**

Run: `python -m pytest tests/test_project/test_operations.py -v`
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/ave/project/operations.py tests/test_project/test_operations.py
git commit -m "feat: add color pipeline GES operations (Phase 5 Task 7)"
```

---

## Task 8: Docker updates — bump libplacebo, update meson constraints

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `plugins/meson.build`

**Step 1: Bump libplacebo version in Dockerfile**

Change `ENV LIBPLACEBO_VERSION=7.349.0` to `ENV LIBPLACEBO_VERSION=7.360.0`.

**Step 2: Update meson version constraints**

In `plugins/meson.build`:
- Change `dependency('libplacebo', version : '>= 7.349')` to `'>= 7.360'`
- Change `dependency('OpenColorIO', version : '>= 2.3')` to `'>= 2.5.1'`

In `plugins/gst-libplacebo/meson.build`:
- Change `dependency('libplacebo', version : '>= 7.349')` to `'>= 7.360'`

**Step 3: Add compound pipeline smoke test to Dockerfile**

After the existing smoke test line (`RUN gst-inspect-1.0 placebofilter && gst-inspect-1.0 ociofilter`), add:

```dockerfile
# Smoke test: verify custom plugins accept enum properties
RUN gst-launch-1.0 -e videotestsrc num-buffers=1 ! \
    video/x-raw,width=320,height=240 ! \
    glupload ! glcolorconvert ! \
    placebofilter tone-mapping=0 ! \
    glcolorconvert ! gldownload ! \
    video/x-raw,format=I420 ! fakesink \
    && echo "placebofilter compound bin: OK"
```

**Step 4: Commit**

```bash
git add docker/Dockerfile plugins/meson.build plugins/gst-libplacebo/meson.build
git commit -m "build: bump libplacebo to 7.360.0, update meson constraints (Phase 5 Task 8)"
```

---

## Task 9: Agent tool registration for color pipeline

**Files:**
- Modify: `src/ave/agent/tools/color.py`

**Step 1: Add new tools to the agent registry**

Add after existing tool registrations:

```python
    @registry.tool(
        domain="color",
        requires=["timeline_loaded", "clip_exists"],
        provides=["lut_applied"],
        tags=["LUT", "lookup table", "cube file", "color look", "film emulation",
              "creative LUT", "apply LUT"],
        modifies_timeline=True,
    )
    def apply_lut(lut_path: str, intensity: float = 1.0):
        """Apply a .cube LUT to a clip via GPU (placebofilter)."""
        from ave.tools.color import compute_lut_application

        return compute_lut_application(lut_path, intensity)

    @registry.tool(
        domain="color",
        requires=["timeline_loaded", "clip_exists"],
        provides=["color_transformed"],
        tags=["color space", "colour space", "transform", "convert",
              "OCIO", "OpenColorIO", "ACES", "Rec.709", "sRGB"],
        modifies_timeline=True,
    )
    def color_transform(src_colorspace: str, dst_colorspace: str, config_path: str = ""):
        """Apply an OCIO color space transform to a clip."""
        from ave.tools.color import compute_color_transform

        return compute_color_transform(src_colorspace, dst_colorspace, config_path or None)

    @registry.tool(
        domain="color",
        requires=["timeline_loaded", "clip_exists"],
        provides=["color_pipeline_applied"],
        tags=["color pipeline", "IDT", "ODT", "ACES pipeline", "tone map",
              "gamut map", "HDR", "SDR", "display transform", "camera log",
              "LogC", "S-Log", "V-Log", "REDLog"],
        modifies_timeline=True,
    )
    def color_pipeline(
        idt_src: str,
        working_space: str = "ACEScg",
        odt_display: str = "sRGB",
        odt_view: str = "ACES 2.0 - SDR Video",
        config_path: str = "ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
        lut_path: str = "",
        lut_intensity: float = 1.0,
        tone_mapping: str = "spline",
        gamut_mapping: str = "perceptual",
    ):
        """Apply full IDT→grade→ODT color pipeline (compound GPU effect)."""
        from ave.tools.color import compute_color_pipeline, compute_tone_mapping

        pipeline = compute_color_pipeline(
            idt_src=idt_src, working_space=working_space,
            odt_display=odt_display, odt_view=odt_view,
            config_path=config_path,
        )
        tone_map = compute_tone_mapping(algorithm=tone_mapping, gamut_mapping=gamut_mapping)
        return {"pipeline": pipeline, "tone_mapping": tone_map, "lut_path": lut_path or None, "lut_intensity": lut_intensity}
```

**Step 2: Commit**

```bash
git add src/ave/agent/tools/color.py
git commit -m "feat: register color pipeline agent tools (Phase 5 Task 9)"
```

---

## Summary

| Task | What | Depends on | Testable locally |
|------|------|-----------|-----------------|
| 0 | Validate compound GL bin (BLOCKING) | — | No (Docker+GPU) |
| 1 | Pure logic dataclasses + validation | — | Yes |
| 2 | Bin description builders | 1 | Yes |
| 3 | Fix pl_color_repr bug | — | No (Docker) |
| 4 | placebofilter tone mapping + color space | 3 | No (Docker) |
| 5 | ocio_processor multi-texture + URIs | — | No (Docker) |
| 6 | ociofilter multi-texture + alpha + FBO cache | 5 | No (Docker) |
| 7 | Python GES operations | 1, 2, 4, 6 | No (Docker) |
| 8 | Docker + meson updates | 4, 6 | No (Docker) |
| 9 | Agent tool registration | 1, 2 | Yes |

**Parallelizable:** Tasks 3+4 (placebofilter) and Tasks 5+6 (ociofilter) are independent and can run in parallel. Tasks 1+2 (Python) can also run in parallel with the C tasks.
