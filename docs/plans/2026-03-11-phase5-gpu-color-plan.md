# Phase 5: GPU Color Pipeline — Implementation Plan

> Date: 2026-03-11
> Status: Ready for implementation
> Dependencies: Phase 3 (preview) complete, Phase 4 (agent tools) complete

---

## Current State Inventory

### What's DONE (do NOT rebuild)

| Component | File | Status |
|-----------|------|--------|
| libplacebo GStreamer plugin | `plugins/gst-libplacebo/gstplacebofilter.c` | 90% — LUT load/apply works, intensity blending TODO |
| OCIO GStreamer plugin | `plugins/gst-ocio/gstociofilter.c` + `ocio_processor.cpp` | 85% — transforms work, error handling gaps |
| Color pure logic | `src/ave/tools/color.py` | 100% — CubeLUT, ColorGradeParams, CDLParams, LUTParams, ColorTransformParams, GLSL generation |
| Color GES ops | `src/ave/tools/color_ops.py` | 100% — apply_lut, apply_color_grade, apply_cdl, apply_color_transform, apply_idt |
| Probe metadata | `src/ave/ingest/probe.py` | 100% — color_space, color_transfer, color_primaries extracted |
| Asset registry IDT fields | `src/ave/ingest/registry.py` | 100% — camera_color_space, camera_transfer, idt_reference fields |
| Docker GPU deps | `docker/Dockerfile` | 100% — libplacebo254, libopencolorio2.3, CUDA 12.8 |
| Agent color tools | `src/ave/agent/tools/color.py` | 100% — color_grade, cdl, lut_parse registered |
| Unit tests | `tests/test_tools/test_color.py` | 100% — parse_cube_lut, validation, GLSL gen |
| E2E tests | `tests/test_e2e/test_color_e2e.py` | 100% — LUT + grade render verification |

### What's REMAINING

1. **libplacebo intensity blending** — C plugin stores `intensity` property but never applies it (line 251-265 in gstplacebofilter.c). Needs temporary texture + shader blending.
2. **OCIO error handling** — `ocio_processor.cpp` has limited error propagation; sampler naming may break with some OCIO configs (expects `ocio_lut3d_0Sampler` or `ocio_lut3d`).
3. **ACES 2.0 config bundle** — No built-in ACES config shipped. Currently relies on user providing `config_path` or `$OCIO` env var.
4. **IDT auto-detection** — `apply_idt()` in color_ops.py reads `agent:camera-color-space` metadata but there's no automatic detection from probe metadata (color_space/color_transfer/color_primaries → OCIO color space name mapping).
5. **10-bit pipeline verification** — No tests verify 10-bit+ passthrough (no 8-bit truncation at any stage).
6. **Docker OCIO version** — Dockerfile installs `libopencolorio2.3` from apt; may need 2.5.1 for ACES 2.0 built-in configs.
7. **Overlay/SoftLight blend mode shaders** — `compositing.py` has approximations (standard alpha blending) for OVERLAY and SOFT_LIGHT. Need proper GLSL shaders.

---

## Subtask Breakdown

### Subtask 5-1: libplacebo Intensity Blending (C Plugin)

**Agent type:** Single agent with C/GStreamer expertise
**Estimated scope:** ~100 LOC C changes

**Context files to read:**
- `plugins/gst-libplacebo/gstplacebofilter.c` (354 LOC) — the existing plugin
- `plugins/gst-libplacebo/gstplacebofilter.h` — struct definition

**What to implement:**
The `intensity` property (0.0–1.0) is already stored in the filter struct but never used in rendering. When intensity < 1.0, the filter should blend between the original frame and the LUT-applied frame.

**Implementation approach:**
1. In `gst_placebo_filter_filter_texture()`, after applying the LUT to produce the output texture:
   - If `self->intensity < 1.0`:
     - Keep a copy of the input texture (or render to a temporary FBO)
     - Use `glBlendFunc` or a simple mix shader to blend: `result = mix(original, lut_applied, intensity)`
   - If `self->intensity == 1.0`: current behavior (no change needed)

**Key constraint:** Must work within GstGLFilter's `filter_texture` vfunc. The input texture is `in_tex`, output is `out_tex`. Allocate a temp texture only when intensity < 1.0.

**Test plan (TDD):**
- Test file: `tests/test_plugins/test_libplacebo_intensity.py`
- Test: Apply LUT with intensity=0.0 → output identical to input
- Test: Apply LUT with intensity=1.0 → output matches full LUT application
- Test: Apply LUT with intensity=0.5 → output pixel values between input and full-LUT
- Tests require `@pytest.mark.gpu` and Docker environment

**Acceptance criteria:**
- `intensity` property actually affects output
- intensity=0.0 is passthrough
- intensity=1.0 is full LUT (no regression)
- No temp texture allocation when intensity=1.0

---

### Subtask 5-2: OCIO Plugin Error Handling & Sampler Robustness

**Agent type:** Single agent with C++/OCIO expertise
**Estimated scope:** ~80 LOC C++ changes

**Context files to read:**
- `plugins/gst-ocio/ocio_processor.cpp` (201 LOC) — C++ OCIO wrapper
- `plugins/gst-ocio/ocio_processor.h` (79 LOC) — C interface
- `plugins/gst-ocio/gstociofilter.c` (490 LOC) — GStreamer filter

**What to implement:**

1. **Error propagation in ocio_processor.cpp:**
   - `ocio_processor_new()` catches exceptions but returns NULL with only a thread-local error buffer. Add structured error codes.
   - `ocio_processor_get_shader_text()` should validate the returned shader is non-empty.
   - `ocio_processor_get_lut3d_data()` should validate LUT size > 0.

2. **Sampler naming robustness in gstociofilter.c:**
   - Currently hardcodes search for `ocio_lut3d_0Sampler` or `ocio_lut3d` in the shader text.
   - OCIO 2.5.x may generate different sampler names depending on config.
   - Fix: Parse the actual sampler name from `GpuShaderDesc::getTextureName()` instead of hardcoding.

3. **Config validation:**
   - Validate that src/dst color spaces exist in the loaded config before creating the processor.
   - Return meaningful error if color space not found (currently throws cryptic OCIO exception).

**Test plan (TDD):**
- Test file: `tests/test_plugins/test_ocio_errors.py`
- Test: Invalid config path → clean error message
- Test: Non-existent color space name → clean error message
- Test: Valid transform → shader text is non-empty, LUT size > 0
- Tests require `@pytest.mark.gpu` and Docker environment

**Acceptance criteria:**
- All OCIO exceptions caught with user-friendly error messages
- Sampler name dynamically extracted from OCIO shader descriptor
- Invalid color space names produce clear errors before GPU execution

---

### Subtask 5-3: ACES 2.0 Config Bundle

**Agent type:** Single agent, Python + OCIO config knowledge
**Estimated scope:** ~150 LOC Python, config files

**Context files to read:**
- `src/ave/tools/color_ops.py` — current `apply_idt()` and `apply_color_transform()`
- `src/ave/tools/color.py` — `compute_color_transform()`
- `src/ave/ingest/registry.py` — `AssetEntry.camera_color_space`, `idt_reference`
- `docker/Dockerfile` — current OCIO installation

**What to implement:**

1. **Bundle ACES 2.0 config:**
   - Create `data/ocio/aces_2.0/` directory with the ACES 2.0 config
   - OCIO 2.5.x ships built-in configs accessible via `OCIO::Config::CreateFromBuiltinConfig("ocio://cg-config-v2.2.0_aces-v2.0_no-nesting")`
   - Add a Python helper: `get_builtin_aces_config() -> str` that returns the config path or builtin name

2. **Default project color space:**
   - Add `DEFAULT_WORKING_SPACE = "ACES - ACEScct"` constant
   - Add `DEFAULT_OCIO_CONFIG = "ocio://cg-config-v2.2.0_aces-v2.0_no-nesting"` constant
   - Update `compute_color_transform()` to accept `config_path=None` meaning "use builtin ACES"

3. **Config validation helper:**
   - `list_colorspaces(config_path: str | None) -> list[str]` — list available color spaces in a config
   - `validate_colorspace(name: str, config_path: str | None) -> bool` — check if a color space exists

**Test plan (TDD):**
- Test file: `tests/test_tools/test_aces_config.py`
- Test: `get_builtin_aces_config()` returns a valid string
- Test: `list_colorspaces(None)` returns a non-empty list containing "ACES - ACEScct"
- Test: `validate_colorspace("ACES - ACEScct", None)` returns True
- Test: `validate_colorspace("NotARealSpace", None)` returns False
- Tests requiring OCIO library: `@pytest.mark.skipif` if PyOpenColorIO not importable

**Acceptance criteria:**
- ACES 2.0 config accessible without user providing a file path
- Default working space is ACEScct (AP1)
- Color space validation before transform execution

---

### Subtask 5-4: IDT Auto-Detection from Probe Metadata

**Agent type:** Single agent, Python
**Estimated scope:** ~120 LOC Python

**Context files to read:**
- `src/ave/ingest/probe.py` — `VideoStream` fields: `color_space`, `color_transfer`, `color_primaries`
- `src/ave/ingest/registry.py` — `AssetEntry` fields
- `src/ave/tools/color_ops.py` — `apply_idt()` reads `agent:camera-color-space`

**What to implement:**

1. **Camera metadata → OCIO color space mapping table:**
   Create `src/ave/tools/idt_detect.py`:
   ```python
   # Mapping from (color_space, color_transfer, color_primaries) → OCIO IDT name
   IDT_MAP = {
       ("bt2020nc", "arib-std-b67", "bt2020"): "Input - Sony - S-Log3 - S-Gamut3.Cine",
       ("bt709", "bt709", "bt709"): "Utility - Linear - sRGB",
       ("bt2020nc", "smpte2084", "bt2020"): "Utility - Linear - Rec.2020",
       # ... expand with common camera profiles
   }
   ```

2. **Detection function:**
   ```python
   def detect_idt(video_stream: VideoStream) -> str | None:
       """Detect OCIO IDT name from ffprobe color metadata.
       Returns None if no match found."""
   ```

3. **Auto-populate on ingest:**
   - Add function `auto_detect_and_set_idt(entry: AssetEntry, video_stream: VideoStream) -> AssetEntry`
   - If detected, sets `camera_color_space` and `idt_reference` on the entry
   - If not detected, leaves fields as-is (user must set manually)

4. **Common camera profiles to map:**
   - Sony S-Log3/S-Gamut3.Cine
   - Canon C-Log2/Cinema Gamut
   - RED Log3G10/REDWideGamutRGB
   - ARRI LogC3/AWG (ALEXA)
   - ARRI LogC4/AWG4 (ALEXA 35)
   - Panasonic V-Log/V-Gamut
   - Blackmagic Film Gen 5
   - Rec.709 (passthrough — no IDT needed)
   - Rec.2020 PQ (HDR passthrough)

**Test plan (TDD):**
- Test file: `tests/test_tools/test_idt_detect.py`
- Test: Sony S-Log3 metadata → correct IDT name
- Test: Rec.709 metadata → returns None (or passthrough IDT)
- Test: Unknown metadata → returns None
- Test: `auto_detect_and_set_idt` populates entry fields correctly
- Pure logic tests — no Docker/GPU required

**Acceptance criteria:**
- At least 8 common camera profiles mapped
- Unknown metadata returns None (not an error)
- Integrates with existing AssetEntry without breaking API

---

### Subtask 5-5: 10-Bit Pipeline Verification

**Agent type:** Single agent, Docker/GStreamer testing
**Estimated scope:** ~100 LOC test code

**Context files to read:**
- `src/ave/tools/color_ops.py` — GES effect wrappers
- `plugins/gst-libplacebo/gstplacebofilter.c` — pixel format handling
- `plugins/gst-ocio/gstociofilter.c` — GL texture format
- `docker/Dockerfile` — GStreamer build flags

**What to implement:**

1. **Test 10-bit passthrough:**
   - Create a 10-bit test source (ProRes 4444, or GStreamer videotestsrc with format=I420_10LE)
   - Apply color operations (LUT, grade, OCIO transform)
   - Probe output to verify bit depth is preserved (not truncated to 8-bit)

2. **Verify GL texture formats:**
   - libplacebo plugin must use `GL_RGBA16F` or `GL_RGB10_A2` internal format for 10-bit+
   - OCIO plugin must use `GL_RGBA16F` for 3D LUT texture
   - Check that GstGLFilter negotiates 10-bit capable caps

3. **Pipeline caps negotiation test:**
   - `video/x-raw(memory:GLMemory),format=RGBA` at minimum
   - Verify no `videoconvert` element silently truncates to 8-bit

**Test plan (TDD):**
- Test file: `tests/test_e2e/test_10bit_pipeline.py`
- All tests: `@pytest.mark.gpu` + `@requires_ges`
- Test: 10-bit source → placebofilter → probe output bit depth
- Test: 10-bit source → ociofilter → probe output bit depth
- Test: Full chain (IDT → grade → LUT → ODT) preserves 10-bit

**Acceptance criteria:**
- No 8-bit truncation in the color pipeline
- Tests document the minimum GL format requirements
- Any truncation issues are identified and fixed

---

### Subtask 5-6: Docker GPU Dependencies Update

**Agent type:** Single agent, Docker/build expertise
**Estimated scope:** ~30-50 LOC Dockerfile changes

**Context files to read:**
- `docker/Dockerfile` — current multi-stage build
- `plugins/gst-libplacebo/meson.build` — build deps
- `plugins/gst-ocio/meson.build` — build deps

**What to implement:**

1. **Verify OCIO version:**
   - Current: `libopencolorio2.3` from Ubuntu 24.04 apt
   - Required: OCIO 2.5.1+ for built-in ACES 2.0 configs and stable GPU API
   - If apt version is insufficient: build OCIO 2.5.1 from source in builder stage

2. **Verify libplacebo version:**
   - Current: `libplacebo254` from apt
   - Required: v7.360.0+ for stable `pl_renderer` API (PL_API_VER >= 309)
   - If apt version is insufficient: build from source

3. **Plugin build integration:**
   - Add meson build commands for both plugins to the Dockerfile
   - Install built plugins to `$GST_PLUGIN_PATH`

4. **Smoke test:**
   - Add a `RUN` step that verifies `gst-inspect-1.0 placebofilter` and `gst-inspect-1.0 ociofilter` succeed

**Test plan:**
- Build the Docker image successfully
- `gst-inspect-1.0 placebofilter` shows properties including `intensity`
- `gst-inspect-1.0 ociofilter` shows properties including `src-colorspace`, `dst-colorspace`

**Acceptance criteria:**
- Both plugins build and register in the Docker image
- OCIO version supports built-in ACES configs
- Smoke test passes in CI

---

### Subtask 5-7: Overlay/SoftLight Blend Mode Shaders

**Agent type:** Single agent, GLSL + Python
**Estimated scope:** ~100 LOC Python + GLSL

**Context files to read:**
- `src/ave/tools/compositing.py` — current BlendMode enum and `compute_blend_params()`
- `src/ave/tools/color.py` — GLSL generation pattern (reference for style)

**What to implement:**

The OVERLAY and SOFT_LIGHT blend modes currently fall back to standard alpha compositing (see compositing.py lines 154-174). These require per-pixel conditional math that can't be expressed with GL blend functions alone.

1. **Add GLSL shader generation for complex blend modes:**
   Create two functions in `src/ave/tools/compositing.py`:

   ```python
   def generate_overlay_glsl() -> str:
       """GLSL overlay: if dst < 0.5: 2*src*dst, else 1 - 2*(1-src)*(1-dst)"""

   def generate_soft_light_glsl() -> str:
       """GLSL soft light (Photoshop formula)"""
   ```

2. **Update `compute_blend_params()` to flag shader-required modes:**
   - Add a `requires_shader: bool` field to `BlendFuncParams`
   - OVERLAY and SOFT_LIGHT return `requires_shader=True`
   - The GES execution layer checks this and uses `glshader` effect instead of GL blend functions

3. **Add `BlendShaderInfo` dataclass:**
   ```python
   @dataclass(frozen=True)
   class BlendShaderInfo:
       blend_mode: BlendMode
       requires_shader: bool
       glsl_source: str | None  # None if GL blend functions suffice
       blend_params: BlendFuncParams | None  # None if shader required
   ```

**Test plan (TDD):**
- Test file: `tests/test_tools/test_compositing_shaders.py`
- Test: `generate_overlay_glsl()` returns valid GLSL with `texture2D` calls
- Test: `generate_soft_light_glsl()` returns valid GLSL
- Test: OVERLAY `compute_blend_params()` returns `requires_shader=True`
- Test: SOURCE/OVER/MULTIPLY still return `requires_shader=False`
- Pure logic tests — no GPU required

**Acceptance criteria:**
- OVERLAY and SOFT_LIGHT have proper mathematical implementations
- Shader-based path is clearly distinguished from GL blend path
- No regression on existing blend modes

---

## Batch Execution Plan

```
Batch 1 (parallel — no dependencies):
├── S5-1: libplacebo intensity blending  [C plugin, GPU]
├── S5-2: OCIO error handling            [C++ plugin, GPU]
├── S5-3: ACES 2.0 config bundle         [Python, optional OCIO dep]
├── S5-4: IDT auto-detection             [Python, pure logic]
└── S5-7: Overlay/SoftLight shaders      [Python + GLSL, pure logic]

Batch 2 (depends on Batch 1):
├── S5-5: 10-bit pipeline verification   [depends on S5-1, S5-2 fixes]
└── S5-6: Docker GPU deps update         [depends on S5-1, S5-2 builds]
```

Batch 1 subtasks are all independent: different files, different languages, no shared state. Run all 5 agents in parallel.

Batch 2 requires the plugin fixes from Batch 1 to be in place before verification and Docker integration can proceed.

---

## Integration Test Plan

After all subtasks complete, run integration tests:

1. **Full color pipeline E2E** (`@pytest.mark.gpu`):
   - Ingest video → auto-detect IDT → apply IDT → apply grade → apply LUT (intensity=0.7) → apply ODT → render → probe output
   - Verify: output exists, codec correct, duration matches, bit depth >= 10

2. **ACES roundtrip** (`@pytest.mark.gpu`):
   - ACEScct working space → grade → export to Rec.709
   - Verify color values are within expected ranges

3. **Agent tool integration** (pure logic):
   - Use `EditingSession.call_tool("color_grade", ...)` → verify state tracking
   - Use `EditingSession.call_tool("lut_parse", ...)` → verify LUT data returned

---

## Risk Factors

| Risk | Mitigation |
|------|------------|
| OCIO 2.5 not available in Ubuntu apt | Build from source in Docker (add ~3min to build) |
| GL format negotiation fails for 10-bit | Fall back to RGBA16F explicitly in plugin caps |
| ACES config too large to bundle | Use OCIO builtin config API (no file needed) |
| Camera IDT mapping incomplete | Ship known mappings, allow manual override |
