# Phase 5: GPU Color Pipeline — Design

> Designed 2026-03-15. Informed by web research into libplacebo v7.360.0, OCIO 2.5.1, and GES GL filter integration. Revised after architectural critique.

---

## Research Findings

### libplacebo v7.360.0
- `pl_render_image()` handles tone mapping + gamut mapping automatically when source/target `pl_color_space` differ on `pl_frame` structs
- `pl_color_space` has primaries for ACES AP0/AP1, BT.2020, Display P3, etc. Transfers include PQ, HLG, V-Log, S-Log1/2, sRGB, BT.1886
- 10+ tone mapping algorithms: spline (default), BT.2390, BT.2446a, ST.2094-40/10, Reinhard, Mobius, Hable, gamma, linear
- Gamut mapping via `pl_gamut_map_function`: perceptual, softclip, relative, absolute, desaturate, darken, highlight, linear
- Peak detection for dynamic HDR via `pl_peak_detect_params`
- `pl_opengl_wrap()` supports `GL_RGBA16F` — can upgrade from RGBA8
- Presets: `pl_render_fast_params`, `pl_render_default_params`, `pl_render_high_quality_params`
- Breaking changes since v7.349: `skip_target_clearing` → `background`/`border`, cubic filter consolidation, XYZ linearization change

### OCIO 2.5.1
- **ABI break from 2.5.0**: `addToDeclareShaderCode()` split into parameter/texture variants; `addTexture()`/`add3DTexture()` now return binding indices
- **No C API in v2** — C++ wrapper with `extern "C"` required (already have this)
- Multiple 3D LUT textures possible per transform — must iterate all via `getNum3DTextures()`
- 1D/2D textures also generated — must iterate via `getNumTextures()`
- Per-frame uniforms via `getNumUniforms()` for future DynamicProperty support
- **ACES 2.0 built-in configs**: `ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5` — URI strings work anywhere config path accepted
- IDT/ODT chains via `DisplayViewTransform` (recommended) or `GroupTransform` (explicit control)

### GES + GL Filter Integration
- `GES.Effect.new(bin_description)` — custom GstGLFilter elements work as effects
- **Must wrap with `glupload ! ... ! gldownload`** — GES auto-inserts `videoconvert` which breaks GL caps
- **Compound effect approach**: chain IDT+grade+ODT in single bin description for zero-copy GL texture passing and float precision preservation
- **All properties baked into bin description string at construction time** — no dynamic modification after creation (avoids dual-instance disambiguation problem)
- Effect ordering: higher index = runs first (source-nearest); `clip.set_top_effect_index()` to reorder
- `GstGLFilter` elements automatically share GL context via `GstGLDisplay` propagation
- GStreamer has NO native float video formats — precision preserved by staying in GL textures internally, not through caps negotiation
- GStreamer 1.28: CUDA/GL interop fix, new 10-bit RGB formats (BGR10x2_LE, RGB10x2_LE)
- **GL context propagation in NLE is unvalidated** — must verify experimentally before committing to compound approach

---

## Current State

### Existing code
- `gstplacebofilter` (494 lines) — .cube LUT loading via `pl_lut_parse_cube()`, intensity blending via custom `pl_shader_custom()` GLSL, `pl_render_image()` for rendering. **Bug:** uses `pl_color_repr_sdtv` (YCbCr) on RGB data. Missing: tone mapping, HDR, gamut mapping, color space conversion, float precision
- `gstociofilter` (526 lines) — OCIO processor creation, GLSL shader compilation, single 3D LUT upload as GL_TEXTURE_3D, fullscreen quad rendering. **Bug:** FBO+VAO+VBO created/destroyed per frame. Missing: multiple texture support, 1D texture support, alpha preservation
- `ocio_processor.cpp` (290 lines) — C++ wrapper with C API. Creates `GpuShaderDesc`, extracts shader text + first 3D LUT data. Missing: multiple 3D textures, 1D/2D textures, uniform enumeration, DisplayViewTransform, `ocio://` URI support
- `tools/color.py` — pure logic: grade params, CDL params, LUT parse, GLSL generation, color transform params. **Bug:** `compute_color_transform()` uses `os.path.isfile()` which rejects `ocio://` URIs
- `agent/tools/color.py` — tool registration: color_grade, cdl, lut_parse
- `operations.py` — NO color operations wired to GES yet
- Docker: already builds OCIO 2.5.1 + libplacebo 7.349.0 + both plugins

---

## Architecture

### Core principle: Compound effect with float-precision chain

The IDT→processing→ODT chain runs as a **single compound `GES.Effect`** bin description:

```
glupload ! glcolorconvert !
  ociofilter[IDT] ! placebofilter[grade/LUT/tonemap] ! ociofilter[ODT]
! glcolorconvert ! gldownload
```

- Consecutive `GstGLFilter` elements share GL context and pass `GLMemory` textures with zero-copy
- Each element's internal FBO uses `GL_RGBA16F` for float-precision processing
- No 8-bit truncation between stages (would happen with separate effects due to `videoconvert` round trips)
- **All properties baked into the bin description string at construction time** — avoids the dual-ociofilter disambiguation problem (cannot use `ClassName::property-name` to distinguish two instances of the same class)
- To change parameters: remove old effect, create new compound effect with updated bin string
- Color space names with spaces escaped in bin description using backslash: `src-colorspace=ARRI\\ LogC4` (Python string: `"ARRI\\\\ LogC4"` → gst-launch sees `ARRI\ LogC4`)

### Standalone effects for simple operations

Single LUT application or single color space transform can be individual `GES.Effect`s when the full pipeline isn't needed:

```
glupload ! glcolorconvert ! placebofilter lut-path=/path intensity=0.8 ! glcolorconvert ! gldownload
```

### Validation-first approach

**Task 0 (before any implementation):** Verify that the compound GL bin approach works inside GES/NLE. Test that `glupload`/`gldownload` within a `GES.Effect` bin correctly propagates GL context. If this fails, fall back to individual effects (accepting 8-bit boundary precision loss) or investigate NLE GL context forwarding fixes.

---

## Design

### Section 1 — Pure logic layer (`tools/color.py`) [implement first — testable locally]

**New dataclasses:**

```python
@dataclass(frozen=True)
class ColorPipelineParams:
    idt_src: str           # Camera color space (e.g., "ARRI LogC4")
    working_space: str     # Working space (default "ACEScg")
    odt_display: str       # Display (e.g., "sRGB")
    odt_view: str          # View transform (e.g., "ACES 2.0 - SDR Video")
    config_path: str       # OCIO config path or ocio:// URI

@dataclass(frozen=True)
class ToneMappingParams:
    algorithm: str         # spline, bt2390, hable, reinhard, etc.
    gamut_mapping: str     # perceptual, softclip, relative, etc.
    peak_detect: bool

@dataclass(frozen=True)
class LUTApplicationParams:
    lut_path: str
    intensity: float
    tone_mapping: ToneMappingParams | None  # optional tone mapping config
```

**New validation functions:**

```python
VALID_TONE_MAPPING = {"spline", "bt2390", "bt2446a", "reinhard", "mobius", "hable", "gamma", "linear", "clip"}
VALID_GAMUT_MAPPING = {"perceptual", "softclip", "relative", "absolute", "desaturate", "darken", "highlight", "linear"}
VALID_PRIMARIES = {"bt709", "bt2020", "display-p3", "ap0", "ap1", "bt601-525", "bt601-625"}
VALID_TRANSFERS = {"srgb", "bt1886", "pq", "hlg", "vlog", "slog2", "linear", "gamma22", "gamma28"}

def compute_color_pipeline(idt_src, working_space="ACEScg",
                           odt_display="sRGB", odt_view="ACES 2.0 - SDR Video",
                           config_path="ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5") -> ColorPipelineParams:
    """Validate and compute full color pipeline parameters."""

def compute_tone_mapping(algorithm="spline", gamut_mapping="perceptual", peak_detect=False) -> ToneMappingParams:
    """Validate tone mapping parameters."""
```

**Fix existing bug:** Update `compute_color_transform()` to accept `ocio://` URIs — skip `os.path.isfile()` check when path starts with `ocio://`.

### Section 2 — Upgrade `gstplacebofilter` (C) [parallel with Section 3]

**Goal:** Tone mapping, gamut mapping, HDR→SDR, color space conversion, float precision.

**New properties (using `g_param_spec_enum` with registered GType, not strings):**

| Property | Type | Default | Purpose |
|----------|------|---------|---------|
| `tone-mapping` | enum `GstPlaceboToneMapping` | `SPLINE` | Algorithm selection |
| `gamut-mapping` | enum `GstPlaceboGamutMapping` | `PERCEPTUAL` | Gamut mapping mode |
| `peak-detect` | boolean | `FALSE` | Dynamic HDR peak detection (off by default — simpler first) |
| `src-primaries` | enum `GstPlaceboPrimaries` | `BT709` | Source color primaries |
| `src-transfer` | enum `GstPlaceboTransfer` | `SRGB` | Source transfer function |
| `dst-primaries` | enum `GstPlaceboPrimaries` | `BT709` | Destination color primaries |
| `dst-transfer` | enum `GstPlaceboTransfer` | `SRGB` | Destination transfer function |

**Critical bug fixes:**
- **Fix `pl_color_repr`**: Change from `pl_color_repr_sdtv` (YCbCr) to explicit RGB repr: `.sys = PL_COLOR_SYSTEM_RGB, .levels = PL_COLOR_LEVELS_FULL`. Current code corrupts colors by applying an incorrect YCbCr-to-RGB matrix.
- **Remove stale `src-colorspace`/`dst-colorspace` string properties**: Currently marked "for metadata/future use" — replace with the new enum-typed primaries/transfer properties to avoid naming collisions with `ociofilter`.

**Key changes:**
- Set `image.color` and `target.color` on `pl_frame` structs from enum properties → mapped to `pl_color_primaries` + `pl_color_transfer` constants
- Configure `pl_color_map_params` with selected tone mapping function and gamut mapping function
- Use `pl_render_high_quality_params` as base (enables dithering, debanding)
- Wrap textures with `GL_RGBA16F` instead of `GL_RGBA8` in `pl_opengl_wrap_params.iformat`
- Keep existing LUT/intensity functionality intact

### Section 3 — Upgrade `gstociofilter` + `ocio_processor.cpp` (C/C++) [parallel with Section 2]

**Goal:** Multiple textures, ACES 2.0 built-in configs, DisplayViewTransform, correctness fixes.

**New properties on `gstociofilter`:**

| Property | Type | Default | Purpose |
|----------|------|---------|---------|
| `display` | string | NULL | OCIO display name (alternative to dst-colorspace) |
| `view` | string | NULL | OCIO view name (used with display) |

**Deferred to Phase 7 (scope reduction):** DynamicProperty exposure/contrast/gamma. Adds significant complexity (per-frame uniform upload, optimization-level constraints with `OPTIMIZATION_LOSSLESS`). Not needed for the core IDT→ODT pipeline.

**`ocio_processor.cpp` changes:**
- Support `ocio://` URI strings → `Config::CreateFromBuiltinConfig()` when path starts with `ocio://`
- **New function** `ocio_processor_new_display_view(config_path, src_cs, display, view, error_out)` — creates `DisplayViewTransform` instead of `ColorSpaceTransform`
- Extract ALL 3D textures (loop `getNum3DTextures()`) — store as vector of `{sampler_name, edge_len, data}`
- Extract ALL 1D/2D textures (loop `getNumTextures()`) — store as vector of `{sampler_name, width, height, channel, data}`
- New C API: `ocio_processor_get_num_3d_textures()`, `ocio_processor_get_3d_texture_info(index, ...)`, `ocio_processor_get_num_textures()`, `ocio_processor_get_texture_info(index, ...)`

**`gstociofilter.c` changes:**
- Upload all textures (3D + 1D/2D) to consecutive GL texture units (starting from unit 1)
- Bind all sampler uniforms by name from OCIO's `GpuShaderDesc`
- Check `GL_MAX_TEXTURE_IMAGE_UNITS` at startup; warn if < needed count
- **Fix alpha preservation**: Save and restore alpha channel in fragment shader: `float saved_alpha = col.a; OCIOColor(col); col.a = saved_alpha;`
- **Cache FBO, VAO, VBO** — create once in `gl_start`, destroy in `gl_stop` (currently created/destroyed per frame)
- FBO internal format → `GL_RGBA16F`

**Mode selection in `gl_start`:**
- If `display` + `view` are set → use `ocio_processor_new_display_view(config, src-colorspace, display, view)`. Note: `src-colorspace` is required in this mode as the input to the DisplayViewTransform.
- Else if `src-colorspace` + `dst-colorspace` are set (and `display`/`view` are NOT set) → use existing `ocio_processor_new()`
- Else → error

### Section 4 — Python GES operations (`operations.py`) [after Sections 1-3]

**New functions:**

```python
def apply_lut(timeline, clip_id, lut_path, intensity=1.0):
    """Apply a .cube LUT via placebofilter as a GES.Effect.

    Returns effect_id.
    Bin: glupload ! glcolorconvert ! placebofilter lut-path=<path> intensity=<val>
         ! glcolorconvert ! gldownload
    """

def apply_color_transform(timeline, clip_id, src_colorspace, dst_colorspace, config_path=None):
    """Apply OCIO color space transform as a GES.Effect.

    Returns effect_id.
    Bin: glupload ! glcolorconvert ! ociofilter config-path=<path>
         src-colorspace=<src> dst-colorspace=<dst> ! glcolorconvert ! gldownload
    """

def apply_color_pipeline(timeline, clip_id, idt_src, working_space="ACEScg",
                         odt_display="sRGB", odt_view="ACES 2.0 - SDR Video",
                         config_path="ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5",
                         lut_path=None, lut_intensity=1.0,
                         tone_mapping="spline", gamut_mapping="perceptual"):
    """Apply full IDT→grade→ODT compound effect. Returns effect_id.

    Constructs a single GES.Effect bin description with all properties baked in:
      glupload ! glcolorconvert !
      ociofilter config-path=<cfg> src-colorspace=<idt_src> dst-colorspace=<ws> !
      placebofilter tone-mapping=<tm> gamut-mapping=<gm> [lut-path=<lut>] !
      ociofilter config-path=<cfg> display=<disp> view=<view> src-colorspace=<ws> !
      glcolorconvert ! gldownload

    Color space names with spaces are escaped: 'ARRI LogC4' → 'ARRI\\ LogC4'
    To modify: remove effect by effect_id, re-create with new parameters.
    """
```

All functions: find clip → validate params via pure logic layer → build bin description string → create GES.Effect → add to clip → verify → return effect_id.

### Section 5 — ACES 2.0 default config

- Default config: `ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5` (built into OCIO 2.5)
- Default working space: ACEScg (AP1 primaries, linear) — standard for VFX/compositing
- IDT auto-detection: read `agent:camera_colorspace` from asset metadata (set at ingest)
- Store config choice in project metadata: `agent:ocio_config` key on timeline
- Available camera IDTs in studio config: ARRI LogC3/4, RED Log3G10, Sony S-Log3, Canon CanonLog2/3, Panasonic V-Log, BMD Film Gen5
- Available displays: sRGB, Display P3, Rec.1886/709, Rec.2100-PQ, Rec.2100-HLG

### Section 6 — Docker updates

- Bump libplacebo: 7.349.0 → 7.360.0 (latest stable, fixes XYZ linearization)
- Update meson version constraints: `plugins/meson.build` OCIO `>= 2.3` → `>= 2.5.1`, libplacebo `>= 7.349` → `>= 7.360` to match actual requirements
- Verify OCIO 2.5.1 built-in configs work: `ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5`
- Add compound pipeline smoke test to Dockerfile
- Verify CUDA-GL interop on GStreamer 1.28.1 (fix shipped in 1.28)

---

## Testing Strategy

### Local tests (no GES/GPU required)
- Unit tests for all new dataclasses and validation functions in `tools/color.py`
- Validate known-good and known-bad algorithm names, color space names, config URIs
- Test `ocio://` URI handling in `compute_color_transform()`
- Test bin description string construction with special characters (spaces, quotes)

### Docker/GES tests
- **Task 0 validation test**: compound GL bin in GES — verify `glupload`/`gldownload` propagates GL context in NLE pipeline
- Integration tests for each `operations.py` function with real GES timeline
- Render a test clip through `apply_color_pipeline()` and probe output for non-black, non-corrupt frames
- Verify standalone `apply_lut()` and `apply_color_transform()` produce correct output

### Docker/GPU tests
- C plugin smoke tests via `gst-launch-1.0`:
  - `placebofilter` with tone mapping (HDR source → SDR output)
  - `ociofilter` with ACES 2.0 built-in config
  - Compound bin with both plugins chained
- Pixel-level regression tests: render known input → compare output to reference within tolerance
- Multiple texture test: verify OCIO transforms that generate >1 LUT texture work correctly
- Alpha preservation test: verify alpha channel survives OCIO transform

---

## Implementation Order

```
Task 0: Validate compound GL bin in GES (BLOCKING — if fails, redesign)
    ↓
Section 1: Pure logic (tools/color.py) — local tests
    ↓
Section 2: placebofilter upgrade ─┐
                                  ├── parallel, independent C code
Section 3: ociofilter upgrade ────┘
    ↓
Section 4: Python GES operations — depends on 1, 2, 3
    ↓
Section 5: ACES config wiring — depends on 3, 4
    ↓
Section 6: Docker updates — depends on 2, 3
```

---

## Out of Scope (YAGNI)

- Vulkan backend for libplacebo (OpenGL sufficient, CUDA-GL interop works)
- OCIO Python bindings (C++ wrapper sufficient)
- Custom skiafilter (Phase 6 text overlay)
- HDR display output (Wayland HDR — render to HDR file formats instead)
- CUDA decode → GL direct path (Phase 7 performance)
- DynamicProperty exposure/contrast/gamma (Phase 7 — adds per-frame uniform complexity)
- Peak detection enabled by default (off by default, enable as follow-up)
- Per-frame uniform caching/optimization (premature)

---

## Risks

1. **GL context propagation in NLE**: `glupload`/`gldownload` in compound `GES.Effect` may not receive GL context messages from NLE. Mitigation: Task 0 validates this experimentally before any implementation.
2. **OCIO shader GLSL compatibility**: Generated GLSL may not compile on all GL versions. Mitigation: target `GPU_LANGUAGE_GLSL_1_3`, test in Docker container.
3. **Multiple texture unit exhaustion**: Complex OCIO transforms can need 5+ texture units. Mitigation: check `GL_MAX_TEXTURE_IMAGE_UNITS` at startup, warn if insufficient.
4. **libplacebo GL context wrapping**: `pl_opengl_create()` wraps current GL context — may conflict with OCIO's direct GL calls. Mitigation: GstGLFilter runs all GL ops on single thread; libplacebo wraps don't take ownership of context.
5. **RGBA16F memory bandwidth**: 2x bandwidth vs RGBA8. Acceptable for color pipeline — not applied to every frame operation.
6. **Bin description escaping**: Color space names with spaces need `\` escaping in gst-launch syntax. Mitigation: helper function that escapes all special characters.
7. **Compound effect immutability**: Properties baked at creation time; changing requires remove+recreate. Acceptable tradeoff vs. unsolvable dual-instance disambiguation.

---

## Sources

- [libplacebo renderer docs](https://libplacebo.org/renderer/)
- [libplacebo tone_mapping.h](https://github.com/haasn/libplacebo/blob/master/src/include/libplacebo/tone_mapping.h)
- [libplacebo v7.360.0 release](https://code.videolan.org/videolan/libplacebo/-/tags/v7.360.0)
- [OCIO 2.5 release notes](https://opencolorio.readthedocs.io/en/latest/releases/ocio_2_5.html)
- [OCIO GPU shader API](https://opencolorio.readthedocs.io/en/latest/api/shaders.html)
- [OCIO studio config](https://opencolorio.readthedocs.io/en/latest/configurations/aces_studio.html)
- [GES.Effect docs](https://gstreamer.freedesktop.org/documentation/gst-editing-services/geseffect.html)
- [GStreamer GL design](https://gstreamer.freedesktop.org/documentation/additional/design/opengl.html)
- [GStreamer 1.28 release notes](https://gstreamer.freedesktop.org/releases/1.28/)
