# Phase 3: Color & Compositing — Design

## Overview
Color pipeline, compositing, and motion graphics. Follows same three-layer pattern as Phase 2: pure logic (testable locally), GES execution (Docker), E2E verification (Docker+GPU).

## Modules

### Color Pure Logic (`src/ave/tools/color.py`)
- `parse_cube_lut(path)` → CubeLUT (title, size, domain, 1D/3D table)
- `compute_color_grade(lift, gamma, gain, saturation, offset)` → ColorGradeParams
- `compute_cdl(slope, offset, power, saturation)` → CDLParams (ASC CDL)
- `generate_grade_glsl(params)` → GLSL source for glshader element
- `generate_cdl_glsl(params)` → GLSL source for glshader element
- `compute_lut_application(lut_path, intensity)` → LUTParams
- `compute_color_transform(src_cs, dst_cs, config_path)` → ColorTransformParams

### C GStreamer Plugins

**gst-libplacebo** (upgrade from Phase 1 passthrough):
- Load .cube LUTs via pl_lut_parse_cube()
- Apply LUT in render via pl_render_params.lut
- Color space properties for tone mapping
- Intensity property for LUT blend (0.0-1.0)

**gst-ocio** (upgrade from Phase 1 passthrough):
- C++ helper (ocio_processor.cpp) bridging C struct → OCIO C++ API
- Load OCIO config, create processor for src→dst transform
- Generate GPU shader via GpuShaderCreator
- Upload 3D LUT texture, compile+apply GLSL program

### Color GES Operations (`src/ave/tools/color_ops.py`)
- `apply_lut(timeline, clip_id, lut_path, intensity)` → placebofilter effect
- `apply_color_grade(timeline, clip_id, grade_params)` → glshader effect
- `apply_cdl(timeline, clip_id, cdl_params)` → glshader effect
- `apply_color_transform(timeline, clip_id, src_cs, dst_cs, config)` → ociofilter effect
- `apply_idt(timeline, clip_id)` → reads clip metadata, applies IDT via OCIO

### Compositing (`src/ave/tools/compositing.py`)
- `compute_layer_params(layers_config)` → LayerParams (validated ordering, alpha, blend, position)
- `apply_compositing(timeline, layers_config)` → configures GES SmartMixer pads
- Blend modes: source, over, multiply, screen, overlay, soft-light, add
- GES SmartMixer selects compositor by rank (glvideomixer export, skiacompositor preview)

### Motion Graphics (`src/ave/tools/motion_graphics.py`)
- `compute_text_overlay(text, font, size, position, color, duration_ns)` → TextOverlayParams
- `render_text_frame(params, width, height)` → numpy RGBA array via skia-python
- `create_text_clip(timeline, params)` → feeds frames via appsrc on dedicated layer
- Templates: lower_third, title_card, subtitle

## Testing Strategy

| Layer | Count | Environment |
|-------|-------|-------------|
| Pure logic (color.py) | ~35 | Local |
| Pure logic (compositing.py) | ~10 | Local |
| Pure logic (motion_graphics.py) | ~10 | Local |
| GLSL generation | ~10 | Local |
| LUT parsing | ~10 | Local |
| GES integration (color_ops.py) | ~15 | Docker |
| C plugins | ~10 | Docker+GPU |
| E2E: ingest→grade→render→probe | ~10 | Docker+GPU |
| **Total** | **~110** | |

## Test Data
- Synthetic .cube LUTs: identity, warm shift, cool shift, high contrast (generated in Python)
- OCIO config: minimal test config with sRGB ↔ linear ↔ ACEScg
- Existing fixtures: color_bars_1080p24.mp4, color_bars_720p30.mp4

## Docker Integration
- Add skia-python to Dockerfile
- Add meson build step for gst-libplacebo and gst-ocio plugins
- Add OCIO test config to test fixtures
