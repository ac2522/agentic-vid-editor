# Agentic Video Editor: Architecture Design v2

> Revised 2026-03-08. Incorporates findings from two external architecture reviews, GPU API research, project format analysis, preview system design, current SOTA model research, GES capability analysis, and intermediate codec evaluation.
>
> Changes from v1: GPU API decision (CUDA+OpenGL, not mixed), project format (GES/XGES, not MLT XML or OTIO), preview system (segment-cached LL-HLS + WebSocket frame serving), visual analysis in ingest, error recovery, Skia for motion graphics, current 2026 model stack, frame rate conforming policy, tool-first architecture with optional MCP, non-destructive IDTs, DNxHR HQX default intermediate, dual compositor strategy, Qwen3-ForcedAligner for alignment.

## Vision

A coding-agent-driven video editing system where you describe what you want in plain language (or from a script), and the agent assembles, trims, layers, grades, and exports the timeline. Every tool is validated automatically via TDD without manual review.

The interface follows the Claude Code CLI model: a chat/prompt interface with a lightweight preview panel showing proxy renders and a timeline scrubber. Users can select frames or time ranges and pass that context to the agent alongside instructions.

## Guiding Principles

1. **Ingest-normalise first.** The agent never works on camera-native formats. Every clip gets transcoded to a known intermediate on ingest. Camera log encoding is preserved in the intermediate; IDTs are applied non-destructively at render time.

2. **Tools are pure functions.** Every tool takes explicit inputs, returns a result, and has no side effects outside its declared output path. Trivially testable.

3. **The timeline is data, not state.** The edit lives in a GES/XGES project file. The agent reads and writes this file via GES Python bindings; rendering uses GES's native GStreamer pipeline construction.

4. **Previews are always cheap.** Segment-cached proxy renders update incrementally. Only affected segments re-render after a change. Full-quality export is explicit and separate.

5. **The rendering engine is pluggable.** The agent's intelligence layer is engine-agnostic. The primary engine is GES/GStreamer. OTIO export enables optional finishing in Resolve, Premiere, or FCPX.

6. **Workflow-agnostic.** The framework handles talking-head, documentary, short-form, and cinematic content equally. The user's system instructions and prompts define the workflow, not the tool architecture.

7. **Single GPU API.** CUDA for decode/encode, OpenGL for compositing and processing. CUDA-OpenGL interop is established (requires GStreamer >= 1.28.1). No Vulkan-CUDA boundary crossings.

8. **Verify before reporting success.** Every tool execution is followed by validation. Errors are caught and retried or escalated, not silently swallowed.

9. **Local-first AI.** All neural network inference (transcription, vision analysis, segmentation, detection) runs locally on GPU. Cloud APIs (Claude Opus 4.6, GPT-5.4) are used only for the orchestration agent. No footage leaves the machine unless the user explicitly exports.

10. **Current models only.** Use 2026 SOTA models, not legacy versions. The model stack is a living document that updates as better models release.

11. **Accept any footage.** The project has a configured frame rate. All incoming footage gets conformed to project fps at ingest, regardless of source frame rate. No restrictions on source material.

---

## GPU API Decision

### The problem (from v1 review)

The v1 design crossed three incompatible GPU APIs: PyNvVideoCodec (CUDA), libplacebo (Vulkan), GStreamer compositor (CUDA/GL). Each boundary requires explicit memory export/import with no existing framework to automate it.

### The decision: CUDA + OpenGL

| Role | Technology | GPU API |
|---|---|---|
| **Decode** | GStreamer nvcodec (nvh264dec, nvh265dec, etc.) | CUDA |
| **Upload to GL** | GStreamer `cudaupload` → `glupload` | CUDA→OpenGL (established interop) |
| **Compositing (export)** | `glvideomixer` | OpenGL |
| **Compositing (preview)** | `skiacompositor` (29 blend modes, CPU) | CPU |
| **Color processing** | libplacebo via OpenGL backend | OpenGL |
| **Text/graphics** | skia-python v144 rendering to numpy → GStreamer appsrc | CPU → GL upload |
| **Color grading** | Custom GLSL shaders via `glshader` | OpenGL |
| **IDT application** | OCIO GStreamer element (custom `GstGLFilter`) | OpenGL |
| **Download from GL** | `gldownload` → `cudadownload` | OpenGL→CUDA |
| **Encode** | GStreamer nvcodec (nvh264enc, nvh265enc, nvav1enc) | CUDA |

**Hard requirement: GStreamer >= 1.28.1.** CUDA-GL interop bugs were fixed in this version (February 2026). MR !614 optimized GL resource registration from per-frame to one-time.

### Why not other options

- **Pure CUDA:** `cudacompositor` has no blend mode support (only `source` and `over` operators). No color processing, no text rendering, no custom shader mechanism.
- **Pure Vulkan:** No Vulkan compositor exists in GStreamer. H.265/AV1 encode not available. Early-stage maturity. Realistic timeline for a complete pipeline is 2027+.
- **CUDA-Vulkan interop:** Requires `VK_KHR_external_memory` + `cudaImportExternalMemory` + semaphore sync. No production video pipeline has shipped with this. Unproven and fragile.
- **CUDA-OpenGL interop:** Established via `cuGraphicsGLRegisterImage`. GStreamer handles this internally between its CUDA and GL elements.

### Why not Vulkan for libplacebo?

libplacebo's **OpenGL backend** supports nearly all the same features as its Vulkan backend: tone mapping, color management, HDR processing, gamut mapping, ICC profiles, 3D LUT application, and advanced scaling. Requires GL 4.3+ for compute shaders and SSBOs. Missing vs Vulkan: no subgroup operations (affects optimized scaling), no native push constants (falls back to UBO at binding point 0). These are performance differences, not correctness differences. For offline/near-offline rendering in a video editor, this is negligible. Staying in OpenGL avoids the entire CUDA-Vulkan interop problem.

### Dual compositor strategy

**Export (GPU):** `glvideomixer` — the only GStreamer compositor with real blend mode control:
- Per-pad `blend-function-src-rgb/alpha`, `blend-function-dst-rgb/alpha` (15 OpenGL blend functions each)
- `blend-equation-rgb/alpha` (add, subtract, reverse-subtract — note: `GL_MIN`/`GL_MAX` missing from GStreamer enum)
- Full `glBlendFuncSeparate` + `glBlendEquationSeparate` control
- **Known crash bugs:** #728 (crash with `gltransformation`), #786 (linking error on resolution change), #622 (null buffer dereference), #713 (SIGSEGV). These affect dynamic resolution changes and transform combinations. Mitigated by: fixed resolution in export, skiacompositor fallback, monitoring upstream fixes.

**Preview (CPU):** `skiacompositor` (GStreamer 1.28+) — 29 blend modes (source, over, multiply, screen, overlay, color-dodge, soft-light, hue, saturation, luminosity, etc.), per-pad positioning/sizing/alpha, anti-aliasing, 100+ pixel formats. CPU-only (no GPU acceleration yet). Handles 480p preview rendering easily. No known crash bugs.

GES's SmartMixer makes this switchable — compositor is selected by GStreamer element factory rank. Change rank to swap compositor for the render pipeline.

---

## Project Format Decision

### The problem (from v2 review)

The v2 architecture described two rendering paths (MLT/melt and GStreamer GPU) that are **architecturally incompatible separate frameworks**. MLT uses FFmpeg internally, has its own plugin system, and most effects are 8-bit CPU-only. There is no native way to route MLT rendering through a GStreamer GPU pipeline. Building a translator was estimated at 10,000-18,000 lines.

### The decision: GES/XGES as internal format, OTIO as interchange

**GES (GStreamer Editing Services)** eliminates the MLT-GStreamer impedance mismatch entirely. GES natively constructs GStreamer pipelines from its project model, giving us direct access to the GPU pipeline.

**What GES provides:**

- **Native GStreamer pipeline construction** — GES builds GStreamer pipelines directly. Our custom elements (libplacebo, GLSL shaders, OCIO) are automatically available as effects.
- **Keyframe animation** — via `GstInterpolationControlSource`. Four interpolation modes: none (step/hold), linear, cubic (natural spline, may overshoot), cubic-monotonic (guaranteed no overshoot). Keyframes stored as nanosecond timestamp:value pairs.
- **Any GStreamer element as an effect** — single-sink/single-source elements usable via `GES.Effect.new("element_name property=value")`. GES auto-inserts format converters around effects.
- **70+ SMPTE transitions** — bar wipes, box wipes, diagonals, clock wipes, iris, crossfade, fade-in. Custom transitions via shapewipe with grayscale bitmaps.
- **Pluggable compositor** — GES SmartMixer wraps the active compositor. Supports `compositor` (CPU), `glvideomixer` (OpenGL), `cudacompositor` (CUDA), `skiacompositor` (CPU, 29 blend modes). Selected by element factory rank.
- **Blend modes** — `framepositioner` exposes `operator` property per clip (since GStreamer 1.20). Maps to compositor blend modes.
- **Custom metadata** — `GESMetaContainer` interface on clips, layers, tracks, timelines, projects. Typed key-value pairs: `ges_meta_container_set_string(container, "agent:edit-intent", "Best take of line 3")`.
- **XGES XML format** — full serialization of timelines, assets, effects, keyframes, metadata.
- **Python bindings** — mature via GObject Introspection. Pitivi (full NLE) is written entirely in Python using GES. Proven completeness.
- **Sub-timelines** — via `gessrc` (since GStreamer 1.24). Enables nested compositions.

**OTIO** is used for:
- **Import:** Bring in timelines from other NLEs (via OTIO adapters for FCP XML, AAF, EDL)
- **Export:** Send timelines to Resolve, Premiere, FCPX, Avid for finishing
- **XGES adapter** exists (`otio-xges-adapter`, part of OTIO PR #609)

### What the agent reads and writes

The agent manipulates GES timelines via Python bindings (`gi.repository.GES`). The project structure:

```
project/
├── project.xges             # XGES project file (source of truth)
├── assets/
│   ├── registry.json         # Asset metadata (paths, codecs, color spaces, IDTs, transcriptions)
│   └── media/
│       ├── working/          # DNxHR HQX intermediates in MXF (disk format, camera log preserved)
│       └── proxy/            # 480p H.264 proxies
├── cache/
│   ├── segments/             # Rendered preview segments (.ts files)
│   └── thumbnails/           # Timeline thumbnail strips
├── luts/                     # .cube LUT library
├── transcriptions/           # Voxtral/Qwen3-FA JSON transcripts
└── exports/                  # Final rendered output
```

### GES limitations (accepted tradeoffs)

| Limitation | Impact | Mitigation |
|---|---|---|
| Constant-rate speed only (no variable speed ramps) | Cannot do smooth speed curves | Acceptable for MVP. RIFE v4.25 for interpolated slow-mo. Variable ramps via manual keyframed playback rate is future work. |
| Max 2 sources overlapping at any timeline position | Limits complex multi-layer compositing | Workable for most editing. Complex compositing uses nested timelines via `gessrc`. |
| Nanosecond timestamps (rounding at some frame rates) | Potential frame-accuracy issues at 23.976fps, 29.97fps | GES issue #61 tracking frame-based addressing. Acceptable; Pitivi handles this in production. |
| Single maintainer (Thibault Saunier) | Bus factor risk | GES is in GStreamer monorepo (Collabora, Igalia backing). Maintain fork capability. |
| Colorimetry bug #111 | Wrong colors with mixed-colorspace inputs | Solved at ingest — all clips enter pipeline in camera log, IDT applied uniformly at render time. |

### What happens to MLT

MLT XML can be imported via OTIO (MLT XML → OTIO → GES). `melt` remains available as a debugging/validation tool but is not part of the primary rendering pipeline.

---

## High-Level Architecture

```
+------------------------------------------------------------------+
|                        USER INTERFACE                              |
|  +-------------------+  +-------------------------------------+  |
|  | Chat / Prompt     |  | Preview Panel (browser)             |  |
|  | (CLI or web)      |  | - LL-HLS video player (hls.js)      |  |
|  |                   |  | - WebSocket frame serving (scrub)    |  |
|  | System instrs     |  | - Timeline scrubber + track lanes    |  |
|  | Workflow prompts   |  | - Frame/range selection -> context   |  |
|  +-------------------+  +-------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                      AGENT INTELLIGENCE LAYER                     |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | Tool Registry    |  | Workflow Engine  |  | Understanding   | |
|  | (Tool RAG +      |  | (orchestration)  |  |                 | |
|  |  optional MCP)   |  |                  |  | Voxtral Mini 3B | |
|  | Deferred loading |  | Role-based       |  | + Qwen3-FA      | |
|  | Semantic search  |  | agent routing    |  | Qwen3-VL 8B     | |
|  +------------------+  |                  |  | Script matching | |
|                         | Verification     |  +-----------------+ |
|  +------------------+  | loop after each  |                      |
|  | Error Recovery   |  | tool execution   |                      |
|  | Validate -> Retry|  +------------------+                      |
|  | or Escalate      |                                            |
|  +------------------+                                            |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                      PROJECT DATA LAYER                           |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | GES Timeline     |  | Asset Registry   |  | LUT Library     | |
|  | (project.xges)   |  | (media metadata, |  | (.cube index,   | |
|  |                  |  |  proxies, paths,  |  |  tagged by type) | |
|  | Source of truth   |  |  color spaces,   |  |                 | |
|  | Includes effects, |  |  IDT references) |  | V-Log, ACES,   | |
|  | keyframes, grades |  |                  |  | creative, etc.  | |
|  | agent: metadata  |  | Transcriptions   |  |                 | |
|  +------------------+  | Visual analysis  |  +-----------------+ |
|                         +------------------+                      |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     RENDERING ENGINE LAYER                        |
|                                                                   |
|  GES constructs GStreamer pipelines natively                      |
|  GPU Pipeline: CUDA (decode/encode) + OpenGL (processing)        |
|                                                                   |
|  +----------------------------------------------------------+   |
|  | Decode: NVDEC via GStreamer nvcodec                        |   |
|  | Upload: cudaupload -> glupload (CUDA-GL interop)          |   |
|  | IDT:    OCIO GStreamer element (camera log -> working)     |   |
|  | Color:  libplacebo via OpenGL backend                     |   |
|  | Comp:   glvideomixer (export) / skiacompositor (preview)  |   |
|  | Text:   skia-python -> appsrc                             |   |
|  | Grade:  GLSL shaders (lift/gamma/gain, curves, CDL)       |   |
|  | ODT:    OCIO GStreamer element (working -> display)        |   |
|  | Download: gldownload -> cudadownload                      |   |
|  | Encode: NVENC via GStreamer nvcodec                        |   |
|  +----------------------------------------------------------+   |
|                                                                   |
|  +----------------------------------------------------------+   |
|  | Export Targets (via OTIO interchange)                      |   |
|  | - OTIO -> DaVinci Resolve (optional finishing/grading)     |   |
|  | - OTIO -> Premiere/FCPX/Avid (via adapters)               |   |
|  | - Direct: MP4/MOV/MKV via GStreamer + NVENC                |   |
|  +----------------------------------------------------------+   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     PREVIEW SYSTEM                                |
|                                                                   |
|  Segment-based cache with hybrid serving                         |
|                                                                   |
|  Playback: LL-HLS with partial segments (hls.js)                |
|  Scrubbing: WebSocket frame serving (JPEG/WebP per frame)        |
|  Thumbnails: Pre-generated strips at ingest                      |
|                                                                   |
|  - Timeline divided into 2-second segments                       |
|  - Cache key: SHA-256(clip refs + effect params + timeline state)|
|  - On edit: invalidate affected segments, re-render in priority  |
|    order (current playhead position first, then outward)         |
|  - Render via GES pipeline for each dirty segment -> .ts         |
|  - Serve LL-HLS with partial segments (200-500ms parts)          |
|  - Target: <2s update for playback, <200ms for scrub frames      |
|  - Visual indicators: blue (cached/ready), red (dirty/rendering) |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     INFRASTRUCTURE LAYER                          |
|                                                                   |
|  Docker + NVIDIA Container Toolkit                                |
|  GStreamer >= 1.28.1 (CUDA-GL interop fixes)                      |
|  CUDA + OpenGL GPU access (headless EGL)                          |
|  Vulkan ICD files (for FFmpeg ProRes RAW decode only)             |
|  NVIDIA_DRIVER_CAPABILITIES=all (not nvidia-headless driver)      |
|  GStreamer nvcodec built from source (gst-plugins-bad)            |
|  Volume mounts: /media, /project, /luts, /export                  |
+------------------------------------------------------------------+
```

---

## Core Components

### 1. Tool Registry

**Architecture: Tools first, MCP optional.**

Tools are the primary interface. The internal tool registry follows deferred-loading patterns (load 3-5 tools per step, not all 200) but is not literally an MCP server. MCP is an optional integration point for users who want to connect external MCP-compatible clients.

The agent has access to many tools (50-200+) but loads only 3-5 per step. Two-tier architecture:

**Core tools (always loaded):**
- `ingest` -- transcode any footage to working intermediate + proxy
- `project_read` / `project_write` -- read/write GES timeline (XGES serialization)
- `render_proxy` -- generate low-quality preview (triggers segment cache invalidation)
- `render_export` -- full-quality final render

**Indexed tools (loaded on demand via semantic search):**
- Color grading tools (apply_lut, lift_gamma_gain, curves, cdl)
- Editing tools (trim, split, slip, slide, ripple_delete)
- Compositing tools (overlay, blend, opacity, transform, crop)
- Audio tools (volume, fade, mix, normalize, noise_reduce)
- Text/graphics tools (title, lower_third, subtitle, watermark)
- Analysis tools (scene_detect, transcribe, match_script, detect_faces, segment_object)
- Effects tools (speed_change, stabilize, denoise, sharpen, upscale, interpolate_frames)
- Export tools (export_resolve, export_fcpx, export_premiere)

Tool metadata includes: input/output types, parameter schemas, GPU requirements (which model, how much VRAM), processing cost estimate. Tools are discovered via deferred loading with semantic search.

**Optional MCP integration:** An MCP server wrapper can expose the tool registry to external clients (Claude Code, Cursor, etc.). This is not the primary interface -- it's an integration option for users who want it. The concern with MCP-only is context window flooding; the tool-based approach keeps context lean.

### 2. Ingest Pipeline

Every piece of media goes through ingest before the agent touches it:

```
Camera-native file (ProRes RAW, H.265, H.264, any fps)
    |
    +-- FFmpeg 8.0 decode (ProRes RAW via Vulkan compute [experimental], others via NVDEC)
    |
    +---> Frame rate conforming
    |     - All footage conformed to project fps (user-configured, e.g. 24, 25, 30, 60)
    |     - Mixed frame rate sources (24p interviews + 60p B-roll + 30p phone) all conform
    |     - Same behavior as DaVinci Resolve timeline conforming
    |
    +---> Disk intermediate: DNxHR HQX in MXF (default) or ProRes 422 HQ in MOV (option)
    |     - Camera log encoding PRESERVED (V-Log stays V-Log, S-Log3 stays S-Log3)
    |     - IDT info stored in asset registry, applied non-destructively at render time
    |     - DNxHR HQX default: faster FFmpeg encode, no metadata rejection, MXF cross-NLE compat
    |     - ProRes 422 HQ option: for workflows requiring Apple ecosystem compatibility
    |     - Both codecs are CPU-only encode; this is an ingest cost, not a per-frame cost
    |     - 4K encode speed: ~7-15 fps (expect 15-35 min per 10-min 4K clip)
    |     - In GPU memory during processing, frames are NV12/RGBA surfaces (not intermediate codec)
    |
    +---> Proxy: 480p H.264 (.mp4)
    |     - For preview panel scrubbing and thumbnail generation
    |     - Fast to generate via NVENC, small files
    |
    +---> Thumbnail strips
    |     - Pre-generated at ingest for visual timeline scrubbing
    |     - One strip per clip, configurable density (e.g., 1 thumb per second)
    |
    +---> Transcription (local, on GPU)
    |     - Primary: Voxtral Mini 3B (Apache 2.0, ~9.5GB VRAM, multilingual)
    |     - Alternative: faster-whisper with large-v3-turbo (MIT, ~6GB VRAM, Whisper-compatible)
    |     - Fast bulk option: NVIDIA Parakeet TDT 1.1B (RTFx >2000, English-only)
    |     - Word-level alignment: Qwen3-ForcedAligner-0.6B (Apache 2.0, ~1.3GB VRAM, 63x RT)
    |       - 32-43ms AAS (3x more accurate than NFA, 100x more accurate than WhisperX on long-form)
    |       - Does not degrade on long-form audio (52.9ms AAS at 300s vs NFA's 246.7ms)
    |     - Speaker diarization: pyannote (ON-DEMAND, not mandatory)
    |       - 20-30+ min per hour of audio on RTX 4090 — dominates ingest time
    |       - Triggered only when user requests speaker-based editing
    |     - All inference runs locally on GPU. No audio leaves the machine.
    |
    +---> Visual analysis (local, on GPU)
    |     - Scene understanding: Qwen3-VL 8B (Apache 2.0, ~16GB VRAM, native video temporal modeling)
    |       - Shot type classification (wide, medium, close-up, cutaway)
    |       - Scene content descriptions with temporal grounding
    |       - Dominant colors / exposure / lighting assessment
    |       - Action and motion descriptions
    |     - Object detection/tracking: YOLO26 (AGPL-3.0, <8GB VRAM)
    |       - NMS-free end-to-end detection, segmentation, pose estimation
    |     - Face detection/recognition: InsightFace (MIT, <1GB VRAM)
    |       - SCRFD detection + ArcFace 512-D embeddings
    |       - Face tracking across clips for continuity
    |     - Shot boundary detection: PySceneDetect (BSD-3, CPU) + TransNetV2 (MIT, <1GB)
    |     - All inference runs locally. No footage leaves the machine.
    |
    +---> Asset registry entry: JSON metadata
          - Original path, working path, proxy path
          - Original and conformed frame rates
          - Duration, resolution, codec
          - Camera color space and transfer function (e.g., V-Gamut / V-Log)
          - IDT reference (which OCIO transform to apply at render time)
          - Transcription reference
          - Visual analysis reference (shot types, scene descriptions, faces, objects)
```

**VRAM sequencing note:** Qwen3-VL 8B alone takes ~16GB VRAM. Models run sequentially during ingest, not in parallel. Typical sequence: decode → transcode (CPU) → transcription (9.5GB) → alignment (1.3GB) → visual analysis (16GB) → face/object detection (1-8GB). Each model is loaded and unloaded in turn.

### 3. Project Data Model (GES/XGES)

The edit lives in a GES timeline. The agent reads and writes it via Python bindings.

**GES timeline construction:**
```python
import gi
gi.require_version('GES', '1.0')
gi.require_version('GstController', '1.0')
from gi.repository import Gst, GES, GstController

Gst.init(None)
GES.init()

# Create timeline with video + audio tracks
timeline = GES.Timeline.new_audio_video()
layer = timeline.append_layer()

# Add clip from asset
asset = GES.UriClipAsset.request_sync("file:///project/assets/media/working/interview.mxf")
clip = layer.add_asset(asset, start=0, inpoint=Gst.SECOND * 5,
                       duration=Gst.SECOND * 30, track_types=GES.TrackType.UNKNOWN)

# Add color grading effect (custom GLSL element)
grade = GES.Effect.new("glshader location=shaders/lift_gamma_gain.glsl")
clip.add_top_effect(grade, 0)

# Keyframe animation on effect property
cs = GstController.InterpolationControlSource()
cs.set_property('mode', GstController.InterpolationMode.CUBIC_MONOTONIC)
for te in clip.get_children(False):
    if isinstance(te, GES.Effect):
        te.set_control_source(cs, "saturation", "direct")
        cs.set(0, 0.5)                    # start: desaturated
        cs.set(2 * Gst.SECOND, 1.0)       # ramp to full color
        break

# Store agent metadata
clip.set_string("agent:edit-intent", "Best take of opening line")
clip.set_string("agent:generation-id", "conv-turn-42")

# Save
timeline.save_to_uri("file:///project/project.xges", None, True)
```

**XGES file structure:**
```xml
<ges version='0.7'>
  <project metadatas='agent:project-name=(string)Interview_Cut_v3'>
    <ressources>
      <asset id='file:///project/assets/media/working/interview.mxf'
             extractable-type-name='GESUriClip'
             metadatas='agent:camera-colorspace=(string)V-Gamut/V-Log,
                        agent:idt=(string)aces_vlog_to_ap1' />
    </ressources>
    <timeline metadatas='agent:working-colorspace=(string)ACEScct_AP1'>
      <track caps='video/x-raw(ANY)' track-type='4' />
      <track caps='audio/x-raw(ANY)' track-type='2' />
      <layer priority='0'>
        <clip id='0' asset-id='file:///project/assets/media/working/interview.mxf'
              type-name='GESUriClip' layer-priority='0' track-types='6'
              start='0' duration='30000000000' inpoint='5000000000' rate='0'
              metadatas='agent:edit-intent=(string)Best_take_of_opening_line'>
          <effect asset-id='glshader location=shaders/lift_gamma_gain.glsl'
                  type-name='GESEffect' track-type='4'>
            <binding type='direct' source_type='interpolation'
                     property='saturation' mode='3' track_id='0'
                     values='0:0.5 2000000000:1.0' />
          </effect>
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
```

**Agent-specific metadata** uses `agent:` prefixed keys via `GESMetaContainer`:
- `agent:edit-intent` -- why the agent made this edit (for explainability)
- `agent:generation-id` -- links edits to the conversation turn that produced them
- `agent:camera-colorspace` -- original camera color space (for IDT selection)
- `agent:idt` -- OCIO transform to apply (e.g., `aces_vlog_to_ap1`)

GES ignores unknown metadata keys during rendering. This follows the same principle as Kdenlive's `kdenlive:` convention in MLT XML.

### 4. Color Pipeline

Built on publicly available color science. Non-destructive IDTs applied at render time. All GPU processing in OpenGL via libplacebo, OCIO, and custom GLSL.

```
Camera Log preserved on disk (V-Log, S-Log3, LogC, etc.)
    |
    +-- Input Device Transform (IDT) — applied at RENDER TIME, not ingest
    |   Via OCIO GStreamer element (custom GstGLFilter)
    |   Transform defined in asset registry and XGES metadata
    |   Camera log -> project working space
    |   Non-destructive: original log encoding preserved in intermediate
    |   If wrong IDT assigned, change metadata and re-render — no re-ingest
    |
    +-- Working Space: ACES AP1 (ACEScct) [default] or DaVinci Wide Gamut
    |   ACES is vendor-neutral and standardized — correct for multi-NLE pipelines
    |   ACES 2.0 Output Transform significantly improved (2024-2025)
    |   DWG available as user option (primaries and transfer function fully published)
    |
    +-- Grading Operations (all in working space, all on GPU)
    |   - Lift/gamma/gain (GLSL shader via glshader)
    |   - Curves (spline interpolation, GLSL)
    |   - ASC CDL (open standard)
    |   - HSL qualification + secondary corrections (GLSL)
    |   - Creative LUTs (libplacebo 3D LUT via OpenGL)
    |
    +-- Output Device Transform (ODT) — working space -> display
    |   Via ACES 2.0 Output Transform, OpenDRT, or custom LUT
    |   Applied via OCIO GStreamer element
    |   libplacebo handles HDR tone mapping, gamut mapping
    |
    +-- Display: Rec.709, Rec.2020/PQ (HDR), etc.
```

**Bit depth throughout the pipeline:** GPU textures are 10-bit or 16-bit float (no 8-bit bottleneck). DNxHR HQX intermediates are 10-bit 4:2:2. OCIO and libplacebo operate at full precision in OpenGL.

### 5. Preview System

Segment-based caching with hybrid serving, modelled after DaVinci Resolve's Smart Cache and FCPX's background rendering.

**Architecture:**
1. Timeline is divided into **2-second segments** aligned to GOP boundaries
2. Each segment gets a **cache key**: SHA-256 of all inputs (clip references, in/out points, effect parameters, transitions, track state)
3. When an edit occurs:
   - Compute which segments are affected by the change
   - Mark those segments as dirty
   - Re-render in **priority order**: current playhead position first, then expand outward
4. Rendered segments are `.ts` files in the cache directory
5. **Dual serving protocol:**
   - **LL-HLS** (Low-Latency HLS) for continuous playback — partial segments (200-500ms parts), `EXT-X-MEDIA-SEQUENCE` for updates, no `#EXT-X-ENDLIST` (treated as live stream). Achieves <2s update latency on localhost.
   - **WebSocket** for interactive scrubbing — client sends timecode, server renders single frame via GES pipeline, returns JPEG/WebP. Target <200ms per frame.
   - **Thumbnail strips** for visual timeline scrubbing (pre-generated at ingest, lightweight)
6. The browser plays via **hls.js** (battle-tested JavaScript HLS player)
7. Visual indicators show cache status: **blue** (cached/ready), **red** (dirty/rendering)

**Rendering each segment:**
- Use GES pipeline with `skiacompositor` (CPU, 29 blend modes) at proxy quality
- Output to `.ts` (MPEG-TS) suitable for HLS
- Proxy quality: 480p, fast H.264 preset via NVENC
- Segment filenames versioned on re-render (`segment_005_v2.ts`) to avoid browser/hls.js caching stale content

**hls.js integration notes:**
- Treat preview as **live stream** (no `#EXT-X-ENDLIST`) to enable playlist auto-reload
- Use `EXT-X-MEDIA-SEQUENCE` to indicate segment changes
- Version segment filenames on re-render to prevent stale 404s after `stopLoad()`/`startLoad()`
- Use LL-HLS partial segments to hit <2s latency target (standard HLS gives 4-6s)

**Idle-triggered background rendering:**
- After 5 seconds of no edits, begin background rendering of remaining dirty segments
- Priority: segments near playhead first, then outward in both directions

### 6. Agent Interaction Model

**Target user: both professionals and amateurs, like Claude Code.**

Claude Code works for senior engineers and beginners. This editor works the same way. The framework provides powerful tools; the user's system instructions determine the level of autonomy and handholding.

- A professional colorist configures detailed system instructions ("always use ACES AP1, apply V-Log IDT, prefer parallel node structure") and gives precise commands.
- A YouTuber configures simple instructions ("make it look cinematic, remove silence, add lower thirds") and lets the agent do more work.
- Same engine, different prompts.

**Interaction patterns (modelled after Claude Code):**

- **System instructions** -- user-customizable project-level context (camera type, color space, preferred LUTs, editing style, level of autonomy)
- **Workflow prompts** -- reusable prompt templates for common tasks ("rough cut from script", "color grade interview", "add lower thirds")
- **Selection as context** -- when the user selects a frame or range in the preview panel, the timecodes and a thumbnail are included in the agent's context
- **Conversational** -- multi-turn: "trim 2 seconds off the start of that clip" → agent identifies "that clip" from selection context

Users can customize:
- System instructions (project settings, style preferences, autonomy level)
- Workflow prompts (reusable editing recipes)
- Tool configurations (preferred codecs, default LUTs, export presets)
- Agent behavior (how autonomous vs. confirmatory)
- Model selection (which STT model, which VLM, which cloud API)

**Orchestration agent:** Claude Opus 4.6 (`claude-opus-4-6`) as primary, with GPT-5.4 and Claude Sonnet 4.6 as alternatives. The framework is model-agnostic — any LLM with tool-use capability can drive the agent. Local LLMs are an option for users who don't want cloud API calls, though tool-use quality will vary.

### 7. Verification Loop

After each tool execution:

1. **Validate project file** -- load XGES via `GES.Timeline.new_from_uri()`, verify it loads without errors
2. **Verify outputs exist** -- check that referenced media files are present and have expected properties (duration, codec, resolution)
3. **For color operations** -- compare before/after histograms to verify the grade was applied
4. **For cuts** -- verify continuity of adjacent clips (no gaps, no overlaps unless intentional)
5. **For renders** -- verify output file is valid (probe with ffprobe)
6. **On failure** -- retry the operation once, then escalate to the user with a clear error message

This addresses the error amplification identified in multi-agent research (Kim et al., arXiv 2512.08296). Uncoordinated multi-agent systems show 17.2x error amplification; **centralized orchestration with verification reduces this to 4.4x**.

---

## Intermediate Format Clarification

| Context | Format | Why |
|---|---|---|
| **On disk (working)** | DNxHR HQX in MXF (default) | Faster FFmpeg encode than ProRes, no metadata rejection risk, 10-bit 4:2:2, cross-NLE compatible in MXF container. Camera log encoding preserved. |
| **On disk (working, option)** | ProRes 422 HQ in MOV | For Apple ecosystem workflows. FFmpeg `prores_ks` encoder (reverse-engineered, 14+ years without legal challenge). Broadcast delivery may reject non-Apple metadata — use Resolve Studio for final ProRes encode. |
| **On disk (proxy)** | H.264 480p (.mp4) | Small, fast to decode, good enough for preview. GPU encode via NVENC. |
| **In GPU memory** | NV12 / P010 / RGBA surfaces | Native GPU texture formats. Decode outputs these directly; processing operates on them; encode reads them directly. |
| **In OpenGL** | GL textures (10-bit or 16-bit float) | Compositing, color processing, text rendering all operate on GL textures. No 8-bit bottleneck. |

---

## Why Not Remotion?

1. **No GPU-accelerated video decode.** WebCodecs support is limited and CPU-bound for professional codecs. The user's footage is from a Panasonic S1 II.
2. **No professional color pipeline.** No OCIO, no wide-gamut working space, no HDR, no V-Log/V-Gamut workflow.
3. **Browser rendering is inherently slower** than native GPU pipelines for video compositing at professional resolutions.
4. **No 3D LUT support** in the browser rendering path.
5. **No control over GPU memory.** Cannot keep frames GPU-resident across operations.

Remotion is excellent for programmatic motion graphics and social media content. It is wrong for a professional camera-to-output pipeline.

---

## Custom Engineering Required

### Must Build

| Component | What It Is |
|---|---|
| **libplacebo GStreamer element** | Wraps libplacebo's OpenGL backend for color management, 3D LUT application, HDR tone mapping, and high-quality scaling inside GStreamer pipelines. ~2000-4000 lines C or Rust. Target `pl_renderer` + `pl_options` API at PL_API_VER 309. Reference: FFmpeg `vf_libplacebo.c` (1845 lines, Vulkan-only — our element targets OpenGL). |
| **OCIO GStreamer element** | Custom `GstGLFilter` subclass for OpenColorIO integration. Loads OCIO config, creates processor for IDT/ODT, generates GPU shader code via `GpuShaderDesc`, uploads LUT textures, applies OCIO shader in the GL pipeline. OCIO 2.5 has mature OpenGL shader pipeline. Needed for non-destructive IDT/ODT at render time. |
| **Color grading GLSL shaders** | Lift/gamma/gain, curves, ASC CDL, HSL qualification running as GLSL shaders via GStreamer `glshader` (single-input). Multi-input effects use compositor blend modes. |
| **GES agent interface** | Python library wrapping GES bindings for agent-friendly timeline manipulation. Programmatic clip placement, effect application, keyframe animation, metadata management. Higher-level than raw GES API. |
| **Motion graphics renderer** | skia-python v144 rendering text, lower thirds, titles, watermarks to RGBA numpy arrays, pushed to GStreamer via appsrc. Keyframe animation system for position, opacity, scale. Note: `gst-plugin-skia` does NOT provide text rendering — `skiacompositor` is compositor-only, `skiafilter` does not exist in current codebase. |
| **Preview server** | Segment-based cache manager + LL-HLS server + WebSocket frame server + browser UI with timeline scrubber, hls.js player, thumbnail strip viewer, and selection-to-agent context bridge |
| **Tool registry** | Internal tool registry with deferred loading and semantic search for 50-200+ tools. Optional MCP server wrapper for external client integration. |
| **Visual analysis pipeline** | Local GPU inference: Qwen3-VL 8B for scene understanding, YOLO26 for detection/tracking, InsightFace for faces, PySceneDetect + TransNetV2 for shot boundaries. SAM 3 for on-demand object segmentation. |
| **Verification system** | Post-execution validation for each tool: XGES load test, file integrity, format compliance, histogram comparison, continuity checks |
| **OTIO XGES interchange** | Validate and extend `otio-xges-adapter` for our use cases. Document which properties survive OTIO round-trip (effects, keyframes, speed, CDL — some will be lossy). Build custom MLT XML → OTIO reader if MLT import is needed. |

### Already Exists (Integrate, Don't Build)

**Rendering & Pipeline:**

| Component | Library | License |
|---|---|---|
| Project model + rendering | GES (GStreamer Editing Services) | LGPL |
| GPU video decode/encode | GStreamer nvcodec (NVDEC/NVENC) | LGPL |
| GPU compositing (export) | GStreamer `glvideomixer` | LGPL |
| CPU compositing (preview) | GStreamer `skiacompositor` (29 blend modes) | LGPL |
| Color space management | OpenColorIO 2.5 | BSD-3 |
| Color science algorithms | colour-science (Python) | BSD-3 |
| GPU color processing | libplacebo (OpenGL backend) | LGPL 2.1 |
| HDR tone mapping | libplacebo | LGPL 2.1 |
| Motion graphics / text | skia-python v144 + HarfBuzz + Pango | BSD / LGPL |
| Frame-level Python video | PyAV 16.1 | BSD |
| Timeline interchange | OpenTimelineIO 0.18 | Apache 2.0 |
| OTIO <-> XGES adapter | otio-xges-adapter (OTIO PR #609) | Apache 2.0 |
| Audio mixing | GStreamer audiomixer | LGPL |
| Docker + GPU | NVIDIA Container Toolkit | Apache 2.0 |
| HLS player | hls.js | Apache 2.0 |

**Local AI Models (all run on RTX 4090, no footage leaves the machine):**

| Task | Model | VRAM | License |
|---|---|---|---|
| Speech-to-text | Voxtral Mini 3B (primary) / faster-whisper large-v3-turbo (alternative) | ~9.5GB / ~6GB | Apache 2.0 / MIT |
| Bulk transcription | NVIDIA Parakeet TDT 1.1B | ~2-4GB | NeMo license |
| Word alignment | Qwen3-ForcedAligner-0.6B | ~1.3GB | Apache 2.0 |
| Speaker diarization | pyannote (on-demand, not mandatory) | ~2GB | MIT |
| Video understanding (VLM) | Qwen3-VL 8B | ~16-18GB | Apache 2.0 |
| Object detection/tracking | YOLO26 (nano-xl variants) | 1-8GB | AGPL-3.0 |
| Face detection/recognition | InsightFace (SCRFD + ArcFace) | <1GB | MIT |
| Scene/shot detection | PySceneDetect + TransNetV2 | CPU / <1GB | BSD-3 / MIT |
| Video segmentation | SAM 3 (Segment Anything Model 3) | ~4GB | Apache 2.0 |
| Frame interpolation | Practical-RIFE v4.25 | ~2-4GB | MIT |
| Video denoising | BasicVSR++ (via MMagic) | ~8-12GB | Apache 2.0 |
| Video upscaling | Real-ESRGAN | ~4GB | BSD-3 |

**Cloud APIs (orchestration agent only):**

| Role | Model | ID |
|---|---|---|
| Primary orchestration agent | Claude Opus 4.6 | `claude-opus-4-6` |
| Alternative orchestration | GPT-5.4 | `gpt-5.4` |
| Fast/cheap orchestration | Claude Sonnet 4.6 / GPT-5.3 Instant | `claude-sonnet-4-6` / `gpt-5.3-instant` |

---

## Known Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| libplacebo GStreamer element is substantial custom work (~2-4k lines) | High | Prototype early in Phase 1. Use libplacebo's OpenGL backend (simpler than Vulkan). Target `pl_renderer` + `pl_options` API. FFmpeg `vf_libplacebo.c` (1845 lines) as reference. |
| OCIO GStreamer element is custom work | High | OCIO 2.5 has mature OpenGL shader pipeline. Architecturally feasible (`GstGLFilter` subclass + `GpuShaderDesc`). Prototype alongside libplacebo element. |
| glvideomixer crash bugs (#728 crash w/ gltransformation, #786 linking error on resolution change, #622 null buffer, #713 SIGSEGV) | High | Dual compositor strategy: `skiacompositor` for preview (no crash bugs, 29 blend modes), `glvideomixer` for export (fixed resolution). Monitor and contribute upstream fixes. `compositor` (CPU) as ultimate fallback. |
| GES single maintainer (Thibault Saunier at Igalia) | Medium | GES is part of GStreamer monorepo (Collabora, Igalia backing). Pitivi depends on it. Maintain fork capability. |
| GES 2-source overlap limit | Medium | Workable for most editing. Complex multi-layer compositing uses nested timelines via `gessrc` (since GStreamer 1.24). |
| otio-xges-adapter gaps (speed effects, CDL) | Medium | OTIO export is lossy by design (interchange always is). Document which properties survive export. Extend adapter as needed. |
| FFmpeg ProRes RAW decoder is experimental | Medium | Reverse-engineered, limited real-world validation across all ProRes RAW sources. Fallback: DaVinci Resolve for ProRes RAW. Most users have H.265 or standard ProRes, not ProRes RAW. |
| ProRes licensing on Linux | Low | FFmpeg `prores_ks` has shipped 14+ years without legal action. Risk is broadcast metadata rejection, not litigation. DNxHR HQX is the default to avoid this entirely. ProRes available as option. |
| CUDA-OpenGL interop adds one GPU memory copy at API boundaries | Low | Copy happens once at pipeline start (decode→GL) and once at end (GL→encode). All processing between stays in GL. Negligible for offline rendering. |
| Diarization dominates ingest time (20-30+ min/hour) | Low | Made optional. On-demand step, not mandatory ingest. |

---

## Build Sequence

### Phase 1: Foundation

Dependencies: None

- Docker container with GStreamer >= 1.28.1 + nvcodec + OpenGL + FFmpeg 8.0 + GES
  - `NVIDIA_DRIVER_CAPABILITIES=all`, headless EGL, non-headless NVIDIA driver on host
  - GStreamer nvcodec built from source (`gst-plugins-bad` with `-Dnvcodec=enabled`)
- Ingest tool (any format → DNxHR HQX/MXF + proxy, camera log preserved, IDT stored in registry)
- GES Python interface (timeline CRUD, effect management, keyframe animation, metadata)
- Basic proxy render via GES pipeline → 480p H.264
- Test fixtures (FFmpeg-generated color bars, test tones) and TDD harness
  - GPU tests: `@pytest.mark.gpu` for conditional skipping in CI without GPU
  - Self-hosted GPU runner or cloud GPU CI (RunPod, Lambda) for full test suite
- Prototype libplacebo GStreamer element (OpenGL backend, basic LUT application)
- Prototype OCIO GStreamer element (basic IDT application)

### Phase 2: Core Editing

Dependencies: Phase 1 foundation

- Trim, split, concatenate tools (operating on GES timeline)
- Multi-track timeline assembly (layers, tracks)
- Transitions (GES native: 70+ SMPTE types, crossfade, custom via shapewipe)
- Speed change (GES constant rate; RIFE v4.25 for interpolated slow-mo)
- Audio placement and basic mixing (volume, fades via GES effects on audio tracks)
- Transcription: Voxtral Mini 3B (or faster-whisper) + Qwen3-ForcedAligner + script matching

### Phase 3: Color & Compositing

Dependencies: Phase 1 foundation + libplacebo/OCIO prototypes

- libplacebo GStreamer element (production-grade, OpenGL backend)
- OCIO GStreamer element (production-grade, IDT/ODT pipeline)
- Non-destructive IDT → working space → ODT pipeline
- LUT application (creative LUTs via libplacebo)
- Color grading GLSL shaders (lift/gamma/gain, CDL, curves via glshader)
- Multi-layer compositing via GES SmartMixer (glvideomixer for export, skiacompositor for preview)
- Motion graphics via skia-python → appsrc (animated text, lower thirds, titles)

### Phase 4: Preview & Polish

Dependencies: Phase 2 core editing

- Segment-based preview cache manager
- LL-HLS server with partial segments + hls.js browser player
- WebSocket frame server for interactive scrubbing (<200ms target)
- Timeline scrubber UI with track lanes, thumbnail strips, and cache status indicators
- Selection → agent context bridge (timecodes + thumbnail)
- Full-quality export pipeline (GPU-accelerated via NVENC, glvideomixer compositor)
- OTIO export to Resolve/Premiere/FCPX (via otio-xges-adapter)

### Phase 5: Agent Layer

Dependencies: Phase 2 core editing + Phase 4 preview

- Tool registry with deferred loading
- Tool RAG for semantic search across 50-200+ tools
- Visual analysis integration in ingest (Qwen3-VL 8B)
- Verification loop (post-execution validation)
- Workflow prompt system + system instruction customization
- Multi-turn conversational editing with selection context

**Phase dependencies are explicit.** Phase 3 (color) can run in parallel with Phase 2 (editing) once the libplacebo/OCIO prototypes from Phase 1 work. Phase 4 (preview) requires Phase 2. Phase 5 (agent) requires Phases 2 and 4.

---

## Unsolved Problems (Acknowledged)

These are known challenges in the field that affect our system but are not architecture problems:

1. **The taste/judgment gap.** Current agentic systems score 2.75/5.0 on editing professionalism vs 4.38/5.0 for humans (arXiv 2509.16811). Genre alignment (4.25 vs 4.66) and watchability (3.94 vs 4.28) gaps are much smaller, suggesting agentic pipelines excel at narrative coherence but fall short on technical polish. This improves as AI models improve. Our architecture doesn't need to solve taste — it needs to not prevent good taste when models catch up.

2. **Long-video narrative consistency.** Temporal reasoning and character tracking remain fragile in current models. The visual analysis pipeline helps but doesn't fully solve this.

3. **Multi-agent coordination tax.** Under fixed compute budgets, per-agent reasoning becomes thin beyond 3-4 agents (Kim et al., arXiv 2512.08296). Performance ranges from +80.9% improvement to -70% degradation depending on coordination strategy. Our centralized orchestration with verification addresses this.

4. **No standardized evaluation benchmark** for agentic video editing quality. We define our own metrics in the TDD harness (render accuracy, tool reliability, proxy fidelity) but aesthetic quality remains subjective.

5. **GES nanosecond timestamp rounding.** At frame rates like 23.976fps and 29.97fps, nanosecond timestamps can cause rounding errors (GES issue #61). Pitivi handles this in production, and the issue is under active development. Monitor and contribute upstream.
