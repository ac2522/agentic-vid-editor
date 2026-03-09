# Agentic Video Editor: Architecture Design

> Design document 2026-03-07. Open-source, agent-driven video editing framework with GPU-accelerated rendering, TDD harness, and optional DaVinci Resolve export.

## Vision

A coding-agent-driven video editing system where you describe what you want in plain language (or from a script), and the agent assembles, trims, layers, grades, and exports the timeline. Every tool is validated automatically via TDD without manual review.

The interface follows the Claude Code CLI model: a chat/prompt interface with a lightweight preview panel showing proxy renders and a timeline scrubber. Users can select frames or time ranges and pass that context to the agent alongside instructions.

## Guiding Principles

1. **Ingest-normalise first.** The agent never works on camera-native formats. Every clip gets transcoded to a known intermediate on ingest. All tools downstream deal with that intermediate only.

2. **Tools are pure functions.** Every tool takes explicit inputs, returns a result, and has no side effects outside its declared output path. Trivially testable.

3. **The timeline is data, not state.** The edit lives in an OTIO JSON file, not in a live editor process. The agent reads and writes this file; rendering is a separate step.

4. **Previews are always cheap.** A low-quality proxy render is always available after any change. Full-quality export is explicit and separate.

5. **The rendering engine is pluggable.** The agent's intelligence layer is engine-agnostic. Start with GStreamer + libplacebo. Export to Resolve via OTIO for optional finishing.

6. **Workflow-agnostic.** The framework handles talking-head, documentary, short-form, and cinematic content equally. The user's system instructions and prompts define the workflow, not the tool architecture.

7. **GPU-first.** Keep data on GPU memory throughout the pipeline. Zero-copy where possible.

---

## High-Level Architecture

```
+------------------------------------------------------------------+
|                        USER INTERFACE                              |
|  +-------------------+  +-------------------------------------+  |
|  | Chat / Prompt     |  | Preview Panel                       |  |
|  | (CLI or web)      |  | - Proxy video player                |  |
|  |                   |  | - Timeline scrubber                 |  |
|  | System instrs     |  | - Frame/range selection → context   |  |
|  | Workflow prompts   |  |                                     |  |
|  +-------------------+  +-------------------------------------+  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                      AGENT INTELLIGENCE LAYER                     |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | Tool Registry    |  | Workflow Engine  |  | Script/Speech   | |
|  | (MCP + Tool RAG) |  | (orchestration)  |  | Matching        | |
|  |                  |  |                  |  | (WhisperX)      | |
|  | Deferred loading |  | Role-based       |  |                 | |
|  | Semantic search  |  | agent routing    |  | Take selection  | |
|  +------------------+  +------------------+  +-----------------+ |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                      TIMELINE DATA LAYER                          |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+ |
|  | OpenTimelineIO   |  | Asset Registry   |  | LUT Library     | |
|  | (.otio files)    |  | (media metadata, |  | (.cube index,   | |
|  |                  |  |  proxies, paths)  |  |  tagged by type) | |
|  | Source of truth  |  |                  |  |                 | |
|  | Version-ctrl'd   |  | Transcriptions   |  | V-Log, ACES,   | |
|  +------------------+  +------------------+  | creative, etc.) | |
|                                               +-----------------+ |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     RENDERING ENGINE LAYER                        |
|                                                                   |
|  +----------------------------------------------------------+   |
|  | Primary: GStreamer + GES + Custom GPU Elements            |   |
|  |                                                            |   |
|  |  Decode: PyNvVideoCodec / GStreamer nvcodec                |   |
|  |  Color:  libplacebo (custom GStreamer element) + OCIO      |   |
|  |  Comp:   GStreamer compositor + custom GLSL blend modes    |   |
|  |  Text:   Motion graphics renderer (Cairo/Skia → appsrc)   |   |
|  |  Grade:  Lift/gamma/gain, curves, CDL (custom GPU element) |   |
|  |  Audio:  GStreamer audiomixer                              |   |
|  |  Encode: NVENC via GStreamer or PyNvVideoCodec             |   |
|  +----------------------------------------------------------+   |
|                                                                   |
|  +----------------------------------------------------------+   |
|  | Export Targets                                             |   |
|  |                                                            |   |
|  |  OTIO → DaVinci Resolve (optional finishing/grading)       |   |
|  |  OTIO → Kdenlive/Premiere/FCPX (via OTIO adapters)        |   |
|  |  Direct: MP4/MOV/MKV via GStreamer pipeline                |   |
|  |  Proxy: 480p H.264 for fast preview                        |   |
|  +----------------------------------------------------------+   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     INFRASTRUCTURE LAYER                          |
|                                                                   |
|  Docker + NVIDIA Container Toolkit                                |
|  CUDA / Vulkan GPU access                                         |
|  Volume mounts for media, projects, LUTs                          |
+------------------------------------------------------------------+
```

---

## Core Components

### 1. Tool Registry (MCP + Tool RAG)

The agent has access to many tools (50-200+) but loads only 3-5 per step. Two-tier architecture:

**Core tools (always loaded):**
- `ingest` — transcode any footage to working intermediate + proxy
- `timeline_read` / `timeline_write` — read/write OTIO
- `render_proxy` — generate low-quality preview
- `render_export` — full-quality final render

**Indexed tools (loaded on demand via semantic search):**
- Color grading tools (apply_lut, lift_gamma_gain, curves, cdl)
- Editing tools (trim, split, slip, slide, ripple_delete)
- Compositing tools (overlay, blend, opacity, transform, crop)
- Audio tools (volume, fade, mix, normalize, noise_reduce)
- Text/graphics tools (title, lower_third, subtitle, watermark)
- Analysis tools (scene_detect, transcribe, match_script, detect_faces)
- Effects tools (speed_change, stabilize, denoise, sharpen)
- Export tools (export_resolve, export_fcpx, export_premiere)

Tool metadata includes: input/output types, parameter schemas, GPU requirements, processing cost estimate. Tools are discovered via MCP deferred loading pattern or Tool RAG embedding search.

### 2. Ingest Pipeline

Every piece of media goes through ingest before the agent touches it:

```
Camera-native file (ProRes RAW, H.265, H.264, etc.)
    │
    ├── FFmpeg 8.0 decode (ProRes RAW via Vulkan, others via NVDEC)
    │
    ├──→ Working copy: ProRes 422 HQ (.mov)
    │    - Known codec, known color space
    │    - All downstream tools work with this
    │
    ├──→ Proxy: 480p H.264 (.mp4)
    │    - For preview panel scrubbing
    │    - Fast to generate, small files
    │
    ├──→ Transcription: WhisperX → word-level JSON
    │    - Timestamps, speaker diarization
    │    - Stored alongside media metadata
    │
    └──→ Asset registry entry: JSON metadata
         - Original path, working path, proxy path
         - Duration, resolution, framerate, codec
         - Color space (V-Log/V-Gamut, Rec.709, etc.)
         - Transcription reference
```

### 3. Timeline Data Model (OTIO)

The edit lives in an OpenTimelineIO file — structured JSON. The agent reads and writes it directly. Example structure:

```
Timeline
├── Stack (root composition)
│   ├── Track (V1 - main video)
│   │   ├── Clip (interview_take3.mov, in: 00:01:23, out: 00:02:45)
│   │   ├── Transition (cross_dissolve, 24 frames)
│   │   └── Clip (broll_sunset.mov, in: 00:00:10, out: 00:00:18)
│   ├── Track (V2 - overlay)
│   │   ├── Gap (until 00:01:00)
│   │   └── Clip (lower_third.png, with opacity + animation metadata)
│   ├── Track (A1 - dialogue)
│   │   └── Clip (interview_take3.mov, audio only)
│   └── Track (A2 - music)
│       └── Clip (background_music.wav, with volume metadata)
```

Effects and grades are stored as OTIO metadata on clips. The rendering engine interprets this metadata when building the GStreamer pipeline.

### 4. Color Pipeline

Built on publicly available color science:

```
Camera Log (V-Log, S-Log3, LogC, etc.)
    │
    ├── Input Transform (camera log → working space)
    │   Via OCIO config or LUT
    │
    ├── Working Space: ACES AP1 (ACEScg) or DaVinci Wide Gamut
    │   User configurable per project
    │
    ├── Grading Operations (all in working space)
    │   - Lift/gamma/gain (standard math)
    │   - Curves (spline interpolation)
    │   - ASC CDL
    │   - HSL qualification + secondary corrections
    │   - Creative LUTs
    │
    ├── Output Transform (working space → display)
    │   Via ACES Output Transform, OpenDRT, or custom LUT
    │
    └── Display: Rec.709, Rec.2020/PQ (HDR), etc.
```

All transforms run on GPU via libplacebo (Vulkan) or OCIO GPU path. The agent applies grades by writing metadata to the OTIO timeline; the renderer interprets them.

### 5. Preview System

The preview panel is a lightweight web interface (or electron app) with:

- **Video player** showing the proxy render of the current timeline state
- **Timeline scrubber** with track lanes showing clip thumbnails
- **Frame/range selection** — click a frame or drag a range; timecodes are sent to the agent as context
- **Auto-refresh** — after the agent makes changes, a new proxy render is triggered and the preview updates

The proxy render is always 480p H.264 — fast to generate, fast to stream. The full-quality export is a separate explicit operation.

### 6. Agent Interaction Model

Modelled after Claude Code:

- **System instructions** — user-customizable project-level context (camera type, color space, preferred LUTs, editing style)
- **Workflow prompts** — reusable prompt templates for common tasks ("rough cut from script", "color grade interview", "add lower thirds")
- **Selection as context** — when the user selects a frame or range in the preview panel, the timecodes and a thumbnail are included in the agent's context
- **Conversational** — multi-turn: "trim 2 seconds off the start of that clip" → agent identifies "that clip" from selection context

Users can customize:
- System instructions (project settings, style preferences)
- Workflow prompts (reusable editing recipes)
- Tool configurations (preferred codecs, default LUTs, export presets)
- Agent behavior (how autonomous vs. confirmatory)

---

## Custom Engineering Required

### Must Build

| Component | What It Is |
|---|---|
| **libplacebo GStreamer element** | Wraps libplacebo's Vulkan rendering for color management, 3D LUT application, HDR tone mapping, and high-quality scaling inside GStreamer pipelines |
| **Extended blend mode compositor** | Custom GLSL shaders for professional blend modes (multiply, screen, overlay, soft light, color dodge, etc.) exposed as GStreamer compositor properties |
| **Color grading GPU element** | Lift/gamma/gain, curves, ASC CDL, HSL qualification running as GPU shaders in GStreamer |
| **Motion graphics renderer** | Rendering engine (Cairo or Skia) for animated text, lower thirds, titles — feeds into GStreamer via appsrc |
| **OTIO → GES pipeline** | Interpreter that reads OTIO timeline + effect metadata and constructs the corresponding GStreamer/GES pipeline for rendering |
| **Preview server** | Lightweight web server that serves proxy renders and provides timeline scrubbing UI with selection-to-agent context bridge |
| **Tool registry** | MCP-compatible tool registry with deferred loading and semantic search for 50-200+ editing tools |
| **GPU pipeline optimizer** | Ensures data stays GPU-resident through the entire decode → process → encode chain, minimizing CPU-GPU transfers |

### Already Exists (Integrate, Don't Build)

| Component | Library | License |
|---|---|---|
| Timeline data model | OpenTimelineIO 0.18 | Apache 2.0 |
| GPU video decode/encode | PyNvVideoCodec 2.0 / GStreamer nvcodec | MIT / LGPL |
| Color space management | OpenColorIO 2.5 | BSD-3 |
| Color science algorithms | colour-science (Python) | BSD-3 |
| Speech-to-text | WhisperX (word-level) / faster-whisper | BSD-4 / MIT |
| Frame-level Python video | PyAV 16.1 | BSD |
| Video compositing/mixing | GStreamer compositor | LGPL |
| Timeline editing | GStreamer Editing Services (GES) | LGPL |
| LUT processing | libplacebo | LGPL 2.1 |
| HDR tone mapping | libplacebo | LGPL 2.1 |
| Audio mixing | GStreamer audiomixer | LGPL |
| OTIO ↔ GES adapter | otio-xges-adapter | Apache 2.0 |
| Docker + GPU | NVIDIA Container Toolkit | Apache 2.0 |

---

## Export Targets

### Direct Render (Primary)
GStreamer pipeline → MP4/MOV/MKV with NVENC encoding. Supports ProRes 422 HQ, H.264, H.265, AV1, DNxHR.

### OTIO → DaVinci Resolve (Optional Finishing)
Export the OTIO timeline for import into Resolve. Use cases:
- Fine color grading with Resolve's tools and control panels
- Neural Engine features (Magic Mask, noise reduction)
- Client review in a familiar environment

### OTIO → Other NLEs
Via OTIO adapters: FCP XML, AAF, CMX 3600 EDL. Enables round-tripping with Premiere, FCPX, Avid.

---

## Containerization

```dockerfile
# Base: NVIDIA CUDA runtime
FROM nvidia/cuda:12.x-runtime-ubuntu24.04

# GStreamer + GES + nvcodec plugins
# FFmpeg 8.0 (for ProRes RAW decode, Vulkan)
# Python 3.12+
# PyAV, PyNvVideoCodec, OpenTimelineIO
# OpenColorIO, colour-science
# WhisperX (with CUDA support)
# libplacebo (built from source with Vulkan)
# Custom GStreamer elements

# Volumes:
#   /media    — source footage
#   /project  — OTIO files, asset registry, proxy renders
#   /luts     — .cube LUT library
#   /export   — rendered output
```

GPU access via `--gpus all` or `--gpus '"device=0"'`. Vulkan ICD files mounted for ProRes RAW decode.

---

## TDD Strategy

Every tool has a corresponding test. Fixture clips are generated by FFmpeg (color bars, test tones) — no real footage required.

```
tests/
├── conftest.py              # Shared fixtures: test clips, test LUTs, test OTIO timelines
├── fixtures/
│   ├── generate_fixtures.sh # FFmpeg commands to create test media
│   └── test_lut.cube        # Minimal valid .cube file
├── test_ingest.py           # Ingest tool: accepts various codecs, produces intermediate + proxy
├── test_timeline.py         # OTIO read/write, clip manipulation, track operations
├── test_transcribe.py       # WhisperX transcription, word-level timestamps
├── test_color.py            # LUT application, lift/gamma/gain, CDL, color space transforms
├── test_composite.py        # Overlay, blend modes, opacity, transforms
├── test_audio.py            # Volume, fades, mixing, normalization
├── test_text.py             # Title generation, lower thirds, subtitle rendering
├── test_effects.py          # Speed change, stabilization, denoise
├── test_render.py           # Proxy generation, full export, codec verification
├── test_export.py           # OTIO export to Resolve XML, FCP XML, AAF
└── test_pipeline.py         # Integration: ingest → edit → grade → render end-to-end
```

Test principles:
- `pytest tests/` must pass without any manual review
- Fixtures are deterministic (generated from FFmpeg, not captured footage)
- Each tool test verifies: correct output format, correct metadata, no side effects, handles edge cases
- Integration tests verify multi-tool pipelines produce valid output
- GPU tests are skippable for CI without GPU access (`@pytest.mark.gpu`)

---

## Build Sequence

### Phase 1: Foundation
- Docker container with GStreamer + CUDA + FFmpeg 8.0
- Ingest tool (any format → ProRes 422 HQ + proxy)
- OTIO timeline read/write
- Basic proxy render (GES → 480p H.264)
- Test fixtures and TDD harness

### Phase 2: Core Editing
- Trim, split, concatenate tools
- Multi-track timeline assembly
- Transitions (GES crossfade + SMPTE wipes)
- Speed change
- Audio placement and basic mixing
- WhisperX transcription + script matching

### Phase 3: Color & Compositing
- libplacebo GStreamer element
- OCIO color pipeline integration
- LUT application (V-Log → Rec.709, creative LUTs)
- Basic color grading (lift/gamma/gain, CDL)
- Multi-layer compositing with blend modes

### Phase 4: Polish & Preview
- Motion graphics renderer (animated text, lower thirds)
- Preview server with timeline scrubber
- Selection → agent context bridge
- Full-quality export pipeline
- OTIO export to Resolve/Premiere/FCPX

### Phase 5: Agent Layer
- Tool registry with MCP deferred loading
- Tool RAG for semantic search
- Workflow prompt system
- System instruction customization
- Multi-turn conversational editing

Each phase delivers testable, working functionality. No phase depends on completing all prior phases — colour work can start in parallel with core editing once the foundation is in place.
