# Architecture v2 Review Response

> Response to `docs/review/2026-03-08-architecture-v2-review.md`. Each point is addressed with a verdict (accepted, partially accepted, or disagreed) and the action taken.

## 1. The Central Contradiction: MLT vs. GPU Pipeline

**Verdict: ACCEPTED — This is a critical architectural flaw.**

The reviewer is correct. MLT and GStreamer are separate, incompatible frameworks. The v2 architecture describes both as rendering paths without addressing how MLT XML project data becomes a GStreamer GPU pipeline. `melt` renders on CPU at 8-bit with no color management. The GPU pipeline runs in GStreamer. There is no bridge between them, and building one is estimated at 12,000-18,000 lines minimum.

**Research conducted:** Deep investigation of GES capabilities, MLT internals, existing bridges (none found beyond an abandoned 2013 shared-memory relay), and the scope of building a translator.

**Decision: Switch to GES (GStreamer Editing Services) as both project model and rendering engine.**

GES eliminates the impedance mismatch entirely:
- **Native GStreamer pipeline construction** — GES builds GStreamer pipelines directly. Our custom elements (libplacebo, GLSL shaders) are automatically available as effects via `GES.Effect.new("element_name")`.
- **Keyframe animation** via `GstInterpolationControlSource` — four modes: none (step), linear, cubic, cubic-monotonic. More structured than MLT's string syntax but equally capable.
- **Any GStreamer element as an effect** — single-sink/single-source elements are automatically usable. Our custom libplacebo element, GLSL color grading shaders, etc. just need to be registered in the GStreamer plugin registry.
- **Pluggable compositor** — GES's SmartMixer wraps whatever compositor is available. We can use `glvideomixer` for GPU rendering and `skiacompositor` for CPU fallback by adjusting element rank.
- **70+ SMPTE transitions** plus custom transitions via shapewipe.
- **Custom metadata** — GES's `GESMetaContainer` interface stores typed key-value pairs. We use `agent:` prefixed keys (e.g., `agent:edit-intent`, `agent:generation-id`).
- **XGES format** — XML serialization with full keyframe, effect, and metadata support.
- **Python bindings proven by Pitivi** — Pitivi is an entire NLE written in Python using GES. The bindings are mature.
- **OTIO adapter exists** — `otio-xges-adapter` is actively maintained (part of OTIO's PR #609).

**Tradeoffs accepted:**
- Only constant-rate speed changes (no variable speed ramps) — acceptable for MVP. RIFE handles interpolated slow-mo for creative speed effects.
- Maximum 2 sources overlapping at any timeline position — workable for most editing. Complex compositing uses nested timelines via `gessrc` (since GStreamer 1.24).
- Nanosecond-based timestamps may cause rounding at certain frame rates (GES issue #61) — known issue, under active development. Acceptable.
- Single maintainer — risk acknowledged. GES is in the GStreamer monorepo (Collabora, Igalia backing). We maintain fork capability.
- Colorimetry bug #111 — solved at ingest (color space normalization).

**What happens to MLT:** MLT XML can still be imported via OTIO (MLT XML → OTIO → GES). `melt` remains available as a debugging/validation tool but is not part of the rendering pipeline.

**Updated in:** Architecture Design v2, Project Format Decision, Rendering Engine Layer, and Core Components sections (comprehensive rewrite).

---

## 2. Components That Don't Exist As Described

### 2a. otio-mlt-adapter

**Verdict: ACCEPTED**

The adapter is write-only, stale (2021), and incompatible with current OTIO (requires 0.12-0.13, current is 0.17+). Since we now use GES/XGES, this is replaced by `otio-xges-adapter` which is actively maintained.

**Updated in:** Architecture Design v2, removed otio-mlt-adapter reference, replaced with otio-xges-adapter.

---

### 2b. gst-plugin-skia

**Verdict: ACCEPTED**

Research confirms:
- `skiacompositor` exists (29 blend modes, sub-pixel positioning, per-pad anti-aliasing) — real and useful
- `skiafilter` does **NOT** exist in the current codebase — the review is correct
- No text rendering in the plugin — compositor and generic filter callbacks only
- **CPU-only** — `wrap_pixels` creates software raster surfaces. No GPU context. GPU acceleration is listed as "future work" by Igalia.
- No rounded corners either (mentioned in slides but not implemented)

For text/motion graphics, the plan remains: **skia-python v144** rendering to numpy arrays, pushed to GStreamer via appsrc. This is independent of gst-plugin-skia.

`skiacompositor` is useful as a **preview compositor** (29 blend modes at 480p, CPU is fine) but cannot replace `glvideomixer` for 4K final rendering due to being CPU-only.

**Updated in:** Architecture Design v2, compositor strategy and text rendering sections.

---

### 2c. libplacebo GStreamer element

**Verdict: ACCEPTED (already in Must Build)**

The estimate (2000-4000 lines) is reasonable. The reference code (FFmpeg `vf_libplacebo.c`, 1845 lines) is Vulkan-only, but our approach targets OpenGL which is architecturally simpler. Target the `pl_renderer` + `pl_options` API at PL_API_VER 309.

No change needed — already correctly categorized.

---

### 2d. OCIO GStreamer integration

**Verdict: ACCEPTED — Must be added to "Must Build" list.**

Valid catch. No GStreamer plugin for OpenColorIO exists. Required for the color pipeline. Needs a custom `GstGLFilter` subclass that loads OCIO config, creates processor, generates GPU shader code via `GpuShaderDesc`, uploads LUT textures, and applies the shader. OCIO 2.5 has a mature OpenGL shader pipeline.

**Updated in:** Architecture Design v2, added to Must Build table.

---

## 3. GPU Pipeline Concerns

### 3a. CUDA-GL Interop: Pin GStreamer >= 1.28.1

**Verdict: ACCEPTED**

CUDA-GL interop bugs were fixed in GStreamer 1.28.1 (February 2026). MR !614 optimized GL resource registration from per-frame to once. This is a hard requirement.

**Updated in:** Architecture Design v2, Infrastructure Layer section.

---

### 3b. glvideomixer crash bugs

**Verdict: ACCEPTED as risk, mitigated by compositor strategy**

The bugs are real (#728, #786, #622, #713, #656). Resolution changes and transform combinations are core editing operations.

**Mitigation strategy:**
- GES makes the compositor pluggable. We can swap compositors by changing element rank.
- `skiacompositor` for proxy/preview rendering (480p, CPU-only, 29 blend modes, no crash bugs)
- `glvideomixer` for GPU-accelerated final render (where resolution is fixed and transforms are less dynamic)
- Monitor and contribute upstream fixes for the crash bugs
- `compositor` (CPU, 3 blend modes) as ultimate fallback

**Updated in:** Architecture Design v2, Known Risks section.

---

### 3c. glshader: Single input only

**Verdict: ACCEPTED**

For single-stream color grading (lift/gamma/gain, curves, CDL), `glshader` works. Multi-input effects use `glvideomixer` or the compositor's per-pad blend mode control.

No architecture change needed — single-stream grading is the primary use case for custom GLSL.

---

### 3d. skiacompositor as primary compositor

**Verdict: PARTIALLY ACCEPTED**

Research confirms skiacompositor has superior blend mode support (29 vs 3) and stability (no GL state bugs), but is **CPU-only**. Not viable for 4K60 final rendering.

**Decision: Dual compositor strategy.**
- `skiacompositor` for preview/proxy rendering (480p, CPU handles this easily, 29 blend modes for accurate preview)
- `glvideomixer` for GPU-accelerated final rendering (4K+, where blend mode control is limited to GL blend functions)
- GES SmartMixer makes this switchable — just change the compositor factory rank for the render pipeline

**Updated in:** Architecture Design v2, Rendering Engine Layer.

---

## 4. Intermediate Format and Color Pipeline

### 4a. ProRes licensing

**Verdict: PARTIALLY ACCEPTED**

Research findings:
- FFmpeg ProRes encoder has shipped for 14+ years without legal action from Apple
- Real risk is broadcast delivery rejection (metadata identifies encoder as non-Apple), not litigation
- DNxHR HQX has similar licensing concerns (Avid's VC-3 standard, technically requires license for commercial use)
- DNxHR HQX in MXF avoids gamma/levels interpretation bugs across NLEs
- FFmpeg's DNxHR encoder is faster and better optimized than prores_ks
- DaVinci Resolve 19.1.4 now has native ProRes encoding on Linux (Studio)

**Decision: DNxHR HQX in MXF as default intermediate, ProRes as option.**
- DNxHR HQX: faster encode, better cross-NLE compatibility in MXF, no metadata rejection risk
- ProRes 422 HQ: available as user-selectable option for workflows that require it
- For final delivery requiring Apple-authorized ProRes: use Resolve Studio's native encoder on Linux

Both codecs are "visually lossless" and indistinguishable in blind tests by professional colorists.

**Updated in:** Architecture Design v2, Ingest Pipeline and Intermediate Format sections.

---

### 4b. Non-destructive IDTs

**Verdict: ACCEPTED**

The reviewer and the research are correct. Professional standard is non-destructive IDT application. Baking at ingest is simpler but sacrifices flexibility that professionals expect.

**Research findings:**
- DaVinci Resolve applies IDTs non-destructively (project-level ACES or node-level "ACES sandwich")
- ACES Metadata File (AMF) format exists for storing IDT/ODT references as sidecar metadata
- Applying IDT at render time in GStreamer is feasible via OCIO shader or 3D LUT loaded in GLSL
- Storing in camera log (V-Log) preserves maximum dynamic range
- If done right, this is one extra shader/LUT application at the start of the GPU pipeline — trivial cost

**Decision: Non-destructive IDTs.**
- Store footage transcoded to intermediate codec **preserving camera log encoding** (V-Log stays V-Log)
- Store IDT info in asset registry: camera model, color space, transfer function, intended IDT
- Apply IDT at render time via the OCIO GStreamer element or libplacebo LUT
- Color space normalization still happens — but at render time, not at ingest encode time
- This resolves GES colorimetry bug #111 the same way: all clips get the same IDT applied at the start of the GPU pipeline

**Updated in:** Architecture Design v2, Guiding Principles, Ingest Pipeline, and Color Pipeline sections.

---

### 4c. ProRes RAW experimental

**Verdict: ACCEPTED**

FFmpeg 8.0's ProRes RAW decoder is reverse-engineered with limited real-world validation. Noted as experimental with fallback to DaVinci Resolve for ProRes RAW sources.

**Updated in:** Architecture Design v2, Known Risks.

---

### 4d. 4K encode speed

**Verdict: ACCEPTED**

7-15 fps for prores_ks at 4K confirmed. DNxHR is faster. Both are CPU-only for encode. For hours of daily footage, this is a significant ingest bottleneck. Document this and note that NVENC H.265 is available for fast internal intermediate when ProRes/DNxHR quality isn't needed.

**Updated in:** Architecture Design v2, Ingest Pipeline.

---

### 4e. ACES AP1 as default

**Verdict: ACCEPTED**

ACES AP1 (ACEScct) is the correct default for a multi-NLE export pipeline. Vendor-neutral, standardized, ACES 2.0 significantly improved the Output Transform. DaVinci Wide Gamut remains an option for users who prefer it.

**Updated in:** Architecture Design v2, Color Pipeline.

---

## 5. Preview System

### 5a. HLS alone insufficient for scrubbing

**Verdict: ACCEPTED**

HLS is for continuous playback, not frame-accurate scrubbing. Video editors require instant visual feedback when dragging through the timeline.

**Decision: Hybrid preview protocol.**
- **LL-HLS** (Low-Latency HLS with partial segments) for continuous playback
- **WebSocket frame serving** for interactive scrubbing (request frame at timecode, receive JPEG/WebP)
- **Thumbnail strips** pre-generated at ingest for visual timeline scrubbing (lightweight, fast)

**Updated in:** Architecture Design v2, Preview System section.

---

### 5b. hls.js segment update quirks

**Verdict: ACCEPTED**

Treat preview as live stream (no `#EXT-X-ENDLIST`), use `EXT-X-MEDIA-SEQUENCE` to indicate changes, version segment filenames when re-rendering.

**Updated in:** Architecture Design v2, Preview System section.

---

### 5c. LL-HLS for <2s latency

**Verdict: ACCEPTED**

Standard HLS with 2s segments gives 4-6s latency. LL-HLS with partial segments (200-500ms parts) on localhost achieves 1.5-2s. Required to hit the <2s target.

**Updated in:** Architecture Design v2, Preview System section.

---

## 6. Transcription Pipeline

### 6c. Newer alignment alternatives

**Verdict: ACCEPTED — Replace MFA with Qwen3-ForcedAligner-0.6B**

Research findings:

| Aligner | AAS (ms, lower=better) | Speed (RTF) | GPU | License | VRAM |
|---|---|---|---|---|---|
| **Qwen3-ForcedAligner-0.6B** | 32-43ms | 0.016 (63x RT) | Yes | Apache 2.0 | ~1.3GB |
| NFA | 101-130ms | 0.007 (150x RT) | Yes | Apache 2.0 | ~2-4GB |
| MFA | Reference standard | ~1x RT | No (CPU) | MIT | N/A |
| WhisperX alignment | 133ms+ (degrades on long-form) | Fast | Yes | BSD-4 | ~3GB |

Qwen3-ForcedAligner is 3x more accurate than NFA and doesn't degrade on long-form audio (52.9ms AAS at 300s vs NFA's 246.7ms). Apache 2.0 licensed, 1.3GB VRAM, 63x faster than real-time. Clear winner.

CrisperWhisper is impressive (single-pass transcription + timestamps) but **CC BY-NC 4.0 (NonCommercial)** — ruled out for any commercial use.

**Updated in:** Architecture Design v2, Ingest Pipeline, Local AI Models table.

---

### 6d. Diarization optional

**Verdict: ACCEPTED**

Research confirms pyannote diarization takes 20-30+ minutes per hour of audio on RTX 3090/4090, dominating the ingest pipeline. Many editing workflows don't need speaker identification.

**Decision:** Diarization is an on-demand analysis step, not part of mandatory ingest. Triggered when the user requests speaker-based editing (e.g., "remove all of speaker B's dialogue").

**Updated in:** Architecture Design v2, Ingest Pipeline.

---

## 7. Academic Claims

### 7b. 17x error trap — corrected citation

**Verdict: ACCEPTED**

17.2x is for uncoordinated "Bag of Agents." Centralized orchestration (which our architecture uses) reduces amplification to 4.4x.

**Updated in:** Architecture Design v2, Verification Loop section.

---

## 8. MLT Framework Limitations

**No longer applicable** — we've switched to GES. MLT's 8-bit processing, lack of GPU acceleration, and non-validating XML parser are no longer constraints on the architecture.

---

## 9. Docker + Infrastructure

**Verdict: ACCEPTED**

Requirements documented:
- Host NVIDIA driver must NOT be `nvidia-headless` variant
- Set `NVIDIA_DRIVER_CAPABILITIES=all` or `compute,utility,graphics,video,display`
- GStreamer nvcodec needs building from source (`gst-plugins-bad` with `-Dnvcodec=enabled`)
- Pin GStreamer >= 1.28.1
- NVIDIA deprecated old encoding presets in R550 driver — nvcodec encoders may need updated preset APIs

**Updated in:** Architecture Design v2, Infrastructure Layer section.

---

## 10. Competitive Landscape

**Verdict: PARTIALLY ACCEPTED**

The landscape research document (`docs/plans/2026-03-07-landscape-research.md`) already covers VideoAgent, VideoDB Director, DiffusionStudio, ShortGPT, and others. A separate competitive analysis section in the architecture doc is unnecessary — the differentiation is the architecture itself:

**What differentiates this project:**
1. Professional GPU color pipeline (OCIO, libplacebo, ACES) — no competitor has this
2. ProRes/DNxHR intermediate workflow with camera log preservation — no competitor handles professional camera footage
3. Local AI inference (no footage leaves the machine) — most competitors use cloud APIs
4. TDD-validated tools — no competitor has automated tool validation
5. Pluggable rendering engine with GPU compositing — no open-source competitor does this
6. Both professional and amateur users via system instruction customization

No architecture change needed.

---

## 11-12. Tool RAG and Vision Models

**Confirmed as correctly designed.** No changes needed.

---

## 13. Open Questions — Answered

### Architecture-Critical

1. **Who translates project format → GStreamer pipeline?** GES does this natively. This is why we switched to GES.

2. **Can skiacompositor serve as primary compositor?** For preview at 480p, yes. For 4K final render, no (CPU-only). Dual compositor strategy.

3. **Fallback when glvideomixer crashes?** GES SmartMixer is pluggable. Fall back to `skiacompositor` (CPU, 29 blend modes) or `compositor` (CPU, 3 blend modes).

4. **Visual consistency between preview and export?** Both use GES. Preview uses `skiacompositor` at 480p. Export uses `glvideomixer` at full resolution. Blend mode mapping between Skia's 29 modes and GL's blend functions is documented, with visual discrepancy warnings for modes that don't map exactly.

### Color Pipeline

5. **Non-destructive IDTs.** Apply at render time via OCIO/libplacebo. Store camera log encoding in intermediate. Document tradeoff.

6. **DNxHR HQX as default.** ProRes available as option. Both are "visually lossless."

7. **OCIO GStreamer element.** Added to Must Build.

8. **Bit depth.** GPU pipeline operates at 10-bit+ (GL textures support 10/16-bit). Intermediates are 10-bit (DNxHR HQX) or 10-bit (ProRes 422 HQ). No 8-bit bottleneck.

### Preview

9. **Frame-accurate scrubbing.** WebSocket frame serving for scrub requests. HLS for playback.

10. **What serves individual frames?** WebSocket endpoint. Client sends timecode, server renders single frame via GES pipeline, returns JPEG/WebP.

11. **Target latency.** <2s for playback (LL-HLS), <200ms for scrubbing (WebSocket frame).

### Transcription

12. **Qwen3-ForcedAligner replaces MFA.** 3x more accurate, GPU-accelerated, Apache 2.0.

13. **Diarization is optional.** On-demand, not part of mandatory ingest.

### Competitive

14. **Differentiation.** Professional GPU color pipeline + local AI + TDD + camera log workflow. No competitor combines these.

15. **Leverage existing projects.** GES (from Pitivi ecosystem), frei0r (shared effects), OTIO adapters, libplacebo (from mpv).

### Infrastructure

16. **GStreamer >= 1.28.1.** Hard requirement, documented.

17. **Headless EGL.** Non-headless NVIDIA driver required. Documented.

18. **GPU CI testing.** Self-hosted runner with NVIDIA GPU. `@pytest.mark.gpu` for conditional skipping. Cloud GPU CI (RunPod, Lambda) as backup.

---

## Summary of Architectural Changes

| Change | Impact | Rationale |
|---|---|---|
| **GES replaces MLT XML** | Major (core architecture) | Eliminates MLT-GStreamer impedance mismatch |
| **Non-destructive IDTs** | Major (color pipeline) | Professional standard, preserves flexibility |
| **DNxHR HQX default** | Medium (ingest) | Better encoder, no metadata rejection, MXF compatibility |
| **OCIO element added to Must Build** | Medium (color pipeline) | Required but was missing |
| **Dual compositor strategy** | Medium (rendering) | skiacompositor for preview, glvideomixer for export |
| **LL-HLS + WebSocket preview** | Medium (preview) | HLS alone can't do scrubbing |
| **Qwen3-ForcedAligner replaces MFA** | Low (transcription) | 3x more accurate, GPU, Apache 2.0 |
| **Diarization optional** | Low (ingest) | 10-30x slower than transcription, not always needed |
| **GStreamer >= 1.28.1 pinned** | Low (infrastructure) | CUDA-GL interop bug fixes |
