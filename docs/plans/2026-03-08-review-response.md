# Architecture Review Response

> Response to `docs/review/2026-03-08-architecture-review.md`. Each point is addressed with a verdict (accepted, partially accepted, or disagreed) and the action taken.

## Critical Issues

### 1. Zero-Copy GPU Pipeline Is Architecturally Incoherent

**Verdict: ACCEPTED**

The reviewer is correct. The v1 pipeline crossed three incompatible GPU API boundaries (CUDA, Vulkan, OpenGL) with no existing framework to automate the interop.

**Research conducted:** Full investigation of CUDA-only, Vulkan-only, and hybrid pipelines in GStreamer 1.26-1.28. Examined cudacompositor, Vulkan compositor (doesn't exist), glvideomixer, CUDA-Vulkan interop, CUDA-OpenGL interop, and libplacebo's backend options.

**Decision: CUDA for decode/encode + OpenGL for processing.**
- `glvideomixer` is the **only** GStreamer compositor with real blend mode control (15 blend functions, 3 blend equations per pad)
- `cudacompositor` has no blend mode support -- basic alpha only
- No Vulkan compositor exists in GStreamer
- libplacebo supports OpenGL backend with nearly identical features to Vulkan (tone mapping, LUTs, HDR, gamut mapping)
- CUDA-OpenGL interop is well-established (`cuGraphicsGLRegisterImage`), unlike CUDA-Vulkan which is unproven in production video pipelines
- One GPU memory boundary at pipeline start (CUDA decode → GL upload) and one at end (GL download → CUDA encode). All processing between stays in OpenGL.

**Updated in:** Architecture Design v2, "GPU API Decision" section.

---

### 2. OTIO as "Source of Truth" Is Underselling the Problem

**Verdict: ACCEPTED**

The reviewer is correct that OTIO's effect model is nearly empty (only `LinearTimeWarp` and `FreezeFrame`). Using OTIO as the internal format would mean stuffing everything into unstructured metadata dictionaries, creating false interoperability.

**Research conducted:** Investigated Kdenlive's MLT XML format, Pitivi's XGES, Shotcut's MLT XML annotations, Blender's binary format, AAF, and FCP XML. Also examined the otio-mlt-adapter.

**Decision: MLT XML as internal format, OTIO as interchange.**
- MLT XML provides effects, keyframe animation (with Catmull-Rom spline interpolation), transitions, filter chains, and direct renderability via `melt`
- Agent-specific metadata uses `agent:` namespace prefix (following Kdenlive's `kdenlive:` pattern)
- OTIO is used for import from and export to other NLEs -- it's interchange, not the source of truth
- This eliminates the need to build a custom format *and* a custom renderer

**Updated in:** Architecture Design v2, "Project Format Decision" section.

---

### 3. `nvcompositor` Is Jetson-Only

**Verdict: ACCEPTED**

Correct. `nvcompositor` is only available on NVIDIA Jetson platforms. The x86 equivalent is `cudacompositor` (GStreamer 1.26, gst-plugins-bad).

**Action:** Removed `nvcompositor` references. The design now uses `glvideomixer` (OpenGL) as the primary compositor, which is mature, cross-platform, and has full blend mode control. `cudacompositor` is not used because it lacks blend modes.

**Updated in:** Architecture Design v2, GPU API Decision section.

---

### 4. GES Has Known Colorimetry Bugs

**Verdict: ACCEPTED**

GES Issue #111 is a real problem: clips with different color spaces get composited without proper conversion.

**Decision:** Force color space normalization to the project working space at **ingest time**, before any clip enters the pipeline. This was implicit in the v1 "ingest-normalise first" principle but is now explicit. All clips are transcoded to the same working color space during ingest. The rendering engine never sees mixed colorimetry.

**Updated in:** Architecture Design v2, Guiding Principles #1 and Ingest Pipeline section.

---

## Significant Concerns

### 5. ProRes 422 HQ as Intermediate Is Suboptimal for GPU Pipeline

**Verdict: PARTIALLY ACCEPTED**

The reviewer correctly identifies that ProRes encode is CPU-only and conflates disk format with GPU memory format. However, the conclusion -- that we should use H.265 via NVENC as the disk intermediate -- is wrong for our use case.

**Why ProRes 422 HQ remains correct for disk:**
- Broad NLE compatibility (Resolve, Premiere, FCPX, Kdenlive all read it natively)
- Known, predictable quality and color behavior
- Intra-frame only -- every frame is a keyframe, enabling instant random access
- The user explicitly wants ProRes workflow compatibility
- CPU encode cost is paid **once** at ingest, not per-frame during editing

**What was clarified:**
- The disk intermediate (ProRes 422 HQ) is distinct from the GPU working format (NV12/P010/RGBA surfaces)
- ProRes never enters the GPU pipeline -- it's decoded to native GPU textures at pipeline start
- This distinction is now explicit in the design

**Updated in:** Architecture Design v2, "Intermediate Format Clarification" section.

---

### 6. otio-xges-adapter Has Critical Gaps

**Verdict: ACCEPTED**

Speed effects and CDL don't survive OTIO-GES conversion. This would be a problem if OTIO were the internal format.

**Mitigation:** Since we now use MLT XML internally, speed effects and CDL are handled natively by MLT (which has `timewarp` for speed and full CDL support). OTIO export is an interchange operation that is lossy by design -- like exporting to EDL loses effects. The design documents which properties survive export to each format.

**Updated in:** Architecture Design v2, Known Risks section.

---

### 7. Cairo Is the Wrong Choice for Motion Graphics

**Verdict: ACCEPTED**

Cairo is CPU-only for practical purposes and is being replaced in major projects. Skia with GPU backends is the correct choice.

**Decision:** Use Skia via its OpenGL backend. A GStreamer plugin already exists (`gst-plugin-skia` in gst-plugins-rs) providing `skiacompositor` and `skiafilter`. Skia's OpenGL backend fits naturally into the CUDA+OpenGL pipeline. It handles text rendering, vector graphics, shapes, and gradients -- exactly what's needed for titles and lower thirds.

**Updated in:** Architecture Design v2, rendering engine layer and component table.

---

### 8. WhisperX Word-Level Alignment Has Known Reliability Issues

**Verdict: PARTIALLY ACCEPTED**

The reviewer correctly identifies reliability issues (up to 3s drift on numbers, regression since 3.3.3, unreliable last-word alignment). However, abandoning WhisperX is not the right response.

**Decision: Two-pass approach.**
1. **WhisperX** for fast initial alignment + speaker diarization (good enough for rough cuts and browsing)
2. **Montreal Forced Aligner (MFA)** for frame-accurate refinement when precision matters (final cuts, script matching)
3. Confidence scoring on alignment results to flag unreliable segments

WhisperX remains the best option for the initial pass because of its speed and integrated diarization. MFA adds sub-frame accuracy as a refinement step.

**Updated in:** Architecture Design v2, Ingest Pipeline section.

---

### 9. GES Maintainer Bus Factor

**Verdict: ACCEPTED as a risk, limited actionability**

Valid observation. GES has one primary contributor (Thibault Saunier at Igalia).

**Mitigation:**
- The v2 design reduces GES dependency by using MLT XML + `melt` as the primary render path
- GES is part of the GStreamer monorepo (backed by Collabora, Igalia, and others) -- not truly single-maintainer
- Maintain fork capability
- Contribute upstream where possible

This is a risk to monitor, not a design change to make.

---

## Gaps and Missing Pieces

### 10. No Multimodal Understanding in the Agent Layer

**Verdict: ACCEPTED (as a future phase)**

The reviewer is correct that text-only agents cannot effectively edit video for non-trivial tasks. The academic literature (LAVE, ExpressEdit, Vidi) confirms that visual understanding is essential.

**Decision:** Add a visual analysis step to the ingest pipeline. During ingest, a vision model (Gemini, GPT-4o, or open model like LLaVA) analyzes each clip for:
- Shot type classification
- Scene content descriptions
- Dominant colors / exposure
- Face detection and tracking
- Motion/action descriptions

This produces searchable metadata the agent can query. However, this is a **Phase 5 feature**, not a foundation requirement. For Phases 1-4, transcription + user descriptions are sufficient. The architecture accommodates vision model integration without requiring it from day one.

**Updated in:** Architecture Design v2, Ingest Pipeline and Phase 5 sections.

---

### 11. No Error Recovery or Self-Correction

**Verdict: ACCEPTED**

The "17x error trap" in multi-agent systems is well-documented. The v1 design had no verification mechanism.

**Decision:** Add a verification loop after each tool execution:
1. Validate MLT XML well-formedness and schema compliance
2. Verify output files exist with expected properties
3. For color operations, compare before/after histograms
4. For cuts, verify continuity of adjacent clips
5. For renders, verify output via ffprobe
6. On failure: retry once, then escalate to user

**Updated in:** Architecture Design v2, "Verification Loop" section.

---

### 12. Preview System Architecture Is Underspecified

**Verdict: ACCEPTED**

The v1 design said "lightweight web interface" with "auto-refresh" but didn't address latency, incremental rendering, streaming, or concurrency.

**Research conducted:** Investigated incremental rendering approaches (segment-based caching as used by FCPX and Resolve), HLS vs WebCodecs+MSE for streaming, GStreamer segment rendering, and professional editor cache architectures.

**Decision: Segment-based cache + HLS serving.**
- Timeline divided into 2-second segments
- Cache key per segment: SHA-256 of all inputs
- On edit: invalidate only affected segments, re-render in priority order (playhead position first)
- Serve via HLS with dynamic `.m3u8` playlist
- Browser plays via hls.js
- Visual indicators for cache status (blue = cached, red = dirty)
- Target: <2 second update for single-clip changes with proxy media
- Background rendering starts 5 seconds after last edit (following Resolve's Smart Cache pattern)

**Updated in:** Architecture Design v2, "Preview System" section.

---

### 13. Build Sequence Phase Dependencies Are Understated

**Verdict: PARTIALLY ACCEPTED**

The reviewer is correct that Phase 3 requires the libplacebo prototype from Phase 1, and Phase 5 requires the preview system from Phase 4. The phases are more sequential than v1 claimed.

**Correction:** Phase dependencies are now explicitly stated:
- Phase 1: no dependencies
- Phase 2: depends on Phase 1
- Phase 3: depends on Phase 1 + libplacebo prototype (can run parallel with Phase 2)
- Phase 4: depends on Phase 2
- Phase 5: depends on Phases 2 + 4

The claim that "colour work can start in parallel with core editing" remains true -- Phase 3 only needs the foundation from Phase 1, not the editing tools from Phase 2.

**Updated in:** Architecture Design v2, Build Sequence section.

---

### 14. No Consideration of Remotion/Browser-Based Alternative

**Verdict: DISAGREED**

The reviewer suggests the architecture should justify why GStreamer was chosen over Remotion. The reasons are clear and fundamental:

1. **No GPU-accelerated video decode** -- WebCodecs is limited and CPU-bound for ProRes
2. **No professional color pipeline** -- no OCIO, no wide-gamut, no HDR, no V-Log/V-Gamut workflow
3. **No 3D LUT support** in browser rendering
4. **Browser rendering is inherently slower** than native GPU pipelines for video compositing at professional resolutions
5. **No control over GPU memory** -- cannot keep frames GPU-resident

The user explicitly requires ProRes footage handling, V-Log grading, and professional color science. Remotion provides none of these. It is excellent for programmatic motion graphics and social media content, but wrong for this use case.

This is now stated explicitly in the v2 design.

**Updated in:** Architecture Design v2, "Why Not Remotion?" section.

---

## Verified Technical Claims

The reviewer flagged several claims for verification. Response to each:

| Claim | Reviewer Status | Our Response |
|---|---|---|
| Zero-copy PyNvVideoCodec → libplacebo → GStreamer → NVENC | **False** | Agreed. Corrected to CUDA+OpenGL pipeline with explicit boundary crossings. |
| nvcompositor for GPU compositing | **Misleading** | Agreed. Corrected to glvideomixer (OpenGL). |
| 85% of Resolve's color science is public | **Plausible** | Stand by this. DWG is fully published, YRGB concept is public, all grading algorithms are standard math, ACES is open. The ~15% that isn't published (tone/gamut mapping curves, Neural Engine) has open equivalents. |
| WhisperX 4.8% WER | **Partially confirmed** | Agreed alignment has issues. Added MFA two-pass approach. |
| VideoAgent 87-98% orchestration success | **Unverified** | Noted. We don't depend on this claim for any design decision. |
| ITR paper results (95% token reduction) | **Unverified** | Noted. MCP's deferred loading (which we use) is proven in production (Claude Code itself). We don't depend on the ITR paper specifically. |

---

## Questions From the Review -- Answered

1. **CUDA or Vulkan?** CUDA for decode/encode, OpenGL for compositing/processing. See GPU API Decision section.

2. **Target hardware?** RTX 4090 (primary), A1000 (secondary). Both with 64GB RAM. RTX 30-series minimum for community users (NVENC + NVDEC baseline). AV1 encode available on 40-series.

3. **Acceptable preview latency?** <2 seconds for single-clip changes. Achieved via segment-based caching with proxy media.

4. **How does the agent understand footage beyond transcription?** Vision model integration at ingest (Phase 5). Shot classification, scene descriptions, face detection, motion analysis. Phase 1-4 uses transcription + user descriptions.

5. **Project format vs interchange format?** MLT XML is the project format. OTIO is the interchange format. Clearly separated.

6. **Target user?** Both professional editors wanting AI assistance AND non-editors wanting automated editing. The system instruction customization allows different levels of autonomy. The framework is workflow-agnostic.

7. **What happens when tool execution fails?** Verification loop: validate → retry once → escalate to user. See Verification Loop section.

8. **Mixed frame rates?** MLT handles mixed frame rates natively (it's a broadcast framework designed for this). Needs testing and documentation but is architecturally supported.

9. **Licensing strategy?** All core components are LGPL or more permissive. Dynamic linking preserves the ability to build a permissively-licensed application layer on top. The agent intelligence layer (our code) can be any license.

10. **GPU CI testing?** Self-hosted runner with NVIDIA GPU for GPU-dependent tests. CPU-only tests run on standard CI. `@pytest.mark.gpu` for conditional skipping. Consider cloud GPU CI (e.g., RunPod, Lambda) for cost-effective GPU testing.
