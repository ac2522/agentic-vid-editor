# Critical Review: Architecture Design v2

> Review conducted 2026-03-08 via parallel deep research across 7 domains: GPU pipeline, MLT framework, AI model stack, preview system, academic research, color pipeline, and infrastructure. Each finding cites specific sources, versions, and evidence.

## Overall Assessment

The v2 design is a substantial improvement over v1 — the GPU API decision is coherent, the OTIO-to-MLT shift is well-reasoned, and the verification loop addresses a real problem. However, deep technical research reveals **four architecture-level issues that must be resolved before implementation**, several significant concerns, and numerous gaps that need filling.

The design is strongest in its vision, principles, and high-level component decomposition. It is weakest in the concrete rendering pipeline details, where several claims don't survive scrutiny against actual GStreamer/MLT/libplacebo documentation.

---

## Critical Issues (Must Address)

### 1. `glvideomixer`'s "Blend Modes" Are Not Professional Blend Modes

**The design's central compositing claim is misleading.**

The "15 blend functions, 3 blend equations per pad" are raw OpenGL `glBlendFuncSeparate` + `glBlendEquationSeparate` parameters. These map to fixed-function blending operations (`GL_SRC_ALPHA`, `GL_ONE_MINUS_SRC_ALPHA`, `GL_ADD`, etc.) — NOT to the blend modes video editors use.

**What you CAN do:** Normal (alpha over) compositing, additive blend.

**What you CANNOT do with fixed-function blending:**
- Multiply (`result = src × dst`)
- Screen (`result = 1 - (1-src)(1-dst)`)
- Overlay, Soft Light, Hard Light
- Color Dodge, Color Burn
- Difference, Exclusion
- Hue, Saturation, Color, Luminosity

These all require per-pixel shader math — fragment shaders, not fixed-function blend state. Every professional NLE (Premiere, Resolve, FCPX) implements 20-30+ blend modes as shader programs.

**Impact:** The compositing architecture needs a custom fragment-shader-based compositor, not `glvideomixer`. This is a fundamental shift from "configure existing element" to "build custom element."

**Sources:** [GstGLVideoMixerInput docs](https://gstreamer.freedesktop.org/documentation/opengl/GstGLVideoMixerInput.html), [glvideomixer docs](https://gstreamer.freedesktop.org/documentation/opengl/glvideomixer.html)

---

### 2. HLS Is Architecturally Wrong for Interactive Preview

**The segment-based HLS preview system cannot achieve <2 second latency. Expect 4-8 seconds minimum.**

Latency breakdown:
| Step | Time |
|------|------|
| Detect changed segments | 5-20ms |
| Spawn melt, parse project, seek | 150-500ms |
| Render 2s at 480p | 200-800ms |
| Write .ts to disk | 5-20ms |
| hls.js detects playlist change | 500-3000ms |
| hls.js fetches + buffers segment | 50-200ms |
| **Total** | **~1.5-4.5s best case** |

The killer is hls.js's polling model. HLS was designed for CDN content delivery, not interactive editing. Even Low-Latency HLS (LL-HLS) targets 2-5 seconds. Additional problems:

- **hls.js does not re-fetch VOD playlists.** The `#EXT-X-ENDLIST` tag tells it content is complete. You'd need live-style playlists or custom loaders, fighting the library's design.
- **Segment replacement mid-playback is invisible to hls.js** — old data stays in the MSE SourceBuffer.
- **Professional NLEs don't use streaming protocols for preview.** Resolve, Premiere, and FCPX all use real-time GPU rendering as the primary preview path, with optional background caching as secondary. The design inverts this: caching is primary with no real-time path.

**Recommendation:** Replace HLS with **WebCodecs + Canvas** for preview. WebCodecs provides sub-frame latency (~16ms at 60fps) via `VideoDecoder` → `<canvas>`. This is exactly what Clipchamp (now Microsoft) uses, as described in their [W3C presentation](https://www.w3.org/2021/03/media-production-workshop/talks/soeren-balko-clipchamp-webcodecs.html). MSE (Media Source Extensions) is a fallback for browsers with incomplete WebCodecs support.

**Sources:** [HLS Low Latency Guide](https://www.videosdk.live/developer-hub/hls/hls-low-latency), hls.js issues [#2351](https://github.com/video-dev/hls.js/issues/2351), [#3913](https://github.com/video-dev/hls.js/issues/3913)

---

### 3. Ingest-Time Color Space Normalization Destroys Information

**Normalizing all clips to the project working space at ingest contradicts professional practice and is architecturally wrong.**

Problems:
1. **Log footage is permanently transformed.** The original camera log encoding (V-Log, S-Log3, LogC) is lost. Subsequent grading operates on transformed data.
2. **Every professional system does it differently.** DaVinci Resolve, Baselight, and Nuke keep original media untouched and apply Input Transforms (IDTs) non-destructively at display/render time. ACES is explicitly designed as a non-destructive pipeline.
3. **Information loss is real.** Log-to-linear transforms involve floating-point quantization. If the user later wants to change the project working space, every clip must be re-ingested.
4. **The claimed fix for GES colorimetry bug #111 doesn't require baking.** You can normalize at render time — just ensure all clips pass through an IDT node before compositing.

**The correct approach:** Store original media unchanged. At render time, apply: `Source ColorSpace → Working Space → Grading Ops → Output ColorSpace`. The color space transform should be a node in the processing graph, not a preprocessing step.

**Sources:** [ACES workflow guide](https://blog.frame.io/2019/09/09/guide-to-aces/), standard practice in Resolve/Baselight/Nuke

---

### 4. MLT XML as Agent Interface Is Fragile; GES Python API Is Architecturally Superior

**The design forces an LLM agent to do text-level XML manipulation instead of making API calls.**

MLT XML problems for programmatic generation:
- **Underspecified schema.** No catalog of valid `mlt_service` values. No specification of property schemas per service. Implicit defaults are undocumented.
- **Ordering traps.** Track priority is inverted ("bottom-most track takes highest priority"). All references must follow the definition. The last producer is implicitly the default for playout.
- **Non-validating parser.** The DTD exists but melt doesn't validate against it. Malformed XML produces silent failures.
- **No concurrency support.** No locking, no atomicity. Agent writing XML while preview reads it → undefined behavior.
- **Python bindings are thin SWIG wrappers** with no dedicated documentation.

**GES provides what the agent actually needs:**
- A proper Python API via GObject Introspection (`gi.repository.GES`)
- Programmatic calls: `timeline.add_layer()`, `layer.add_clip()`, `clip.add_effect()`
- Built-in timeline operations (trim, ripple, roll)
- GStreamer pipeline integration for GPU-accelerated rendering
- Active maintenance as part of the GStreamer monorepo

**MLT's additional weaknesses:**
- **Single maintainer** (Dan Dennedy, who also maintains Shotcut). Bus factor = 1.
- **Weak GPU support.** Only Movit/GLSL, no CUDA, no hardware encoding, poor headless support. CPU-only melt in practice.
- **melt parses entire project per invocation.** No daemon mode. Unsuitable for rendering many small segments.
- **Poor multi-core scaling.** Users report 12% CPU utilization on 8-core systems. `real_time=-N` gives ~25% gain with 4 threads, far from linear.

**Recommendation:** Use GES with its Python API for timeline construction and manipulation. Use GStreamer pipelines for GPU-accelerated rendering. If MLT compatibility is needed for Shotcut/Kdenlive import/export, use it as a serialization format, not the internal representation.

**Sources:** [MLT XML docs](https://www.mltframework.org/docs/mltxml/), [GES Python API](https://lazka.github.io/pgi-docs/GES-1.0/index.html), [MLT releases](https://github.com/mltframework/mlt/releases), [MLT parallelization discussion #1078](https://github.com/mltframework/mlt/discussions/1078)

---

## Significant Concerns

### 5. CUDA-OpenGL Interop Is Functional But Historically Fragile

- A GStreamer 1.28.1 fix (February 2026) is described as "Fix CUDA/GL interop copy path" — meaning this was **broken or degraded as recently as Feb 2026**.
- An earlier MR (#614 in gst-plugins-bad) revealed `cuGraphicsGLRegisterImage` was being called per-frame instead of cached, causing dramatic performance degradation.
- [GStreamer Discourse](https://discourse.gstreamer.org/t/graphics-api-interoperability-and-zero-copy-memory/464) states: "GPU APIs do not interop in GStreamer" with the NVIDIA CUDA-GL case as the **only** exception, and recommends examining upload element source code because "there are no design documents."
- Whether the transfer is truly zero-copy or a device-to-device copy is undocumented.

**Flag:** The claim of "seamless" CUDA-GL interop is optimistic. It works, but it has been buggy, is undocumented at the design level, and performance characteristics are unquantified.

---

### 6. libplacebo's OpenGL Backend Is a Second-Class Citizen

- mpv 0.41.0 (February 2026) switched to **Vulkan-based gpu-next as the default renderer**. The project direction is clearly Vulkan-first.
- OpenGL backend lacks: push constants (falls back to UBOs → global uniforms), subgroup operations (needed for histogram-based tone mapping), timeline semaphores for async compute.
- Compute shaders are optional in OpenGL (require GL 4.3+) but mandatory in Vulkan. libplacebo uses compute for polar (EWA) scaling, peak detection for HDR tone mapping, and debanding.
- The `pl_opengl_params.no_compute` flag exists precisely because OpenGL compute is unreliable.

**The "nearly all the same features" claim is overstated.** OpenGL is a compatibility fallback, not a first-class peer. The design should acknowledge degraded capabilities or plan for Vulkan.

**Sources:** [libplacebo OpenGL gpu.c](https://github.com/haasn/libplacebo/blob/master/src/opengl/gpu.c), [mpv 0.41.0 release](https://github.com/mpv-player/mpv/issues/17400)

---

### 7. VRAM Budget Does Not Fit on 24GB RTX 4090

Realistic VRAM allocation:
| Component | VRAM |
|-----------|------|
| NVIDIA driver + display | 300-500 MB |
| NVDEC decode buffers (4K) | 200-400 MB |
| GStreamer GL textures | 500 MB - 1 GB |
| OpenGL compositing FBOs (4K RGBA) | 200-500 MB |
| NVENC encode buffers | 200-400 MB |
| Qwen3-VL 8B (FP16) | **18-20 GB** |
| **Total** | **~21-25 GB** |

**This does not fit.** The design says models run sequentially, but even loading Qwen3-VL 8B alone leaves only 4-6 GB for the entire video pipeline.

**Required mitigations (must pick at least one):**
1. Mandate INT4/INT8 quantization for Qwen3-VL (Q4_K_M reduces to ~5-6 GB with acceptable quality loss)
2. Time-multiplex: never run AI inference and video rendering simultaneously
3. Use a smaller VLM
4. Include a concrete VRAM budget table in the design with specific quantization levels

**Sources:** [Qwen3-VL VRAM requirements](https://apxml.com/models/qwen3-8b), [Qwen3-VL GitHub](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

---

### 8. AI Model Stack Has License and Accuracy Issues

| Model | Issue |
|-------|-------|
| **YOLO26** | AGPL-3.0 is a legal poison pill. Any project integrating YOLO26 must be AGPL-3.0 or purchase an Enterprise license. Replace with **RT-DETR** (Apache 2.0) or similar. |
| **InsightFace** | Code is MIT, but **pre-trained model weights are non-commercial research only**. Commercial use requires separate licensing or training your own models. |
| **Parakeet TDT 1.1B** | License is **CC-BY-4.0**, not "NeMo license" as stated. Actually better than claimed. |
| **BasicVSR++** | Designed for **video super-resolution**, not denoising. Using it for denoising is a misapplication. Replace with a purpose-built video denoising model. |
| **pyannote** | v4.0.3 has a known VRAM regression — uses ~9.5 GB instead of ~2 GB. Must pin to v3.x. |
| **Voxtral Mini 3B** | Reasonable but **NVIDIA Canary-Qwen-2.5B** tops the HuggingFace Open ASR Leaderboard at 5.63% WER, runs at 418 RTFx, and uses less VRAM. |

**Model loading/unloading overhead:** Each 16GB model swap (load from NVMe → GPU, PyTorch init, CUDA warmup) takes 10-30 seconds. For a typical ingest touching 5-6 models, expect **60-120 seconds of pure model-swapping overhead**. The design should specify: safetensors format, pre-compiled TensorRT engines, and system RAM caching for weights.

**Sources:** [YOLO26 license](https://www.ultralytics.com/license), [InsightFace GitHub](https://github.com/deepinsight/insightface), [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b), [pyannote VRAM issue #1963](https://github.com/pyannote/pyannote-audio/issues/1963)

---

### 9. No Existing libplacebo GStreamer Element OR OCIO GStreamer Plugin Exists Anywhere

Both are novel engineering. Zero prototypes, zero discussions, zero MRs found.

**libplacebo GStreamer element complexity:**
- The FFmpeg `vf_libplacebo.c` is ~1500-1600 lines but benefits from FFmpeg's mature `FFVulkanContext` abstraction. GStreamer has no equivalent.
- GStreamer's GL pipeline is built around OpenGL; libplacebo is Vulkan-native. Bridging requires either running libplacebo on its degraded OpenGL backend, or building a Vulkan GStreamer element outside GStreamer's standard GL infrastructure.
- The 2000-4000 line estimate is optimistic for production quality. A fully-featured element with buffer pool negotiation and zero-copy paths could exceed 4000 lines.

**OCIO GStreamer integration complexity:**
- OCIO 2.x provides a `GpuShaderCreator` API that generates GLSL from transforms. Integration requires: create Processor → extract GPU shader → upload LUT textures → compile GLSL → apply via `GstGLFilter` subclass.
- Estimated 1500-2500 additional lines.
- Dynamic transforms (changing color space mid-stream) require shader recompilation.

**Total custom rendering code: likely 6000-10000+ lines** across libplacebo element, OCIO element, color grading shaders, and compositor.

---

### 10. The "One Boundary Crossing" Claim Only Holds for Trivial Pipelines

The claim that data crosses CUDA-GL boundaries only at pipeline start and end breaks in:
- **Multi-stream editing:** Each decoded stream exits CUDA independently. N inputs = N boundary crossings.
- **Mid-pipeline CUDA operations:** Denoising, optical flow for retiming, or CUDA-based LUT application each require GL→CUDA→GL round-trips.
- **Forked output paths:** Simultaneous encode (CUDA/NVENC) + preview (OpenGL) requires the pipeline to fork, adding per-frame crossings on the encode side.
- **Multi-pass rendering:** Some compositing operations require reading back results for the next pass.

---

## Gaps and Missing Pieces

### 11. Missing Academic Reference: EditDuet (SIGGRAPH 2025)

[EditDuet](https://research.adobe.com/publication/editduet-a-multi-agent-system-for-video-non-linear-editing/) (Adobe Research, SIGGRAPH 2025) is the most directly comparable system to this design. It uses an Editor + Critic dual-agent architecture for NLE, evaluated on EditStock (56 real filmmaking projects). The design should position itself relative to EditDuet.

Other missing papers: EGAgent (arXiv:2601.18157, Jan 2026), LongVideoAgent (arXiv:2512.20618, Dec 2025), "Rewriting Video" (arXiv:2601.08565, Jan 2026).

---

### 12. Verification Must Be Interleaved, Not Post-Hoc

The "17x error trap" is from ["Towards a Science of Scaling Agent Systems"](https://arxiv.org/abs/2512.08296) (Google DeepMind + MIT, Dec 2025). Key quantitative finding:
- **Independent agents amplify errors 17.2x**
- **Centralized coordination with inline verification reduces this to 4.4x**

The design's verification loop runs after each tool execution, which is good. But the DeepMind paper shows that step-level corrections during execution outperform post-hoc correction. The verification should be integrated into the tool execution itself, not a separate pass.

Also: for sequential reasoning tasks (narrative construction, shot ordering), **all multi-agent variants degraded performance by 39-70%**. The orchestration agent should handle narrative decisions solo, using multi-agent only for parallelizable analysis.

---

### 13. Histogram-Based Color Verification Is Insufficient

Histograms capture value distribution per channel but miss:
- Spatial relationships (two different images can have identical histograms)
- Hue accuracy (a hue shift preserving luminance distribution is invisible)
- Specific color accuracy (skin tones, brand colors)

**Replace with:** Delta E (dE2000) against reference test charts (ColorChecker), per-pixel color difference maps, SSIM for structural comparison, waveform/vectorscope analysis.

---

### 14. OpenGL in Docker Requires Explicit EGL Configuration

GStreamer's GL elements require EGL headless rendering in a container. This is **not** default behavior:
- Must use `nvidia/opengl` or `nvidia/cudagl` base images with libglvnd
- `NVIDIA_DRIVER_CAPABILITIES` must include `graphics`
- GStreamer must be compiled with EGL support (`GST_GL_PLATFORM=egl`, `GST_GL_WINDOW=gbm`)
- GStreamer can **silently fall back to software rendering** (LLVMpipe) if EGL initialization fails — pipeline appears to work at 1/100th performance
- `nvidia/cudagl` images on Docker Hub are **deprecated and no longer maintained**. Must build custom base image.

---

### 15. Ingest Pipeline Missing Analysis Dimensions

The visual analysis pipeline (shot classification + scene description + face detection) is necessary but insufficient. Missing:
- **Emotion/sentiment analysis** — critical for editing decisions (pacing, music selection, shot choice). See EmoVid (arXiv:2511.11002).
- **Audio scene classification** — music detection, ambient sound type, dialogue quality assessment. EGAgent highlights hybrid visual+audio search as critical.
- **Motion/camera movement classification** — pan, tilt, dolly, handheld, static. Important for cut decisions.
- **Aesthetic quality scoring** — standard in automated editing pipelines.
- **Dialogue quality assessment** — signal-to-noise ratio, room tone consistency, clipping detection.

---

### 16. "No Footage Leaves the Machine" Is Misleading

The cloud orchestration agent (Claude/GPT) receives:
- User prompts describing editing intent
- Transcription text (runs locally, but text sent to orchestrator)
- Scene descriptions and shot classifications (from local VLM, but summaries sent)
- Timeline metadata (clip names, in/out points, effect parameters)
- File paths, directory structures, project names

For confidential content (unreleased films, corporate video, legal depositions), this metadata is itself sensitive. The design should document exactly which data fields reach cloud APIs and offer a fully-local orchestration option.

---

### 17. Security Is Underspecified

- **CVE-2025-23266 ("NVIDIAScape"):** Critical (CVSS 9.0) container escape vulnerability in NVIDIA Container Toolkit. All versions up to 1.17.7 affected. Public exploit exists.
- The agent executing arbitrary `melt`/GStreamer commands is **arbitrary code execution**. GStreamer's `gst-launch` can load arbitrary shared libraries via plugin paths. MLT's `melt` can execute shell commands via certain filters.
- **Mandatory mitigations:** Pin NVIDIA Container Toolkit ≥1.17.8, run as non-root user, drop all unnecessary capabilities, whitelist allowed GStreamer elements and MLT filters.

---

### 18. Node-Based Grading Pipeline Is Missing

Professional color grading is not "apply one shader." It's executing a DAG of operations where each node can be a correction, qualifier, window, LUT, resize, key mixer, or layer composite. The design has individual GLSL shaders but no **graph execution engine** to compose them. This is the core of a professional grading tool and represents the majority of the color engineering work.

---

### 19. Cache Invalidation Is the Hardest Problem (Underspecified)

The design handwaves cache invalidation. Real scenarios:
- **Track-level filter change** (e.g., LUT on V1): invalidates every segment on that track
- **Speed change on a clip:** invalidates that clip's segments AND shifts all subsequent time positions
- **Ripple edit:** every segment after the edit point shifts, potentially invalidating the entire timeline
- **A 10-minute color grade change:** 300 segments × 400ms each = 120 seconds sequential render, 30 seconds with 4 workers

The invalidation graph is a dependency DAG where filters, transitions, and track-level effects create complex chains. Building a correct and efficient system is arguably harder than the rendering itself. The design needs a render queue with priority scheduling, bounded concurrency, cancellation of stale renders, and GPU session limits (consumer NVIDIA cards support only 2-3 concurrent NVENC sessions).

---

## Alternative Approaches the Design Should Consider

### A. Vulkan Video Pipeline (Emerging)

GStreamer 1.28 now supports Vulkan H.264 encode, H.265 decode (10-bit), AV1 decode, VP9 decode. A full Vulkan pipeline would eliminate ALL API boundary crossings and give access to compute shaders, subgroup ops, push constants. Current limitation: H.265/AV1 encode not yet available. But this is the clear direction — Khronos has published [Vulkan Video Encode AV1 extensions](https://www.khronos.org/blog/khronos-announces-vulkan-video-encode-av1-encode-quantization-map-extensions).

### B. AMD HIP Plugin (GStreamer 1.28)

New [HIP plugin](https://centricular.com/devlog/2025-07/amd-hip-integration/) provides `hipcompositor`, `hipconvert`, etc. HIP code runs on both AMD (ROCm) and NVIDIA (CUDA translation). Vendor-neutral GPU compute path the design didn't consider.

### C. WebCodecs + Canvas for Preview Instead of HLS

Sub-frame latency, direct decode-to-canvas, instant seeking. Used by Clipchamp in production. Browser support: Chrome 94+, Edge 94+, Firefox 130+, Safari 16.4+.

### D. GES API Instead of MLT XML Text Manipulation

Proper Python API for timeline construction. Active maintenance as part of GStreamer monorepo. GPU-accelerated rendering via GStreamer hardware elements.

### E. In-Process libmlt Render Daemon Instead of melt-per-Segment

Eliminates 150-500ms startup overhead per segment. Keeps source media open and MLT graph warm. Accepts render requests over IPC. Shotcut and Kdenlive use this pattern (libmlt as in-process library).

---

## Questions That Need Answering

1. **Compositing architecture:** If `glvideomixer` can't do professional blend modes, are you building a custom compositor from scratch? Or chaining `glshader` elements? What's the actual compositing pipeline?

2. **Vulkan timeline:** Given libplacebo is Vulkan-first and GStreamer is adding Vulkan Video support, should the design target Vulkan as the primary GPU API with OpenGL as fallback, rather than the reverse?

3. **GES vs MLT:** Given that the agent is programmatically constructing edits, why use XML text manipulation (fragile, underspecified) instead of GES's Python API (typed, documented, maintained)?

4. **Non-destructive color:** How do you handle users wanting to change the project working space after ingest if color transforms are baked in?

5. **VRAM budget:** What specific quantization levels are mandated for each model? What's the maximum concurrent GPU memory allocation? Is time-multiplexing sufficient or does the pipeline need redesign?

6. **AGPL contamination:** YOLO26 is AGPL-3.0. What's the replacement? RT-DETR? YOLO-World with a different license? A permissively-licensed detection model?

7. **Preview latency target:** If HLS is replaced with WebCodecs, what's the new architecture? Does the segment-based caching concept survive, or does the design shift to real-time GPU rendering (like actual NLEs)?

8. **Container vs bare metal:** Professional VFX pipelines run bare metal. Is containerization actually required for the primary workflow, or only for CI/CD and distribution?

9. **Model swap overhead:** 60-120 seconds of model loading/unloading during ingest. Is this acceptable? Should models be pre-loaded into system RAM? TensorRT pre-compilation?

10. **Render pipeline ownership:** Who owns the rendering pipeline — GStreamer (via GES), or MLT (via melt), or a custom engine? The design currently has three render paths (GStreamer GPU, melt CPU, melt GPU/Movit) without clear arbitration.

11. **Evaluation framework:** Will you adopt EditStock as the evaluation dataset and VBench-2.0 dimensions as the quality framework?

12. **A1000 degraded mode:** The laptop A1000 has only 4GB VRAM — insufficient for any AI model and barely adequate for 4K rendering. What's the minimum viable feature set per GPU tier?

---

## Verified Technical Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| glvideomixer has 15 blend functions, 3 equations | **True but misleading** | These are raw GL params, not professional blend modes |
| libplacebo OpenGL has "nearly all" Vulkan features | **Overstated** | Missing push constants, subgroups, compute optional. Vulkan-first direction. |
| CUDA-GL interop is "well-established" | **Partially true** | Works but bugs patched as recently as GStreamer 1.28.1 (Feb 2026) |
| MLT XML has Catmull-Rom spline animation | **True** | 3 modes only: discrete, linear, Catmull-Rom. No bezier, no easing. |
| Voxtral Mini 3B is ~9.5GB VRAM | **Confirmed** | Apache 2.0, bf16/fp16 |
| YOLO26 exists | **Confirmed** | Released Jan 14, 2026 by Ultralytics. AGPL-3.0. |
| SAM 3 exists and is ~4GB | **Confirmed** | Released Nov 20, 2025 by Meta. Apache 2.0. |
| Qwen3-VL 8B has native video temporal modeling | **Confirmed** | Text-Timestamp Alignment + interleaved-MRoPE |
| BasicVSR++ is for denoising | **Misleading** | Designed for super-resolution; denoising is a secondary research application |
| DaVinci Wide Gamut is "fully published" | **Substantially true** | Whitepaper has primaries, transfer function, whitepoint. Minor green primary discrepancy (0.1618 vs 0.1682). |
| ProRes encode is CPU-only | **Confirmed** | No GPU ProRes encoder exists in FFmpeg |
| Melt can render segment ranges | **Confirmed** | Via in=/out= parameters. But full project parse on every invocation. |
| "No footage leaves the machine" | **Misleading** | Semantic metadata, transcriptions, scene descriptions sent to cloud APIs |

---

## Risk Summary

| Risk | Severity | Recommendation |
|------|----------|----------------|
| glvideomixer can't do professional blend modes | **Critical** | Build custom fragment-shader compositor |
| HLS adds 4-8s latency to preview | **Critical** | Replace with WebCodecs + Canvas |
| Ingest-time color baking loses information | **Critical** | Switch to non-destructive pipeline transforms |
| MLT XML text manipulation is fragile for agents | **Critical** | Switch to GES Python API |
| CUDA-GL interop has unquantified performance | **High** | Benchmark early; plan for Vulkan migration |
| libplacebo OpenGL is second-class | **High** | Plan for Vulkan backend |
| VRAM budget exceeds 24GB | **High** | Mandate quantization, add VRAM budget table |
| YOLO26 AGPL contaminates project | **High** | Replace with permissively-licensed model |
| No OCIO or libplacebo GStreamer element exists | **High** | Budget 6000-10000+ lines custom code |
| OpenGL in Docker needs explicit EGL config | **High** | Document EGL requirements, verify hardware GL active |
| Container security (NVIDIAScape CVE) | **High** | Pin toolkit ≥1.17.8, run non-root, whitelist elements |
| Cache invalidation is underspecified | **High** | Design dependency DAG, render queue, cancellation |
| InsightFace weights are non-commercial | **Medium** | Document restriction or train own models |
| BasicVSR++ wrong for denoising | **Medium** | Replace with purpose-built denoiser |
| pyannote v4 VRAM regression | **Medium** | Pin to v3.x |
| MLT single maintainer (bus factor = 1) | **Medium** | Reduce MLT dependency if kept |
| Model swap overhead (60-120s ingest) | **Medium** | TensorRT pre-compilation, safetensors, RAM caching |
| Missing emotion/audio/motion analysis in ingest | **Medium** | Add analysis dimensions |
| Histogram-based color verification insufficient | **Medium** | Use dE2000, SSIM, test charts |
| No node-based grading engine | **Medium** | Design DAG execution engine |
| Missing EditDuet reference | **Low** | Add to academic context |
