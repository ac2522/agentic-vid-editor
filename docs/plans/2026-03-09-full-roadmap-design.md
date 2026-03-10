# AVE Full Roadmap Design

> Designed 2026-03-09. Informed by parallel research into GES API, preview systems, MCP/agent frameworks, GPU color pipeline, and competitive landscape.

---

## Research Findings (informing all phases)

### GES API
- **Transitions:** `timeline.set_auto_transition(True)` + overlap clips on same layer. No manual TransitionClip needed. Set `vtype` on auto-created TransitionClip for wipes.
- **Speed:** `clip.set_child_property("rate", 2.0)` — built into GES sources. `pitch` element for audio rate, `videorate` for video.
- **Volume:** `GES.Effect.new("volume")` for static. `GstInterpolationControlSource` + `DirectControlBinding` for keyframed fades.
- **Fade caveat:** `DirectControlBinding` normalizes 0-1 → volume 0-10. Unity gain = control source value 0.1.

### Preview Systems
- **LL-HLS:** Overkill for localhost. 1-2s minimum latency even with partial segments. Drop it.
- **WebRTC:** `webrtcsink` (gst-plugins-rs) gives sub-500ms playback latency. Simplest GStreamer path for continuous playback.
- **WebSocket frame serving:** No existing NLE library. Simple pattern: client sends timecode, server decodes frame, returns JPEG. Must build.
- **Segment rendering:** GES pipeline seek with start+stop positions. Render to fMP4 chunks.
- **WebCodecs:** Browser-side codec access for fast client-side scrubbing from cached segments.

### Agent Infrastructure
- **Claude Agent SDK:** Python `claude-agent-sdk` on PyPI. Production-ready. Supports subagents, MCP native. Does NOT yet support `defer_loading` (open feature request).
- **FastMCP 3.0:** `@mcp.tool()` decorator, auto JSON schema from type hints. De facto standard for tool serving.
- **Tool discovery at scale (critical):** Flat MCP listing degrades at 20-40 tools. At 50+, Opus 4 accuracy dropped from 79.5% to 49%. Three production patterns exist:
  - *Deferred + Search* (Anthropic API native): BM25/regex search, ~85% token reduction. 60% retrieval accuracy at 4K tools.
  - *Progressive Disclosure / Meta-tools* (Speakeasy, SynapticLabs BCPs): Register 2 meta-tools (`getTools`/`useTools`), agent navigates domain → tool summary → full definition. 96-160x token reduction, 100% success rate.
  - *Embedding RAG* (LangGraph BigTool): For 1000+ tools. Fragile — semantic similarity mismatch, embedding drift, no confidence reasoning.
- **AutoTool** (arXiv:2511.14650, Nov 2025): Directed graph of tool transition probabilities from usage patterns. 30% inference cost reduction.
- **MCP server composition:** Production gateways exist (Microsoft, MetaMCP, IBM ContextForge). Pattern: one MCP server per domain, gateway aggregates.
- **Competitive landscape:** No competitor uses GES/GStreamer. VideoAgent/Director use FFmpeg/MoviePy. AVE occupies distinct professional-grade niche.

### GPU Color Pipeline
- **libplacebo v7.360.0:** No upstream GStreamer element exists. Custom plugin confirmed necessary. Target `pl_renderer` + `pl_options` API (stable since PL_API_VER 309).
- **OCIO 2.5.1:** GPU API changed (ABI break from 2.5.0). C++17 required. Built-in ACES 2.0 configs included.
- **skiacompositor:** Still no text rendering in GStreamer 1.28. Custom `skiafilter` callback or `textoverlay`/pango needed.
- **glvideomixer:** Crash bugs #728, #786 still open. Use skiacompositor or CPU compositor instead.

### Transcription
- **"Whisper Has an Internal Word Aligner"** (Sep 2025, arXiv 2509.09987): Select attention heads + character-level decoding → word timestamps without separate aligner. May eliminate MFA/NFA dependency.
- **CrisperWhisper:** Research-grade, not production-ready. Skip.
- **MFA 3.x:** Rewrote internals (Kalpy), much faster. NFA requires full NeMo stack.

---

## Phase 2B: GES Execution Completion

**Goal:** Wire up the 3 remaining pure-logic tools to GES. Complete the editing toolkit.

**Scope:** ~300-500 lines. 1-2 sessions.

| Task | What | Essential Detail |
|------|------|-----------------|
| 2B-1 | Speed execution | `clip.set_child_property("rate", rate)` + adjust clip duration. Use `pitch` element for audio with `preserve_pitch=True`, `videorate` for video. |
| 2B-2 | Volume execution | `GES.Effect.new("volume")` + `set_child_property("volume", linear_gain)`. Static volume only. |
| 2B-3 | Fade execution | `GstInterpolationControlSource` + `DirectControlBinding` on volume property. Keyframes at fade boundaries. Note: control source normalizes 0-1 maps to volume 0-10, so unity = 0.1. |
| 2B-4 | Transition execution | `timeline.set_auto_transition(True)` at creation. Move clip_b start to create overlap. Set `vtype` on auto-created TransitionClip for wipes. |
| 2B-5 | Tests + E2E | Unit tests for each operation. E2E: ingest → apply speed/volume/fade/transition → render → probe output. |

---

## Phase 3: Preview System

**Goal:** Render timeline segments, serve them for browser playback and scrubbing.

**Architecture decision:** Segment cache + WebSocket scrubbing for v1. WebRTC as optional enhancement. Drop LL-HLS entirely — adds complexity with no benefit on localhost.

| Task | What | Essential Detail |
|------|------|-----------------|
| 3-1 | Segment renderer | Extend `render/` with `render_segment(xges_path, output, start_ns, stop_ns)`. Uses GES pipeline seek with start/stop. Output: fMP4 (fragmented MP4) segments. |
| 3-2 | Segment cache manager | Track segment state: dirty/rendering/clean. Invalidate on timeline edit. Priority: viewport-visible segments first. Store in `cache/segments/`. |
| 3-3 | Preview server (HTTP + WebSocket) | HTTP serves cached segments as static files. WebSocket handles: frame requests (timecode → JPEG), playback state sync, segment invalidation notifications. Python `aiohttp` or `websockets`. |
| 3-4 | Frame extractor | Given timecode, decode single frame from cached segment (or direct GES pipeline seek to PAUSED + screenshot). Return JPEG/WebP over WebSocket. For scrubbing. |
| 3-5 | Browser preview client | Minimal HTML/JS. MSE (Media Source Extensions) for segment playback. WebSocket for scrubbing frames. Canvas for frame display. Transport controls. |
| 3-6 | WebRTC live path (stretch) | `webrtcsink` for real-time playback preview. Lower latency than segment cache for continuous play. Optional — segments + MSE may be sufficient for v1. |

---

## Phase 4: Tool Architecture + Agent Orchestration

**Goal:** Expose all ave tools via domain-scoped MCP servers with progressive disclosure. Build orchestrator that takes natural language editing instructions.

### Tool Discovery Architecture

**Pattern: Domain-scoped MCP servers + Progressive Disclosure meta-tools.**

AVE will have ~40-80 tools across 6 domains. Flat listing would consume 8K-16K tokens and degrade selection accuracy. Instead:

**Layer 1 — Domain servers (one FastMCP server per module):**
- `ave-editing` — trim, split, concatenate, speed, transitions
- `ave-audio` — volume, fade, normalize
- `ave-color` — IDT, LUT, grade (Phase 5+)
- `ave-transcription` — transcribe, align, search transcript
- `ave-render` — render segment, render export, proxy
- `ave-project` — timeline CRUD, clip management, metadata, asset registry

**Layer 2 — Meta-tools (always loaded, ~500 tokens):**
- `list_domains()` → returns domain names + 1-line summaries
- `list_tools(domain)` → returns tool names + descriptions for one domain
- `use_tool(domain, tool, params)` → executes a tool, loading its full schema on demand

The agent navigates: intent → domain → tool → execute. No embedding search. No flat listing. Deterministic.

**Layer 3 — Graph dependencies (tool prerequisites):**
Some tools require others. Example: `apply_transition` needs two adjacent clips (requires `concatenate` or manual placement first). `render_segment` needs a saved timeline. Dependencies expressed as metadata on each tool: `requires: ["timeline_loaded"]`, `provides: ["clip_added"]`. The meta-tool validates prerequisites before execution.

**Why not RAG:** At AVE's scale (40-80 tools), RAG's failure modes (semantic mismatch between `trim` vs `split`, embedding drift) outweigh benefits. Progressive disclosure is deterministic and debuggable.

**Why not flat Anthropic deferred loading:** The Agent SDK doesn't support `defer_loading` yet (open feature request). Even when it does, BM25 search hits only 60% retrieval accuracy at scale. Domain scoping is more reliable.

**Migration path:** If tool count exceeds 100+ (Phase 7+), add semantic search within domains as an optimization. The domain-scoped architecture makes this a per-domain enhancement, not a rearchitecture.

### Tasks

| Task | What | Essential Detail |
|------|------|-----------------|
| 4-1 | Domain MCP servers | One FastMCP 3.0 server per module. `@mcp.tool()` on each function. Stdio transport. Each server independently testable. |
| 4-2 | Meta-tool layer | `list_domains()`, `list_tools(domain)`, `use_tool(domain, tool, params)`. Always-loaded. Validates prerequisites from tool dependency graph before execution. |
| 4-3 | Tool dependency graph | JSON/YAML definition of tool prerequisites and provisions. Example: `trim` requires `["timeline_loaded", "clip_exists"]`, provides `["clip_trimmed"]`. Validated by meta-tool layer. |
| 4-4 | Tool descriptions | Each tool gets a description optimized for LLM selection. Include parameter constraints, units (nanoseconds), and examples. Test with real queries — iterate on names/descriptions per Anthropic's ACI guidance. |
| 4-5 | Project state resource | MCP resource exposing current timeline state: clip list, durations, effects, metadata. Agents query this before making edits. Exposed via `ave-project` server. |
| 4-6 | Agent orchestrator | Claude Agent SDK. Single orchestrator receives user intent, uses meta-tools to discover and invoke domain tools. Sees domain summaries in system prompt, drills down as needed. |
| 4-7 | Editing session manager | Manages lifecycle: load project → accept commands → apply edits → trigger preview update → save. Bridges agent output to GES execution. |
| 4-8 | Transcript-driven editing | Tool in `ave-transcription` domain. Takes transcript + edit instruction ("remove ums", "cut from X to Y") and translates to trim/split operations using word timestamps. |

---

## Phase 5: GPU Color Pipeline

**Goal:** Real color science — libplacebo tone mapping, OCIO transforms, non-destructive IDTs.

**Prerequisite:** GStreamer >= 1.28.1 (CUDA-GL interop fixes).

**Risk:** Largest custom C code in the project (~4000-6000 lines across both elements). Plan for iterative development — passthrough → tone mapping → full color pipeline.

| Task | What | Essential Detail |
|------|------|-----------------|
| 5-1 | libplacebo element (real) | Upgrade C prototype to functional `GstGLFilter`. Target `pl_renderer` + `pl_options` API (stable since PL_API_VER 309). OpenGL 4.3+ backend. Tone mapping, HDR→SDR, gamut mapping, scaling. |
| 5-2 | OCIO element (real) | Upgrade C prototype. OCIO 2.5.1 GPU API (note ABI break). Load config → create processor → `GpuShaderDesc` → upload LUT textures → apply in GL pipeline. C++17 required. |
| 5-3 | IDT application at render | Apply IDTs from asset registry at render time. Chain: source (camera log) → OCIO IDT → libplacebo (working space processing) → OCIO ODT → output. Non-destructive — original files untouched. |
| 5-4 | Color pipeline integration | Wire libplacebo + OCIO elements into GES render pipeline via custom `GES.Effect` wrappers. Ensure 10-bit+ throughout (no 8-bit truncation). |
| 5-5 | ACES 2.0 config | Ship built-in ACES 2.0 config (included in OCIO 2.5). Default project color space: ACEScct (AP1). IDT auto-detection from probe metadata. |
| 5-6 | Docker: GPU color deps | Add libplacebo, OCIO 2.5.1 builds to Dockerfile. Verify CUDA-GL interop on GStreamer 1.28.1+. |

---

## Phase 6: Advanced Editing Features

**Goal:** Text/graphics, visual analysis, multi-format export.

| Task | What | Essential Detail |
|------|------|-----------------|
| 6-1 | Text overlay system | `textoverlay` (pango) for simple titles. For animated lower thirds: evaluate Qt/QML via `qml6glsink` vs custom `skiafilter` callback. Agent generates text overlay parameters. |
| 6-2 | Scene detection | PySceneDetect or custom FFmpeg-based scene detection at ingest. Store cut points in asset registry. Agent uses for automatic rough cut. |
| 6-3 | Shot classification | CLIP-based zero-shot classification (wide/medium/close-up/etc). Run at ingest, store in registry. Inform agent editing decisions. |
| 6-4 | OTIO export | Build XGES→OTIO reader. OTIO has existing adapters for AAF (Resolve/Avid), FCPXML (FCPX), OTIO→Premiere via AAF. Enables "export to Resolve" workflow. |
| 6-5 | OTIO import | Build OTIO→XGES writer. Accept timelines from other NLEs for agent-assisted editing. |
| 6-6 | Multi-format render | Export presets: H.264 web, H.265 archive, ProRes/DNxHR master, per-platform social media specs. Wraps GES render pipeline with codec/container selection. |

---

## Phase 7: Multi-Agent + Production Hardening

**Goal:** Specialized agent roles, error recovery, performance.

| Task | What | Essential Detail |
|------|------|-----------------|
| 7-1 | Role-based agents | Subagents via Claude Agent SDK: Editor (trim/split/arrange), Colorist (color/LUTs), Sound Designer (volume/fade/normalize), Transcriptionist (transcribe/align). Orchestrator delegates by domain. |
| 7-2 | Intra-domain search | If any single domain exceeds ~30 tools: add semantic or BM25 search within that domain's `list_tools()`. Per-domain enhancement, not a rearchitecture. Also evaluate AutoTool-style transition graphs from usage data. |
| 7-3 | Verification loop | Agent verifies each edit: render affected segment → probe → compare to intent. Catches the 4.4x error amplification (centralized orchestration mitigation from Kim et al.). |
| 7-4 | Whisper internal aligner | Evaluate "Whisper Has an Internal Word Aligner" (Sep 2025) — select attention heads + character-level decoding for word timestamps. Could eliminate MFA/NFA dependency. |
| 7-5 | Error recovery | Undo stack via XGES snapshots. Rollback on failed operations. Graceful degradation on GPU unavailability (CPU fallback paths). |
| 7-6 | Performance | Parallel segment rendering. GPU-accelerated ingest (NVDEC decode → NVENC re-encode). Proxy-first editing with online conform. |
| 7-7 | Compositor strategy | Evaluate `skiacompositor` as primary (avoids glvideomixer crash bugs #728, #786). Fallback to CPU `compositor` if skia unavailable. |

---

## Phase 8: Web UI + Integration

**Goal:** Usable interface beyond CLI/agent.

| Task | What | Essential Detail |
|------|------|-----------------|
| 8-1 | Timeline UI | Browser-based timeline visualization. Display clips, transitions, effects. Drag-and-drop editing sends commands to agent. |
| 8-2 | Chat interface | Natural language editing commands. Powered by Phase 4 orchestrator. Show agent reasoning + preview updates in real-time. |
| 8-3 | Asset browser | Grid/list view of ingested assets with thumbnails, metadata, transcripts. Drag to timeline. |
| 8-4 | Color grading UI | Wheels/curves/scopes interface driving OCIO+libplacebo parameters. Real-time preview via WebRTC path. |

---

## Dependency Graph

```
Phase 2B (GES completion)
    ↓
Phase 3 (Preview) ←→ Phase 4 (Agent/MCP)
    ↓                    ↓
Phase 5 (GPU Color)   Phase 6 (Advanced Editing)
    ↓                    ↓
    └──→ Phase 7 (Multi-Agent + Hardening) ←──┘
                   ↓
            Phase 8 (Web UI)
```

Phases 3 and 4 can run in parallel. Phase 5 and 6 can run in parallel. Phase 7 depends on both. Phase 8 depends on 7.

---

## Sources

### GES API
- [GES TransitionClip docs](https://gstreamer.freedesktop.org/documentation/gst-editing-services/gestransitionclip.html)
- [GES Timeline (auto-transition)](https://gstreamer.freedesktop.org/documentation/gst-editing-services/gestimeline.html)
- [GStreamer Dynamic Controllable Parameters](https://gstreamer.freedesktop.org/documentation/application-development/advanced/dparams.html)
- [GstInterpolationControlSource](https://gstreamer.freedesktop.org/documentation/controller/gstinterpolationcontrolsource.html)
- [Bringing slow motion to Pitivi (GES rate)](https://suhas2go.github.io/gnome/pitivi/2018/06/02/GES_Slow_Motion/)

### Preview Systems
- [webrtcsink documentation](https://gstreamer.freedesktop.org/documentation/rswebrtc/webrtcsink.html)
- [WebCodecs API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API)
- [GESPipeline documentation](https://gstreamer.freedesktop.org/documentation/gst-editing-services/gespipeline.html)
- [GStreamer Seeking design](https://gstreamer.freedesktop.org/documentation/additional/design/seeking.html)

### Agent Infrastructure & Tool Discovery
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)
- [claude-agent-sdk on PyPI](https://pypi.org/project/claude-agent-sdk/)
- [FastMCP 3.0](https://gofastmcp.com/tutorials/create-mcp-server)
- [Anthropic: Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Tool Search Tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Speakeasy: 100x Token Reduction via Progressive Disclosure](https://www.speakeasy.com/blog/100x-token-reduction-dynamic-toolsets)
- [SynapticLabs: Bounded Context Packs / Meta-Tool Pattern](https://blog.synapticlabs.ai/bounded-context-packs-meta-tool-pattern)
- [AutoTool: Graph-based Tool Selection (arXiv:2511.14650)](https://arxiv.org/abs/2511.14650)
- [Arcade: Anthropic Tool Search at 4000 Tools](https://blog.arcade.dev/anthropic-tool-search-4000-tools-test)
- [MCP server composition gateways](https://skywork.ai/blog/mcp-server-vs-mcp-gateway-comparison-2025/)
- [LangGraph BigTool](https://github.com/langchain-ai/langgraph-bigtool)
- [VideoAgent graph orchestration](https://github.com/HKUDS/VideoAgent)

### GPU Color Pipeline
- [libplacebo GitHub](https://github.com/haasn/libplacebo)
- [OCIO 2.5 Release Notes](https://opencolorio.readthedocs.io/en/latest/releases/ocio_2_5.html)
- [skiacompositor documentation](https://gstreamer.freedesktop.org/documentation/skia/index.html)
- [glvideomixer issue #728](https://gitlab.freedesktop.org/gstreamer/gst-plugins-base/-/issues/728)

### Transcription
- ["Whisper Has an Internal Word Aligner" (arXiv 2509.09987)](https://arxiv.org/abs/2509.09987)
- [MFA 3.0 Changelog](https://montreal-forced-aligner.readthedocs.io/en/latest/changelog/news_3.0.html)

### Competitive Landscape
- [VideoAgent (GitHub)](https://github.com/HKUDS/VideoAgent)
- [Director (GitHub)](https://github.com/video-db/Director)
- [a16z: It's time for agentic video editing](https://a16z.com/its-time-for-agentic-video-editing/)
