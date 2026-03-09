# Agentic Video Editor: Landscape Research

> Research conducted 2026-03-07. Covers existing projects, tech stack options, agent orchestration patterns, color science replicability, and rendering engine analysis.

## Existing Agentic Video Editing Projects

### Most Relevant

| Project | What It Does | Stack | Maturity |
|---|---|---|---|
| [HKUDS/VideoAgent](https://github.com/HKUDS/VideoAgent) | 30+ specialized editing agents, graph-powered workflow generation | Claude 3.7 Sonnet, GPT-4o, Python | Research project, 87-98% orchestration success |
| [VideoDB Director](https://github.com/video-db/Director) | 20+ built-in agents for video tasks, reasoning engine, chat UI | Python, VideoDB cloud, MIT license | Production-oriented but cloud-dependent |
| [logsv/llm-video-editor](https://github.com/logsv/llm-video-editor) | Full local pipeline: transcription, scene detection, LLM EDL planning, FFmpeg render | Python 3.11, Ollama, FFmpeg | Working CLI, runs locally |
| [burningion/video-editing-mcp](https://github.com/burningion/video-editing-mcp) | MCP server for video editing, creates OTIO projects for Resolve | Python, OTIO, MCP | 243 GitHub stars |
| [burningion/pydantic-video-editing-agent](https://github.com/burningion/pydantic-video-editing-agent) | End-to-end agent: research, voiceover, source media, assemble edit | Pydantic AI, Python | Proof of concept |
| [digitalsamba/claude-code-video-toolkit](https://github.com/digitalsamba/claude-code-video-toolkit) | Claude Code video production: Remotion components, transitions, themes, Playwright recording | TypeScript, Remotion, Python | Most complete Claude Code video integration |
| [samuelgursky/davinci-resolve-mcp](https://github.com/samuelgursky/davinci-resolve-mcp) | MCP server for Resolve with 202 API features | Python, MCP, MIT license | Only 8% verified working, Linux unsupported |

### Also Notable

| Project | What It Does |
|---|---|
| [Diffusion Studio Agent](https://github.com/diffusionstudio/agent) | Browser-based video compositing with agent layer (WebCodecs) |
| [RayVentura/ShortGPT](https://github.com/RayVentura/ShortGPT) | Automated short-form content pipeline with "LLM oriented video editing language" |
| [WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor) | Automated silence/motion editing, exports Resolve XML |
| [mitch7w/ai-video-editor](https://github.com/mitch7w/ai-video-editor) | LLM rough-cut editing with Claude 3.5 Sonnet |
| [Remotion Skills](https://remotion.dev/) | Claude Code + React = rendered video. Natural language to code to MP4 |

### Key Gaps in Existing Projects

None of them combine:
- TDD-driven tool development
- Docker + CUDA containerization
- GPU-accelerated rendering pipeline
- MCP-style searchable tool registry
- Open-source color science (OCIO/ACES)
- Flexible workflow customization (system instructions, prompts)
- Professional-grade compositing and color grading

---

## Agent Orchestration Patterns for Large Tool Sets

### How to Handle 100+ Tools

The "lots of tools" problem is solved. Three key patterns from research:

**1. MCP Deferred Loading (Claude Code's own approach)**
- Tools marked `defer_loading: true` when definitions exceed 10K tokens
- A single "Tool Search" meta-tool replaces all tool schemas in context
- Agent searches and loads only 3-5 relevant tools per step
- 85% reduction in token consumption (77K → 8.7K)
- Two search modes: regex (precise) and BM25 (semantic)

**2. Tool RAG (Red Hat, 2025)**
- Tool descriptions embedded as vectors in a knowledge base
- Semantic search retrieves top-k most relevant tools per step
- 3x improvement in tool invocation accuracy, 50% prompt length reduction

**3. ITR — Instruction-Tool Retrieval (arXiv, February 2026)**
- Per-step retrieval of minimal system-prompt fragments AND minimal tool subsets
- 95% fewer tokens per step, 32% better tool routing accuracy, 70% cost reduction
- Gains compound with number of agent steps — critical for long editing sessions

### Role-Based Agent Partitioning

The a16z framework identifies five agent roles for video editing:
1. **Processing agents** — ingest, normalize, identify footage types
2. **Orchestration agents** — coordinate between tools, manage workflow
3. **Polish agents** — micro-refinements (audio cleanup, color touch-ups)
4. **Adaptation agents** — repurpose across formats/platforms
5. **Optimization agents** — high-level creative decisions

Each role sees only its relevant tool subset, naturally limiting search space.

### Relevant Frameworks

| Framework | Pattern | Tool Management |
|---|---|---|
| MCP (Model Context Protocol) | Federated registry + deferred loading | `.well-known/mcp.json` capability advertisement |
| CrewAI | Role-based collaboration | Tools bound to agent roles |
| LangGraph | Graph-based state machines | Structured tool routing per node |
| OpenFX | Suite-and-action plugin standard | Describe-before-instantiate, typed metadata |

### Sources

- [MCP Tool Search Guide](https://www.atcyrus.com/stories/mcp-tool-search-claude-code-context-pollution-guide)
- [MCP Registry](https://github.com/modelcontextprotocol/registry)
- [Tool RAG — Red Hat](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/)
- [ITR Paper](https://arxiv.org/abs/2602.17046)
- [Tool-to-Agent Retrieval](https://arxiv.org/html/2511.01854v1)
- [a16z: Time for Agentic Video Editing](https://a16z.com/its-time-for-agentic-video-editing/)
- [OpenFX Documentation](https://openfx.readthedocs.io/en/main/Reference/ofxStructure.html)

---

## DaVinci Resolve as a Rendering Backend

### What the API Can Do

| Capability | Status |
|---|---|
| Project/timeline creation | Fully supported |
| Media import and organization | Fully supported |
| Add clips to timeline | Fully supported (with recordFrame positioning) |
| Apply LUTs to nodes | Fully supported |
| Apply CDL corrections | Fully supported |
| Apply DRX grade templates | Fully supported |
| Insert text/titles | Fully supported |
| Set opacity | Fully supported |
| Set composite/blend modes | Fully supported (33 modes) |
| Trigger renders | Fully supported |
| Import OTIO timelines | Fully supported |

### What the API Cannot Do (Showstoppers)

| Gap | Severity | Workaround |
|---|---|---|
| **Add transitions** | Showstopper | None via scripting |
| **Change clip speed/retiming** | Showstopper | None via scripting |
| **Per-clip audio volume/fades** | High | External audio mixing |
| **Move clips on timeline** | Medium-High | Delete + re-add (loses clip state) |
| **Create color nodes** | Medium | Bugged; use DRX templates |
| **Keyframe Edit page properties** | Medium | Must use Fusion sub-API |

### Headless Mode

- `resolve -nogui` works but requires NVIDIA GPU, supported Linux distro, and correct drivers
- Linux scripting is explicitly unsupported by davinci-resolve-mcp
- Resolve Studio ($295) required for external scripting
- Documented reliability issues on Linux

### Verdict

Resolve cannot serve as a complete rendering backend due to missing transitions, speed changes, and audio control. Useful as an optional export target for manual finishing/grading.

---

## Color Science: Replicability Assessment

### The Bottom Line

Resolve's color science advantage is ~85% excellent engineering of publicly known algorithms, ~10% undisclosed implementation curves (with open equivalents), ~5% genuinely proprietary neural networks.

### What Is Fully Public and Already Implemented

| Component | Status | Open Implementation |
|---|---|---|
| DaVinci Wide Gamut color space | Published whitepaper with full spec | colour-science Python, OCIO ACES config |
| DaVinci Intermediate transfer function | Published constants and formulas | colour-science Python |
| YRGB luminance-preservation | Well-understood principle since 1980s | Implementable from first principles |
| Lift/gamma/gain | Standard math | Everywhere |
| ASC CDL (Slope/Offset/Power/Sat) | Open ASC standard | Everywhere |
| Curves, HSL qualification | Standard techniques | Textbook computer graphics |
| ACES color management | Fully open, Hollywood standard | OpenColorIO 2.5 |
| HDR tone mapping | Open standards | libplacebo (BT.2390, BT.2446a, ST.2094-40) |
| Debayering | Academic literature | RawTherapee, darktable, colour-demosaicing |

### What Is Genuinely Proprietary

| Component | Impact | Open Alternative |
|---|---|---|
| DaVinci tone mapping curves | Aesthetic preference | ACES 2.0 Output Transform, OpenDRT, libplacebo |
| DaVinci gamut mapping curves | Aesthetic preference | ACES 2.0 gamut mapping |
| Neural Engine (Magic Mask, UltraNR, Speed Warp) | AI features | RIFE, segmentation models (different, not drop-in) |

### Key Sources

- [Blackmagic DWG Whitepaper](https://documents.blackmagicdesign.com/InformationNotes/DaVinci_Resolve_17_Wide_Gamut_Intermediate.pdf)
- [colour-science Python](https://github.com/colour-science/colour)
- [OpenColorIO](https://opencolorio.org/)
- [ACES 2.0](https://docs.acescentral.com/)
- [OpenDRT](https://github.com/jedypod/open-display-transform)
- [libplacebo](https://github.com/haasn/libplacebo)

---

## Rendering Engine Options

### Why FFmpeg + MoviePy Is Not Enough

FFmpeg is a codec tool, not a compositor. Its GPU filter set is ~12 Vulkan filters vs hundreds of CPU filters. MoviePy wraps FFmpeg's CLI. Neither provides real-time multi-track compositing, professional blend modes, or integrated color management.

### Why DaVinci Resolve's API Is Not Enough

Missing transitions, speed changes, and audio control. Only 8% of the MCP wrapper's features verified working. Linux unsupported.

### GStreamer + GES: The Strongest Open-Source Foundation

**Strengths:**
- GPU compositing (`nvcompositor`, `glvideomixer`)
- Editing services library (GES) with timeline, tracks, clips, transitions
- Python bindings via GObject Introspection
- Recently stabilized clip speed control
- OTIO adapter exists (`otio-xges-adapter`)
- Docker + NVIDIA GPU is a solved problem (DeepStream proves it)
- LGPL licensed

**Gaps requiring custom work:**
- No 3D LUT support, no OCIO integration → build libplacebo GStreamer element
- Limited blend modes (Over, Add only) → custom GLSL shaders
- No color grading tools → build lift/gamma/gain/curves as GPU elements
- No motion graphics engine → build renderer (Cairo/Skia) feeding through appsrc
- Fragmented GPU pipeline → careful construction to keep data GPU-resident

### libplacebo: The Key Color/Compositing Library

Extracted from mpv's rendering core. Vulkan-native, production-grade:
- Full color management: ICC profiles, BT.1886, custom 1D/3D LUTs, V-Gamut support
- Dynamic HDR tone mapping with histogram analysis and scene change detection
- Gamut mapping for wide-gamut/SDR conversions
- High-quality scaling (polar/"Jinc" filters, anti-aliasing, anti-ringing)
- Integrated into FFmpeg and VLC
- **No GStreamer plugin exists** — this is the highest-priority custom component

### Other Key Technologies

| Technology | Role | Status |
|---|---|---|
| PyNvVideoCodec 2.0 | GPU decode/encode, zero-copy surfaces | Production-ready, MIT license |
| OpenColorIO 2.5 | Color space management, ACES 2.0 | Industry standard, BSD-3 |
| WhisperX | Word-level transcription + speaker diarization | 4.8% WER, BSD-4 |
| faster-whisper | Segment-level transcription (lighter) | 4-6x faster than vanilla Whisper, MIT |
| PyAV 16.1 | Frame-level Python video manipulation | BSD, best for programmatic work |
| OpenTimelineIO 0.18 | Timeline interchange format | Apache 2.0, ASWF incubation |
| colour-science | Color science algorithms (DWG, ACES, debayering) | BSD-3, comprehensive |

### What Makes Professional NLEs Fast

The key insight from Resolve's architecture: **keep everything on GPU memory throughout the entire pipeline**. The moment frames download to CPU RAM and re-upload, performance collapses. The open-source equivalent:

```
PyNvVideoCodec (decode to GPU) → libplacebo (process on GPU) → GStreamer compositor (GPU textures) → NVENC (encode from GPU)
```

This zero-copy pipeline is architecturally competitive with Resolve.

### Sources

- [GStreamer Editing Services](https://gstreamer.freedesktop.org/documentation/gst-editing-services/)
- [NVIDIA Accelerated GStreamer](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/Multimedia/AcceleratedGstreamer.html)
- [libplacebo](https://github.com/haasn/libplacebo)
- [PyNvVideoCodec 2.0](https://developer.nvidia.com/pynvvideocodec)
- [OpenColorIO](https://opencolorio.readthedocs.io/)
- [WhisperX](https://github.com/m-bain/whisperX)
- [otio-xges-adapter](https://github.com/OpenTimelineIO/otio-xges-adapter)
- [GStreamer Docker NVCODEC](https://gist.github.com/m1k1o/28c73fc15cd1fba59b73364c3b7a5d0a)
