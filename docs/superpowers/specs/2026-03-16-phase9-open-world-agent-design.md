# Phase 9: Open-World Agent Architecture

> Designed 2026-03-16. Informed by research into MCP server design patterns (Cloudflare Code Mode, Anthropic best practices, enterprise MCP servers), Claude Code skills architecture, progressive disclosure, and AI segmentation landscape.

---

## Overview

Phase 9 transforms AVE from a self-contained video editor into an extensible, open-world agent platform. Four pillars, implemented sequentially:

1. **P1: Plugin/Skill System** — Markdown skills + Python plugins with lazy loading and domain namespacing
2. **P2: MCP Server Exposure** — 6 outcome-oriented tools for external agent consumption
3. **P3: Web Search + Research Agent** — Built-in search with replaceable provider, research/execute subagent separation
4. **P4: AI Rotoscoping/Keying** — RotoscopeBackend protocol with SAM 2 + RVM backends, agent-driven keyframe feedback loop

### Design Principles

- **Postel's Law**: Liberal in what AVE accepts (any workflow, any experience level, any plugin), conservative in what the core produces (stable, reliable, validated results).
- **Progressive Disclosure**: Only load what's needed. Manifests at startup, full content on demand.
- **Separation of Concerns**: Research agents can't modify timelines. Editor agents can't access the web.
- **Plugin-first**: New capabilities ship as plugins, not core features. Core provides the machinery; plugins provide the behaviors.

---

## Pillar 1: Plugin/Skill System

### Two-Layer Architecture

**Layer 1: Markdown Skills** — Workflow and process customization. No coding required.

Skills are `.md` files with YAML frontmatter. The agent reads and follows them as playbooks for specific workflows.

**Directories (searched in priority order):**
```
project/.ave/skills/          # Project-specific skills (highest priority)
~/.ave/skills/                # User skills
ave/skills/                   # Built-in skills (shipped with ave)
```

**Skill format:**
```yaml
---
name: cinematic-grain
description: Apply cinematic film grain using texture overlay technique
domain: color
triggers: ["film grain", "grain effect", "analog look"]
---

## Steps
1. Search for film grain overlay textures...
2. Apply as compositing layer with blend mode...
...
```

**Loading strategy:** At startup, only frontmatter is indexed (~50 tokens per skill). When the agent's intent matches a skill's triggers or description, the full markdown body loads on-demand. This mirrors Claude Code's skill architecture.

**Layer 2: Python Plugins** — Capability extension. Registers new tools, backends, or agent behaviors.

**Directories (searched in priority order):**
```
project/.ave/plugins/          # Project plugins (highest priority)
~/.ave/plugins/                # User plugins
ave/plugins/                   # Built-in plugins
```

**Plugin structure:**
```
my-plugin/
    plugin.yaml       # Manifest (always loaded)
    __init__.py        # Entry point (loaded on demand)
    ...                # Implementation files
```

**Manifest format (`plugin.yaml`):**
```yaml
name: rotoscope-sam2
description: SAM 2 video segmentation backend
version: 1.0.0
domain: vfx
tools:
  - name: segment_video
    summary: Segment objects in video using SAM 2
  - name: refine_mask
    summary: Refine segmentation mask at specific frames
requires:
  python: ["torch", "segment-anything-2"]
  system: ["cuda"]  # checked via SystemCapabilities
```

### Domain Namespacing

Every tool has a namespaced identity:
- Built-in: `ave:editing.trim`, `ave:color.grade`
- User plugin: `user:rotoscope-sam2.segment_video`
- Community: `community:film-grain.apply_grain`

The `search_tools` meta-tool searches across all namespaces. Results indicate origin so the agent (and user) know where a tool comes from.

Namespace collisions are impossible — the registry enforces uniqueness within each namespace, and namespaces are structurally separated.

### Plugin Lifecycle

```
discover → load manifest → register summaries → (on first tool invocation) → load code → register full tools
```

1. **Discovery**: On startup, scan plugin directories for `plugin.yaml` files.
2. **Manifest loading**: Parse YAML, register tool summaries (name + one-line description) into the ToolRegistry. ~50 tokens per plugin.
3. **Dependency check**: Validate Python and system requirements against `SystemCapabilities`. If unmet, tools register as unavailable with a clear reason (e.g., "requires CUDA but no GPU detected").
4. **Lazy code loading**: When a tool from the plugin is first invoked via `call_tool`, import the plugin's Python module, run its `register()` function, which adds full tool definitions (schemas, implementations) to the registry.
5. **Unloading**: Not supported in v1. Plugins load once per session. Restart to pick up changes.

### Skill/Plugin Interaction

Skills can reference plugin tools. A skill might say "use `user:rotoscope-sam2.segment_video` with prompts targeting the subject's face." The agent resolves the namespaced tool through the registry.

Plugins can ship bundled skills. A plugin's directory can contain a `skills/` subdirectory with markdown skills that are auto-discovered alongside the plugin.

---

## Pillar 2: MCP Server Exposure

### Design Rationale

Research findings driving this design:
- Cloudflare Code Mode: 2 tools expose an entire 1,100-endpoint API in ~1,000 tokens (99.9% reduction).
- Anthropic guidance: "Do the orchestration in your code, not in the LLM's context window." Target 5-8 tools per server.
- Enterprise servers (GitHub, Notion, Slack): All use outcome-oriented tools, 5-8 per server.
- Claude Code as MCP server: Exposes ~8 tools at the right granularity.

AVE follows this consensus: small set of outcome-oriented tools for casual consumers, with drill-down for power users.

### 6 MCP Tools

```python
@mcp.tool()
def edit_video(instruction: str, options: dict | None = None) -> EditResult:
    """Natural language video editing. AVE's internal orchestrator handles
    tool discovery, role-based routing, execution, and verification.

    Examples:
      edit_video("remove all filler words from the interview")
      edit_video("add a 2-second cross dissolve between clips 3 and 4")
      edit_video("color grade to match ARRI LogC4 to Rec.709")
    """

@mcp.tool()
def get_project_state(include: list[str] | None = None) -> ProjectState:
    """Current timeline structure, clips, effects, metadata, assets.
    Optional filter: include=['clips', 'effects', 'metadata', 'assets']
    Returns compact JSON — not the full XGES XML."""

@mcp.tool()
def render_preview(segment: str | None = None, format: str = "jpeg") -> PreviewResult:
    """Render a preview frame or segment.
    segment: time range as "start_ns-stop_ns" or None for current position.
    format: "jpeg", "png", or "mp4" (for segment video).
    Returns file path to rendered output."""

@mcp.tool()
def ingest_asset(path: str, options: dict | None = None) -> AssetInfo:
    """Bring media into the project. Auto-probes codec, resolution,
    frame rate, color space, duration. Generates proxy if configured."""

@mcp.tool()
def search_tools(query: str, domain: str | None = None) -> list[ToolSummary]:
    """Power user: discover AVE's granular tools by keyword or domain.
    Returns tool names, summaries, and namespaces."""

@mcp.tool()
def call_tool(name: str, params: dict) -> ToolResult:
    """Power user: execute any registered AVE tool directly by name.
    Use search_tools first to discover available tools and their schemas."""
```

### Internal Flow for `edit_video`

When a consuming agent calls `edit_video("add film grain to the intro")`:

1. Instruction enters AVE's `Orchestrator`
2. Orchestrator uses internal `search_tools` to find relevant tools
3. Role-based routing: color instruction → Colorist subagent
4. Colorist selects and executes tools (may trigger skill if "film grain" matches a skill)
5. Verification loop: render affected segment → probe → validate against intent
6. Returns `EditResult` with: what was done, preview path, verification status

The consuming agent never touches AVE's internal domain model. One call in, one result out.

### MCP Resources

```
ave://timeline/current      # Current timeline state as JSON
ave://assets/list           # Ingested assets with metadata
ave://capabilities          # GPU, codecs, plugins, backends available
```

### Transport

```bash
# Stdio transport (for Claude Code, nanobot, local agents)
ave mcp serve

# Streamable HTTP transport (for remote/hosted agents)
ave mcp serve --transport http --port 8420
```

**Claude Code configuration** (`~/.claude/mcp_servers.json`):
```json
{
  "ave": {
    "command": "ave",
    "args": ["mcp", "serve"],
    "env": {"AVE_PROJECT": "/path/to/project.xges"}
  }
}
```

### Concurrency

MCP tool calls route through the existing `EditingSession` which has `threading.Lock` for concurrent access safety. XGES snapshots provide rollback on failure. The MCP server and local agent can operate on the same project simultaneously — access is serialized.

---

## Pillar 3: Web Search + Research Agent

### SearchBackend Protocol

```python
class SearchBackend(Protocol):
    async def search(
        self, query: str, max_results: int = 10
    ) -> list[SearchResult]:
        """Execute a web search."""
        ...

    async def fetch_page(self, url: str) -> PageContent:
        """Fetch and extract readable content from a URL."""
        ...

@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # domain name

@dataclass(frozen=True)
class PageContent:
    url: str
    text: str
    headings: list[str]
```

### Default Backend

Ships with a backend wrapping a search API (Brave Search or Tavily — both have free tiers, lightweight, no browser required). User provides their API key via environment variable (`AVE_SEARCH_API_KEY`).

Replaceable via the plugin system: a Python plugin can register an alternative `SearchBackend` (e.g., one backed by a search MCP server the user has configured).

### Domain-Aware Search

The search tool is video-editing-aware with a default bias toward high-quality sources:

```python
VIDEO_EDITING_SOURCES = [
    "forum.blackmagicdesign.com",
    "community.frame.io",
    "liftgammagain.com",
    "reddit.com/r/colorgrading",
    "reddit.com/r/VideoEditing",
    "cinematography.com",
    "docs.arri.com",
    "sony.com/en/articles/technical",
]
```

This is a bias (boosted ranking), not a filter. Results from any source come through. Users can customize the source list via config or skill.

### Research/Execute Subagent Separation

When the agent needs to research a technique, the workflow splits into two isolated subagents:

**Researcher subagent** (read-only — no timeline access, has web access):
1. Receives research question
2. Searches web, fetches and reads relevant pages
3. Synthesizes into a structured `ResearchBrief`

```python
@dataclass(frozen=True)
class Approach:
    name: str           # "ARRI Official LUT + Manual Offset"
    description: str    # What to do
    tool_mapping: str   # How this maps to AVE tools (natural language)
    source: str         # URL where technique was found
    trade_offs: str     # Pros/cons noted in forums

@dataclass(frozen=True)
class ResearchBrief:
    question: str
    approaches: list[Approach]  # 1-3 techniques found
    sources: list[str]          # All URLs consulted
    confidence: float           # How well-supported the findings are
```

**Editor subagent** (timeline access — no web access):
1. Receives the `ResearchBrief`
2. Translates approaches into AVE tool calls
3. Executes chosen approach (or multiple if user's workflow skill requests comparison)
4. Runs verification loop

**Why this separation:**
- **Security**: Researcher can access web but not modify project. Editor can modify project but not access web. No confused deputy.
- **Context efficiency**: Researcher's context is forum text. Editor's context is timeline state. Neither pollutes the other.
- **Auditability**: Research findings are a structured artifact. The user can see exactly what sources informed the edit.

### Tool Registration

Two tools in the `research` domain:

| Tool | What |
|------|------|
| `web_search(query, sources?)` | Raw web search. Returns `list[SearchResult]`. |
| `research_technique(question)` | Full research workflow — triggers researcher subagent, returns `ResearchBrief`. |

`research_technique` is the high-level tool that triggers the subagent pipeline. `web_search` is the raw primitive for skills/plugins that want custom research workflows.

---

## Pillar 4: AI Rotoscoping / Keying

### RotoscopeBackend Protocol

```python
class RotoscopeBackend(Protocol):
    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask:
        """Segment a single frame given point/box/text prompts."""
        ...

    def segment_video(
        self, frames: Iterator[np.ndarray],
        prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
    ) -> Iterator[SegmentationMask]:
        """Segment across frames with temporal consistency."""
        ...

    def refine_mask(
        self, frame: np.ndarray, mask: SegmentationMask,
        corrections: list[MaskCorrection],
    ) -> SegmentationMask:
        """Refine a mask with point additions/removals."""
        ...
```

### Data Models

```python
@dataclass(frozen=True)
class SegmentPrompt:
    kind: Literal["point", "box", "text"]
    value: Any  # (x,y), (x1,y1,x2,y2), or "the person in the red shirt"

@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray        # H×W binary or soft alpha mask
    confidence: float       # 0-1 per-frame quality estimate
    frame_index: int
    metadata: dict          # Backend-specific info

@dataclass(frozen=True)
class MaskCorrection:
    kind: Literal["include_point", "exclude_point", "include_region", "exclude_region"]
    value: Any
```

### Two Backends

**Sam2Backend** — Meta's Segment Anything Model 2:
- State-of-the-art video segmentation with temporal memory
- Supports point, box, and text prompts
- Handles complex scenes, multiple objects, occlusion
- Requires CUDA GPU, ~4-8GB VRAM
- Best for: complex scenes, multiple subjects, production quality

**RvmBackend** — Robust Video Matting:
- Purpose-built for human foreground/background separation
- Produces soft alpha mattes (superior edge quality for compositing)
- Lighter weight, real-time capable on modest GPUs
- No prompts needed — assumes human subject
- Best for: interviews, talking heads, green screen replacement, fast iteration

### ChromaKeyBackend

Traditional chroma keying as a third backend sharing the same `SegmentationMask` output:

```python
class ChromaKeyBackend:
    """Deterministic color-space thresholding. Not ML-based."""

    def key_frame(
        self, frame: np.ndarray, key_color: str = "green",
        tolerance: float = 0.3, spill_suppression: float = 0.5,
    ) -> SegmentationMask:
        ...
```

Same output type means the feedback loop, evaluation, and compositing work identically regardless of whether the mask came from SAM 2, RVM, or chroma key.

### Agent-Driven Keyframe Feedback Loop

The core innovation. The agent doesn't just run segmentation — it evaluates and iterates:

```
1. ANALYZE    → Agent picks keyframes (scene cuts, motion peaks, lighting changes)
2. SEGMENT    → Run backend on keyframes only (fast — 5-10 frames)
3. EVALUATE   → Agent assesses mask quality via MaskEvaluator:
                 - Edge smoothness (gradient analysis at boundary)
                 - Temporal consistency (mask IoU between adjacent keyframes)
                 - Coverage ratio (foreground area vs expected)
                 - Confidence scores from backend
4. DECIDE     → If quality acceptable → propagate to full clip
                 If problems found → adjust parameters, add corrections, go to step 2
5. PROPAGATE  → Run full video segmentation with tuned parameters
6. VERIFY     → Sample frames from full result, re-evaluate
```

### Quality Evaluation

```python
@dataclass(frozen=True)
class MaskQuality:
    edge_smoothness: float     # 0-1, gradient analysis at mask boundary
    temporal_stability: float  # 0-1, mask IoU between adjacent keyframes
    coverage_ratio: float      # foreground area vs expected
    confidence_mean: float     # average backend confidence
    problem_frames: list[int]  # frames below quality threshold

class MaskEvaluator:
    def evaluate(
        self, masks: list[SegmentationMask], frames: list[np.ndarray]
    ) -> MaskQuality:
        """Evaluate segmentation quality across keyframes."""
        ...
```

The agent reads `MaskQuality` and reasons about what to do. For example: "edge_smoothness is 0.6 on frame 142 where the subject's hair is backlit — add an include_point correction at the hair boundary and re-run." This is where the LLM's reasoning ability transforms a tedious manual process.

### Human-in-the-Loop Checkpoints (Plugin Behavior)

The core provides the feedback loop machinery. Whether to auto-converge or checkpoint with the user is a workflow choice implemented as a skill:

- **Auto-converge skill** (default): Agent iterates until MaskQuality meets thresholds, then applies.
- **Checkpoint skill** (user opt-in): Agent does initial pass, presents keyframe results to user for approval. User flags problem areas. Agent focuses refinement on those areas.

### Tool Registration

Four tools in the `vfx` domain:

| Tool | What | `modifies_timeline` |
|------|------|---------------------|
| `segment_video(asset, prompts, backend?)` | Run segmentation, returns mask asset path | No |
| `refine_mask(mask_asset, corrections, frames?)` | Refine specific frames | No |
| `evaluate_mask(mask_asset, asset)` | Assess quality, return MaskQuality | No |
| `apply_mask(clip, mask_asset, operation)` | Apply mask to timeline (remove bg, composite) | Yes |

Only `apply_mask` triggers the verification loop and XGES snapshot. The iterative refine cycle (`segment → evaluate → refine → evaluate`) is read-only until the agent is satisfied.

### Plugin Structure

Ships as a built-in plugin (exemplar for community plugin authors):

```
ave/plugins/vfx-rotoscope/
    plugin.yaml
    __init__.py
    backends/
        sam2.py
        rvm.py
        chroma_key.py
    evaluator.py
    tools.py
    skills/
        auto-rotoscope.md        # Auto-converge workflow
        guided-rotoscope.md      # Human checkpoint workflow
```

---

## Implementation Sequence

All four pillars implemented sequentially with end-to-end TDD:

### P1: Plugin/Skill System
1. Skill discovery and frontmatter indexing
2. Skill loading on-demand and agent integration
3. Plugin manifest discovery and lazy loading
4. Domain namespacing in ToolRegistry
5. Plugin dependency checking via SystemCapabilities
6. Built-in example skill and plugin
7. End-to-end: skill triggers on intent → loads → agent follows → tools execute

### P2: MCP Server Exposure
1. FastMCP server with stdio transport
2. `get_project_state` and `ingest_asset` tools
3. `search_tools` and `call_tool` tools (wrapping existing registry)
4. `edit_video` tool with internal orchestrator routing
5. `render_preview` tool
6. MCP resources (timeline, assets, capabilities)
7. Streamable HTTP transport
8. End-to-end: Claude Code connects → calls edit_video → gets result

### P3: Web Search + Research Agent
1. SearchBackend protocol and data models
2. Default search backend (Brave/Tavily)
3. `web_search` tool registration
4. Researcher subagent with web access isolation
5. ResearchBrief data model and synthesis
6. `research_technique` tool with subagent orchestration
7. Domain-aware source biasing
8. Replaceable backend via plugin system
9. End-to-end: user asks technique question → research → synthesis → execution

### P4: AI Rotoscoping/Keying
1. RotoscopeBackend protocol and data models
2. MaskEvaluator (quality assessment — no ML dependency)
3. ChromaKeyBackend (deterministic, no ML)
4. Sam2Backend
5. RvmBackend
6. Keyframe selection logic (leveraging existing scene detection)
7. Feedback loop orchestration (analyze → segment → evaluate → decide → propagate)
8. VFX tools registration (segment_video, refine_mask, evaluate_mask, apply_mask)
9. Auto-rotoscope and guided-rotoscope skills
10. End-to-end: agent segments subject → evaluates → refines → applies to timeline

### Test Strategy

- **TDD throughout**: Failing test first, then implementation.
- **Unit tests**: Each backend, evaluator, tool function tested in isolation.
- **Integration tests**: Plugin loading → tool registration → tool execution chains.
- **E2E tests with LLM** (`@pytest.mark.llm`): Real Anthropic API calls testing:
  - Agent discovers and uses a plugin tool via natural language
  - MCP server receives instruction → orchestrator routes → result returned
  - Researcher subagent searches → synthesizes → editor executes
  - Rotoscope agent iterates through feedback loop to convergence
- **Mock backends for CI**: Lightweight mock implementations of SearchBackend, RotoscopeBackend for tests that don't need real ML inference.

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|-----------|
| MCP or direct registry? | Direct registry (Phase 4 decision). MCP only for external exposure. |
| How many MCP tools? | 6, following Anthropic's 5-8 guideline and Cloudflare pattern. |
| Plugin loading strategy? | Lazy — manifest at startup, code on first invocation. |
| Research agent isolation? | Researcher: web access, no timeline. Editor: timeline access, no web. |
| Rotoscope feedback: auto or human? | Both — auto-converge as default, human checkpoints as plugin skill. |
| External MCP consumption? | User-controlled. Users explicitly opt-in to external MCPs. Not auto-discovery. |

---

## Sources

### MCP Server Design
- [Cloudflare Code Mode: give agents an entire API in 1,000 tokens](https://blog.cloudflare.com/code-mode-mcp/)
- [MCP is Not the Problem, It's your Server - Phil Schmid](https://www.philschmid.de/mcp-best-practices)
- [Anthropic: Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [Docker: Top 5 MCP Server Best Practices](https://www.docker.com/blog/mcp-server-best-practices/)
- [Stainless: API MCP Server Architecture Guide](https://www.stainless.com/mcp/api-mcp-server-architecture-guide)
- [Notion MCP Server](https://github.com/makenotion/notion-mcp-server)

### Context Window & Tool Discovery
- [MCP and Context Overload - EclipseSource](https://eclipsesource.com/blogs/2026/01/22/mcp-context-overload/)
- [MCP vs AI Agent Skills - MarkTechPost](https://www.marktechpost.com/2026/03/13/model-context-protocol-mcp-vs-ai-agent-skills-a-deep-dive-into-structured-tools-and-behavioral-guidance-for-llms/)
- [Progressive Disclosure Might Replace MCP - MCPJam](https://www.mcpjam.com/blog/claude-agent-skills)
- [How to Prevent MCP Tool Overload - Lunar](https://www.lunar.dev/post/why-is-there-mcp-tool-overload-and-how-to-solve-it-for-your-ai-agents)
- [LLM Agent Tools Best Practices - LlamaIndex](https://www.llamaindex.ai/blog/building-better-tools-for-llm-agents-f8c5a6714f11)

### Agent Architecture
- [Claude Code Skills Architecture](https://code.claude.com/docs/en/skills)
- [Anthropic: Skill Authoring Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [Claude Code as MCP Server](https://github.com/steipete/claude-code-mcp)
- [Nanobot - Ultra-Lightweight Agent](https://github.com/HKUDS/nanobot)
- [Agent Wars 2026: OpenClaw vs Memu vs Nanobot](https://evoailabs.medium.com/agent-wars-2026-openclaw-vs-memu-vs-nanobot-which-local-ai-should-you-run-8ef0869b2e0c)

### AI Segmentation
- [SAM 2 - Segment Anything Model 2](https://github.com/facebookresearch/sam2)
- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)
