# Phase 3 & 4: Preview System + Agent Tool Architecture

> Designed 2026-03-10. Builds on full roadmap (2026-03-09).

---

## Phase 3: Preview System

### Goal
Render timeline segments, serve them for browser playback and scrubbing.

### Architecture
Segment cache + aiohttp server (HTTP for segments, WebSocket for frame scrubbing). No WebRTC in v1. No LL-HLS.

### Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `segment.py` | `src/ave/render/` | Render timeline sub-ranges to fMP4 via GES pipeline seek |
| `cache.py` | `src/ave/preview/` | Track segment state (dirty/rendering/clean), invalidate on edits |
| `frame.py` | `src/ave/preview/` | Extract single JPEG frame at timecode via FFmpeg |
| `server.py` | `src/ave/preview/` | aiohttp HTTP+WebSocket server |
| `client/` | `src/ave/preview/client/` | Minimal HTML/JS browser preview (MSE + WS) |

### Data Flow

1. Timeline edit → cache marks affected segments dirty
2. Cache triggers re-render of dirty segments (priority: viewport-visible)
3. HTTP serves clean segments as static fMP4 files
4. WebSocket handles: frame requests (timecode → JPEG), playback state, invalidation notifications
5. Browser uses MSE to play segments, WS for scrubbing frames

### Segment Strategy

- Fixed-duration segments (default 5s)
- Filename: `{timeline_id}_{start_ns}_{end_ns}.mp4`
- Format: fragmented MP4 (fMP4) for MSE compatibility
- Segments align to nearest frame boundary

---

## Phase 4: Agent Tool Architecture (No MCP)

### Goal
Progressive tool discovery with minimal context window cost. Direct Python function registry, not MCP servers.

### Why Not MCP

- Flat MCP listing: 8K-16K tokens for ~40 tools, always loaded
- Progressive MCP: still ~1,250 tokens baseline + protocol overhead
- MCP requires subprocess management, IPC serialization
- Research shows progressive discovery without MCP achieves 98-99% token reduction

### Architecture: Direct Tool Registry + Progressive Discovery

**3 meta-functions in agent context (~150 tokens):**

```python
search_tools(query: str, domain: str | None = None) -> list[ToolSummary]
get_tool_schema(tool_name: str) -> ToolSchema
call_tool(tool_name: str, params: dict) -> ToolResult
```

**Tool registration via decorator:**

```python
@registry.tool(domain="editing", requires=["timeline_loaded"], provides=["clip_trimmed"])
def trim(timeline, clip_id: str, in_ns: int, out_ns: int) -> TrimResult:
    """Trim a clip to new in/out points (nanoseconds)."""
    ...
```

### Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `registry.py` | `src/ave/agent/` | ToolRegistry: register, search, get_schema, call |
| `dependencies.py` | `src/ave/agent/` | Tool dependency graph (prerequisites/provisions) |
| `tools_editing.py` | `src/ave/agent/tools/` | Register: trim, split, concat, speed, transitions |
| `tools_audio.py` | `src/ave/agent/tools/` | Register: volume, fade, normalize |
| `tools_color.py` | `src/ave/agent/tools/` | Register: LUT, grade, CDL, transform |
| `tools_transcription.py` | `src/ave/agent/tools/` | Register: transcribe, search transcript, transcript-edit |
| `tools_render.py` | `src/ave/agent/tools/` | Register: render segment, render export, proxy |
| `tools_project.py` | `src/ave/agent/tools/` | Register: timeline CRUD, clips, assets |
| `orchestrator.py` | `src/ave/agent/` | Claude Agent SDK orchestrator with 3 meta-tools |
| `session.py` | `src/ave/agent/` | Editing session lifecycle manager |
| `transcript_edit.py` | `src/ave/tools/` | Transcript-driven editing (pure logic) |

### Tool Dependency Graph

```json
{
  "trim": {
    "domain": "editing",
    "requires": ["timeline_loaded", "clip_exists"],
    "provides": ["clip_trimmed"]
  },
  "render_segment": {
    "domain": "render",
    "requires": ["timeline_loaded", "timeline_saved"],
    "provides": ["segment_rendered"]
  }
}
```

Meta-tool validates prerequisites before execution.

### Token Cost

| Approach | Tokens at init | Per-tool lookup |
|----------|---------------|-----------------|
| Flat MCP (6 servers) | ~8,000-16,000 | 0 (pre-loaded) |
| **AVE ToolRegistry** | **~150** | **~100-200** |

---

## Implementation Plan

### Batch 1 (Parallel — no dependencies)

| Stream | Module | Test File |
|--------|--------|-----------|
| S1: Segment Renderer | `src/ave/render/segment.py` | `tests/test_render/test_segment.py` |
| S2: Segment Cache | `src/ave/preview/cache.py` | `tests/test_preview/test_cache.py` |
| S3: Frame Extractor | `src/ave/preview/frame.py` | `tests/test_preview/test_frame.py` |
| S4: Tool Registry | `src/ave/agent/registry.py` + `dependencies.py` | `tests/test_agent/test_registry.py` + `test_dependencies.py` |
| S5: Transcript Editing | `src/ave/tools/transcript_edit.py` | `tests/test_tools/test_transcript_edit.py` |

### Batch 2 (Depends on Batch 1)

| Stream | Module | Depends On |
|--------|--------|------------|
| S6: Domain Tool Registration | `src/ave/agent/tools/*.py` | S4 (registry) |
| S7: Preview Server + Client | `src/ave/preview/server.py` + `client/` | S1, S2, S3 |

### Batch 3 (Depends on Batch 2)

| Stream | Module | Depends On |
|--------|--------|------------|
| S8: Orchestrator + Session | `src/ave/agent/orchestrator.py` + `session.py` | S4, S6 |

### Final: Integration Tests

- Preview E2E: edit → cache invalidation → re-render → serve → verify
- Agent E2E: search_tools → get_schema → call_tool → verify timeline
- Full stack: agent command → edit → preview update → verify

---

## New Dependencies

```toml
preview = ["aiohttp>=3.9"]
agent = ["claude-agent-sdk>=0.1"]
```

---

## Sources

- [Progressive Tool Discovery Pattern](https://agentic-patterns.com/patterns/progressive-tool-discovery/)
- [mcp2cli: 99% Token Cost Reduction](https://topaiproduct.com/2026/03/09/mcp2cli-the-tool-that-cuts-mcp-token-costs-by-99-just-hit-hacker-news/)
- [OpenClaw ContextEngine](https://www.epsilla.com/blogs/2026-03-09-openclaw-2026-3-7-contextengine-agentic-architecture)
- [Agent Skills Specification](https://agentskills.io/specification)
- [Cloudflare Code Mode](https://blog.cloudflare.com/code-mode-mcp/)
