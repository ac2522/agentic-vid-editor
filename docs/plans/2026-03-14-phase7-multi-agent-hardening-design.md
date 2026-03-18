# Phase 7: Multi-Agent + Production Hardening — Design Spec

> Designed 2026-03-14. Builds on Phase 1-6 foundation (155+ tests, 34+ tools, 11 domains).

---

## Overview

Phase 7 transforms AVE from a single-orchestrator tool system into a production-hardened multi-agent editing framework. The key additions:

1. **Role-based agents** — domain-specialized subagents that isolate context and improve accuracy
2. **Enhanced search** — BM25-style scoring + tool transition tracking for growing tool sets
3. **Verification loop** — render-and-probe after edits to catch error amplification
4. **Whisper internal aligner** — evaluate attention-head word alignment to eliminate MFA dependency
5. **Error recovery** — XGES snapshot undo stack with rollback and graceful degradation
6. **Performance** — parallel segment rendering scheduler, proxy-first editing workflow
7. **Compositor strategy** — skiacompositor preference with CPU compositor fallback

### Design Principles

- **Pure logic first**: All new modules have testable pure-logic layers (no GES/GPU required)
- **Protocol-based extensibility**: New backends follow existing Protocol patterns (VisionBackend, SceneBackend)
- **Existing patterns preserved**: Tool registration, domain scoping, session state, nanosecond timestamps
- **Incremental enhancement**: Builds on existing Orchestrator/Session/Registry, doesn't replace them
- **Memory-efficient**: Snapshot management with optional disk persistence; parallel rendering caps concurrency
- **Serial execution**: All tool calls are serialized through the session — no concurrent mutations

### Concurrency Model

**All subagent tool calls are serialized through the single `EditingSession`.** The Claude Agent SDK may invoke subagents concurrently, but each subagent's tool calls go through `Orchestrator.handle_tool_call()` → `EditingSession.call_tool()`, which is protected by a `threading.Lock`. This ensures snapshot-before-execute and provision tracking remain consistent. This is documented and tested explicitly.

### Known Limitation: Clip Identity

The current tool set uses abstract parameters (e.g., `clip_duration_ns`, `in_ns`, `out_ns`) rather than clip identifiers. For the E2E workflow "trim the first 5 seconds and make it warmer," the agent must first query project state to resolve clip references. Phase 7 does not add a full clip identity system — this is deferred to Phase 8 (Web UI) where timeline state is directly queryable. For now, the agent resolves clip context through `probe_media` and timeline metadata tools.

---

## Task 7-5: Error Recovery

**Priority: Implement first** — other tasks depend on reliable rollback.

### 7-5a: XGES Snapshot Manager

**File:** `src/ave/project/snapshots.py`

```python
class Snapshot:
    """Immutable record of timeline state at a point in time."""
    snapshot_id: str          # UUID
    timestamp: float          # time.time() — wall-clock, NOT GES nanoseconds
    label: str                # Human-readable label (e.g., "before trim clip_001")
    xges_content: str         # Full XGES XML content
    provisions: frozenset[str]  # Session state at snapshot time (restored on rollback)
    tool_name: str | None     # Tool that triggered this snapshot

class SnapshotManager:
    """Manages XGES snapshots for undo/rollback."""

    def __init__(self, max_snapshots: int = 50, persist_dir: Path | None = None):
        self._snapshots: list[Snapshot] = []
        self._max_snapshots = max_snapshots
        self._persist_dir = persist_dir  # Optional disk persistence for crash recovery

    def capture(self, xges_path: Path, label: str, provisions: frozenset[str],
                tool_name: str | None = None) -> Snapshot
    def restore(self, snapshot_id: str, xges_path: Path) -> tuple[Snapshot, frozenset[str]]
        """Restore XGES content AND return provisions to be restored to SessionState."""
    def restore_latest(self, xges_path: Path) -> tuple[Snapshot, frozenset[str]] | None
    def list_snapshots(self) -> list[SnapshotSummary]
    def clear(self) -> int  # Returns count cleared
```

**Design decisions:**
- Full XGES content stored in memory (XGES files are small XML, typically <100KB even for complex timelines)
- Max 50 snapshots by default (configurable) — oldest evicted BEFORE adding new (no off-by-one)
- Optional `persist_dir`: if set, each snapshot is also written to disk as `{snapshot_id}.xges` for crash recovery
- `restore()` returns both the snapshot AND the provisions frozenset — the caller (session) MUST restore `SessionState` from this
- `SnapshotSummary` is a compact view (~30 tokens) for LLM consumption

### 7-5b: Enhanced Session with Snapshot Integration

**File:** `src/ave/agent/session.py` (extend existing)

The existing `EditingSession.call_tool()` gains snapshot-before-execute behavior and a threading lock:

```python
import threading

class EditingSession:
    def __init__(self) -> None:
        # ... existing init ...
        self._snapshot_manager: SnapshotManager | None = None
        self._lock = threading.Lock()  # Serialize all tool calls

    def call_tool(self, tool_name: str, params: dict) -> Any:
        with self._lock:
            # 1. Capture snapshot BEFORE execution (if project loaded)
            if self._project_path and self._snapshot_manager:
                self._snapshot_manager.capture(
                    self._project_path,
                    label=f"before {tool_name}",
                    provisions=self._state.provisions,
                    tool_name=tool_name,
                )
            # 2. Execute tool (existing logic)
            try:
                result = self._registry.call_tool(tool_name, params, self._state)
            except Exception:
                # 3. On failure, auto-restore latest snapshot INCLUDING provisions
                if self._snapshot_manager:
                    restored = self._snapshot_manager.restore_latest(self._project_path)
                    if restored:
                        _, provisions = restored
                        self._state.reset()
                        self._state.add(*provisions)
                raise
            # 4. Record in history (existing logic)
```

### 7-5c: Graceful Degradation

**File:** `src/ave/tools/capabilities.py`

```python
class SystemCapabilities:
    """Detect available system capabilities for graceful degradation."""

    gpu_available: bool
    cuda_version: str | None
    ges_available: bool
    ffmpeg_available: bool
    whisper_available: bool

    @classmethod
    def detect(cls) -> SystemCapabilities

    def fallback_for(self, capability: str) -> str | None
        """Return fallback description for missing capability."""
```

This follows the same pattern as `conftest.py` detection functions but packaged for runtime use.

---

## Task 7-1: Role-Based Agents

### 7-1a: Agent Role Definitions

**File:** `src/ave/agent/roles.py`

```python
class AgentRole:
    """Definition of a specialized agent role."""
    name: str                    # e.g., "editor", "colorist"
    description: str             # When to use this agent
    system_prompt: str           # Role-specific expertise
    domains: list[str]           # Tool domains this agent can access
    tool_names: list[str] | None # Specific tool restrictions (None = all in domains)

# Predefined roles
EDITOR_ROLE = AgentRole(
    name="editor",
    description="Expert video editor for timeline operations: trimming, splitting, "
                "arranging clips, speed changes, transitions, and compositing.",
    system_prompt="""You are a professional video editor...""",
    domains=["editing", "compositing", "motion_graphics", "scene"],
)

COLORIST_ROLE = AgentRole(
    name="colorist",
    description="Color grading specialist for color correction, LUT application, "
                "CDL adjustments, and color space management.",
    system_prompt="""You are a professional colorist...""",
    domains=["color"],
)

SOUND_DESIGNER_ROLE = AgentRole(
    name="sound_designer",
    description="Audio specialist for volume adjustment, fades, normalization, "
                "and audio mixing.",
    system_prompt="""You are a professional sound designer...""",
    domains=["audio"],
)

TRANSCRIPTIONIST_ROLE = AgentRole(
    name="transcriptionist",
    description="Transcription specialist for speech-to-text, word alignment, "
                "transcript search, filler word removal, and text-based editing.",
    system_prompt="""You are a transcription specialist...""",
    domains=["transcription"],
)

ALL_ROLES = [EDITOR_ROLE, COLORIST_ROLE, SOUND_DESIGNER_ROLE, TRANSCRIPTIONIST_ROLE]
```

### 7-1b: Multi-Agent Orchestrator

**File:** `src/ave/agent/multi_agent.py`

```python
class MultiAgentOrchestrator:
    """Orchestrator that delegates to role-based subagents.

    Extends the single-orchestrator pattern with domain-aware routing.
    For use with Claude Agent SDK's AgentDefinition.
    """

    def __init__(self, session: EditingSession, roles: list[AgentRole] | None = None):
        self._session = session
        self._roles = roles or ALL_ROLES
        self._base_orchestrator = Orchestrator(session)

    def get_agent_definitions(self) -> dict[str, AgentDefinition]:
        """Generate Claude Agent SDK AgentDefinition for each role."""
        # Each role becomes a subagent with:
        # - domain-scoped system prompt
        # - restricted tool access (only its domains)
        # - the 3 meta-tools (search, schema, call) as MCP tools

    def get_system_prompt(self) -> str:
        """System prompt that describes available specialist agents."""

    def get_role_tools(self, role: AgentRole) -> list[str]:
        """Get tool names accessible to a specific role."""
```

**Key design:**
- `MultiAgentOrchestrator` wraps the existing `Orchestrator`, not replaces it
- **No explicit `route_request()` method** — routing is handled by Claude itself via the `description` field in each `AgentDefinition`. The LLM reads agent descriptions and decides which subagent to invoke. This is more robust than keyword matching and handles cross-domain requests naturally (the LLM can invoke multiple subagents sequentially or handle the request directly).
- Each role's subagent gets a scoped view of the tool registry (only its domains)
- Cross-domain requests: the LLM orchestrator handles these directly using the full tool registry, or invokes multiple specialist subagents in sequence
- If a subagent is invoked for an inappropriate task, it degrades gracefully — its scoped tool search returns no results, and it reports back to the orchestrator
- Claude Agent SDK's `AgentDefinition` is generated dynamically from `AgentRole` definitions
- **Subagents use progressive disclosure**: each subagent gets the same 3 meta-tools (search, schema, call) but scoped to its domains only

### 7-1c: SDK Integration Layer

**File:** `src/ave/agent/sdk_bridge.py`

```python
def create_ave_agent_options(
    session: EditingSession,
    roles: list[AgentRole] | None = None,
    model: str = "opus",
) -> dict:
    """Create Claude Agent SDK ClaudeAgentOptions-compatible configuration.

    Returns a dict that can be unpacked into ClaudeAgentOptions(**result).
    This avoids a hard dependency on claude-agent-sdk at import time.
    """

def create_meta_tool_server(session: EditingSession, role: AgentRole | None = None):
    """Create an MCP server config for the 3 meta-tools, optionally scoped to a role."""
```

**Why a bridge module:** The `claude-agent-sdk` package is an optional dependency (requires Claude Code CLI). The bridge produces plain dicts/configs that work with or without the SDK installed. Tests can validate the configuration structure without the SDK.

---

## Task 7-3: Verification Loop

### 7-3a: Verification Models

**File:** `src/ave/tools/verify.py`

```python
class EditIntent:
    """Captures what an edit was supposed to accomplish."""
    tool_name: str
    description: str            # Natural language intent
    expected_changes: dict      # Key metrics that should change
    # e.g., {"duration_ns": 2_000_000_000, "has_audio": True}

class VerificationResult:
    """Result of verifying an edit against intent."""
    passed: bool
    intent: EditIntent
    actual_metrics: dict        # Probed metrics from rendered output
    discrepancies: list[str]    # Human-readable list of mismatches
    confidence: float           # 0.0-1.0

class VerificationBackend(Protocol):
    """Protocol for edit verification strategies."""
    def verify(self, intent: EditIntent, segment_path: Path) -> VerificationResult: ...
```

### 7-3b: Probe-Based Verifier

**File:** `src/ave/tools/verify_probe.py`

```python
class ProbeVerifier:
    """Verifies edits by probing rendered segments.

    Implements VerificationBackend.
    Uses ffprobe to check duration, codecs, resolution, audio presence.
    """

    def verify(self, intent: EditIntent, segment_path: Path) -> VerificationResult:
        """Probe the segment and compare against intent."""

    def _probe_segment(self, path: Path) -> dict:
        """Extract metrics from a rendered segment via ffprobe."""

    def _compare_metrics(self, expected: dict, actual: dict) -> tuple[bool, list[str]]:
        """Compare expected vs actual metrics. Returns (passed, discrepancies)."""
```

### 7-3c: Verification-Aware Session

**File:** `src/ave/agent/verification.py`

```python
class VerifiedSession:
    """Wraps EditingSession with post-edit verification.

    After each tool call that modifies the timeline:
    1. Capture snapshot (via SnapshotManager)
    2. Execute tool
    3. Render affected segment
    4. Probe and verify
    5. If verification fails: rollback to snapshot, report error
    """

    def __init__(self, session: EditingSession, verifier: VerificationBackend | None = None):
        self._session = session
        self._verifier = verifier

    def call_tool_verified(self, tool_name: str, params: dict,
                           intent: EditIntent | None = None) -> VerificationResult | Any:
        """Execute tool with optional verification."""
```

**Design decisions:**
- **Verification is deferred to end-of-turn**, not after each individual tool call. The orchestrator runs the full agent turn (which may involve multiple tool calls), then verifies the final state against the original intent. This avoids rendering 6-8 segments during a single multi-step edit.
- A `modifies_timeline: bool` flag is added to `@registry.tool()` decorator (default `False`). Only tools with `modifies_timeline=True` (editing, color, audio domain tools) trigger verification at turn end.
- `EditIntent` is constructed by the orchestrator from the user's original prompt before the agent turn begins. The orchestrator extracts expected outcomes (duration change, color shift, etc.) as part of the system prompt.
- Fallback: if no verifier configured, behaves like normal `call_tool()`
- Verification failure does NOT auto-rollback — it reports the discrepancy to the LLM, which can decide to retry or accept

---

## Task 7-2: Intra-Domain Search Enhancement

### 7-2a: BM25-Style Scoring

**File:** `src/ave/agent/search.py`

```python
class ToolSearchEngine:
    """Enhanced tool search with BM25-style scoring.

    Improves on the current word-count scoring in ToolRegistry.search_tools()
    with term frequency, inverse document frequency, and field weighting.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self._documents: dict[str, SearchDocument] = {}
        self._k1 = k1
        self._b = b

    def reindex_all(self, registry: ToolRegistry)
        """Primary entry point: index all tools and compute IDF. Must be called
        after all tools are registered. IDF is recomputed on every call."""
    def search(self, query: str, domain: str | None = None, limit: int = 10) -> list[SearchHit]

class SearchDocument:
    """Indexed tool document with pre-computed term frequencies."""
    name: str
    domain: str
    terms: dict[str, float]  # term -> TF score

class SearchHit:
    """Search result with relevance score."""
    tool_name: str
    domain: str
    score: float
    description: str
```

### 7-2b: Tool Transition Tracking

**File:** `src/ave/agent/transitions.py`

```python
class ToolTransitionGraph:
    """Tracks tool usage patterns for AutoTool-style transition probabilities.

    Records which tools are called after which, building a directed graph
    of transition probabilities. Used to suggest likely next tools.
    """

    def __init__(self):
        self._transitions: dict[str, dict[str, int]] = {}  # from -> {to -> count}
        self._total_from: dict[str, int] = {}

    def record(self, from_tool: str, to_tool: str)
    def suggest_next(self, current_tool: str, limit: int = 5) -> list[tuple[str, float]]
    def to_json(self) -> str
    @classmethod
    def from_json(cls, data: str) -> ToolTransitionGraph
```

**Integration:** The `EditingSession.call_tool()` records transitions automatically. The transition graph is serialized to `{project_dir}/.ave/transitions.json` on session close and loaded on session start, accumulating usage patterns across sessions.

---

## Task 7-4: Whisper Internal Aligner

### Design

**File:** `src/ave/tools/whisper_aligner.py`

```python
class WhisperAlignerBackend(Protocol):
    """Protocol for Whisper attention-head word alignment.

    Based on "Whisper Has an Internal Word Aligner" (arXiv 2509.09987):
    - Teacher-forces Whisper with character-level tokens
    - Selects attention heads that capture word boundaries
    - Uses Dynamic Time Warping (DTW) to extract timestamps
    - Achieves 39.5% improvement on TIMIT over baseline
    """

    def align_words(
        self,
        audio_path: Path,
        transcript: str,
        language: str = "en",
    ) -> list[WordAlignment]: ...

class WordAlignment:
    """Word-level timestamp from Whisper attention alignment."""
    word: str
    start_seconds: float
    end_seconds: float
    confidence: float

class WhisperAlignerConfig:
    """Configuration for Whisper internal aligner."""
    model_size: str = "large-v3"  # Whisper model variant
    head_selection: str = "auto"  # "auto" or specific head indices
    dtw_backend: str = "numpy"   # "numpy" or "scipy"
```

**Status:** This is an evaluation task. The protocol and models are defined here so they integrate with the existing transcription pipeline. Actual implementation requires:
1. Access to Whisper model weights with attention output — **requires PyTorch + transformers** (whisper.cpp does NOT expose attention heads)
2. Attention head selection heuristics from the paper
3. DTW implementation (numpy-based, ~50 lines)

**Dependency note:** Unlike the rest of Phase 7, this task requires `torch` (~2GB) and `transformers` as heavy optional dependencies. These must be lazy-imported and gated behind a `@requires_torch` test marker. The Docker image would need a separate layer for PyTorch. This is why 7-4 is labeled an evaluation task — the protocol and data models are implemented now, but the actual PyTorch-based backend is deferred until the dependency cost is justified.

The existing `TranscriptionBackend` protocol in `tools/transcribe.py` handles transcription; this `WhisperAlignerBackend` is a separate protocol specifically for word alignment, composable with any transcription backend.

---

## Task 7-6: Performance

### 7-6a: Parallel Segment Rendering Scheduler

**File:** `src/ave/render/parallel.py`

```python
class RenderJob:
    """A single segment render job."""
    segment_id: str
    start_ns: int
    stop_ns: int
    priority: int          # Lower = higher priority
    status: Literal["pending", "rendering", "complete", "failed"]
    output_path: Path | None

class RenderScheduler:
    """Schedules parallel segment rendering with priority ordering.

    Priority order:
    1. Segments near playhead (lowest distance = highest priority)
    2. Dirty segments in viewport
    3. Remaining dirty segments outward from playhead

    Concurrency capped at max_workers to avoid VRAM exhaustion.
    """

    def __init__(self, max_workers: int = 2):
        self._queue: list[RenderJob] = []
        self._active: dict[str, RenderJob] = {}
        self._max_workers = max_workers

    def enqueue(self, jobs: list[RenderJob])
    def next_batch(self) -> list[RenderJob]
        """Return up to max_workers jobs from the queue, moving them to active state."""
    def mark_complete(self, segment_id: str, output_path: Path)
    def mark_failed(self, segment_id: str, error: str)
    def pending_count(self) -> int
    def active_count(self) -> int
```

**Note:** `RenderScheduler` is a **pure queue/priority data structure**. It does not execute renders. The executor is the existing `render_segment()` function from the render module. A higher-level `ParallelRenderExecutor` (Phase 8, when the preview server drives rendering) will call `next_batch()` → spawn workers → call `mark_complete()`/`mark_failed()`. Phase 7 delivers the scheduler logic; Phase 8 wires it to actual GES pipelines.

### 7-6b: Proxy-First Editing Workflow

**File:** `src/ave/tools/proxy.py`

```python
class ProxyConfig:
    """Configuration for proxy-first editing workflow."""
    proxy_width: int = 854        # 480p width
    proxy_height: int = 480
    proxy_codec: str = "libx264"
    proxy_preset: str = "ultrafast"
    online_conform: bool = True   # Swap proxies for full-res at export time

class ProxyWorkflow:
    """Manages proxy-first editing: edit on proxies, conform to full-res for export.

    Workflow:
    1. Ingest creates both full-res intermediate and proxy
    2. Editing operates on proxy paths (fast)
    3. At export time, conform swaps proxy paths for full-res paths
    4. Final render uses full-res intermediates
    """

    def get_editing_path(self, asset_entry: dict) -> Path:
        """Return proxy path for editing, full-res path if proxy unavailable."""

    def conform_timeline(self, xges_path: Path, registry_path: Path) -> ConformResult:
        """Swap all proxy references to full-res.
        Validates all full-res paths exist BEFORE writing.
        Raises ConformError with list of missing files if any are absent."""

class ConformResult:
    swaps: int                    # Number of paths swapped
    warnings: list[str]          # Non-fatal issues

class ConformError(Exception):
    missing_files: list[Path]    # Full-res files that don't exist
```

---

## Task 7-7: Compositor Strategy

### Design

**File:** `src/ave/render/compositor.py`

```python
class CompositorStrategy:
    """Compositor selection strategy.

    Preference order:
    1. skiacompositor — avoids glvideomixer crash bugs #728, #786
    2. compositor (CPU) — universal fallback, always available in GStreamer
    3. glvideomixer — only if explicitly requested (known crash bugs)

    Detection is lazy: check GStreamer element registry at first use.
    """

    strategy: Literal["skia", "cpu", "gl", "auto"]

    @classmethod
    def detect_available(cls) -> list[str]:
        """Detect which compositors are available in the GStreamer registry.
        Returns ["cpu"] as safe default if GStreamer is not importable."""

    @classmethod
    def select(cls, preference: str = "auto") -> str:
        """Select best available compositor element name.
        Falls back to "compositor" (CPU) if preferred option unavailable."""

    @classmethod
    def get_element_name(cls, strategy: str) -> str:
        """Return GStreamer element name for a strategy."""
```

**Integration with render pipeline:** The `RenderPreset` dataclass in `render/presets.py` gains an optional `compositor` field. When rendering, the pipeline builder uses `CompositorStrategy.select()` to choose the compositor element.

---

## File Map

| New File | Purpose | Lines (est.) |
|----------|---------|-------------|
| `src/ave/project/snapshots.py` | XGES snapshot manager | ~120 |
| `src/ave/tools/capabilities.py` | System capability detection | ~80 |
| `src/ave/agent/roles.py` | Agent role definitions | ~120 |
| `src/ave/agent/multi_agent.py` | Multi-agent orchestrator | ~150 |
| `src/ave/agent/sdk_bridge.py` | Claude Agent SDK bridge | ~80 |
| `src/ave/tools/verify.py` | Verification models + protocol | ~80 |
| `src/ave/tools/verify_probe.py` | Probe-based verifier | ~100 |
| `src/ave/agent/verification.py` | Verification-aware session | ~80 |
| `src/ave/agent/search.py` | BM25-style tool search | ~120 |
| `src/ave/agent/transitions.py` | Tool transition graph | ~80 |
| `src/ave/tools/whisper_aligner.py` | Whisper aligner protocol + models | ~60 |
| `src/ave/render/parallel.py` | Parallel render scheduler | ~100 |
| `src/ave/tools/proxy.py` | Proxy-first workflow | ~80 |
| `src/ave/render/compositor.py` | Compositor strategy | ~70 |
| **Tests** | |
| `tests/test_tools/test_snapshots.py` | Snapshot manager tests | ~150 |
| `tests/test_tools/test_capabilities.py` | Capability detection tests | ~60 |
| `tests/test_agent/test_roles.py` | Role definition tests | ~100 |
| `tests/test_agent/test_multi_agent.py` | Multi-agent orchestrator tests | ~150 |
| `tests/test_tools/test_verify.py` | Verification tests | ~120 |
| `tests/test_agent/test_search.py` | BM25 search tests | ~100 |
| `tests/test_agent/test_transitions.py` | Transition graph tests | ~80 |
| `tests/test_tools/test_whisper_aligner.py` | Whisper aligner model tests | ~60 |
| `tests/test_render/test_parallel.py` | Parallel scheduler tests | ~100 |
| `tests/test_tools/test_proxy.py` | Proxy workflow tests | ~80 |
| `tests/test_render/test_compositor.py` | Compositor strategy tests | ~60 |
| **Total** | | ~2,460 |

---

## Implementation Order

```
7-5 Error Recovery (foundation — other tasks need snapshot/rollback)
  ├── 7-5a: SnapshotManager
  ├── 7-5b: Session integration
  └── 7-5c: Capabilities detection
      ↓
7-1 Role-Based Agents (core multi-agent feature)
  ├── 7-1a: AgentRole definitions
  ├── 7-1b: MultiAgentOrchestrator
  └── 7-1c: SDK bridge
      ↓
7-3 Verification Loop (quality assurance)
  ├── 7-3a: Models + protocol
  ├── 7-3b: ProbeVerifier
  └── 7-3c: VerifiedSession
      ↓
7-2 Intra-Domain Search (incremental enhancement)
  ├── 7-2a: BM25 search engine
  └── 7-2b: Transition graph
      ↓
7-6 Performance (parallel rendering, proxy workflow)
  ├── 7-6a: RenderScheduler
  └── 7-6b: ProxyWorkflow
      ↓
7-7 Compositor Strategy
      ↓
7-4 Whisper Internal Aligner (evaluation + protocol)
      ↓
Wire into registry + full test suite
```

---

## Testing Strategy

All tests follow TDD: write failing test first, then implement.

**Pure logic tests** (run anywhere, no markers):
- Snapshot capture/restore round-trip
- BM25 scoring correctness
- Transition graph recording and suggestion
- RenderScheduler priority ordering
- ProxyWorkflow path selection
- CompositorStrategy selection logic
- AgentRole definition validation
- MultiAgentOrchestrator routing
- Verification model creation and comparison

**Integration tests** (require ffprobe/ffmpeg):
- ProbeVerifier against real rendered segments (`@requires_ffmpeg`)
- ProxyWorkflow conform with actual media files (`@requires_ffmpeg`)

**E2E tests** (require GES):
- Snapshot → edit → rollback → verify state restored (`@requires_ges`)
- Full verification loop: edit → render → probe → compare (`@requires_ges`, `@requires_ffmpeg`)

**LLM tests** (require API key):
- Multi-agent routing with real Claude calls (`@pytest.mark.llm`)
- Role-based tool selection accuracy (`@pytest.mark.llm`)

---

## Dependencies

**Core (no new deps):** All implementations except 7-4 use:
- Python stdlib (`uuid`, `time`, `math`, `json`, `subprocess`, `dataclasses`, `threading`)
- Existing project dependencies (`pydantic`, `pytest`)
- Optional: `claude-agent-sdk` (for SDK bridge, lazy-imported)

**Task 7-4 only (heavy optional):**
- `torch` (~2GB) + `transformers` — required for Whisper attention head access
- Gated behind `@requires_torch` marker, lazy-imported, separate Docker layer
- Not required for any other Phase 7 task

The `claude-agent-sdk` remains optional — the bridge module produces plain dicts that can be validated without installing the SDK.

---

## Review Feedback Applied

This spec incorporates feedback from automated code review (2026-03-14):
1. **Concurrency**: Added `threading.Lock` to `EditingSession` and explicit serialization guarantee
2. **Provisions restoration**: `restore()` returns provisions; session restores `SessionState` on rollback
3. **Verification timing**: Changed from per-tool-call to end-of-turn verification to avoid performance issues
4. **Routing**: Removed explicit `route_request()` — LLM handles routing via `AgentDefinition.description`
5. **BM25 IDF**: `reindex_all()` is the primary entry point; IDF computed on every call
6. **Conform safety**: `conform_timeline()` validates all paths before writing; raises `ConformError` on missing files
7. **Whisper deps**: Corrected dependency section — 7-4 requires PyTorch (heavy optional)
8. **Snapshot persistence**: Added optional `persist_dir` for crash recovery
9. **Transition storage**: Specified `{project_dir}/.ave/transitions.json` as storage path
10. **Compositor fallback**: `detect_available()` returns `["cpu"]` when GStreamer unavailable
11. **RenderScheduler scope**: Documented as pure queue; executor deferred to Phase 8
12. **`modifies_timeline` flag**: Added to `@registry.tool()` for verification gating
13. **Clip identity gap**: Documented as known limitation, deferred to Phase 8
