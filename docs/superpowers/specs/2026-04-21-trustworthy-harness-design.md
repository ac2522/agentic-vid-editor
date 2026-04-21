# Trustworthy Harness & Safety Foundation — Design Spec

**Date:** 2026-04-21
**Status:** Draft — pending user review

## Problem Statement

AVE has substantial infrastructure (1100+ passing tests, 9 development phases, Web UI, MCP server, multi-agent orchestrator, context optimizer). But three gaps prevent it from being trustworthy in the face of imperfect LLM agents:

1. **No end-to-end validation that the system actually produces usable videos.** Existing tests validate parts in isolation. Nothing asserts that "user types 'clean up the ums in this clip'" results in a watchable MP4 file.
2. **Tests that an LLM-assisted development loop can silently fake.** Unit/integration tests check "tool was called with X args" or "function returned Y." A hallucinating agent or bad tool can pass these without ever producing correct output. We need assertions on *artifacts* (rendered video, timeline structural state) that cannot be faked.
3. **No safety substrate for "agent made a mistake."** LLM agents make wrong tool calls; this is endemic, not a bug. Today the system has per-tool-call snapshot rollback internally, but no user-facing undo/redo, no state-sync protocol after rollback, no cross-agent activity log, and no structural guarantee that agents can't touch domains they shouldn't.

As the feature catalog (`docs/feature-catalog/`) grows and tool count increases, these gaps compound. Every new tool is a new vector for silent failure and a new surface the agent can mis-select from.

## Goals

1. Build a **safety foundation** that makes agent mistakes cheap to recover from and visible across agents.
2. Build a **tiered evaluation harness** with three trust rungs (plan → executed → rendered+judged) that assert correctness at levels an agent cannot fake.
3. Reuse maximum existing open-source frameworks to minimize implementation cost.
4. Enable "TDD at the scenario level" — a new tool or behavior starts with a failing scenario and ships when the scenario passes across all applicable rungs.
5. Feed harness failures back into the existing prompt-optimization loop (`src/ave/optimize/`) as training signal.

## Non-Goals (for this spec)

- Replacing the existing `optimize/` module. Prompt optimization and regression-gating evaluation are different concerns and remain separate modules.
- Full git-style branch/merge semantics for XGES timelines. User-initiated experimentation via named snapshots is deferred to a future spec.
- Covering every section of the feature catalog. The harness seeds with three flagship scenarios and grows incrementally.
- Fully parallel multi-agent execution with true concurrency. Interleaved sessions with domain scoping is sufficient.

## Design Decisions (resolved during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Evaluation layers | All three rungs, layered (plan → execute → render) | Cost-appropriate testing; most scenarios stay cheap, golden path gets deep |
| Undo granularity | Hybrid: per-tool-call snapshots (internal auto-revert) + per-turn checkpoints (user-facing) | Matches user mental model; preserves existing auto-revert |
| State-sync protocol | Mix of always-on summary + on-demand query | Cheap default safety, deep detail when agent needs it |
| Concurrency model | Interleaved serialization + domain-partitioned scopes | Uses existing lock; structural conflict prevention |
| Cross-agent visibility | Append-only activity log, not git branches | Solves the stated need without merge complexity |
| Harness framework | Built on Inspect AI | MCP-native, agent-first, Apache 2.0, actively maintained (UK AISI) |
| Judge strategy | Multi-model ensemble, routed by rubric dimension (static/still/temporal) | Correctly handles temporal artifacts (motion blur, grain, animation) via video-native models; ensemble reduces model-specific bias |
| CI render hardware | Self-hosted runner on user's 4090; enables local video-native VLM judges at zero marginal cost | Avoids GitHub hosted GPU runner cost; unlocks local Qwen3-VL / Molmo 2 / VideoScore |
| Rung C gating | VLM failures authoritative; golden-scenario passes flagged "pending human review"; 10% random sample of other passes flagged too | Humans focus review effort on creative-intent questions; VLM catches temporal artifacts |
| Scenario governance | Light at creation (PR template) + metric-driven pruning of non-discriminating scenarios after 90 days | Prevents bit-rot without heavy process early |
| Scenario authoring format | YAML (our schema) → Inspect Dataset adapter | Human-authorable, diffable, tool-agnostic |

## Shared Foundation

Five pieces of infrastructure ship before any harness code, because everything else depends on them. These are additive to existing code.

### 1. Turn Checkpoint Layer

Extends existing `src/ave/project/snapshots.py` `SnapshotManager`:

- Per-tool-call snapshots remain (auto-revert on failure).
- New concept: **turn checkpoint** — a named snapshot captured when a user message enters the agent session, tagged with a prompt ID and timestamp.
- Public API additions: `SnapshotManager.create_turn_checkpoint(prompt_id)`, `SnapshotManager.rollback_to_turn(prompt_id)`, `SnapshotManager.redo_turn(prompt_id)`.
- Undo from the user's perspective operates on turn checkpoints. Redo re-applies the turn's recorded tool chain (replay) or restores the post-turn snapshot (preferred — faster).

### 2. User-Facing Undo/Redo in Web UI

- Undo/redo buttons in the chat header; ⌘Z / ⌘⇧Z keyboard shortcuts.
- WebSocket event `timeline_rollback` emitted on undo/redo so frontend timeline/preview refresh.
- Chat pane shows a rollback marker ("↶ undone: 'apply a warm color grade' — 14:22") so history is clear.

### 3. State-Sync Protocol

Every agent turn starts with a synthetic system-slot message:

```
STATE SUMMARY (as of 14:23):
  Timeline: 3 clips, duration 42.8s, 1 audio track, 0 video effects
  Last action: user pressed undo (rolled back 'apply color grade')
  Activity since your last turn:
    - 14:22 (music-agent): boosted bass +2dB on audio track 2 @ 00:14-00:28
    - 14:23 (user): undo

For full state, call get_project_state.
```

- Summary is compact (target ≤300 tokens), always injected.
- `get_project_state` MCP tool remains for deep queries.
- Protocol is defined in `src/ave/agent/state_sync.py` (new) — consumed by `ChatSession` and `MultiAgentOrchestrator`.

### 4. Domain-Scoped Agents

- New field on `AgentRole`: `owned_domains: list[Domain]` where `Domain` enum includes `AUDIO`, `VIDEO`, `SUBTITLE`, `VFX_MASK`, `COLOR`, `TIMELINE_STRUCTURE`, `METADATA`.
- `EditingSession.dispatch(tool_name, args, agent_role)` checks tool's domain tag against `agent_role.owned_domains`; raises `ScopeViolationError` if disallowed.
- Each built-in tool declares `domains_touched: list[Domain]` via registry metadata.
- Single-agent sessions bypass scope checks (backward compat — not a regression for simple users).

### 5. Activity Log

- Append-only per-session log: `src/ave/agent/activity.py` — `ActivityLog` class.
- Each entry: `(timestamp, agent_id, tool_name, terse_summary, snapshot_id)`.
- Terse summary is generated by the tool itself via a new `summarize_for_log(args, result) -> str` method on `Tool`. Falls back to `"{tool_name}({key_args})"` if tool doesn't implement.
- Log drives both state-sync (per-agent diffs) and harness assertions ("scenario must emit ≥N log entries").
- Persisted alongside XGES in project directory: `.ave/activity-log.jsonl`.

### Source-Asset Immutability

Not a module, a discipline enforced at dispatch:

- `EditingSession.dispatch` refuses tool calls whose resolved arg paths fall under `assets/media/source/`.
- Raises `SourceAssetWriteError`. No agent can produce an irreversible edit to original footage.
- Proxies, intermediates, and render outputs remain writable.

## Harness Architecture

All new code lives under `src/ave/harness/`:

```
src/ave/harness/
├── __init__.py
├── scenarios/                   # YAML scenario library (grows over time)
│   ├── reel.filler-word-trim.yaml
│   ├── short.highlight-reel-from-long.yaml
│   └── talking-head.subtitled-vertical.yaml
├── schema.py                    # YAML parsing + validation + Inspect Dataset adapter
├── fixtures/
│   ├── lavfi/                   # FFmpeg lavfi-generated synthetic clips (deterministic)
│   ├── corpus/                  # small CC-licensed real-video fixtures (Rung C only)
│   └── builder.py               # on-demand lavfi generation
├── solvers/
│   ├── plan.py                  # Rung A solver
│   ├── execute.py               # Rung B solver
│   └── render.py                # Rung C solver
├── scorers/
│   ├── tool_selection.py        # AST-based tool-call + arg comparison (BFCL pattern)
│   ├── state_diff.py            # τ-bench-style XGES state comparison
│   ├── safety.py                # reversibility, scope, state-sync, asset-immutability, log-completeness
│   └── vlm_judge.py             # Claude VLM with VBench-2.0-style rubric
├── judges/
│   ├── _protocol.py             # JudgeBackend Protocol (swappable)
│   ├── ensemble.py              # Routes dimensions to tier (static/still/temporal); aggregates via majority/minority-veto
│   ├── router.py                # Per-dimension capability table (which judges handle what)
│   ├── deterministic.py         # FFmpeg/ffprobe-based checks for static dimensions (duration, resolution, RMS)
│   ├── claude_vlm.py            # Claude Haiku/Sonnet/Opus via anthropic SDK — still-composition tier
│   ├── gemini_vlm.py            # Gemini 3.1 Pro/Flash via google-genai — video-native tier
│   ├── kimi_vlm.py              # Kimi K2.6 via moonshot/openai-compat API — video-native tier
│   ├── glm_vlm.py               # GLM-4.6/GLM-5 via zhipu API — still-composition tier (video TBD)
│   ├── qwen_vl_local.py         # Qwen3-VL local via transformers — video-native tier (runs on 4090)
│   ├── molmo2_local.py          # Molmo 2 local via transformers — video-native tier (runs on 4090)
│   └── videoscore_local.py      # VideoScore (Mantis-8B) local — video-quality scoring tier
├── artifacts/                   # rendered outputs + judge traces (Rung C)
│   └── store.py
├── cli.py                       # `ave-harness run <scenario_id> [--tier plan|execute|render]`
└── pytest_plugin.py             # pytest fixtures for harness-based tests
```

### Framework Reuse

| Component | Provided by | Our addition |
|---|---|---|
| Orchestration, caching, parallel exec, result logging | Inspect AI (`inspect_ai.task`, `Dataset`, `Solver`, `Scorer`) | Adapters only |
| Tool-call correctness patterns | BFCL-style AST comparison (we implement), DeepEval `ToolCorrectnessMetric` pattern (inspiration) | Own implementation in `scorers/tool_selection.py` |
| State comparison | τ-bench *pattern* (not a library) — compare end-state against expected goal-state | Implementation in `scorers/state_diff.py` |
| VLM evaluation dimensions | VBench-2.0 rubric taxonomy (adapted) | Scenario-specific rubrics in YAML, judged by Claude |
| Synthetic test fixtures | FFmpeg `lavfi` (`testsrc`, `sine`, `color`) | Builder helpers |
| LLM-as-judge plumbing | Inspect AI's `model_graded_qa` scorer (extended) | VBench-derived rubric |

### Optional Dependencies

Added to `pyproject.toml`:

```
harness = [
    "inspect-ai>=0.3",
    "pyyaml>=6.0",
]
judge-cloud = [
    "anthropic>=0.43",          # Claude Haiku/Sonnet/Opus
    "google-genai>=0.8",        # Gemini 3.1 Pro/Flash
    # Kimi K2.6 via openai-compat (moonshot endpoint) — no extra dep
    # GLM via zhipu-compat (openai SDK) — no extra dep
]
judge-local = [
    "torch>=2.1",               # local VLMs on user's 4090
    "transformers>=4.49",       # Qwen3-VL, Molmo 2, VideoScore
]
```

Existing `[web]` extra already covers `anthropic`; `judge-cloud` duplicates the pin to make the harness stand-alone.

### Judge Routing Policy

The ensemble router classifies each rubric dimension into one of three categories based on what is being measured:

| Dimension type | Examples | Judge tier | Default ensemble |
|---|---|---|---|
| **Static / deterministic** | duration, resolution, aspect ratio, audio RMS/LUFS, caption presence, format | `deterministic` (no VLM) | ffprobe + simple metrics |
| **Still-composition** | framing, caption legibility, text readability, color palette, visual balance | `still` (frame-sampling VLMs OK) | 2 of: Claude Haiku 4.5, Claude Sonnet 4.6, GLM-4.6 |
| **Temporal / motion** | pacing, cut rhythm, motion blur quality, film grain evolution, flicker, animation smoothness | `temporal` (video-native only) | 1 local + 1 cloud: (Qwen3-VL or Molmo 2) + (Gemini 3.1 Pro or Kimi K2.6) |

Aggregation rules:
- **Majority vote** (default) — ≥2 of 3 judges must agree for a pass.
- **Minority veto** (safety-critical dimensions, e.g., "content preservation" on filler-word-trim) — any single judge failure fails the dimension.
- **Disagreement logging** — when judges disagree, the full trace is saved to the artifact store and surfaced in the human review queue as high-value calibration data.

Cost policy:
- Local video-native judges (Qwen3-VL, Molmo 2, VideoScore) run on every applicable scenario at zero marginal cost on the self-hosted 4090 runner.
- Cloud judges are called for ensemble redundancy; rate-limited per scenario (max 2 cloud judges per run) to cap budget.
- Rotation across cloud judges over time mitigates any single provider's model drift.

## Scenario Schema (YAML)

Full schema with all fields:

```yaml
# Required
id: string                        # unique, dotted namespace (e.g., "reel.filler-word-trim")
description: string
tiers: [plan, execute, render]    # subset — not every scenario needs all three
prompt: string                    # colloquial user prompt sent to agent

# Scope enforcement
scope:
  allowed_agents: [role_name, ...]        # which roles may handle; others reject
  forbidden_layers: [layer_id, ...]       # scope violation if touched

# Fixtures
inputs:
  assets:
    - id: string
      ref: string   # fixture://name OR lavfi://expression OR corpus://name

# Rung-specific expectations (all optional — scenario runs only tiers it declares)
expected:
  plan:
    tools_required:
      all_of: [tool_name, ...]            # every listed tool must appear
      any_of: [tool_name, ...]            # at least one must appear
    tools_forbidden: [tool_name, ...]
    irrelevance_allowed: bool              # if true, agent refusing is a pass
    arg_constraints:                       # optional per-tool arg validation
      tool_name: {field: predicate}
  execute:
    timeline:
      clip_count: {min: int, max: int}
      duration_seconds: {min: float, max: float}
      effects_applied: [effect_name, ...]  # must include these
      effects_forbidden: [effect_name, ...]
    snapshots_created: {min: int}
    activity_log_entries: {min: int}
    custom_assertions: [path.to.callable, ...]   # escape hatch for scenario-specific checks
  render:
    preset: string                         # render preset ID
    rubric:
      - dimension: string                  # one of VBench-2.0 dimensions or custom
        prompt: string                     # natural-language rubric for the VLM
        pass_threshold: float              # 0-1
    artifact_retention: days               # how long to keep the rendered MP4

# Cross-cutting safety invariants (always checked when applicable)
safety:
  must_be_reversible: bool      # every tool call has working inverse via undo
  must_respect_scope: bool      # no out-of-domain tool dispatches
  state_sync_after_undo: bool   # agent reads injected state change correctly
  source_asset_immutable: bool  # source media hash unchanged
```

### Scenario Evaluation Semantics

- A scenario passes a tier iff all declared assertions pass AND all applicable safety invariants hold.
- Tiers are independent — a scenario can pass `plan` and fail `execute`.
- CI gating: PR gating runs `plan` tier; merge-to-master runs `execute` tier; nightly runs `render` tier.
- A scenario without a declared tier is not run at that level (no implicit passes).

## Flagship Golden Scenarios (initial library)

Three scenarios seed the library. Each exercises different agent skills and covers primary YouTube/Reel/TikTok output formats.

### `reel.filler-word-trim`
- **Prompt:** "clean up the ums and ahs in this interview clip"
- **Skills exercised:** transcription, text-driven editing, trim operations, undo granularity
- **Expected tool chain:** transcribe → find_filler_words → compute_text_cut → trim_clip (×N)
- **Render rubric:** audio continuity, prompt alignment, content preservation
- **Agents:** transcriptionist + editor

### `short.highlight-reel-from-long`
- **Prompt:** "make me a 30-second highlight from this 5-minute vlog"
- **Skills exercised:** scene detection, visual analysis, rough cut assembly, duration targeting, preset rendering
- **Expected tool chain:** detect_scenes → analyze_scenes (vision) → compute_rough_cut → render (instagram_reel preset)
- **Render rubric:** pacing, visual coherence, duration (28-32s), format (1080×1920)
- **Agents:** editor (primary)

### `talking-head.subtitled-vertical`
- **Prompt:** "turn this horizontal talking-head into a vertical Reel with burnt-in captions"
- **Skills exercised:** transcription, text overlay/subtitle generation, aspect ratio conversion, multi-agent coordination
- **Expected tool chain:** transcribe → (parallel) format_conversion + apply_text_overlay
- **Render rubric:** caption timing accuracy, readability, speaker centered in frame, aspect ratio
- **Agents:** transcriptionist + editor

## Integration with Existing Modules

| Module | Harness uses it for | Change required |
|---|---|---|
| `EditingSession` | Dispatch at Rung B/C | Add `scope` validation hook; emit `activity_log` events; add source-asset write-guard |
| `SnapshotManager` | Rung B reversibility scorer | Add `turn_checkpoint()`, `rollback_to_turn()`, `redo_turn()` API |
| `MCP server` | Agent tool interface at Rung B/C | No change — use existing 6 outcome-oriented tools |
| `multi_agent.py` + `roles.py` | Scope enforcement | Add `owned_domains: list[Domain]` to `AgentRole`; wire scope check in session dispatch |
| `optimize/` | Receives harness failures as training signal | Add `harness_run → opik_dataset` converter (one-way) |
| `web/chat.py` + `web/client/` | User-facing undo/redo + rollback notifications | Undo/redo button + keyboard shortcuts; WS event; chat rollback markers |
| `agent/verification.py` | Rung B uses its verification receipts | Extend `VerifiedSession` to emit machine-readable receipts |
| `tools/registry.py` | Per-tool `domains_touched` metadata | Add `domains_touched: list[Domain]` + `summarize_for_log()` to tool registration |

No existing module gets more than a handful of lines added. Backward compatibility preserved (scope checks skip when `owned_domains` is not set).

## Safety Invariants (Cross-Cutting)

Property-based tests that run against every Rung B scenario and also as standalone Hypothesis-driven tests.

- **Reversibility:** for every recorded tool call, `undo` restores exact pre-call XGES state. Comparison is structural, not byte-level — normalize to a canonical form before diff: sort clips by `(layer, start_time)`, sort effects on each clip by `type`, strip GES-internal IDs and timestamps that vary per save. Two timelines are equivalent iff their canonical forms match.
- **Scope:** no out-of-domain tool dispatch. Domain tags on tools are authoritative; violations are hard failures.
- **State-sync:** inject an undo between agent turns, assert the agent reads the injected state summary on its next turn and doesn't reference stale state. Implemented as a mock-agent test harness that checks whether the model's next tool call references pre-undo state elements.
- **Source-asset immutability:** SHA-256 hash source asset files before and after scenario run — must be identical.
- **Activity-log completeness:** every tool call that mutated state emitted at least one log entry. No silent state changes.

Standalone Hypothesis tests generate random tool-call sequences and assert all five invariants hold for any sequence.

## Phasing

Each phase ships independently and is independently useful.

### Phase 1 — Safety Foundation
Shared plumbing only. No harness yet.

- Turn checkpoint layer on `SnapshotManager`
- User-facing undo/redo in Web UI (buttons, keyboard, WS event, chat markers)
- State-sync protocol (`state_sync.py` module, `ChatSession` integration)
- Domain-scoped agents (`Domain` enum, `AgentRole.owned_domains`, session dispatch check)
- Activity log (`ActivityLog` class, per-session `.ave/activity-log.jsonl`, tool `summarize_for_log` default impl)
- Source-asset immutability enforcement

### Phase 2 — Harness Rung A
Plan-level tool selection.

- `src/ave/harness/` skeleton
- YAML schema + Pydantic validators
- Inspect AI adapter (`schema.py` → Inspect `Dataset`)
- `solvers/plan.py` — plan-only solver
- `scorers/tool_selection.py` — AST-based tool-call + arg comparison
- `scorers/safety.py` — scope invariant (applicable at plan level)
- First 3 scenarios at plan tier
- `fixtures/lavfi/` builder
- `cli.py` minimum (run one scenario)
- Pytest plugin (runs plan tier by default)

### Phase 3 — Harness Rung B
Executed state assertions.

- `solvers/execute.py` — drives real `EditingSession`
- `scorers/state_diff.py` — XGES structural comparison
- `scorers/safety.py` — full invariant suite (reversibility, scope, state-sync, asset immutability, log completeness)
- Hypothesis-driven standalone safety property tests
- 3 flagship scenarios at execute tier
- CI Docker workflow (GES/GStreamer required)

### Phase 4 — Harness Rung C
Rendered output and VLM judge.

- `solvers/render.py`
- `judges/claude_vlm.py` (default)
- `judges/videoscore.py` (optional local)
- `scorers/vlm_judge.py` — rubric-driven scoring with VBench-2.0 dimension taxonomy
- `artifacts/store.py` — rendered MP4 + judge trace retention
- Nightly GitHub Actions workflow (GPU-equipped runner)
- 3 flagship scenarios at render tier
- `cli.py` complete (all tiers)

### Post-Phase 4 (Ongoing)
- Grow scenario library as tools are added (TDD at scenario level — new tool starts with failing scenario).
- Wire Opik feedback loop (`optimize/` consumes harness failures).
- Expand VBench rubric taxonomy as needed.

## Resolved Decisions (moved from Open Questions)

- **Judge strategy:** Multi-model ensemble routed per rubric dimension. Static checks deterministic (no VLM), still-composition served by frame-sampling VLMs (Claude, GLM), temporal/motion served by video-native models (Gemini 3.1 Pro, Kimi K2.6, Qwen3-VL, Molmo 2, VideoScore). Majority vote default, minority-veto for safety-critical dimensions. Two-judge minimum per dimension.
- **CI render hardware:** Self-hosted GitHub Actions runner on user's RTX 4090 machine. Enables local video-native judges at zero marginal cost; cloud judges called sparingly for ensemble redundancy.
- **Human-in-the-loop gating:** Asymmetric — Rung C failures are authoritative (no override). Golden-scenario passes flagged "pending human review" in artifact store. Additional 10% random sample of non-golden passes flagged for calibration data. Disagreements between judges auto-promoted to review queue.
- **Scenario library governance:** Light at creation (PR template with "what behavior does this scenario lock in?" field) + discrimination-power metric in Phase 4+ (scenarios that never fail anything over 90 days get auto-archived, must be explicitly resurrected).

## Open Questions (remaining)

- **Judge model drift:** cloud VLM providers update models periodically without notice, which can shift scores. Mitigation: pin to specific model versions when APIs allow; log model version in judge traces; re-run a calibration suite monthly.
- **Scenario cross-contamination:** if the same model serves as both the agent and a judge member, there's a risk of correlated errors. Enforce: agent model ≠ any judge in its own scenario's ensemble. Logged as invariant.
- **Self-hosted runner availability:** if the user's 4090 machine is offline, Rung C fails to run entirely. Consider a fallback policy (skip-with-warning vs. fail-suite).
- **VBench-2.0 rubric adaptation:** VBench is designed for generated video (text-to-video), not edited video. Some dimensions transfer directly (temporal stability, subject consistency), others don't (prompt adherence from scratch). Phase 4 includes building an AVE-specific rubric taxonomy influenced by but not copying VBench.

## Success Criteria

The design is successful if:

1. A new contributor can write a YAML scenario and have it running at Rung A within 30 minutes of reading the schema doc.
2. The three flagship scenarios pass all three rungs on the current codebase (post-Phase 4). If any fail, that's actual missing functionality — the harness is working.
3. Every new tool added to the registry requires at least one accompanying scenario that exercises it.
4. Undo/redo works cleanly for end users; a wrong agent tool call is recoverable with a single click.
5. Multi-agent sessions cannot corrupt each other's domains (scope violations are structural impossibilities, not runtime guard-rails).
6. Harness failures produce training signal that demonstrably improves tool-selection accuracy over time through the Opik feedback loop.
