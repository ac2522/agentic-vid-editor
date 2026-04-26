# AVE Context Optimizer — Design Spec

## Problem Statement

AVE sends many text artifacts to LLMs: agent role system prompts, tool descriptions, and orchestrator instructions. These artifacts directly impact agent quality — a vague tool description causes wrong tool selection, a missing convention in a system prompt causes incorrect parameter formatting.

Currently these artifacts are hand-written and never systematically evaluated or improved. We need a framework that:
1. Extracts optimizable text artifacts from the AVE codebase
2. Evaluates their quality against video-editing task datasets
3. Optimizes them using automated prompt optimization (Opik Agent Optimizer)
4. Versions improvements with regression guards
5. Provides a standalone eval path for CI (no Opik server required)

## Technology Choice: Opik Agent Optimizer

**Why Opik over alternatives:**
- Only production-grade OSS framework (Apache 2.0, 17.6k stars) with native tool description optimization
- MetaPromptOptimizer + EvolutionaryOptimizer both support combined prompt+tool optimization
- LiteLLM backend supports Anthropic/Claude models natively
- Self-hostable via Docker, evaluation dashboard included
- Python-native SDK (`pip install opik-optimizer`)

**Why not DSPy:** Doesn't optimize tool description text directly — only instructions around tool use.
**Why not Promptfoo:** Eval-only, no automatic optimization. Good for CI regression but not generation.
**Why not TextGrad:** Research-grade, less production tooling. Could be a future backend.

## Architecture

```
src/ave/optimize/
├── __init__.py           # Public API: optimize_artifacts, evaluate_artifacts
├── artifacts.py          # ContextArtifact model + ArtifactExtractor
├── datasets.py           # EvalDataset builder + built-in video editing datasets
├── metrics.py            # AVE-specific evaluation metrics
├── campaign.py           # OptimizationCampaign orchestrator
├── store.py              # Versioned artifact storage + diff tracking
├── evaluate.py           # StandaloneEvaluator (no Opik dependency, CI-friendly)
├── data/                 # Built-in JSONL eval datasets
│   ├── tool_selection.jsonl
│   └── role_routing.jsonl
└── backends/
    ├── __init__.py
    ├── _protocol.py      # OptimizerBackend Protocol
    └── _opik.py          # Opik Agent Optimizer implementation (lazy import)
```

### Dependency Strategy

`opik-optimizer` is an **optional dependency** (like onnxruntime, scenedetect). The framework:
- Imports opik lazily only in `backends/_opik.py` via `_compat.import_optional`
- Core types (ContextArtifact, EvalDataset, metrics) work without opik installed
- `evaluate.py` provides standalone evaluation using `litellm` or `anthropic` SDK directly — no Opik server needed
- `backends/_opik.py` requires both `opik-optimizer` and an Opik server/API key
- `_compat.INSTALL_MAPPING` updated with `"opik_optimizer": "opik-optimizer"` for consistent error messages
- pyproject.toml gets `[optimize]` optional dep group

## Core Types

### ContextArtifact

```python
class ArtifactKind(str, Enum):
    SYSTEM_PROMPT = "system_prompt"         # AgentRole.system_prompt
    ROLE_DESCRIPTION = "role_description"   # AgentRole.description
    TOOL_DESCRIPTION = "tool_description"   # Tool full docstring (used for both search summary and schema)
    ORCHESTRATOR_PROMPT = "orchestrator"    # Multi-agent orchestrator instructions

@dataclass(frozen=True)
class ContextArtifact:
    id: str                    # Unique ID: "role.editor.system_prompt", "tool.trim_clip.description"
    kind: ArtifactKind
    content: str               # Current text content
    source_location: str       # File path + optional line range for apply-back
    metadata: dict[str, Any]   # domain, tags, role_name, etc. (frozen via tuple conversion)
```

**ID format:** Dot-separated (not colons) to be filesystem-safe: `role.editor.system_prompt`, `tool.trim_clip.description`. The `ArtifactStore` uses these directly as directory names.

**MVP scope:** Only `SYSTEM_PROMPT`, `ROLE_DESCRIPTION`, `TOOL_DESCRIPTION`, and `ORCHESTRATOR_PROMPT`. Memory docs and skills are deferred — they are freeform prose that is hard to write metrics for, and the highest-value targets are system prompts and tool descriptions.

### ArtifactExtractor

Pulls artifacts from live AVE code. Does NOT duplicate text — reads from source of truth.

```python
class ArtifactExtractor:
    def extract_from_roles(self, roles: Sequence[AgentRole]) -> list[ContextArtifact]:
        """Extract system_prompt and description from each AgentRole.

        Produces two artifacts per role:
        - role.<name>.system_prompt (kind=SYSTEM_PROMPT)
        - role.<name>.description (kind=ROLE_DESCRIPTION)
        """

    def extract_from_registry(self, registry: ToolRegistry) -> list[ContextArtifact]:
        """Extract full docstrings from registered tools.

        Produces one artifact per tool:
        - tool.<name>.description (kind=TOOL_DESCRIPTION)

        Content is the full docstring (inspect.getdoc), which includes
        parameter documentation if present (Google-style Args: sections).
        """

    def extract_all(
        self,
        roles: Sequence[AgentRole] | None = None,
        registry: ToolRegistry | None = None,
    ) -> list[ContextArtifact]:
        """Extract all artifacts from all provided sources."""
```

**Note on TOOL_PARAM:** Dropped as a separate kind. `ParamInfo` in the registry has no `description` field — parameter docs are part of the tool's full docstring. The full docstring is the optimizable unit, which is consistent with how Opik's `optimize_tools=True` works (it modifies tool and param description text together).

### LLMResponse

```python
@dataclass(frozen=True)
class LLMResponse:
    """Structured LLM output for metric scoring."""
    text: str                          # Raw text output
    tool_calls: list[ToolCall] = ()    # Tools selected/called

@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
```

Deterministic metrics (ToolSelectionAccuracy) use `tool_calls`; text-based metrics use `text`.

### EvalDataset

```python
@dataclass(frozen=True)
class EvalItem:
    """Single evaluation item — a task + expected behavior."""
    id: str
    task: str                              # User request / editing instruction
    expected_tools: list[str]              # Expected tools (ordered for multi-step tasks)
    expected_output_pattern: str | None     # Regex or substring for compliance checking
    context: dict[str, Any]                # Additional context (project state, available tools, etc.)

@dataclass
class EvalDataset:
    name: str
    items: list[EvalItem]

    @classmethod
    def from_jsonl(cls, path: Path) -> EvalDataset:
        """Load dataset from JSONL file."""

    def split(
        self, train_ratio: float = 0.8, seed: int | None = None
    ) -> tuple[EvalDataset, EvalDataset]:
        """Split into train/validation. seed=None means deterministic by position."""
```

### Built-in Datasets

The framework ships with curated eval datasets for video editing:

1. **tool_selection** — "Trim 2 seconds from the start" → expects `["trim_clip"]`
2. **role_routing** — Tasks that should route to specific agent roles

Datasets stored as JSONL in `src/ave/optimize/data/`.

### Metrics

```python
from typing import ClassVar

class Metric(Protocol):
    name: ClassVar[str]
    def score(self, item: EvalItem, response: LLMResponse) -> MetricResult: ...

@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float         # 0.0 to 1.0
    reason: str          # Explanation (critical for Opik HRPO root cause analysis)
```

**Built-in metrics (MVP):**

| Metric | Type | What it measures |
|--------|------|-----------------|
| `ToolSelectionAccuracy` | Deterministic | Does the LLM pick the correct tool(s)? Jaccard similarity on `expected_tools` vs `response.tool_calls`. |
| `ConventionCompliance` | Deterministic | Does output follow AVE conventions? Checks ns timestamps, `agent:` prefix, codec defaults via regex. |
| `InstructionFollowing` | LLM-as-judge | Does the response follow system prompt directives? Uses a cheap model call. |

**Metric → Opik adapter:**
The Opik backend includes `_to_opik_metric(metrics: list[Metric])` that returns a callable `(dataset_item: dict, llm_output: str) -> ScoreResult`. This adapter:
1. Converts `dict` → `EvalItem` and `str` → `LLMResponse` (parsing tool_use JSON if present)
2. Runs all AVE metrics, combines scores (weighted mean, configurable weights)
3. Returns `opik.evaluation.metrics.score_result.ScoreResult(value=combined, reason=per_metric_breakdown)`

### OptimizerBackend Protocol

```python
class OptimizerBackend(Protocol):
    def optimize(
        self,
        artifact: ContextArtifact,
        dataset: EvalDataset,
        metrics: list[Metric],
        config: OptimizationConfig,
    ) -> OptimizationResult: ...

@dataclass(frozen=True)
class OptimizationConfig:
    model: str = "anthropic/claude-sonnet-4-6"    # Reasoning/optimizer model (stable alias)
    target_model: str = "anthropic/claude-haiku-4-5"  # Model artifacts are used with in production
    max_trials: int = 10
    n_samples: int | None = None        # Items per trial (None = all)
    algorithm: str = "meta_prompt"      # "meta_prompt" | "evolutionary" | "hrpo"
    n_threads: int = 8
    seed: int = 42
    min_improvement: float = 0.01       # Minimum score improvement to accept optimization
    on_regression: str = "reject"       # "reject" | "warn" | "store_anyway"

@dataclass(frozen=True)
class OptimizationResult:
    original_score: float
    optimized_score: float
    improvement: float              # Percentage improvement
    optimized_artifact: ContextArtifact  # New content (or original if rejected)
    accepted: bool                  # Whether improvement met threshold
    trial_history: list[dict]       # Per-trial scores for analysis
```

**Per-artifact optimization:** The protocol optimizes one artifact at a time. This maps cleanly to Opik's `optimize_prompt()` — each artifact becomes a single `ChatPrompt`. The campaign layer handles iteration over all artifacts. A batched variant can be added later for artifacts that interact (e.g., system prompt + tool descriptions together).

### StandaloneEvaluator

For CI regression testing without Opik server dependency.

```python
class StandaloneEvaluator:
    """Evaluate artifacts against datasets using direct LLM calls.

    No Opik dependency. Uses litellm or anthropic SDK directly.
    """

    def __init__(self, model: str = "anthropic/claude-haiku-4-5"):
        self.model = model

    def evaluate(
        self,
        artifacts: list[ContextArtifact],
        dataset: EvalDataset,
        metrics: list[Metric],
    ) -> EvaluationResult:
        """Run each dataset item against the artifacts, score with metrics.

        For each EvalItem:
        1. Build a prompt using the artifacts (system prompt + tool descriptions)
        2. Send to LLM via litellm.completion()
        3. Parse response into LLMResponse
        4. Score with each metric
        5. Aggregate results
        """

@dataclass(frozen=True)
class EvaluationResult:
    overall_score: float
    per_metric: dict[str, float]         # metric_name -> average score
    per_item: list[dict[str, Any]]       # Per-item breakdown
    artifact_scores: dict[str, float]    # artifact_id -> score contribution
```

### ArtifactStore

Versions optimized artifacts for tracking improvement over time.

```python
class ArtifactStore:
    """Versioned storage for optimized artifacts.

    Storage layout:
        .ave/optimized/
        ├── campaigns.jsonl          # Campaign log: {id, timestamp, config_hash, scores}
        └── artifacts/
            ├── role.editor.system_prompt/
            │   ├── v1.txt           # Original
            │   ├── v2.txt           # First optimization
            │   └── meta.json        # Version history: [{version, score, campaign_id, timestamp}]
            └── tool.trim_clip.description/
                ├── v1.txt
                └── meta.json

    Campaign ID format: "{iso_date}_{uuid4_hex8}" e.g. "2026-03-16_a1b2c3d4"
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / "optimized"

    def save(self, artifact: ContextArtifact, score: float, campaign_id: str) -> int:
        """Save optimized artifact. Only promotes to best if score > current best.
        Returns version number."""

    def load_best(self, artifact_id: str) -> ContextArtifact | None:
        """Load the highest-scoring version of an artifact."""

    def current_best_score(self, artifact_id: str) -> float | None:
        """Return the score of the current best version, or None if no versions."""

    def history(self, artifact_id: str) -> list[dict]:
        """Return version history with scores."""

    def diff(self, artifact_id: str, v1: int, v2: int) -> str:
        """Return unified diff between two versions."""
```

### OptimizationCampaign

Top-level orchestrator that ties everything together.

```python
class OptimizationCampaign:
    """Runs a complete optimization cycle: extract → evaluate → optimize → validate → store."""

    def __init__(
        self,
        backend: OptimizerBackend,
        store: ArtifactStore,
        extractor: ArtifactExtractor,
    ): ...

    def run(
        self,
        dataset: EvalDataset,
        metrics: list[Metric],
        config: OptimizationConfig,
        artifact_filter: Callable[[ContextArtifact], bool] | None = None,
        validation_dataset: EvalDataset | None = None,
    ) -> CampaignResult:
        """
        Full optimization cycle:
        1. Extract artifacts from codebase
        2. Evaluate baseline score per artifact
        3. For each artifact: run optimizer, get optimized candidate
        4. Apply regression guard: only accept if improvement >= config.min_improvement
        5. If validation_dataset provided, validate accepted artifacts against holdout
        6. Store accepted artifacts
        7. Return comparison report with both improved and regressed artifacts
        """

    def evaluate_only(
        self,
        dataset: EvalDataset,
        metrics: list[Metric],
    ) -> EvaluationResult:
        """Evaluate current artifacts without optimizing. For CI regression.
        Uses StandaloneEvaluator — no Opik dependency."""

@dataclass(frozen=True)
class CampaignResult:
    campaign_id: str                    # Format: "{iso_date}_{uuid4_hex8}"
    baseline: EvaluationResult
    optimized: EvaluationResult
    validation: EvaluationResult | None
    artifacts_improved: list[tuple[ContextArtifact, ContextArtifact, float]]  # (before, after, delta)
    artifacts_rejected: list[tuple[ContextArtifact, float, float]]  # (artifact, original, optimized)
    duration_seconds: float
```

## Public API

```python
from ave.optimize import optimize_artifacts, evaluate_artifacts

# Evaluate current artifacts (CI-friendly, no Opik needed)
result = evaluate_artifacts(
    roles=ALL_ROLES,
    registry=registry,
    dataset_path=Path("src/ave/optimize/data/tool_selection.jsonl"),
)

# Run full optimization campaign (requires opik-optimizer)
result = optimize_artifacts(
    roles=ALL_ROLES,
    registry=registry,
    dataset_path=Path("src/ave/optimize/data/tool_selection.jsonl"),
    config=OptimizationConfig(algorithm="meta_prompt", max_trials=10),
    store_dir=Path(".ave"),
)
```

## Test Strategy

### Test Markers

| Marker | When |
|--------|------|
| None | Pure unit tests (models, extractors, store, datasets, deterministic metrics) |
| `@pytest.mark.llm` | Requires ANTHROPIC_API_KEY (LLM-as-judge metrics, standalone eval) |
| `@requires_opik` | Requires opik-optimizer + Opik server/API key |
| `@pytest.mark.slow` | >10 seconds (optimization campaigns) |

`requires_opik` is a new skip marker added to `conftest.py` following the existing `requires_X` pattern.

### Test Matrix

| Layer | Tests | Marker |
|-------|-------|--------|
| ContextArtifact, EvalItem, LLMResponse | Construction, serialization | None |
| ArtifactExtractor | Extracts from mock roles/registry | None |
| EvalDataset | Load JSONL, split train/val, deterministic split | None |
| ToolSelectionAccuracy | Score with known tool calls | None |
| ConventionCompliance | Score with known convention patterns | None |
| InstructionFollowing | LLM-as-judge scoring | `@pytest.mark.llm` |
| StandaloneEvaluator | Full eval pipeline with real LLM | `@pytest.mark.llm` |
| ArtifactStore | Save, load_best, regression guard, history, diff | None |
| OpikOptimizerBackend | Opik SDK integration | `@requires_opik` + `@pytest.mark.llm` + `@pytest.mark.slow` |
| OptimizationCampaign | Full campaign with regression guard | `@requires_opik` + `@pytest.mark.llm` + `@pytest.mark.slow` |

## Conventions

- Follows AVE patterns: Protocol-based backends, frozen dataclasses
- Optional dependency: `opik-optimizer` lazy-imported only in `backends/_opik.py` via `_compat.import_optional`
- All functions pure where possible — campaign orchestrator is the only stateful component
- Eval datasets are JSONL files (easy to version, diff, extend)
- Metric results always include `reason` string
- Artifact IDs use dots (not colons) for filesystem safety
- Regression guard is mandatory: worse artifacts are never silently promoted

## Out of Scope (MVP)

- `ArtifactKind.MEMORY` and `ArtifactKind.SKILL` (deferred — freeform prose, hard to metric)
- `ArtifactKind.TOOL_PARAM` (deferred — `ParamInfo` has no description field; full docstring is the unit)
- Automatic apply-back of optimized artifacts to source files
- CLI entry point (`ave optimize run`, `ave optimize eval`)
- CI integration (GitHub Actions)
- Multi-objective optimization (quality vs. token count Pareto front)
- Batched multi-artifact optimization (e.g., system prompt + tools together)
- Extracting orchestrator system prompt from `MultiAgentOrchestrator.get_system_prompt()` (needs orchestrator instance)
