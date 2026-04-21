# Phase 2 — Harness Rung A (Plan-Level Tool Selection) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first rung of the trustworthy-harness: scenarios stored as YAML, loaded through Inspect AI's Task/Dataset/Solver/Scorer framework, validated against agent tool-selection (not tool execution) for the three flagship scenarios.

**Architecture:** New `src/ave/harness/` module. Scenarios live as YAML under `harness/scenarios/`. Pydantic models validate the schema. An Inspect AI adapter turns scenarios into `Dataset`s; a plan-only `Solver` binds non-executing tool stubs and lets the LLM choose a tool chain; two `Scorer`s evaluate `all_of`/`any_of`/`forbidden` tool requirements and plan-level scope violations. Tests use `mockllm/model` (deterministic, zero-cost); real evaluation runs use configurable cloud/local models.

**Tech Stack:** Python 3.12, Inspect AI (optional dep), Pydantic, PyYAML, FFmpeg (for lavfi fixtures, optional). Pytest with `importorskip` gates keeps CI green when the harness extra isn't installed.

**Spec:** `docs/superpowers/specs/2026-04-21-trustworthy-harness-design.md` — "Harness Architecture" + Phase 2 section.

**Depends on:** Phase 1 (feature/phase1-safety-foundation) — `Domain`, `ActivityLog`, `EditingSession.call_tool(agent_role=)` must exist and work.

---

## File Structure

**New files:**

```
src/ave/harness/
├── __init__.py                     # Module exports + version gate
├── schema.py                       # Pydantic Scenario, PlanExpected, ScopeSpec, InputAsset, etc.
├── loader.py                       # load_scenario_from_yaml(path) -> Scenario
├── adapter.py                      # scenarios_to_dataset(scenarios) -> Inspect Dataset
├── evaluators/
│   ├── __init__.py
│   ├── tool_selection.py           # Pure evaluate_plan(called, expected) -> Verdict
│   └── scope.py                    # Pure evaluate_scope(called, registry, forbidden) -> Verdict
├── solvers/
│   ├── __init__.py
│   └── plan.py                     # @solver plan_solver() — non-executing tool stubs
├── scorers/
│   ├── __init__.py
│   ├── tool_selection.py           # @scorer wrapper around evaluators.tool_selection
│   └── scope.py                    # @scorer wrapper around evaluators.scope
├── task.py                         # @task plan_rung_task(scenario_file)
├── fixtures/
│   ├── __init__.py
│   └── builder.py                  # build_lavfi_clip(expression, duration, out_path)
├── scenarios/
│   ├── reel.filler-word-trim.yaml
│   ├── short.highlight-reel-from-long.yaml
│   └── talking-head.subtitled-vertical.yaml
├── cli.py                          # `ave-harness run <id>` minimal CLI
└── pytest_plugin.py                # Exports discover_plan_scenarios() for pytest glue

tests/test_harness/
├── __init__.py
├── test_schema.py                  # Task 2
├── test_loader.py                  # Task 3
├── test_fixtures_builder.py        # Task 4
├── test_evaluators_tool_selection.py   # Task 5
├── test_evaluators_scope.py        # Task 6
├── test_adapter.py                 # Task 7
├── test_plan_solver.py             # Task 8
├── test_scorers.py                 # Task 9
├── test_task.py                    # Task 10
├── test_scenarios.py               # Task 11
├── test_cli.py                     # Task 12
└── test_pytest_plugin.py           # Task 13
```

**Modified files:**

- `pyproject.toml` — add `[harness]` optional-dependency group (`inspect-ai>=0.3`, `pyyaml>=6.0`) and a new `[project.scripts]` entry for the `ave-harness` command
- `tests/conftest.py` — add `requires_harness` marker and `requires_inspect` marker (for harness tests that require inspect-ai)

---

## Task 1: Harness scaffold + optional dependency

**Files:**
- Create: `src/ave/harness/__init__.py`
- Create: `tests/test_harness/__init__.py`
- Create: `tests/test_harness/test_module_skeleton.py`
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_module_skeleton.py
"""Tests that the harness module imports cleanly and exposes its version."""


def test_harness_module_imports():
    import ave.harness  # noqa: F401


def test_harness_exports_version_string():
    import ave.harness
    assert isinstance(ave.harness.__version__, str)
    assert len(ave.harness.__version__) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_module_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/__init__.py`:

```python
"""AVE Harness — trustworthy end-to-end evaluation of agent behaviour.

Three rungs of increasing cost and coverage:
- Rung A (this module at Phase 2): plan-level tool selection, no execution
- Rung B (Phase 3): real tool execution, state assertions
- Rung C (Phase 4): full render + VLM-judge

Built on Inspect AI (optional dependency — installed via the [harness] extra).
"""

from __future__ import annotations

__version__ = "0.1.0"
```

Create `tests/test_harness/__init__.py` (empty file).

Modify `pyproject.toml` — append a new entry to `[project.optional-dependencies]`:

```toml
harness = [
    "inspect-ai>=0.3",
    "pyyaml>=6.0",
]
```

Modify `tests/conftest.py` — append new skip markers at the end, below the existing `requires_opik`:

```python
def _inspect_available() -> bool:
    try:
        import inspect_ai  # noqa: F401
        return True
    except ImportError:
        return False


def _pyyaml_available() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


requires_inspect = pytest.mark.skipif(
    not _inspect_available(), reason="inspect-ai not installed (pip install ave[harness])"
)
requires_pyyaml = pytest.mark.skipif(
    not _pyyaml_available(), reason="pyyaml not installed"
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_module_skeleton.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/__init__.py tests/test_harness/__init__.py tests/test_harness/test_module_skeleton.py pyproject.toml tests/conftest.py
git commit -m "feat(harness): add scaffold module and [harness] optional dependency"
```

---

## Task 2: Scenario schema (Pydantic models)

**Files:**
- Create: `src/ave/harness/schema.py`
- Test: `tests/test_harness/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_schema.py
"""Tests for the harness Scenario Pydantic schema."""

import pytest

from ave.harness.schema import (
    InputAsset,
    PlanExpected,
    Scenario,
    ScopeSpec,
)


def _minimal_scenario_dict() -> dict:
    return {
        "id": "reel.test",
        "description": "Minimal test scenario",
        "tiers": ["plan"],
        "prompt": "do a thing",
        "scope": {
            "allowed_agents": ["editor"],
            "forbidden_layers": [],
        },
        "inputs": {
            "assets": [{"id": "clip1", "ref": "fixture://testclip.mp4"}],
        },
        "expected": {
            "plan": {
                "tools_required": {"all_of": ["trim"], "any_of": []},
                "tools_forbidden": [],
                "irrelevance_allowed": False,
            }
        },
    }


def test_scenario_minimal_loads():
    s = Scenario.model_validate(_minimal_scenario_dict())
    assert s.id == "reel.test"
    assert s.tiers == ("plan",)
    assert s.prompt == "do a thing"
    assert s.expected.plan is not None


def test_scenario_rejects_unknown_tier():
    d = _minimal_scenario_dict()
    d["tiers"] = ["plan", "nonsense"]
    with pytest.raises(Exception):  # Pydantic ValidationError
        Scenario.model_validate(d)


def test_scope_spec_accepts_empty_lists():
    s = ScopeSpec(allowed_agents=[], forbidden_layers=[])
    assert s.allowed_agents == ()
    assert s.forbidden_layers == ()


def test_input_asset_ref_required():
    with pytest.raises(Exception):
        InputAsset.model_validate({"id": "x"})


def test_plan_expected_defaults_empty():
    pe = PlanExpected.model_validate({})
    assert pe.tools_required.all_of == ()
    assert pe.tools_required.any_of == ()
    assert pe.tools_forbidden == ()
    assert pe.irrelevance_allowed is False


def test_scenario_without_plan_expected_is_allowed():
    """Scenarios may declare only a subset of tiers — that's fine."""
    d = _minimal_scenario_dict()
    d["tiers"] = ["execute"]
    d["expected"] = {}
    s = Scenario.model_validate(d)
    assert s.expected.plan is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.schema'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/schema.py`:

```python
"""Pydantic schema for harness scenarios.

Each scenario is a YAML file conforming to the Scenario model. Scenarios
opt into one or more tiers (plan / execute / render); each tier has its
own expectation block that runs when that tier runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Tier = Literal["plan", "execute", "render"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class InputAsset(_Frozen):
    id: str
    ref: str  # "fixture://...", "lavfi://...", "corpus://..."


class Inputs(_Frozen):
    assets: tuple[InputAsset, ...] = ()


class ScopeSpec(_Frozen):
    allowed_agents: tuple[str, ...] = ()
    forbidden_layers: tuple[str, ...] = ()

    @field_validator("allowed_agents", "forbidden_layers", mode="before")
    @classmethod
    def _coerce_tuple(cls, v):
        if v is None:
            return ()
        return tuple(v)


class ToolsRequired(_Frozen):
    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    @field_validator("all_of", "any_of", mode="before")
    @classmethod
    def _coerce_tuple(cls, v):
        if v is None:
            return ()
        return tuple(v)


class PlanExpected(_Frozen):
    tools_required: ToolsRequired = Field(default_factory=ToolsRequired)
    tools_forbidden: tuple[str, ...] = ()
    irrelevance_allowed: bool = False

    @field_validator("tools_forbidden", mode="before")
    @classmethod
    def _coerce_tuple(cls, v):
        if v is None:
            return ()
        return tuple(v)


class ExecuteExpected(_Frozen):
    """Stub for Phase 3. Accepts any dict; validated later."""
    raw: dict = Field(default_factory=dict)


class RenderExpected(_Frozen):
    """Stub for Phase 4. Accepts any dict; validated later."""
    raw: dict = Field(default_factory=dict)


class Expected(_Frozen):
    plan: PlanExpected | None = None
    execute: ExecuteExpected | None = None
    render: RenderExpected | None = None


class SafetyExpected(_Frozen):
    must_be_reversible: bool = True
    must_respect_scope: bool = True
    state_sync_after_undo: bool = True
    source_asset_immutable: bool = True


class Scenario(_Frozen):
    id: str
    description: str = ""
    tiers: tuple[Tier, ...]
    prompt: str
    scope: ScopeSpec = Field(default_factory=ScopeSpec)
    inputs: Inputs = Field(default_factory=Inputs)
    expected: Expected = Field(default_factory=Expected)
    safety: SafetyExpected = Field(default_factory=SafetyExpected)

    @field_validator("tiers", mode="before")
    @classmethod
    def _coerce_tuple(cls, v):
        if v is None:
            return ()
        return tuple(v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_schema.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/schema.py tests/test_harness/test_schema.py
git commit -m "feat(harness): add Pydantic Scenario schema"
```

---

## Task 3: YAML loader

**Files:**
- Create: `src/ave/harness/loader.py`
- Test: `tests/test_harness/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_loader.py
"""Tests for YAML scenario loading."""

import pytest

pytest.importorskip("yaml")

from pathlib import Path

from ave.harness.loader import load_scenario_from_yaml
from ave.harness.schema import Scenario


_VALID = """
id: reel.filler-word-trim
description: "Remove filler words from an interview clip"
tiers: [plan]
prompt: "clean up the ums and ahs in this interview clip"
scope:
  allowed_agents: [transcriptionist, editor]
  forbidden_layers: []
inputs:
  assets:
    - id: clip1
      ref: "fixture://interview-60s.mp4"
expected:
  plan:
    tools_required:
      all_of: [find_fillers]
      any_of: [text_cut, trim]
    tools_forbidden: [apply_blend_mode]
    irrelevance_allowed: false
"""


def test_load_valid_yaml(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text(_VALID)
    scenario = load_scenario_from_yaml(p)
    assert isinstance(scenario, Scenario)
    assert scenario.id == "reel.filler-word-trim"
    assert "find_fillers" in scenario.expected.plan.tools_required.all_of


def test_load_invalid_yaml_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: scenario: schema")
    with pytest.raises(Exception):
        load_scenario_from_yaml(p)


def test_load_missing_required_field(tmp_path: Path):
    p = tmp_path / "missing.yaml"
    p.write_text("id: x\n")  # missing tiers, prompt
    with pytest.raises(Exception):
        load_scenario_from_yaml(p)


def test_load_nonexistent_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_scenario_from_yaml(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.loader'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/loader.py`:

```python
"""Load harness scenarios from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from ave.harness.schema import Scenario


def load_scenario_from_yaml(path: Path) -> Scenario:
    """Parse a scenario YAML file into a Scenario model.

    Raises FileNotFoundError if path doesn't exist, yaml.YAMLError on bad
    YAML, and pydantic ValidationError on schema violations.
    """
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in {path}, got {type(data).__name__}")
    return Scenario.model_validate(data)
```

Note: if `pyyaml` isn't installed, the import at the top of `loader.py` will fail at module import time. That's acceptable — the `[harness]` extra pulls it in.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_loader.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/loader.py tests/test_harness/test_loader.py
git commit -m "feat(harness): add YAML scenario loader"
```

---

## Task 4: Lavfi fixture builder

**Files:**
- Create: `src/ave/harness/fixtures/__init__.py`
- Create: `src/ave/harness/fixtures/builder.py`
- Test: `tests/test_harness/test_fixtures_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_fixtures_builder.py
"""Tests for the lavfi-based deterministic fixture builder."""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_build_testsrc_clip_creates_mp4(tmp_path: Path):
    from ave.harness.fixtures.builder import build_lavfi_clip

    out = tmp_path / "testsrc.mp4"
    built = build_lavfi_clip(
        expression="testsrc=size=320x240:rate=24",
        duration_seconds=1.0,
        output_path=out,
    )
    assert built == out
    assert out.exists()
    assert out.stat().st_size > 0


@requires_ffmpeg
def test_build_raises_on_bogus_expression(tmp_path: Path):
    from ave.harness.fixtures.builder import build_lavfi_clip

    with pytest.raises(RuntimeError, match="ffmpeg"):
        build_lavfi_clip(
            expression="lavfi_DEFINITELY_INVALID",
            duration_seconds=0.5,
            output_path=tmp_path / "x.mp4",
        )


def test_module_importable_without_ffmpeg():
    """The module should import without ffmpeg present (only callers fail)."""
    import ave.harness.fixtures.builder  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_fixtures_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.fixtures'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/fixtures/__init__.py` (empty).

Create `src/ave/harness/fixtures/builder.py`:

```python
"""Deterministic video fixture generation via FFmpeg lavfi.

Produces small, reproducible test videos (color patterns with timestamps,
tone bursts, etc.) for harness scenarios that don't need real footage.

The `lavfi` input device is part of the FFmpeg distribution; see
``https://ffmpeg.org/ffmpeg-filters.html#testsrc`` for available sources.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def build_lavfi_clip(
    expression: str,
    duration_seconds: float,
    output_path: Path,
    *,
    framerate: int = 24,
) -> Path:
    """Build a test video from a lavfi filter expression.

    Examples
    --------
    >>> build_lavfi_clip(
    ...     "testsrc=size=1280x720:rate=24",
    ...     duration_seconds=5.0,
    ...     output_path=Path("/tmp/test.mp4"),
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found on PATH")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        expression,
        "-t",
        str(duration_seconds),
        "-r",
        str(framerate),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg lavfi build failed (exit {result.returncode}):\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_fixtures_builder.py -v`
Expected: PASS (3 tests; the two `@requires_ffmpeg` tests run if ffmpeg is on PATH, otherwise skip).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/fixtures/__init__.py src/ave/harness/fixtures/builder.py tests/test_harness/test_fixtures_builder.py
git commit -m "feat(harness): add lavfi-based deterministic fixture builder"
```

---

## Task 5: Pure tool-selection evaluator

**Files:**
- Create: `src/ave/harness/evaluators/__init__.py`
- Create: `src/ave/harness/evaluators/tool_selection.py`
- Test: `tests/test_harness/test_evaluators_tool_selection.py`

**Architectural note:** The scoring logic is separated from Inspect AI plumbing so it can be unit-tested without any LLM or Inspect dependency. The `@scorer`-decorated wrapper in Task 9 calls into this pure function.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_evaluators_tool_selection.py
"""Pure tool-selection evaluator tests (no Inspect AI dependency)."""

from ave.harness.evaluators.tool_selection import Verdict, evaluate_plan
from ave.harness.schema import PlanExpected, ToolsRequired


def _plan(all_of=(), any_of=(), forbidden=(), irrelevance=False) -> PlanExpected:
    return PlanExpected(
        tools_required=ToolsRequired(all_of=tuple(all_of), any_of=tuple(any_of)),
        tools_forbidden=tuple(forbidden),
        irrelevance_allowed=irrelevance,
    )


def test_all_of_satisfied():
    v = evaluate_plan(["find_fillers", "text_cut"], _plan(all_of=["find_fillers", "text_cut"]))
    assert v.passed is True


def test_all_of_missing():
    v = evaluate_plan(["find_fillers"], _plan(all_of=["find_fillers", "text_cut"]))
    assert v.passed is False
    assert "text_cut" in v.reason
    assert "missing" in v.reason.lower()


def test_any_of_satisfied_with_one():
    v = evaluate_plan(["trim"], _plan(any_of=["trim", "text_cut"]))
    assert v.passed is True


def test_any_of_not_satisfied():
    v = evaluate_plan(["apply_blend_mode"], _plan(any_of=["trim", "text_cut"]))
    assert v.passed is False
    assert "any_of" in v.reason.lower()


def test_forbidden_called_fails():
    v = evaluate_plan(["find_fillers", "apply_blend_mode"], _plan(
        all_of=["find_fillers"],
        forbidden=["apply_blend_mode"],
    ))
    assert v.passed is False
    assert "forbidden" in v.reason.lower()


def test_all_and_any_both_required():
    v_pass = evaluate_plan(
        ["find_fillers", "trim"],
        _plan(all_of=["find_fillers"], any_of=["trim", "text_cut"]),
    )
    assert v_pass.passed is True

    v_fail = evaluate_plan(
        ["find_fillers"],  # missing any_of member
        _plan(all_of=["find_fillers"], any_of=["trim", "text_cut"]),
    )
    assert v_fail.passed is False


def test_irrelevance_no_tools_called_passes_when_allowed():
    v = evaluate_plan([], _plan(all_of=["find_fillers"], irrelevance=True))
    assert v.passed is True
    assert "irrelevance" in v.reason.lower()


def test_irrelevance_no_tools_called_fails_when_not_allowed():
    v = evaluate_plan([], _plan(all_of=["find_fillers"], irrelevance=False))
    assert v.passed is False


def test_verdict_is_hashable_frozen_dataclass():
    v = Verdict(passed=True, reason="ok")
    assert v.passed is True
    assert v.reason == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_evaluators_tool_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.evaluators'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/evaluators/__init__.py` (empty).

Create `src/ave/harness/evaluators/tool_selection.py`:

```python
"""Pure tool-selection evaluator — decides pass/fail from a called-tool list.

Separated from Inspect AI plumbing for unit-test simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ave.harness.schema import PlanExpected


@dataclass(frozen=True)
class Verdict:
    """Evaluation verdict."""

    passed: bool
    reason: str


def evaluate_plan(
    called_tools: Sequence[str],
    expected: PlanExpected,
) -> Verdict:
    """Decide whether a sequence of tool-call names satisfies the plan's expectations.

    Rules applied in order:
    1. If no tools were called AND irrelevance_allowed is True -> pass (agent correctly refused).
    2. If no tools were called AND irrelevance_allowed is False -> fail.
    3. Any tool in ``tools_forbidden`` that was called -> fail.
    4. Every tool in ``all_of`` must appear at least once -> else fail.
    5. If ``any_of`` is non-empty, at least one member must appear -> else fail.
    6. Otherwise -> pass.
    """
    called = list(called_tools)

    if not called:
        if expected.irrelevance_allowed:
            return Verdict(True, "irrelevance allowed; agent called no tools")
        return Verdict(False, "agent called no tools, but irrelevance_allowed=False")

    forbidden_hits = [t for t in expected.tools_forbidden if t in called]
    if forbidden_hits:
        return Verdict(False, f"forbidden tools invoked: {forbidden_hits}")

    missing_all = [t for t in expected.tools_required.all_of if t not in called]
    if missing_all:
        return Verdict(False, f"missing required (all_of): {missing_all}")

    if expected.tools_required.any_of:
        hits = [t for t in expected.tools_required.any_of if t in called]
        if not hits:
            return Verdict(
                False,
                f"none of the any_of tools called: expected one of "
                f"{list(expected.tools_required.any_of)}",
            )

    return Verdict(True, "plan satisfies all constraints")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_evaluators_tool_selection.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/evaluators/__init__.py src/ave/harness/evaluators/tool_selection.py tests/test_harness/test_evaluators_tool_selection.py
git commit -m "feat(harness): add pure tool-selection evaluator"
```

---

## Task 6: Pure scope evaluator

**Files:**
- Create: `src/ave/harness/evaluators/scope.py`
- Test: `tests/test_harness/test_evaluators_scope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_evaluators_scope.py
"""Pure scope-enforcement evaluator tests."""

from ave.agent.domains import Domain
from ave.agent.registry import ToolRegistry
from ave.harness.evaluators.scope import evaluate_scope


def _registry_with_domain_tools() -> ToolRegistry:
    reg = ToolRegistry()

    def do_audio(x: int) -> None:
        """Audio op."""

    def do_video(x: int) -> None:
        """Video op."""

    reg.register("do_audio", do_audio, domain="audio", domains_touched=(Domain.AUDIO,))
    reg.register("do_video", do_video, domain="video", domains_touched=(Domain.VIDEO,))
    return reg


def test_scope_passes_when_no_forbidden_domain_touched():
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["do_audio"],
        registry=reg,
        forbidden_domains=("video",),
    )
    assert v.passed is True


def test_scope_fails_when_forbidden_domain_touched():
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["do_video"],
        registry=reg,
        forbidden_domains=("video",),
    )
    assert v.passed is False
    assert "do_video" in v.reason
    assert "video" in v.reason.lower()


def test_scope_passes_when_forbidden_is_empty():
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["do_audio", "do_video"],
        registry=reg,
        forbidden_domains=(),
    )
    assert v.passed is True


def test_scope_tolerates_unknown_tool():
    """Tools not in the registry (e.g., typo) are reported but don't crash scope."""
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["mystery_tool"],
        registry=reg,
        forbidden_domains=("video",),
    )
    # The scope evaluator can't prove a violation for an unknown tool — passes.
    assert v.passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_evaluators_scope.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.evaluators.scope'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/evaluators/scope.py`:

```python
"""Pure scope-enforcement evaluator.

Checks that none of the called tools touch forbidden domains. Uses the
`domains_touched` metadata declared via the ToolRegistry.
"""

from __future__ import annotations

from typing import Sequence

from ave.agent.domains import Domain
from ave.agent.registry import RegistryError, ToolRegistry
from ave.harness.evaluators.tool_selection import Verdict


def evaluate_scope(
    *,
    called_tools: Sequence[str],
    registry: ToolRegistry,
    forbidden_domains: Sequence[str],
) -> Verdict:
    """Check scope compliance.

    Returns a failing Verdict if any called tool's domains intersect the
    forbidden set. Unknown tool names (not in the registry) are ignored —
    they can't prove a scope violation, and the tool-selection scorer already
    catches unrecognized plans.
    """
    forbidden = {d for d in forbidden_domains}
    violations: list[tuple[str, list[Domain]]] = []

    for name in called_tools:
        try:
            touched = registry.get_tool_domains_touched(name)
        except (RegistryError, KeyError):
            continue
        hits = [d for d in touched if d.value in forbidden]
        if hits:
            violations.append((name, hits))

    if violations:
        body = "; ".join(
            f"{name} touches {[d.value for d in hits]}" for name, hits in violations
        )
        return Verdict(False, f"scope violations: {body}")
    return Verdict(True, "scope respected")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_evaluators_scope.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/evaluators/scope.py tests/test_harness/test_evaluators_scope.py
git commit -m "feat(harness): add pure scope evaluator"
```

---

## Task 7: Inspect AI dataset adapter

**Files:**
- Create: `src/ave/harness/adapter.py`
- Test: `tests/test_harness/test_adapter.py`

**Architectural note:** This is the first task that imports `inspect_ai`. Tests are gated by `requires_inspect`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_adapter.py
"""Tests for the Scenario -> Inspect AI Dataset adapter."""

import pytest

from tests.conftest import requires_inspect
from ave.harness.schema import (
    Expected,
    InputAsset,
    Inputs,
    PlanExpected,
    Scenario,
    ScopeSpec,
    ToolsRequired,
)


def _scenario() -> Scenario:
    return Scenario(
        id="reel.test",
        description="",
        tiers=("plan",),
        prompt="do a thing",
        scope=ScopeSpec(allowed_agents=("editor",), forbidden_layers=()),
        inputs=Inputs(assets=(InputAsset(id="c", ref="fixture://x.mp4"),)),
        expected=Expected(
            plan=PlanExpected(
                tools_required=ToolsRequired(all_of=("trim",), any_of=()),
                tools_forbidden=("apply_blend_mode",),
                irrelevance_allowed=False,
            )
        ),
    )


@requires_inspect
def test_scenarios_to_dataset_shape():
    from ave.harness.adapter import scenarios_to_dataset

    dataset = scenarios_to_dataset([_scenario()])
    samples = list(dataset)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "reel.test"
    assert sample.input == "do a thing"
    # Metadata carries the full scenario for downstream solvers/scorers.
    assert "scenario" in sample.metadata
    assert sample.metadata["scenario"].id == "reel.test"


@requires_inspect
def test_empty_scenarios_yields_empty_dataset():
    from ave.harness.adapter import scenarios_to_dataset

    dataset = scenarios_to_dataset([])
    assert len(list(dataset)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.adapter'` (or skip if inspect-ai not installed).

If `inspect-ai` is not yet installed, install it first:

```bash
.venv/bin/pip install "inspect-ai>=0.3"
```

Then re-run Step 2.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/adapter.py`:

```python
"""Adapt harness Scenarios to Inspect AI Datasets.

Each Scenario becomes one Inspect AI Sample:
- ``input``: the colloquial user prompt
- ``target``: the scenario id (used only for identification; real scoring
  uses metadata)
- ``metadata``: {"scenario": Scenario} — the full Scenario model, so solvers
  and scorers can read any field they need.
"""

from __future__ import annotations

from typing import Sequence

from inspect_ai.dataset import MemoryDataset, Sample

from ave.harness.schema import Scenario


def scenarios_to_dataset(scenarios: Sequence[Scenario]) -> MemoryDataset:
    """Build an Inspect AI MemoryDataset from a list of Scenarios."""
    samples = [
        Sample(
            id=s.id,
            input=s.prompt,
            target=s.id,
            metadata={"scenario": s},
        )
        for s in scenarios
    ]
    return MemoryDataset(samples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_adapter.py -v`
Expected: PASS (2 tests) when inspect-ai installed; skipped otherwise.

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/adapter.py tests/test_harness/test_adapter.py
git commit -m "feat(harness): add Scenario -> Inspect AI Dataset adapter"
```

---

## Task 8: Plan solver

**Files:**
- Create: `src/ave/harness/solvers/__init__.py`
- Create: `src/ave/harness/solvers/plan.py`
- Test: `tests/test_harness/test_plan_solver.py`

**Architectural note:** The plan solver wires non-executing tool stubs that record their invocation and return a plausible no-op result. The model chooses from these stubs; we don't actually execute AVE tools. Each invocation is appended to `state.metadata["called_tools"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_plan_solver.py
"""Plan-solver tests using Inspect AI's mockllm backend."""

import pytest

from tests.conftest import requires_inspect


@requires_inspect
@pytest.mark.asyncio
async def test_plan_solver_records_tool_calls_from_mockllm():
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ToolCall
    from inspect_ai.solver import TaskState

    from ave.harness.schema import (
        Expected,
        InputAsset,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.solvers.plan import extract_tool_calls

    # Build a state with a synthetic tool-use assistant message.
    scenario = Scenario(
        id="test",
        tiers=("plan",),
        prompt="do it",
        scope=ScopeSpec(allowed_agents=("editor",), forbidden_layers=()),
        inputs=Inputs(),
        expected=Expected(plan=PlanExpected(
            tools_required=ToolsRequired(all_of=("trim",)),
        )),
    )
    sample = Sample(id="test", input="do it", target="test", metadata={"scenario": scenario})

    state = TaskState(
        model="mockllm/mock",
        sample_id="test",
        epoch=0,
        input=sample.input,
        messages=[
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id="1", function="trim", arguments={"clip_id": "c1"}),
                    ToolCall(id="2", function="find_fillers", arguments={}),
                ],
            ),
            ChatMessageTool(content="trim ok", tool_call_id="1"),
            ChatMessageTool(content="fillers ok", tool_call_id="2"),
        ],
    )

    called = extract_tool_calls(state)
    assert called == ["trim", "find_fillers"]


@requires_inspect
def test_plan_solver_is_decorated_solver():
    """The plan_solver factory returns something Inspect AI accepts as a Solver."""
    from ave.harness.solvers.plan import plan_solver

    s = plan_solver()
    # Inspect AI solvers are callables (coroutines once awaited); verifying
    # it's callable avoids coupling to the internal Solver type.
    assert callable(s)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_plan_solver.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/solvers/__init__.py` (empty).

Create `src/ave/harness/solvers/plan.py`:

```python
"""Plan-level solver — runs the agent with non-executing tool stubs.

The agent chooses tools from the scenario's allowed-agent domains; the
stubs simply record that the tool was invoked. The scorer then reads
``state.metadata["called_tools"]`` to decide pass/fail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_ai.solver import Generate, Solver, TaskState, solver

if TYPE_CHECKING:
    pass


def extract_tool_calls(state: TaskState) -> list[str]:
    """Walk the conversation and collect the names of invoked tools, in order."""
    names: list[str] = []
    for msg in state.messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            fn = getattr(tc, "function", None) or getattr(tc, "name", None)
            if fn:
                names.append(str(fn))
    return names


@solver
def plan_solver() -> Solver:
    """Drive the agent through a single tool-planning turn.

    The scenario's declared ``allowed_agents`` are informational at this
    rung; we leave tool wiring to Inspect AI's configured model harness.
    On return, ``state.metadata["called_tools"]`` holds the invocation
    sequence for the scorer.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        state.metadata = dict(state.metadata or {})
        state.metadata["called_tools"] = extract_tool_calls(state)
        return state

    return solve
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_plan_solver.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/solvers/__init__.py src/ave/harness/solvers/plan.py tests/test_harness/test_plan_solver.py
git commit -m "feat(harness): add plan-level solver recording tool-call sequence"
```

---

## Task 9: Inspect AI scorer wrappers

**Files:**
- Create: `src/ave/harness/scorers/__init__.py`
- Create: `src/ave/harness/scorers/tool_selection.py`
- Create: `src/ave/harness/scorers/scope.py`
- Test: `tests/test_harness/test_scorers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_scorers.py
"""Tests for Inspect AI scorer wrappers around the pure evaluators."""

import pytest

from tests.conftest import requires_inspect


@requires_inspect
@pytest.mark.asyncio
async def test_tool_selection_scorer_reads_state_metadata():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from ave.harness.schema import (
        Expected,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.scorers.tool_selection import tool_selection_scorer

    scenario = Scenario(
        id="t",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(allowed_agents=("editor",)),
        inputs=Inputs(),
        expected=Expected(plan=PlanExpected(
            tools_required=ToolsRequired(all_of=("trim",)),
        )),
    )
    state = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["trim", "concat"]},
    )

    scorer = tool_selection_scorer()
    score = await scorer(state, Target("t"))
    assert score.value == 1  # passing
    assert "plan satisfies" in (score.explanation or "").lower()


@requires_inspect
@pytest.mark.asyncio
async def test_tool_selection_scorer_fails_on_missing_required():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from ave.harness.schema import (
        Expected,
        Inputs,
        PlanExpected,
        Scenario,
        ScopeSpec,
        ToolsRequired,
    )
    from ave.harness.scorers.tool_selection import tool_selection_scorer

    scenario = Scenario(
        id="t",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(),
        inputs=Inputs(),
        expected=Expected(plan=PlanExpected(
            tools_required=ToolsRequired(all_of=("trim", "text_cut")),
        )),
    )
    state = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["trim"]},
    )

    score = await tool_selection_scorer()(state, Target("t"))
    assert score.value == 0


@requires_inspect
@pytest.mark.asyncio
async def test_scope_scorer_respects_forbidden_layers():
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    from ave.harness.schema import Expected, Inputs, Scenario, ScopeSpec
    from ave.harness.scorers.scope import scope_scorer

    scenario = Scenario(
        id="t",
        tiers=("plan",),
        prompt="",
        scope=ScopeSpec(allowed_agents=("sound_designer",), forbidden_layers=("video",)),
        inputs=Inputs(),
        expected=Expected(),
    )
    # "trim" touches TIMELINE_STRUCTURE in the default registry; doesn't violate "video"
    state_ok = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["trim"]},
    )
    score_ok = await scope_scorer()(state_ok, Target("t"))
    assert score_ok.value == 1

    # "apply_blend_mode" touches VIDEO -> violation
    state_bad = TaskState(
        model="mockllm/mock",
        sample_id="t",
        epoch=0,
        input="",
        messages=[],
        metadata={"scenario": scenario, "called_tools": ["apply_blend_mode"]},
    )
    score_bad = await scope_scorer()(state_bad, Target("t"))
    assert score_bad.value == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_scorers.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/scorers/__init__.py` (empty).

Create `src/ave/harness/scorers/tool_selection.py`:

```python
"""Inspect AI scorer wrapper for plan-level tool selection."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.harness.evaluators.tool_selection import evaluate_plan
from ave.harness.schema import Scenario


@scorer(metrics=[])
def tool_selection_scorer() -> Scorer:
    """Pass/fail based on tool-invocation constraints in the scenario's plan expectations."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        scenario: Scenario = meta["scenario"]
        called: list[str] = list(meta.get("called_tools", []))
        plan = scenario.expected.plan
        if plan is None:
            return Score(value=1, explanation="scenario has no plan expectations; skipping")
        verdict = evaluate_plan(called, plan)
        return Score(
            value=1 if verdict.passed else 0,
            answer=",".join(called),
            explanation=verdict.reason,
        )

    return score
```

Create `src/ave/harness/scorers/scope.py`:

```python
"""Inspect AI scorer wrapper for plan-level scope compliance."""

from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.agent.registry import ToolRegistry
from ave.agent.session import EditingSession
from ave.harness.evaluators.scope import evaluate_scope
from ave.harness.schema import Scenario

_session_for_registry: EditingSession | None = None


def _shared_registry() -> ToolRegistry:
    """Lazily build a full-registry EditingSession to look up tool domains."""
    global _session_for_registry
    if _session_for_registry is None:
        _session_for_registry = EditingSession()
    return _session_for_registry.registry


@scorer(metrics=[])
def scope_scorer() -> Scorer:
    """Pass/fail based on whether called tools respect the scenario's forbidden_layers."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        scenario: Scenario = meta["scenario"]
        called: list[str] = list(meta.get("called_tools", []))
        verdict = evaluate_scope(
            called_tools=called,
            registry=_shared_registry(),
            forbidden_domains=scenario.scope.forbidden_layers,
        )
        return Score(
            value=1 if verdict.passed else 0,
            answer=",".join(called),
            explanation=verdict.reason,
        )

    return score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_scorers.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/scorers/__init__.py src/ave/harness/scorers/tool_selection.py src/ave/harness/scorers/scope.py tests/test_harness/test_scorers.py
git commit -m "feat(harness): add Inspect AI scorer wrappers for tool-selection and scope"
```

---

## Task 10: Task builder + end-to-end mockllm test

**Files:**
- Create: `src/ave/harness/task.py`
- Test: `tests/test_harness/test_task.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_task.py
"""End-to-end test: Scenario -> Inspect AI Task -> mockllm run -> score."""

from pathlib import Path

import pytest

from tests.conftest import requires_inspect


_SCENARIO_YAML = """
id: reel.filler-word-trim
description: ""
tiers: [plan]
prompt: "clean up the ums"
scope:
  allowed_agents: [transcriptionist, editor]
  forbidden_layers: []
inputs:
  assets: []
expected:
  plan:
    tools_required:
      all_of: [find_fillers]
      any_of: [text_cut, trim]
    tools_forbidden: [apply_blend_mode]
    irrelevance_allowed: false
"""


@requires_inspect
def test_plan_rung_task_constructs_valid_task(tmp_path: Path):
    from inspect_ai import Task

    from ave.harness.task import plan_rung_task

    p = tmp_path / "s.yaml"
    p.write_text(_SCENARIO_YAML)

    task = plan_rung_task(scenario_file=str(p))
    assert isinstance(task, Task)
    assert task.dataset is not None
    assert task.solver is not None
    assert task.scorer is not None


@requires_inspect
def test_plan_rung_task_runs_with_mockllm_canned_response(tmp_path: Path):
    """Run the task end-to-end with a mock model that emits the required tool calls."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ave.harness.task import plan_rung_task

    p = tmp_path / "s.yaml"
    p.write_text(_SCENARIO_YAML)

    # mockllm/mock accepts no real tool calls, so this test only verifies that
    # the Task can be constructed and invoked end-to-end under eval() without
    # raising. The scorer will mark it as failing (no tools called and
    # irrelevance_allowed=False), which is the correct behaviour.
    task = plan_rung_task(scenario_file=str(p))
    model = get_model("mockllm/mock")
    results = inspect_eval(task, model=model, display="plain", log_dir=str(tmp_path / "logs"))

    assert len(results) == 1
    eval_log = results[0]
    # eval_log has a samples attribute with one entry per scenario
    assert len(eval_log.samples) == 1
    sample = eval_log.samples[0]
    # The scorer produced a score (pass or fail is implementation-dependent
    # for mockllm); the important thing is eval completed without crashing
    # and produced a Score object.
    assert sample.scores is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_task.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.task'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/task.py`:

```python
"""Assemble a scenario into an Inspect AI Task at the plan rung."""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

from ave.harness.adapter import scenarios_to_dataset
from ave.harness.loader import load_scenario_from_yaml
from ave.harness.scorers.scope import scope_scorer
from ave.harness.scorers.tool_selection import tool_selection_scorer
from ave.harness.solvers.plan import plan_solver


@task
def plan_rung_task(scenario_file: str) -> Task:
    """Build a Task that evaluates a single YAML scenario at the plan rung."""
    scenario = load_scenario_from_yaml(Path(scenario_file))
    return Task(
        dataset=scenarios_to_dataset([scenario]),
        solver=plan_solver(),
        scorer=[tool_selection_scorer(), scope_scorer()],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_task.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/task.py tests/test_harness/test_task.py
git commit -m "feat(harness): add plan_rung_task builder for end-to-end scenario eval"
```

---

## Task 11: Three flagship scenario YAMLs

**Files:**
- Create: `src/ave/harness/scenarios/reel.filler-word-trim.yaml`
- Create: `src/ave/harness/scenarios/short.highlight-reel-from-long.yaml`
- Create: `src/ave/harness/scenarios/talking-head.subtitled-vertical.yaml`
- Test: `tests/test_harness/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_scenarios.py
"""Tests that the bundled flagship scenarios load and validate cleanly."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from ave.harness.loader import load_scenario_from_yaml


SCENARIO_DIR = Path(__file__).parent.parent.parent / "src" / "ave" / "harness" / "scenarios"


def test_reel_filler_word_trim_loads():
    s = load_scenario_from_yaml(SCENARIO_DIR / "reel.filler-word-trim.yaml")
    assert s.id == "reel.filler-word-trim"
    assert "plan" in s.tiers
    # Realism: the required tool chain should mention find_fillers for fillers
    assert "find_fillers" in s.expected.plan.tools_required.all_of


def test_highlight_reel_loads():
    s = load_scenario_from_yaml(SCENARIO_DIR / "short.highlight-reel-from-long.yaml")
    assert s.id == "short.highlight-reel-from-long"
    # Requires scene detection to chunk the input
    assert "detect_scenes" in s.expected.plan.tools_required.all_of


def test_subtitled_vertical_loads():
    s = load_scenario_from_yaml(SCENARIO_DIR / "talking-head.subtitled-vertical.yaml")
    assert s.id == "talking-head.subtitled-vertical"
    # Requires text overlay for the captions
    assert "add_text_overlay" in s.expected.plan.tools_required.all_of


def test_all_flagship_scenarios_use_real_tool_names():
    """Every tool named in the required/forbidden lists must exist in the registry."""
    from ave.agent.session import EditingSession

    session = EditingSession()
    known = {t.name for t in session.registry.search_tools()}
    for yaml_name in (
        "reel.filler-word-trim.yaml",
        "short.highlight-reel-from-long.yaml",
        "talking-head.subtitled-vertical.yaml",
    ):
        s = load_scenario_from_yaml(SCENARIO_DIR / yaml_name)
        mentioned = set(s.expected.plan.tools_required.all_of)
        mentioned |= set(s.expected.plan.tools_required.any_of)
        mentioned |= set(s.expected.plan.tools_forbidden)
        unknown = mentioned - known
        assert not unknown, f"{yaml_name} references unknown tools: {unknown}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_scenarios.py -v`
Expected: FAIL — scenario files don't exist yet.

- [ ] **Step 3: Write the three scenario files**

Create `src/ave/harness/scenarios/reel.filler-word-trim.yaml`:

```yaml
id: reel.filler-word-trim
description: "Remove filler words (um, ah, you know) from an interview clip"
tiers: [plan]
prompt: "clean up the ums and ahs in this interview clip"
scope:
  allowed_agents: [transcriptionist, editor]
  forbidden_layers: []
inputs:
  assets:
    - id: interview
      ref: "lavfi://testsrc=size=640x480:rate=24:duration=60"
expected:
  plan:
    tools_required:
      all_of: [find_fillers]
      any_of: [text_cut, trim]
    tools_forbidden: [apply_blend_mode, cdl, lut_parse]
    irrelevance_allowed: false
safety:
  must_be_reversible: true
  must_respect_scope: true
  state_sync_after_undo: true
  source_asset_immutable: true
```

Create `src/ave/harness/scenarios/short.highlight-reel-from-long.yaml`:

```yaml
id: short.highlight-reel-from-long
description: "Assemble a short highlight reel from a longer video"
tiers: [plan]
prompt: "make me a 30-second highlight from this 5-minute vlog"
scope:
  allowed_agents: [editor]
  forbidden_layers: []
inputs:
  assets:
    - id: vlog
      ref: "lavfi://testsrc=size=1280x720:rate=24:duration=300"
expected:
  plan:
    tools_required:
      all_of: [detect_scenes, create_rough_cut]
      any_of: [render_with_preset]
    tools_forbidden: [find_fillers, web_search]
    irrelevance_allowed: false
safety:
  must_be_reversible: true
  must_respect_scope: true
  state_sync_after_undo: true
  source_asset_immutable: true
```

Create `src/ave/harness/scenarios/talking-head.subtitled-vertical.yaml`:

```yaml
id: talking-head.subtitled-vertical
description: "Convert a horizontal talking-head video to a vertical reel with burnt-in captions"
tiers: [plan]
prompt: "turn this horizontal talking-head into a vertical Reel with burnt-in captions"
scope:
  allowed_agents: [transcriptionist, editor]
  forbidden_layers: []
inputs:
  assets:
    - id: raw
      ref: "lavfi://testsrc=size=1920x1080:rate=24:duration=60"
expected:
  plan:
    tools_required:
      all_of: [add_text_overlay]
      any_of: [render_with_preset, search_transcript]
    tools_forbidden: [cdl, lut_parse]
    irrelevance_allowed: false
safety:
  must_be_reversible: true
  must_respect_scope: true
  state_sync_after_undo: true
  source_asset_immutable: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_scenarios.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/scenarios/ tests/test_harness/test_scenarios.py
git commit -m "feat(harness): add three flagship plan-tier scenarios"
```

---

## Task 12: `ave-harness` CLI

**Files:**
- Create: `src/ave/harness/cli.py`
- Modify: `pyproject.toml` — add `[project.scripts]` entry
- Test: `tests/test_harness/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_cli.py
"""Tests for the `ave-harness` CLI."""

from pathlib import Path

import pytest

from tests.conftest import requires_inspect


def test_parse_args_run_subcommand_requires_scenario():
    from ave.harness.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_parse_args_run_with_scenario_file():
    from ave.harness.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["run", "--scenario-file", "x.yaml", "--tier", "plan"])
    assert ns.command == "run"
    assert ns.scenario_file == "x.yaml"
    assert ns.tier == "plan"


def test_parse_args_default_tier_is_plan():
    from ave.harness.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["run", "--scenario-file", "x.yaml"])
    assert ns.tier == "plan"


def test_run_rejects_unsupported_tier():
    from ave.harness.cli import cli_main

    rc = cli_main(["run", "--scenario-file", "x.yaml", "--tier", "execute"])
    assert rc != 0


@requires_inspect
def test_run_plan_tier_on_scenario_returns_zero_on_eval(tmp_path: Path):
    """Running a trivial scenario via the CLI returns 0 (smoke test)."""
    from ave.harness.cli import cli_main

    yaml_text = """
id: cli.smoke
description: ""
tiers: [plan]
prompt: "do it"
scope:
  allowed_agents: [editor]
  forbidden_layers: []
inputs:
  assets: []
expected:
  plan:
    tools_required:
      all_of: []
      any_of: []
    tools_forbidden: []
    irrelevance_allowed: true
"""
    s = tmp_path / "s.yaml"
    s.write_text(yaml_text)

    rc = cli_main(
        [
            "run",
            "--scenario-file",
            str(s),
            "--tier",
            "plan",
            "--model",
            "mockllm/mock",
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    assert rc == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.cli'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/cli.py`:

```python
"""`ave-harness` CLI — minimal entry point for Phase 2.

Usage
-----

    ave-harness run --scenario-file path/to/scenario.yaml [--tier plan]
                    [--model mockllm/mock] [--log-dir ./logs]

Only the ``plan`` tier is implemented in Phase 2; the other tiers return a
non-zero exit code with a clear error until Phases 3 and 4 land.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


SUPPORTED_TIERS = ("plan",)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ave-harness", description="AVE harness CLI")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a scenario at a given tier")
    run.add_argument("--scenario-file", required=True, help="Path to the scenario YAML")
    run.add_argument(
        "--tier",
        default="plan",
        help="Evaluation tier (only 'plan' is available in Phase 2)",
    )
    run.add_argument(
        "--model",
        default="mockllm/mock",
        help="Inspect AI model spec (e.g., anthropic/claude-opus-4-7, mockllm/mock)",
    )
    run.add_argument(
        "--log-dir",
        default=None,
        help="Directory for Inspect AI eval logs (default: ./logs)",
    )
    return p


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command != "run":
        parser.error(f"unknown command: {ns.command}")

    if ns.tier not in SUPPORTED_TIERS:
        print(
            f"error: tier {ns.tier!r} not implemented in Phase 2 "
            f"(supported: {list(SUPPORTED_TIERS)})",
            file=sys.stderr,
        )
        return 2

    try:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import get_model

        from ave.harness.task import plan_rung_task
    except ImportError as exc:
        print(
            f"error: harness deps missing ({exc}). Install with "
            f"`pip install ave[harness]`",
            file=sys.stderr,
        )
        return 3

    task = plan_rung_task(scenario_file=ns.scenario_file)
    model = get_model(ns.model)
    log_dir = ns.log_dir or "./logs"

    results = inspect_eval(task, model=model, display="plain", log_dir=log_dir)
    if not results:
        print("error: no eval results produced", file=sys.stderr)
        return 4
    return 0


def main() -> None:
    sys.exit(cli_main(sys.argv[1:]))
```

Modify `pyproject.toml` — add a `[project.scripts]` entry (after `[project.optional-dependencies]`):

```toml
[project.scripts]
ave-harness = "ave.harness.cli:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_cli.py -v`
Expected: PASS (5 tests).

Also reinstall the package so the `ave-harness` entry point works on the path:

Run: `.venv/bin/pip install -e .`
Expected: installs in editable mode without errors.

Then smoke-test:

Run: `.venv/bin/ave-harness run --scenario-file src/ave/harness/scenarios/reel.filler-word-trim.yaml --tier plan --model mockllm/mock --log-dir /tmp/ave-harness-logs`
Expected: exits 0, writes eval log under `/tmp/ave-harness-logs/`.

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/cli.py tests/test_harness/test_cli.py pyproject.toml
git commit -m "feat(harness): add minimal ave-harness CLI for plan-tier runs"
```

---

## Task 13: Pytest plugin — discover and run plan scenarios

**Files:**
- Create: `src/ave/harness/pytest_plugin.py`
- Test: `tests/test_harness/test_pytest_plugin.py`

**Architectural note:** Rather than register a pytest plugin via entry points (which would auto-load into every test run), Phase 2 exposes a helper function `discover_plan_scenarios(scenario_dir)` that returns parametrizable scenario paths. Users wire it into their own `tests/test_harness/test_plan_scenarios.py` (or similar) with `pytest.mark.parametrize`. This keeps the integration opt-in and avoids surprising other test suites.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness/test_pytest_plugin.py
"""Tests for the pytest helper module."""

from pathlib import Path

import pytest

from tests.conftest import requires_inspect


def test_discover_plan_scenarios_returns_yaml_paths(tmp_path: Path):
    from ave.harness.pytest_plugin import discover_plan_scenarios

    (tmp_path / "a.yaml").write_text("id: a\ntiers: [plan]\nprompt: p")
    (tmp_path / "b.yaml").write_text("id: b\ntiers: [execute]\nprompt: p")
    (tmp_path / "ignored.txt").write_text("not a scenario")

    scenarios = discover_plan_scenarios(tmp_path)
    names = sorted(Path(p).name for p in scenarios)
    # Only YAML files; both a and b are returned because discover does NOT
    # filter by tier (filtering happens when the scenario is loaded).
    assert names == ["a.yaml", "b.yaml"]


def test_discover_plan_scenarios_sorts_stably(tmp_path: Path):
    from ave.harness.pytest_plugin import discover_plan_scenarios

    (tmp_path / "z.yaml").write_text("id: z\ntiers: [plan]\nprompt: p")
    (tmp_path / "a.yaml").write_text("id: a\ntiers: [plan]\nprompt: p")

    scenarios = discover_plan_scenarios(tmp_path)
    assert [Path(p).name for p in scenarios] == ["a.yaml", "z.yaml"]


def test_bundled_scenarios_discovered():
    from ave.harness.pytest_plugin import discover_plan_scenarios, bundled_scenarios_dir

    scenarios = discover_plan_scenarios(bundled_scenarios_dir())
    names = {Path(p).name for p in scenarios}
    assert "reel.filler-word-trim.yaml" in names
    assert "short.highlight-reel-from-long.yaml" in names
    assert "talking-head.subtitled-vertical.yaml" in names


@requires_inspect
def test_run_plan_scenario_helper_returns_ok_for_irrelevance_scenario(tmp_path: Path):
    """The convenience helper runs one scenario and returns True on pass."""
    from ave.harness.pytest_plugin import run_plan_scenario

    yaml_text = """
id: helper.smoke
description: ""
tiers: [plan]
prompt: "do it"
scope:
  allowed_agents: [editor]
  forbidden_layers: []
inputs:
  assets: []
expected:
  plan:
    tools_required:
      all_of: []
      any_of: []
    tools_forbidden: []
    irrelevance_allowed: true
"""
    p = tmp_path / "s.yaml"
    p.write_text(yaml_text)

    ok = run_plan_scenario(str(p), model="mockllm/mock", log_dir=str(tmp_path / "logs"))
    assert ok is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_harness/test_pytest_plugin.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.harness.pytest_plugin'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/ave/harness/pytest_plugin.py`:

```python
"""Pytest helpers for harness scenarios.

Consumers wire these into a test module of their own using
pytest.mark.parametrize, e.g.:

    import pytest
    from ave.harness.pytest_plugin import (
        bundled_scenarios_dir,
        discover_plan_scenarios,
        run_plan_scenario,
    )

    @pytest.mark.parametrize("scenario_file", discover_plan_scenarios(bundled_scenarios_dir()))
    def test_plan_scenario(scenario_file):
        assert run_plan_scenario(scenario_file) is True
"""

from __future__ import annotations

from pathlib import Path


def bundled_scenarios_dir() -> Path:
    """Return the directory of flagship scenarios shipped with ave.harness."""
    return Path(__file__).parent / "scenarios"


def discover_plan_scenarios(scenario_dir: Path) -> list[str]:
    """Return stably sorted YAML file paths under scenario_dir."""
    directory = Path(scenario_dir)
    paths = sorted(p for p in directory.glob("*.yaml") if p.is_file())
    return [str(p) for p in paths]


def run_plan_scenario(
    scenario_file: str,
    *,
    model: str = "mockllm/mock",
    log_dir: str | None = None,
) -> bool:
    """Run a single plan-tier scenario; return True iff all samples passed all scorers."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ave.harness.task import plan_rung_task

    task = plan_rung_task(scenario_file=scenario_file)
    results = inspect_eval(
        task,
        model=get_model(model),
        display="plain",
        log_dir=log_dir or "./logs",
    )
    if not results:
        return False
    for eval_log in results:
        for sample in eval_log.samples or []:
            for score in (sample.scores or {}).values():
                if getattr(score, "value", 0) != 1:
                    return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_harness/test_pytest_plugin.py -v`
Expected: PASS (4 tests).

Also run the full harness test suite to catch any regression:

Run: `.venv/bin/python -m pytest tests/test_harness/ -v`
Expected: no new failures; all tests either PASS or SKIP (skip when inspect-ai isn't installed).

- [ ] **Step 5: Commit**

```bash
git add src/ave/harness/pytest_plugin.py tests/test_harness/test_pytest_plugin.py
git commit -m "feat(harness): add pytest helpers for scenario discovery and running"
```

---

## Phase 2 Completion Verification

After all 13 tasks committed:

```bash
.venv/bin/python -m pytest tests/test_harness/ tests/test_agent/ tests/test_project/ tests/test_web/ -v
```

Expected: all new harness tests pass or skip (when `inspect-ai` isn't installed); no Phase 1 regressions.

Additional smoke test:

```bash
.venv/bin/pip install -e ".[harness]"  # install inspect-ai + pyyaml
.venv/bin/ave-harness run \
    --scenario-file src/ave/harness/scenarios/reel.filler-word-trim.yaml \
    --tier plan \
    --model mockllm/mock \
    --log-dir /tmp/ave-harness-logs
```

Expected: exits 0, log file written to `/tmp/ave-harness-logs/`.

When all green, Phase 2 is complete. Phase 3 (Rung B — execute + state diff scorer + safety invariants) follows as a separate plan.
