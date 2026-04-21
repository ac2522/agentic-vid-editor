# Phase 1 — Safety Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the safety substrate the Phase 2–4 harness will rely on: turn checkpoints, user-facing undo/redo, state-sync protocol, domain-scoped agents, activity log, and source-asset immutability. No harness code in this phase.

**Architecture:** Additive changes to existing modules (`SnapshotManager`, `EditingSession`, `AgentRole`, `ChatSession`, Web UI). New standalone modules for `Domain` enum, `ActivityLog`, and `state_sync`. Backward compatibility preserved: callers that don't opt in keep existing behavior.

**Tech Stack:** Python 3.12, pytest, aiohttp (Web UI), vanilla JS frontend. No new runtime dependencies in this phase.

**Spec:** `docs/superpowers/specs/2026-04-21-trustworthy-harness-design.md` — "Shared Foundation" section.

---

## File Structure

**New files:**
- `src/ave/agent/domains.py` — `Domain` enum + `DOMAIN_TO_ROLE_MAP`
- `src/ave/agent/activity.py` — `ActivityLog`, `ActivityEntry`
- `src/ave/agent/state_sync.py` — `build_state_summary()` + protocol types
- `src/ave/agent/errors.py` — `ScopeViolationError`, `SourceAssetWriteError` (new shared errors module)
- `tests/test_agent/test_domains.py`
- `tests/test_agent/test_activity.py`
- `tests/test_agent/test_state_sync.py`
- `tests/test_agent/test_session_scope.py`
- `tests/test_agent/test_session_immutability.py`
- `tests/test_project/test_snapshot_turns.py`
- `tests/test_agent/test_session_turns.py`
- `tests/test_web/test_chat_state_sync.py`
- `tests/test_web/test_undo_redo_api.py`

**Modified files:**
- `src/ave/project/snapshots.py` — add turn-checkpoint API
- `src/ave/agent/roles.py` — add `owned_domains` to `AgentRole`; populate for all 6 roles
- `src/ave/agent/registry.py` — add optional `domains_touched` metadata to tool registration
- `src/ave/agent/session.py` — scope validation, activity log integration, source-asset immutability, turn lifecycle API
- `src/ave/web/chat.py` — state-summary injection + new protocol events (`format_timeline_rollback`)
- `src/ave/web/api.py` — POST `/api/undo`, POST `/api/redo`, GET `/api/state-summary`
- `src/ave/web/client/chat.js` — undo/redo buttons, keyboard shortcuts, rollback marker rendering
- `src/ave/web/client/index.html` — undo/redo button markup
- `src/ave/web/client/styles.css` — styling for rollback markers

---

## Task 1: Domain enum + errors module

**Files:**
- Create: `src/ave/agent/domains.py`
- Create: `src/ave/agent/errors.py`
- Test: `tests/test_agent/test_domains.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent/test_domains.py
"""Tests for the Domain enum."""

from ave.agent.domains import Domain


def test_domain_enum_members():
    """All required domain members exist."""
    assert Domain.AUDIO.value == "audio"
    assert Domain.VIDEO.value == "video"
    assert Domain.SUBTITLE.value == "subtitle"
    assert Domain.VFX_MASK.value == "vfx_mask"
    assert Domain.COLOR.value == "color"
    assert Domain.TIMELINE_STRUCTURE.value == "timeline_structure"
    assert Domain.METADATA.value == "metadata"
    assert Domain.RENDER.value == "render"
    assert Domain.INGEST.value == "ingest"
    assert Domain.RESEARCH.value == "research"


def test_domain_from_string():
    """Domain.from_string maps legacy domain names to enum members."""
    assert Domain.from_string("audio") is Domain.AUDIO
    assert Domain.from_string("editing") is Domain.TIMELINE_STRUCTURE
    assert Domain.from_string("compositing") is Domain.VIDEO
    assert Domain.from_string("motion_graphics") is Domain.SUBTITLE
    assert Domain.from_string("scene") is Domain.TIMELINE_STRUCTURE
    assert Domain.from_string("transcription") is Domain.SUBTITLE
    assert Domain.from_string("vfx") is Domain.VFX_MASK
    assert Domain.from_string("color") is Domain.COLOR
    assert Domain.from_string("research") is Domain.RESEARCH


def test_domain_from_string_unknown_raises():
    """Unknown domain strings raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown domain"):
        Domain.from_string("nonexistent")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_domains.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.agent.domains'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ave/agent/domains.py
"""Domain enum — classifies which part of a project a tool touches.

Used for agent-role scoping: each AgentRole declares owned_domains,
and EditingSession rejects tool calls whose domains fall outside
the dispatching role's ownership.
"""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    """Canonical domains for tool classification and agent scoping."""

    AUDIO = "audio"
    VIDEO = "video"
    SUBTITLE = "subtitle"
    VFX_MASK = "vfx_mask"
    COLOR = "color"
    TIMELINE_STRUCTURE = "timeline_structure"
    METADATA = "metadata"
    RENDER = "render"
    INGEST = "ingest"
    RESEARCH = "research"

    @classmethod
    def from_string(cls, raw: str) -> Domain:
        """Map legacy domain names (editing, compositing, etc.) to canonical domains."""
        legacy_map = {
            "editing": cls.TIMELINE_STRUCTURE,
            "compositing": cls.VIDEO,
            "motion_graphics": cls.SUBTITLE,
            "scene": cls.TIMELINE_STRUCTURE,
            "transcription": cls.SUBTITLE,
            "vfx": cls.VFX_MASK,
        }
        if raw in legacy_map:
            return legacy_map[raw]
        try:
            return cls(raw)
        except ValueError as exc:
            raise ValueError(f"Unknown domain: {raw!r}") from exc
```

```python
# src/ave/agent/errors.py
"""Shared errors for agent-session safety enforcement."""

from __future__ import annotations


class ScopeViolationError(Exception):
    """Raised when an agent attempts a tool call outside its owned domains."""


class SourceAssetWriteError(Exception):
    """Raised when a tool call would write to a source-asset path."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_domains.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/domains.py src/ave/agent/errors.py tests/test_agent/test_domains.py
git commit -m "feat(agent): add Domain enum and shared safety errors"
```

---

## Task 2: Extend AgentRole with owned_domains

**Files:**
- Modify: `src/ave/agent/roles.py`
- Test: `tests/test_agent/test_roles.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_roles.py`:

```python
def test_all_roles_have_owned_domains():
    """Every built-in role declares at least one owned domain."""
    from ave.agent.roles import ALL_ROLES
    from ave.agent.domains import Domain

    for role in ALL_ROLES:
        assert isinstance(role.owned_domains, tuple)
        assert len(role.owned_domains) > 0, f"{role.name} has no owned_domains"
        for d in role.owned_domains:
            assert isinstance(d, Domain), (
                f"{role.name} owned_domains contains non-Domain: {d!r}"
            )


def test_editor_owns_timeline_structure_and_video():
    from ave.agent.roles import EDITOR_ROLE
    from ave.agent.domains import Domain

    assert Domain.TIMELINE_STRUCTURE in EDITOR_ROLE.owned_domains
    assert Domain.VIDEO in EDITOR_ROLE.owned_domains


def test_sound_designer_owns_only_audio():
    from ave.agent.roles import SOUND_DESIGNER_ROLE
    from ave.agent.domains import Domain

    assert SOUND_DESIGNER_ROLE.owned_domains == (Domain.AUDIO,)


def test_transcriptionist_owns_subtitle_and_metadata():
    from ave.agent.roles import TRANSCRIPTIONIST_ROLE
    from ave.agent.domains import Domain

    assert Domain.SUBTITLE in TRANSCRIPTIONIST_ROLE.owned_domains
    assert Domain.METADATA in TRANSCRIPTIONIST_ROLE.owned_domains
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_roles.py::test_all_roles_have_owned_domains -v`
Expected: FAIL with `AttributeError: 'AgentRole' object has no attribute 'owned_domains'`

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/agent/roles.py`. Add import and field, and set `owned_domains` on each role:

```python
# Add at top, after "from dataclasses import dataclass"
from ave.agent.domains import Domain


@dataclass(frozen=True)
class AgentRole:
    """Definition of a specialized agent role."""

    name: str
    description: str
    system_prompt: str
    domains: tuple[str, ...]
    owned_domains: tuple[Domain, ...] = ()
```

Then add `owned_domains=(...)` to each of the six existing role definitions:

```python
# EDITOR_ROLE — add after the "domains=..." line
    owned_domains=(Domain.TIMELINE_STRUCTURE, Domain.VIDEO, Domain.SUBTITLE),
# COLORIST_ROLE
    owned_domains=(Domain.COLOR,),
# SOUND_DESIGNER_ROLE
    owned_domains=(Domain.AUDIO,),
# TRANSCRIPTIONIST_ROLE
    owned_domains=(Domain.SUBTITLE, Domain.METADATA),
# RESEARCHER_ROLE
    owned_domains=(Domain.RESEARCH,),
# VFX_ARTIST_ROLE
    owned_domains=(Domain.VFX_MASK, Domain.VIDEO),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_roles.py -v`
Expected: PASS (all tests including the new ones)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/roles.py tests/test_agent/test_roles.py
git commit -m "feat(agent): add owned_domains to AgentRole for scope enforcement"
```

---

## Task 3: Tool registration accepts `domains_touched` metadata

**Files:**
- Modify: `src/ave/agent/registry.py`
- Test: `tests/test_agent/test_registry.py` (extend existing)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_registry.py`:

```python
def test_register_with_domains_touched():
    """Tool registration accepts an optional domains_touched parameter."""
    from ave.agent.registry import ToolRegistry
    from ave.agent.domains import Domain

    reg = ToolRegistry()

    def add_watermark(clip_id: str) -> str:
        """Add a watermark to a clip."""
        return clip_id

    reg.register(
        "add_watermark",
        add_watermark,
        domain="video",
        domains_touched=(Domain.VIDEO, Domain.METADATA),
    )

    assert reg.get_tool_domains_touched("add_watermark") == (Domain.VIDEO, Domain.METADATA)


def test_domains_touched_defaults_to_legacy_domain_mapping():
    """When domains_touched is omitted, fall back to from_string(domain)."""
    from ave.agent.registry import ToolRegistry
    from ave.agent.domains import Domain

    reg = ToolRegistry()

    def dummy() -> None:
        """Dummy."""

    reg.register("dummy", dummy, domain="editing")
    # 'editing' → TIMELINE_STRUCTURE via Domain.from_string
    assert reg.get_tool_domains_touched("dummy") == (Domain.TIMELINE_STRUCTURE,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_registry.py::test_register_with_domains_touched -v`
Expected: FAIL — `register()` does not accept `domains_touched` kwarg.

- [ ] **Step 3: Write minimal implementation**

In `src/ave/agent/registry.py`:

1. Add to imports near the top:

```python
from ave.agent.domains import Domain
```

2. In the internal tool-record dataclass (search for the registration storage — look for where tools are stored; the class should have a `domain` field). Add a `domains_touched` field to the record class:

```python
# Find the internal dataclass that stores registered tools. It likely has
# fields like: name, func, domain, description. Add:
    domains_touched: tuple[Domain, ...] = ()
```

3. Extend the `register()` method signature to accept `domains_touched: tuple[Domain, ...] | None = None`. Compute the effective tuple:

```python
def register(
    self,
    name: str,
    func: Callable,
    domain: str,
    *,
    domains_touched: tuple[Domain, ...] | None = None,
    # ... existing kwargs ...
) -> None:
    effective_domains = (
        domains_touched
        if domains_touched is not None
        else (Domain.from_string(domain),)
    )
    # ... store record with domains_touched=effective_domains ...
```

4. Add a public accessor:

```python
def get_tool_domains_touched(self, tool_name: str) -> tuple[Domain, ...]:
    """Return the Domain tuple declared by the tool (or inferred from its legacy domain)."""
    record = self._tools[tool_name]  # use existing accessor pattern
    return record.domains_touched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_registry.py -v`
Expected: PASS (both new tests and all existing registry tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/registry.py tests/test_agent/test_registry.py
git commit -m "feat(registry): add optional domains_touched metadata on tool registration"
```

---

## Task 4: Scope validation in `EditingSession.call_tool`

**Files:**
- Modify: `src/ave/agent/session.py`
- Test: `tests/test_agent/test_session_scope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent/test_session_scope.py
"""Scope-enforcement tests for EditingSession.call_tool."""

import pytest

from ave.agent.domains import Domain
from ave.agent.errors import ScopeViolationError
from ave.agent.roles import SOUND_DESIGNER_ROLE, EDITOR_ROLE
from ave.agent.session import EditingSession


def _register_audio_and_video_tool(session: EditingSession) -> None:
    """Register one tool in AUDIO and one in VIDEO for scope testing."""

    def set_volume(clip_id: str, db: float) -> dict:
        """Set clip volume."""
        return {"clip_id": clip_id, "db": db}

    def add_watermark(clip_id: str) -> dict:
        """Add a watermark."""
        return {"clip_id": clip_id}

    session.registry.register(
        "set_volume",
        set_volume,
        domain="audio",
        domains_touched=(Domain.AUDIO,),
    )
    session.registry.register(
        "add_watermark",
        add_watermark,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )


def test_call_tool_without_role_bypasses_scope():
    """No role passed = no scope check (backward compat)."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    # Both should succeed without role
    assert session.call_tool("set_volume", {"clip_id": "c1", "db": -3.0})
    assert session.call_tool("add_watermark", {"clip_id": "c1"})


def test_call_tool_with_role_rejects_out_of_domain():
    """Sound designer may not call a VIDEO tool."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    with pytest.raises(ScopeViolationError, match="add_watermark"):
        session.call_tool(
            "add_watermark",
            {"clip_id": "c1"},
            agent_role=SOUND_DESIGNER_ROLE,
        )


def test_call_tool_with_role_allows_in_domain():
    """Sound designer may call an AUDIO tool."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    result = session.call_tool(
        "set_volume",
        {"clip_id": "c1", "db": -3.0},
        agent_role=SOUND_DESIGNER_ROLE,
    )
    assert result["db"] == -3.0


def test_scope_violation_contains_role_and_tool_and_domain():
    """Error message is actionable."""
    session = EditingSession()
    _register_audio_and_video_tool(session)

    with pytest.raises(ScopeViolationError) as excinfo:
        session.call_tool(
            "add_watermark",
            {"clip_id": "c1"},
            agent_role=SOUND_DESIGNER_ROLE,
        )

    msg = str(excinfo.value)
    assert "sound_designer" in msg.lower()
    assert "add_watermark" in msg
    assert "video" in msg.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_session_scope.py -v`
Expected: FAIL — `call_tool()` does not accept `agent_role` kwarg.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/agent/session.py`:

1. Add imports at top:

```python
from ave.agent.errors import ScopeViolationError
from ave.agent.roles import AgentRole
```

2. Update `call_tool` signature and add scope check before the lock block:

```python
def call_tool(
    self,
    tool_name: str,
    params: dict,
    *,
    agent_role: AgentRole | None = None,
) -> Any:
    """Execute a tool with state tracking, snapshots, history recording, and scope enforcement.

    If agent_role is provided, the tool's domains_touched must be a subset of
    agent_role.owned_domains; otherwise ScopeViolationError is raised.
    """
    if agent_role is not None:
        touched = self._registry.get_tool_domains_touched(tool_name)
        out_of_scope = [d for d in touched if d not in agent_role.owned_domains]
        if out_of_scope:
            raise ScopeViolationError(
                f"Role {agent_role.name!r} cannot call {tool_name!r}: "
                f"tool touches {[d.value for d in out_of_scope]}, "
                f"role owns {[d.value for d in agent_role.owned_domains]}"
            )

    with self._lock:
        # ... existing body unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_session_scope.py -v`
Expected: PASS (4 tests)

Also run existing session tests to confirm no regressions:

Run: `python -m pytest tests/test_agent/test_session.py -v`
Expected: PASS (all existing)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/session.py tests/test_agent/test_session_scope.py
git commit -m "feat(session): enforce agent-role scope on call_tool"
```

---

## Task 5: `ActivityLog` class with JSONL persistence

**Files:**
- Create: `src/ave/agent/activity.py`
- Test: `tests/test_agent/test_activity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent/test_activity.py
"""Tests for the ActivityLog append-only log."""

import json
from pathlib import Path

import pytest

from ave.agent.activity import ActivityEntry, ActivityLog


def test_entry_roundtrip_dict():
    """ActivityEntry serializes and deserializes cleanly."""
    e = ActivityEntry(
        timestamp=1714000000.0,
        agent_id="sound_designer",
        tool_name="set_volume",
        summary="set_volume(clip_id='c1', db=-3.0)",
        snapshot_id="snap-1",
    )
    d = e.to_dict()
    assert d["agent_id"] == "sound_designer"
    assert ActivityEntry.from_dict(d) == e


def test_append_in_memory(tmp_path: Path):
    log = ActivityLog(persist_path=None)
    log.append(
        agent_id="editor",
        tool_name="trim_clip",
        summary="trim_clip(clip='c1', in=1.0, out=4.0)",
        snapshot_id="snap-a",
    )
    log.append(
        agent_id="colorist",
        tool_name="apply_lut",
        summary="apply_lut(clip='c1', lut='FilmLook.cube')",
        snapshot_id="snap-b",
    )
    entries = log.entries()
    assert len(entries) == 2
    assert entries[0].tool_name == "trim_clip"
    assert entries[1].agent_id == "colorist"


def test_append_persisted(tmp_path: Path):
    """Each append writes one JSON line to the persist path."""
    persist = tmp_path / "activity-log.jsonl"
    log = ActivityLog(persist_path=persist)
    log.append(agent_id="editor", tool_name="trim_clip", summary="t", snapshot_id="s1")
    log.append(agent_id="editor", tool_name="split_clip", summary="s", snapshot_id="s2")

    raw = persist.read_text().splitlines()
    assert len(raw) == 2
    assert json.loads(raw[0])["tool_name"] == "trim_clip"
    assert json.loads(raw[1])["tool_name"] == "split_clip"


def test_load_from_persisted_file(tmp_path: Path):
    """Opening an existing log reads prior entries."""
    persist = tmp_path / "activity-log.jsonl"
    log1 = ActivityLog(persist_path=persist)
    log1.append(agent_id="editor", tool_name="trim_clip", summary="t", snapshot_id="s1")

    log2 = ActivityLog(persist_path=persist)
    entries = log2.entries()
    assert len(entries) == 1
    assert entries[0].tool_name == "trim_clip"


def test_entries_since(tmp_path: Path):
    """entries_since(timestamp) returns only entries after the given time."""
    log = ActivityLog(persist_path=None)
    log.append(agent_id="a", tool_name="t1", summary="", snapshot_id="s1")
    cutoff = log.entries()[-1].timestamp
    log.append(agent_id="a", tool_name="t2", summary="", snapshot_id="s2")

    after = log.entries_since(cutoff)
    assert len(after) == 1
    assert after[0].tool_name == "t2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_activity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ave.agent.activity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ave/agent/activity.py
"""Append-only per-session activity log.

Emitted by EditingSession on every successful tool call. Feeds the
state-sync protocol and harness assertions.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ActivityEntry:
    """A single activity record."""

    timestamp: float
    agent_id: str
    tool_name: str
    summary: str
    snapshot_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ActivityEntry:
        return cls(
            timestamp=float(d["timestamp"]),
            agent_id=str(d["agent_id"]),
            tool_name=str(d["tool_name"]),
            summary=str(d["summary"]),
            snapshot_id=str(d["snapshot_id"]),
        )


class ActivityLog:
    """Append-only log persisted as JSONL.

    In-memory when persist_path is None; otherwise each append writes
    one JSON line and the constructor loads any existing file.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._persist_path = persist_path
        self._entries: list[ActivityEntry] = []
        if persist_path is not None and persist_path.exists():
            self._entries = [
                ActivityEntry.from_dict(json.loads(line))
                for line in persist_path.read_text().splitlines()
                if line.strip()
            ]

    def append(
        self,
        *,
        agent_id: str,
        tool_name: str,
        summary: str,
        snapshot_id: str,
    ) -> ActivityEntry:
        entry = ActivityEntry(
            timestamp=time.time(),
            agent_id=agent_id,
            tool_name=tool_name,
            summary=summary,
            snapshot_id=snapshot_id,
        )
        self._entries.append(entry)
        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._persist_path.open("a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def entries(self) -> list[ActivityEntry]:
        return list(self._entries)

    def entries_since(self, timestamp: float) -> list[ActivityEntry]:
        return [e for e in self._entries if e.timestamp > timestamp]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_activity.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/activity.py tests/test_agent/test_activity.py
git commit -m "feat(agent): add append-only ActivityLog with JSONL persistence"
```

---

## Task 6: Integrate `ActivityLog` into `EditingSession`

**Files:**
- Modify: `src/ave/agent/session.py`
- Test: `tests/test_agent/test_session_scope.py` (extend) + new assertions

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent/test_session_scope.py`:

```python
def test_successful_tool_call_appends_activity_entry(tmp_path):
    """Every successful call adds one entry to the session's activity log."""
    from ave.agent.activity import ActivityLog

    log = ActivityLog(persist_path=tmp_path / "activity-log.jsonl")
    session = EditingSession(activity_log=log)
    _register_audio_and_video_tool(session)

    session.call_tool("set_volume", {"clip_id": "c1", "db": -3.0})
    session.call_tool("add_watermark", {"clip_id": "c1"})

    entries = log.entries()
    assert len(entries) == 2
    assert entries[0].tool_name == "set_volume"
    assert entries[1].tool_name == "add_watermark"
    # snapshot_id is present even when snapshot_manager is None (placeholder ok)
    assert all(e.snapshot_id for e in entries)


def test_failed_tool_call_does_not_append_activity_entry(tmp_path):
    """Exceptions abort the call — no activity entry is recorded."""
    from ave.agent.activity import ActivityLog

    log = ActivityLog(persist_path=tmp_path / "activity-log.jsonl")
    session = EditingSession(activity_log=log)

    def always_fails(clip_id: str) -> None:
        """Always fails."""
        raise RuntimeError("boom")

    session.registry.register(
        "always_fails",
        always_fails,
        domain="audio",
        domains_touched=(Domain.AUDIO,),
    )

    with pytest.raises(RuntimeError, match="boom"):
        session.call_tool("always_fails", {"clip_id": "c1"})

    assert log.entries() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_session_scope.py::test_successful_tool_call_appends_activity_entry -v`
Expected: FAIL — `EditingSession.__init__` does not accept `activity_log`.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/agent/session.py`:

1. Add import:

```python
from ave.agent.activity import ActivityLog
```

2. Extend constructor:

```python
def __init__(
    self,
    snapshot_manager: SnapshotManager | None = None,
    transition_graph: ToolTransitionGraph | None = None,
    plugin_dirs: list[Path] | None = None,
    skill_dirs: list[Path] | None = None,
    activity_log: ActivityLog | None = None,
) -> None:
    # ... existing init ...
    self._activity_log = activity_log
```

3. After the successful `self._history.append(...)` and before `return result` in `call_tool`, add:

```python
if self._activity_log is not None:
    snap_id = ""
    if self._snapshot_manager is not None:
        snaps = self._snapshot_manager.list_snapshots()
        if snaps:
            snap_id = snaps[-1].snapshot_id
    self._activity_log.append(
        agent_id=(agent_role.name if agent_role else "anonymous"),
        tool_name=tool_name,
        summary=self._build_log_summary(tool_name, params, result),
        snapshot_id=snap_id,
    )
```

4. Add helper method:

```python
def _build_log_summary(self, tool_name: str, params: dict, result: Any) -> str:
    """Build a terse one-line summary for the activity log.

    Default format: tool_name(key1=val1, key2=val2) with values truncated.
    Tools may override by registering a custom summarizer in the future.
    """
    items = []
    for k, v in params.items():
        s = repr(v)
        if len(s) > 30:
            s = s[:27] + "..."
        items.append(f"{k}={s}")
    return f"{tool_name}({', '.join(items)})"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_session_scope.py -v`
Expected: PASS (new tests plus original scope tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/session.py tests/test_agent/test_session_scope.py
git commit -m "feat(session): emit ActivityLog entries on successful tool calls"
```

---

## Task 7: Source-asset immutability enforcement

**Files:**
- Modify: `src/ave/agent/session.py`
- Test: `tests/test_agent/test_session_immutability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent/test_session_immutability.py
"""Source-asset immutability tests."""

import hashlib
from pathlib import Path

import pytest

from ave.agent.domains import Domain
from ave.agent.errors import SourceAssetWriteError
from ave.agent.session import EditingSession


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_call_tool_rejects_path_under_source_media(tmp_path: Path):
    """A tool call whose resolved path falls under assets/media/source/ is rejected."""
    project = tmp_path / "proj"
    source_dir = project / "assets" / "media" / "source"
    source_dir.mkdir(parents=True)
    original = source_dir / "footage.mxf"
    original.write_bytes(b"untouchable")
    pre = _sha(original)

    session = EditingSession(project_root=project)

    def mutate_file(path: str) -> None:
        """Pretend to mutate a file."""
        Path(path).write_bytes(b"tampered")

    session.registry.register(
        "mutate_file",
        mutate_file,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )

    with pytest.raises(SourceAssetWriteError, match="source"):
        session.call_tool("mutate_file", {"path": str(original)})

    # File unchanged
    assert _sha(original) == pre
    assert original.read_bytes() == b"untouchable"


def test_call_tool_allows_paths_outside_source(tmp_path: Path):
    """Paths not under source/ go through normally."""
    project = tmp_path / "proj"
    working_dir = project / "assets" / "media" / "working"
    working_dir.mkdir(parents=True)
    intermediate = working_dir / "intermediate.mxf"
    intermediate.write_bytes(b"editable")

    session = EditingSession(project_root=project)

    def mutate_file(path: str) -> dict:
        """Mutates the file."""
        Path(path).write_bytes(b"changed")
        return {"path": path}

    session.registry.register(
        "mutate_file",
        mutate_file,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )

    session.call_tool("mutate_file", {"path": str(intermediate)})
    assert intermediate.read_bytes() == b"changed"


def test_no_project_root_means_no_immutability_check(tmp_path: Path):
    """Without project_root, the check is skipped (backward compat)."""
    session = EditingSession()  # no project_root
    suspicious = tmp_path / "assets" / "media" / "source" / "file.mxf"
    suspicious.parent.mkdir(parents=True)
    suspicious.write_bytes(b"irrelevant")

    def noop(path: str) -> None:
        """Noop."""

    session.registry.register(
        "noop",
        noop,
        domain="video",
        domains_touched=(Domain.VIDEO,),
    )
    # Should NOT raise because project_root not set
    session.call_tool("noop", {"path": str(suspicious)})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_session_immutability.py -v`
Expected: FAIL — `EditingSession.__init__` does not accept `project_root`.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/agent/session.py`:

1. Add import:

```python
from ave.agent.errors import SourceAssetWriteError
```

2. Extend constructor with `project_root` parameter:

```python
def __init__(
    self,
    snapshot_manager: SnapshotManager | None = None,
    transition_graph: ToolTransitionGraph | None = None,
    plugin_dirs: list[Path] | None = None,
    skill_dirs: list[Path] | None = None,
    activity_log: ActivityLog | None = None,
    project_root: Path | None = None,
) -> None:
    # ... existing init body ...
    self._project_root = Path(project_root).resolve() if project_root else None
```

3. Add helper method:

```python
def _check_source_asset_immutability(self, params: dict) -> None:
    """Reject tool calls whose params include paths under assets/media/source/."""
    if self._project_root is None:
        return
    source_root = (self._project_root / "assets" / "media" / "source").resolve()
    for value in params.values():
        if not isinstance(value, (str, Path)):
            continue
        try:
            candidate = Path(value).resolve()
        except (OSError, ValueError):
            continue
        try:
            candidate.relative_to(source_root)
        except ValueError:
            continue  # not under source root
        raise SourceAssetWriteError(
            f"Tool call would touch source asset: {candidate} "
            f"(source assets under {source_root} are write-protected)"
        )
```

4. Call it early in `call_tool` — after scope check, before the lock:

```python
def call_tool(
    self,
    tool_name: str,
    params: dict,
    *,
    agent_role: AgentRole | None = None,
) -> Any:
    # ... existing scope check ...

    self._check_source_asset_immutability(params)

    with self._lock:
        # ... existing body ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_session_immutability.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/session.py tests/test_agent/test_session_immutability.py
git commit -m "feat(session): enforce source-asset immutability at dispatch"
```

---

## Task 8: Turn checkpoint API on `SnapshotManager`

**Files:**
- Modify: `src/ave/project/snapshots.py`
- Test: `tests/test_project/test_snapshot_turns.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_project/test_snapshot_turns.py
"""Tests for turn-level checkpoints on SnapshotManager."""

from pathlib import Path

import pytest

from ave.project.snapshots import SnapshotManager


def _write_xges(path: Path, content: str) -> None:
    path.write_text(content)


def test_capture_turn_checkpoint_tags_snapshot(tmp_path: Path):
    xges = tmp_path / "project.xges"
    _write_xges(xges, "<xges>pre</xges>")

    mgr = SnapshotManager()
    snap = mgr.capture_turn_checkpoint(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset({"timeline_loaded"}),
    )
    assert snap.label == "turn_checkpoint:turn-001"
    assert snap.turn_id == "turn-001"


def test_rollback_to_turn_restores_xges(tmp_path: Path):
    xges = tmp_path / "project.xges"
    _write_xges(xges, "<xges>pre</xges>")

    mgr = SnapshotManager()
    mgr.capture_turn_checkpoint(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset({"timeline_loaded"}),
    )

    _write_xges(xges, "<xges>post</xges>")
    snap, provs = mgr.rollback_to_turn(turn_id="turn-001", xges_path=xges)
    assert xges.read_text() == "<xges>pre</xges>"
    assert "timeline_loaded" in provs
    assert snap.turn_id == "turn-001"


def test_redo_turn_restores_post_turn_state(tmp_path: Path):
    xges = tmp_path / "project.xges"
    _write_xges(xges, "<xges>pre</xges>")

    mgr = SnapshotManager()
    mgr.capture_turn_checkpoint(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset(),
    )
    _write_xges(xges, "<xges>post</xges>")
    mgr.capture_post_turn(
        xges_path=xges,
        turn_id="turn-001",
        provisions=frozenset({"edited"}),
    )

    # Roll back
    mgr.rollback_to_turn(turn_id="turn-001", xges_path=xges)
    assert xges.read_text() == "<xges>pre</xges>"

    # Redo
    snap, provs = mgr.redo_turn(turn_id="turn-001", xges_path=xges)
    assert xges.read_text() == "<xges>post</xges>"
    assert "edited" in provs


def test_rollback_to_unknown_turn_raises(tmp_path: Path):
    mgr = SnapshotManager()
    with pytest.raises(KeyError, match="turn-nope"):
        mgr.rollback_to_turn(turn_id="turn-nope", xges_path=tmp_path / "p.xges")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_project/test_snapshot_turns.py -v`
Expected: FAIL — `capture_turn_checkpoint` not defined.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/project/snapshots.py`:

1. Add `turn_id: str | None = None` to the `Snapshot` dataclass:

```python
@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    timestamp: float
    label: str
    xges_content: str
    provisions: frozenset[str]
    tool_name: str | None = None
    turn_id: str | None = None
```

2. Add three new methods on `SnapshotManager`:

```python
def capture_turn_checkpoint(
    self,
    xges_path: Path,
    turn_id: str,
    provisions: frozenset[str],
) -> Snapshot:
    """Capture a pre-turn checkpoint (state before the user's turn runs)."""
    while len(self._snapshots) >= self._max_snapshots:
        evicted = self._snapshots.pop(0)
        self._cleanup_persisted(evicted.snapshot_id)

    content = Path(xges_path).read_text()
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        timestamp=time.time(),
        label=f"turn_checkpoint:{turn_id}",
        xges_content=content,
        provisions=frozenset(provisions),
        tool_name=None,
        turn_id=turn_id,
    )
    self._snapshots.append(snap)
    if self._persist_dir is not None:
        (self._persist_dir / f"{snap.snapshot_id}.xges").write_text(content)
    return snap


def capture_post_turn(
    self,
    xges_path: Path,
    turn_id: str,
    provisions: frozenset[str],
) -> Snapshot:
    """Capture the post-turn state (for redo)."""
    while len(self._snapshots) >= self._max_snapshots:
        evicted = self._snapshots.pop(0)
        self._cleanup_persisted(evicted.snapshot_id)

    content = Path(xges_path).read_text()
    snap = Snapshot(
        snapshot_id=str(uuid.uuid4()),
        timestamp=time.time(),
        label=f"post_turn:{turn_id}",
        xges_content=content,
        provisions=frozenset(provisions),
        tool_name=None,
        turn_id=turn_id,
    )
    self._snapshots.append(snap)
    if self._persist_dir is not None:
        (self._persist_dir / f"{snap.snapshot_id}.xges").write_text(content)
    return snap


def rollback_to_turn(
    self,
    turn_id: str,
    xges_path: Path,
) -> tuple[Snapshot, frozenset[str]]:
    """Restore the pre-turn checkpoint (undo)."""
    for s in self._snapshots:
        if s.turn_id == turn_id and s.label.startswith("turn_checkpoint:"):
            Path(xges_path).write_text(s.xges_content)
            return s, s.provisions
    raise KeyError(f"No turn checkpoint for {turn_id!r}")


def redo_turn(
    self,
    turn_id: str,
    xges_path: Path,
) -> tuple[Snapshot, frozenset[str]]:
    """Restore the post-turn checkpoint (redo)."""
    for s in self._snapshots:
        if s.turn_id == turn_id and s.label.startswith("post_turn:"):
            Path(xges_path).write_text(s.xges_content)
            return s, s.provisions
    raise KeyError(f"No post-turn checkpoint for {turn_id!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_project/test_snapshot_turns.py -v`
Expected: PASS (4 tests)

Also confirm no regression:

Run: `python -m pytest tests/test_project/ -v`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/ave/project/snapshots.py tests/test_project/test_snapshot_turns.py
git commit -m "feat(snapshots): add turn-checkpoint API for user-facing undo/redo"
```

---

## Task 9: `EditingSession` turn lifecycle + undo/redo API

**Files:**
- Modify: `src/ave/agent/session.py`
- Test: `tests/test_agent/test_session_turns.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent/test_session_turns.py
"""Turn lifecycle and undo/redo tests for EditingSession."""

from pathlib import Path

import pytest

from ave.agent.domains import Domain
from ave.agent.session import EditingSession, SessionError
from ave.project.snapshots import SnapshotManager


def _setup_session(tmp_path: Path) -> tuple[EditingSession, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges>initial</xges>")

    mgr = SnapshotManager()
    session = EditingSession(snapshot_manager=mgr, project_root=project)
    session.load_project(xges)
    return session, xges


def test_begin_turn_captures_checkpoint(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    snaps = session.snapshot_manager.list_snapshots()
    assert any(s.label == "turn_checkpoint:turn-001" for s in snaps)


def test_end_turn_captures_post_turn(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after-turn</xges>")
    session.end_turn("turn-001")
    snaps = session.snapshot_manager.list_snapshots()
    assert any(s.label == "post_turn:turn-001" for s in snaps)


def test_undo_turn_restores_pre_turn_state(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after-turn</xges>")
    session.end_turn("turn-001")

    session.undo_turn("turn-001")
    assert xges.read_text() == "<xges>initial</xges>"


def test_redo_turn_restores_post_turn_state(tmp_path: Path):
    session, xges = _setup_session(tmp_path)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after-turn</xges>")
    session.end_turn("turn-001")
    session.undo_turn("turn-001")

    session.redo_turn("turn-001")
    assert xges.read_text() == "<xges>after-turn</xges>"


def test_undo_without_project_raises(tmp_path: Path):
    session = EditingSession(snapshot_manager=SnapshotManager())
    with pytest.raises(SessionError, match="project"):
        session.undo_turn("turn-001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_session_turns.py -v`
Expected: FAIL — `EditingSession.begin_turn` not defined.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/agent/session.py` — add methods:

```python
def begin_turn(self, turn_id: str) -> None:
    """Capture a pre-turn checkpoint. Call once when the user's message starts being processed."""
    if self._project_path is None or self._snapshot_manager is None:
        return  # silently no-op when project not loaded
    self._snapshot_manager.capture_turn_checkpoint(
        xges_path=self._project_path,
        turn_id=turn_id,
        provisions=self._state.provisions,
    )


def end_turn(self, turn_id: str) -> None:
    """Capture a post-turn checkpoint for redo. Call once after the agent finishes the turn."""
    if self._project_path is None or self._snapshot_manager is None:
        return
    self._snapshot_manager.capture_post_turn(
        xges_path=self._project_path,
        turn_id=turn_id,
        provisions=self._state.provisions,
    )


def undo_turn(self, turn_id: str) -> None:
    """Roll back to the pre-turn checkpoint identified by turn_id."""
    if self._project_path is None or self._snapshot_manager is None:
        raise SessionError("Cannot undo: no project loaded or no snapshot manager")
    _, provs = self._snapshot_manager.rollback_to_turn(
        turn_id=turn_id,
        xges_path=self._project_path,
    )
    self._state.reset()
    self._state.add(*provs)


def redo_turn(self, turn_id: str) -> None:
    """Re-apply the post-turn checkpoint identified by turn_id."""
    if self._project_path is None or self._snapshot_manager is None:
        raise SessionError("Cannot redo: no project loaded or no snapshot manager")
    _, provs = self._snapshot_manager.redo_turn(
        turn_id=turn_id,
        xges_path=self._project_path,
    )
    self._state.reset()
    self._state.add(*provs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_session_turns.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/session.py tests/test_agent/test_session_turns.py
git commit -m "feat(session): add begin_turn/end_turn/undo_turn/redo_turn lifecycle"
```

---

## Task 10: State-sync summary builder

**Files:**
- Create: `src/ave/agent/state_sync.py`
- Test: `tests/test_agent/test_state_sync.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent/test_state_sync.py
"""Tests for the state-sync summary builder."""

from pathlib import Path

import pytest

from ave.agent.activity import ActivityLog
from ave.agent.session import EditingSession
from ave.agent.state_sync import StateSummary, build_state_summary
from ave.project.snapshots import SnapshotManager


def _setup_session(tmp_path: Path) -> tuple[EditingSession, ActivityLog, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges></xges>")
    log = ActivityLog(persist_path=tmp_path / "activity-log.jsonl")
    session = EditingSession(
        snapshot_manager=SnapshotManager(),
        activity_log=log,
        project_root=project,
    )
    session.load_project(xges)
    return session, log, xges


def test_summary_includes_timeline_loaded_state(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    summary = build_state_summary(session=session, activity_log=log, since_timestamp=0.0)
    assert isinstance(summary, StateSummary)
    assert "timeline_loaded" in summary.state_provisions


def test_summary_includes_recent_activity(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="trim a", snapshot_id="s1")
    log.append(agent_id="colorist", tool_name="apply_lut", summary="apply lut", snapshot_id="s2")

    summary = build_state_summary(session=session, activity_log=log, since_timestamp=0.0)
    assert len(summary.recent_entries) == 2
    assert summary.recent_entries[0].tool_name == "trim_clip"


def test_summary_respects_since_timestamp(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="old", snapshot_id="s1")
    cutoff = log.entries()[-1].timestamp
    log.append(agent_id="editor", tool_name="split_clip", summary="new", snapshot_id="s2")

    summary = build_state_summary(session=session, activity_log=log, since_timestamp=cutoff)
    assert len(summary.recent_entries) == 1
    assert summary.recent_entries[0].tool_name == "split_clip"


def test_render_produces_compact_text_block(tmp_path: Path):
    session, log, xges = _setup_session(tmp_path)
    log.append(agent_id="editor", tool_name="trim_clip", summary="trim a", snapshot_id="s1")

    summary = build_state_summary(session=session, activity_log=log, since_timestamp=0.0)
    text = summary.render()
    assert "STATE SUMMARY" in text
    assert "trim_clip" in text
    assert "editor" in text
    # Compact — target ≤300 tokens (~1200 chars as a rough bound)
    assert len(text) < 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent/test_state_sync.py -v`
Expected: FAIL — `ave.agent.state_sync` module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ave/agent/state_sync.py
"""State-sync protocol — compact summaries injected into agent turns.

The summary is a concise text block that tells the agent what the current
timeline state is and what has happened since its last turn. It's prepended
to the agent's user-message payload at turn start.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ave.agent.activity import ActivityEntry, ActivityLog

if TYPE_CHECKING:
    from ave.agent.session import EditingSession


@dataclass(frozen=True)
class StateSummary:
    """Compact state summary for agent context injection."""

    generated_at: float
    state_provisions: tuple[str, ...]
    recent_entries: tuple[ActivityEntry, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """Render as a compact text block (≤300 tokens target)."""
        ts = datetime.fromtimestamp(self.generated_at, tz=timezone.utc).strftime("%H:%M:%S")
        lines = [f"STATE SUMMARY (as of {ts} UTC):"]
        provisions = ", ".join(sorted(self.state_provisions)) or "none"
        lines.append(f"  Session provisions: {provisions}")
        if self.recent_entries:
            lines.append("  Activity since your last turn:")
            for e in self.recent_entries:
                etime = datetime.fromtimestamp(e.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
                lines.append(f"    - {etime} ({e.agent_id}): {e.summary}")
        else:
            lines.append("  No activity since your last turn.")
        lines.append("")
        lines.append("For full detail, call get_project_state.")
        return "\n".join(lines)


def build_state_summary(
    *,
    session: EditingSession,
    activity_log: ActivityLog,
    since_timestamp: float,
) -> StateSummary:
    """Construct a state summary for the next agent turn."""
    return StateSummary(
        generated_at=time.time(),
        state_provisions=tuple(sorted(session.state.provisions)),
        recent_entries=tuple(activity_log.entries_since(since_timestamp)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent/test_state_sync.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ave/agent/state_sync.py tests/test_agent/test_state_sync.py
git commit -m "feat(agent): add state-sync summary builder for turn-context injection"
```

---

## Task 11: Wire state summary into `ChatSession`

**Files:**
- Modify: `src/ave/web/chat.py`
- Test: `tests/test_web/test_chat_state_sync.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web/test_chat_state_sync.py
"""Tests that ChatSession injects a state summary into each turn."""

import pytest

from ave.agent.activity import ActivityLog
from ave.web.chat import ChatSession


class _FakeOrchestrator:
    """Minimal orchestrator stub that records the last prepended-state-summary."""

    def __init__(self, session, activity_log):
        self.session = session
        self.activity_log = activity_log
        self.last_user_message: str | None = None

    async def run_turn(self, user_message: str, ws) -> None:
        self.last_user_message = user_message


class _FakeTimeline:
    pass


@pytest.mark.asyncio
async def test_chat_injects_state_summary_prefix(tmp_path):
    from ave.agent.session import EditingSession
    from ave.project.snapshots import SnapshotManager

    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges></xges>")

    log = ActivityLog(persist_path=tmp_path / "log.jsonl")
    ave_session = EditingSession(
        snapshot_manager=SnapshotManager(),
        activity_log=log,
        project_root=project,
    )
    ave_session.load_project(xges)
    log.append(agent_id="editor", tool_name="trim_clip", summary="trim a", snapshot_id="s1")

    orch = _FakeOrchestrator(session=ave_session, activity_log=log)
    chat = ChatSession(orchestrator=orch, timeline_model=_FakeTimeline())

    class _FakeWS:
        async def send_json(self, *args, **kwargs):
            pass

    await chat.handle_message(_FakeWS(), "trim the start of clip-1 by 2s")

    # The orchestrator should have received a message that starts with the summary
    assert orch.last_user_message is not None
    assert orch.last_user_message.startswith("STATE SUMMARY")
    assert "trim_clip" in orch.last_user_message
    # Original user text preserved at the end
    assert "trim the start of clip-1" in orch.last_user_message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web/test_chat_state_sync.py -v`
Expected: FAIL — current `ChatSession` calls a real Anthropic client; `run_turn` on the fake isn't invoked, or the raw text isn't prefixed.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/web/chat.py`:

1. Add import at the top:

```python
from ave.agent.state_sync import build_state_summary
```

2. Modify `ChatSession.__init__` to keep a `_last_summary_timestamp: float = 0.0`:

```python
def __init__(self, orchestrator, timeline_model) -> None:
    self._orchestrator = orchestrator
    self._timeline = timeline_model
    self._messages: list[dict] = []
    self._processing = False
    self._cancel_event = asyncio.Event()
    self._last_summary_timestamp: float = 0.0
```

3. Modify `handle_message` to build a state summary and prepend it to the user message, then advance `_last_summary_timestamp`:

```python
async def handle_message(self, ws, text: str) -> None:
    if self._processing:
        await ws.send_json(format_busy())
        return
    self._processing = True
    self._cancel_event.clear()

    try:
        ave_session = getattr(self._orchestrator, "session", None)
        activity_log = getattr(self._orchestrator, "activity_log", None)
        if ave_session is not None and activity_log is not None:
            summary = build_state_summary(
                session=ave_session,
                activity_log=activity_log,
                since_timestamp=self._last_summary_timestamp,
            )
            prefixed = summary.render() + "\n\n" + text
            self._last_summary_timestamp = summary.generated_at
        else:
            prefixed = text

        await self._orchestrator.run_turn(prefixed, ws)
    finally:
        self._processing = False
```

(If the existing `handle_message` body is more complex than this stub, preserve the existing behavior and only add the prefix build + pass-through. The test's `_FakeOrchestrator` has a `run_turn(user_message, ws)` method matching the signature we call.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web/test_chat_state_sync.py -v`
Expected: PASS

Run full web test suite to catch any regression:

Run: `python -m pytest tests/test_web/ -v`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/chat.py tests/test_web/test_chat_state_sync.py
git commit -m "feat(chat): inject state-sync summary prefix into every user turn"
```

---

## Task 12: Undo/redo pure handlers + WebSocket rollback event

**Files:**
- Modify: `src/ave/web/api.py` (add pure-function handlers — matches existing pure-function style)
- Modify: `src/ave/web/chat.py` (add `format_timeline_rollback`, parse `undo`/`redo` client messages)
- Modify: `src/ave/web/app.py` (wire new message types into `_handle_chat_ws`)
- Test: `tests/test_web/test_undo_redo_api.py`

**Architectural note:** Sessions are per-WebSocket in `web/app.py`'s `sessions` dict. Rather than add stateful HTTP endpoints that would need a session token, pipe undo/redo through the existing WebSocket. The pure-function handlers in `web/api.py` encapsulate the logic; `web/app.py` just calls them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web/test_undo_redo_api.py
"""Tests for pure undo/redo handlers and the rollback WS event."""

from pathlib import Path

import pytest

from ave.agent.activity import ActivityLog
from ave.agent.session import EditingSession
from ave.project.snapshots import SnapshotManager
from ave.web.api import redo_response, undo_response
from ave.web.chat import format_timeline_rollback


def _setup_session_with_completed_turn(tmp_path: Path) -> tuple[EditingSession, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    xges = project / "project.xges"
    xges.write_text("<xges>initial</xges>")

    session = EditingSession(
        snapshot_manager=SnapshotManager(),
        activity_log=ActivityLog(persist_path=tmp_path / "log.jsonl"),
        project_root=project,
    )
    session.load_project(xges)
    session.begin_turn("turn-001")
    xges.write_text("<xges>after</xges>")
    session.end_turn("turn-001")
    return session, xges


def test_undo_response_ok(tmp_path: Path):
    session, xges = _setup_session_with_completed_turn(tmp_path)
    status, body = undo_response(session, "turn-001")
    assert status == 200
    assert body == {"ok": True, "turn_id": "turn-001", "direction": "undo"}
    assert xges.read_text() == "<xges>initial</xges>"


def test_redo_response_ok(tmp_path: Path):
    session, xges = _setup_session_with_completed_turn(tmp_path)
    undo_response(session, "turn-001")
    status, body = redo_response(session, "turn-001")
    assert status == 200
    assert body == {"ok": True, "turn_id": "turn-001", "direction": "redo"}
    assert xges.read_text() == "<xges>after</xges>"


def test_undo_response_missing_turn_id(tmp_path: Path):
    session, _ = _setup_session_with_completed_turn(tmp_path)
    status, body = undo_response(session, "")
    assert status == 400
    assert body["ok"] is False


def test_undo_response_unknown_turn(tmp_path: Path):
    session, _ = _setup_session_with_completed_turn(tmp_path)
    status, body = undo_response(session, "ghost")
    assert status == 404
    assert body["ok"] is False


def test_redo_response_unknown_turn(tmp_path: Path):
    session, _ = _setup_session_with_completed_turn(tmp_path)
    status, body = redo_response(session, "ghost")
    assert status == 404


def test_format_timeline_rollback_event():
    evt = format_timeline_rollback(turn_id="turn-001", direction="undo")
    assert evt == {
        "type": "timeline_rollback",
        "turn_id": "turn-001",
        "direction": "undo",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web/test_undo_redo_api.py -v`
Expected: FAIL — `undo_response`, `redo_response`, and `format_timeline_rollback` don't exist yet.

- [ ] **Step 3: Write minimal implementation**

Edit `src/ave/web/chat.py` — add near the other `format_*` helpers:

```python
def format_timeline_rollback(*, turn_id: str, direction: str) -> dict:
    """Format a state-rollback notification for the client.

    direction: "undo" or "redo"
    """
    return {"type": "timeline_rollback", "turn_id": turn_id, "direction": direction}
```

Edit `src/ave/web/api.py` — add pure handlers:

```python
# Add imports at the top:
from ave.agent.session import EditingSession, SessionError


def undo_response(session: EditingSession, turn_id: str) -> tuple[int, dict]:
    """Pure handler: execute undo and return (status_code, response_body)."""
    if not turn_id:
        return 400, {"ok": False, "error": "missing turn_id"}
    try:
        session.undo_turn(turn_id)
    except KeyError:
        return 404, {"ok": False, "error": "unknown turn"}
    except SessionError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "turn_id": turn_id, "direction": "undo"}


def redo_response(session: EditingSession, turn_id: str) -> tuple[int, dict]:
    """Pure handler: execute redo and return (status_code, response_body)."""
    if not turn_id:
        return 400, {"ok": False, "error": "missing turn_id"}
    try:
        session.redo_turn(turn_id)
    except KeyError:
        return 404, {"ok": False, "error": "unknown turn"}
    except SessionError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "turn_id": turn_id, "direction": "redo"}
```

Edit `src/ave/web/app.py` — in the `async for msg in ws` loop inside `_handle_chat_ws`, extend the dispatch. After the existing `elif parsed.get("type") == "cancel":` block, insert:

```python
elif parsed.get("type") in ("undo", "redo"):
    from ave.web.api import undo_response, redo_response
    from ave.web.chat import format_timeline_rollback

    if chat_session is None:
        await ws.send_json(format_error("Agent not available"))
        continue

    editing_session = chat_session._orchestrator.session  # type: ignore[attr-defined]
    turn_id = parsed.get("turn_id", "")
    handler = undo_response if parsed["type"] == "undo" else redo_response
    status, body = handler(editing_session, turn_id)
    await ws.send_json(body)
    if status == 200:
        await ws.send_json(
            format_timeline_rollback(turn_id=turn_id, direction=parsed["type"])
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web/test_undo_redo_api.py -v`
Expected: PASS (6 tests)

Run full web tests to confirm no regression:

Run: `python -m pytest tests/test_web/ -v`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/api.py src/ave/web/chat.py src/ave/web/app.py tests/test_web/test_undo_redo_api.py
git commit -m "feat(web): add undo/redo pure handlers + timeline_rollback WS event"
```

---

## Task 13: Undo/redo UI buttons, keyboard shortcuts, and rollback markers

**Files:**
- Modify: `src/ave/web/client/index.html`
- Modify: `src/ave/web/client/chat.js`
- Modify: `src/ave/web/client/styles.css`

This task is UI-only and has no Python tests. Verification is manual browser testing. Follow the CLAUDE.md guidance: start the dev server and use the feature in a browser before reporting complete.

- [ ] **Step 1: Add markup for undo/redo buttons**

Edit `src/ave/web/client/index.html`. Find the chat header region (likely a `<div class="chat-header">` or similar near the top of the chat panel). Add:

```html
<div class="undo-controls">
  <button id="undo-btn" title="Undo last turn (⌘Z)" disabled>↶ Undo</button>
  <button id="redo-btn" title="Redo turn (⌘⇧Z)" disabled>↷ Redo</button>
</div>
```

- [ ] **Step 2: Add keyboard + button handlers in `chat.js`**

**Architectural note:** Undo/redo travel over the existing WebSocket as `{"type": "undo" | "redo", "turn_id": "..."}` (matching Task 12). Client tracks turn IDs locally: every successful `done` event pushes a turn onto `turnStack`; Undo pops it and sends the WS message; Redo works in reverse.

Open `src/ave/web/client/chat.js` and locate the existing WS handler (the function that processes incoming JSON messages, likely near the top). Add module-level state:

```javascript
const turnStack = [];         // stack of turn_ids that completed successfully
let redoableTurnIds = [];     // turns that have been undone and can be redone
let lastTurnId = null;        // ID of the in-flight turn (null if idle)
```

Find the existing WS-message dispatch (if the file uses `ws.onmessage`, you'll see a JSON.parse plus type-based dispatch). Add branches for `done` and `timeline_rollback`:

```javascript
// Inside the existing ws.onmessage handler / dispatch:
if (msg.type === "done") {
  if (lastTurnId) {
    turnStack.push(lastTurnId);
    redoableTurnIds = [];
    lastTurnId = null;
    refreshUndoRedoButtons();
  }
}
if (msg.type === "timeline_rollback") {
  renderRollbackMarker(msg.turn_id, msg.direction);
  window.dispatchEvent(new CustomEvent("timeline-stale"));
}
```

Add helper:

```javascript
function refreshUndoRedoButtons() {
  const undoBtn = document.getElementById("undo-btn");
  const redoBtn = document.getElementById("redo-btn");
  if (undoBtn) undoBtn.disabled = turnStack.length === 0;
  if (redoBtn) redoBtn.disabled = redoableTurnIds.length === 0;
}
```

Find the existing function that sends the user's chat message over the WS. The server expects `{"type": "message", "text": "..."}`. Modify the send to generate and remember a turn ID:

```javascript
// Before ws.send(JSON.stringify({ type: "message", text }))
lastTurnId = "turn-" + (crypto.randomUUID ? crypto.randomUUID() : Date.now());
ws.send(JSON.stringify({ type: "message", text, turn_id: lastTurnId }));
```

(The `turn_id` field is currently ignored by the server but is harmless extra data. A future change can have the server echo it back in `done`.)

Button + keyboard handlers (add near the bottom of the file, guarded by `DOMContentLoaded` if that's the existing pattern):

```javascript
function wireUndoRedoControls(ws) {
  const undoBtn = document.getElementById("undo-btn");
  const redoBtn = document.getElementById("redo-btn");
  if (!undoBtn || !redoBtn) return;

  undoBtn.addEventListener("click", () => {
    const turnId = turnStack.pop();
    if (!turnId) return;
    redoableTurnIds.push(turnId);
    ws.send(JSON.stringify({ type: "undo", turn_id: turnId }));
    refreshUndoRedoButtons();
  });

  redoBtn.addEventListener("click", () => {
    const turnId = redoableTurnIds.pop();
    if (!turnId) return;
    turnStack.push(turnId);
    ws.send(JSON.stringify({ type: "redo", turn_id: turnId }));
    refreshUndoRedoButtons();
  });

  document.addEventListener("keydown", (e) => {
    const mod = navigator.platform.includes("Mac") ? e.metaKey : e.ctrlKey;
    if (mod && !e.shiftKey && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (!undoBtn.disabled) undoBtn.click();
    }
    if (mod && e.shiftKey && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (!redoBtn.disabled) redoBtn.click();
    }
  });
}
```

Call `wireUndoRedoControls(ws)` once the WebSocket is established. Look for the existing WS setup (likely in a function that opens the connection and assigns handlers); call `wireUndoRedoControls(ws)` immediately after the WebSocket is assigned.

Rollback marker rendering:

```javascript
function renderRollbackMarker(turnId, direction) {
  const chatLog = document.querySelector(".chat-log") || document.getElementById("chat-log");
  if (!chatLog) return;
  const marker = document.createElement("div");
  marker.className = "chat-marker rollback";
  marker.textContent = (direction === "undo" ? "↶ undone: " : "↷ redone: ") + turnId;
  chatLog.appendChild(marker);
  chatLog.scrollTop = chatLog.scrollHeight;
}
```

- [ ] **Step 3: Add styling for rollback markers**

Edit `src/ave/web/client/styles.css`, append:

```css
.undo-controls {
  display: inline-flex;
  gap: 4px;
  margin-left: 8px;
}
.undo-controls button {
  font-size: 0.85em;
  padding: 2px 8px;
  cursor: pointer;
}
.undo-controls button:disabled {
  opacity: 0.4;
  cursor: default;
}
.chat-marker.rollback {
  font-size: 0.85em;
  color: #888;
  font-style: italic;
  padding: 4px 8px;
  border-left: 2px solid #bbb;
  margin: 4px 0;
}
```

- [ ] **Step 4: Manual browser verification**

Start the dev server:

```bash
python -m ave.web.app --project-dir /tmp/test-project --port 8765
```

Open `http://localhost:8765` in a browser. Verify:
- Undo/Redo buttons appear in the chat header and are disabled initially.
- Sending a chat message then pressing Undo rolls the timeline back (preview refreshes), the rollback marker appears in the chat log, the Undo button disables, the Redo button enables.
- Pressing Redo re-applies and flips the button states again.
- ⌘Z / ⌘⇧Z shortcuts work identically.
- Sending a new message after Undo clears the Redo stack (Redo button disables).

Capture any unexpected behavior and fix inline.

- [ ] **Step 5: Commit**

```bash
git add src/ave/web/client/index.html src/ave/web/client/chat.js src/ave/web/client/styles.css
git commit -m "feat(web-ui): add undo/redo buttons, keyboard shortcuts, and rollback markers"
```

---

## Phase 1 Completion Verification

Once all 13 tasks are committed, run the full test suite:

```bash
python -m pytest tests/test_agent/ tests/test_project/ tests/test_web/ -v
```

Expected: all new tests pass, no regressions in the existing ~1100 tests.

Then manually verify in browser (once more):
- Open project → send chat message → observe Undo button enables → click Undo → state rolls back + marker appears → click Redo → state restored.
- Stop and restart the server; verify `.ave/activity-log.jsonl` is persisted and re-loaded correctly by inspecting the file.

When all above is green, Phase 1 is complete. Phase 2 (Harness Rung A) follows as a separate plan.
