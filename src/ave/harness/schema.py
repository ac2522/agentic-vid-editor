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
    model_config = ConfigDict(frozen=True, extra="forbid")


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


class MinMax(_Frozen):
    min: float | None = None
    max: float | None = None

    @field_validator("min", "max", mode="before")
    @classmethod
    def _coerce_float(cls, v):
        if v is None:
            return None
        return float(v)


class TimelineBounds(_Frozen):
    clip_count: MinMax = Field(default_factory=MinMax)
    duration_seconds: MinMax = Field(default_factory=MinMax)
    effects_applied: tuple[str, ...] = ()
    effects_forbidden: tuple[str, ...] = ()

    @field_validator("clip_count", "duration_seconds", mode="before")
    @classmethod
    def _coerce_minmax(cls, v):
        if v is None:
            return MinMax()
        return v

    @field_validator("effects_applied", "effects_forbidden", mode="before")
    @classmethod
    def _coerce_tuple(cls, v):
        if v is None:
            return ()
        return tuple(v)


class ExecuteExpected(_Frozen):
    timeline: TimelineBounds = Field(default_factory=TimelineBounds)
    snapshots_created: MinMax = Field(default_factory=MinMax)
    activity_log_entries: MinMax = Field(default_factory=MinMax)

    @field_validator("timeline", mode="before")
    @classmethod
    def _coerce_timeline(cls, v):
        if v is None:
            return TimelineBounds()
        return v

    @field_validator("snapshots_created", "activity_log_entries", mode="before")
    @classmethod
    def _coerce_minmax(cls, v):
        if v is None:
            return MinMax()
        return v


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
