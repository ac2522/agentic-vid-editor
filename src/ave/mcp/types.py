"""MCP response data models for AVE."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EditResult:
    success: bool
    description: str
    tools_used: list[str] = field(default_factory=list)
    preview_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProjectState:
    clip_count: int
    duration_ns: int
    layers: int
    clips: list[dict] = field(default_factory=list)
    effects: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreviewResult:
    path: str
    format: str
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class AssetInfo:
    asset_id: str
    path: str
    codec: str
    width: int
    height: int
    duration_ns: int
    color_space: str | None = None
    frame_rate: float | None = None
