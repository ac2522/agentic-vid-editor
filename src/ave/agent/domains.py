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
