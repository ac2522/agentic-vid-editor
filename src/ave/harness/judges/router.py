"""Dimension-type router for the judge ensemble."""

from __future__ import annotations

from typing import Sequence

from ave.harness.judges._protocol import DimensionType, JudgeBackend


_STATIC = frozenset({
    "duration", "resolution", "aspect_ratio", "audio_rms",
    "format", "bitrate", "framerate",
})
_STILL = frozenset({
    "framing", "speaker_framing", "caption_legibility", "text_readability",
    "color_palette", "visual_balance", "composition", "subject_centered",
})
_TEMPORAL = frozenset({
    "pacing", "motion_blur", "audio_continuity", "content_preservation",
    "animation_smoothness", "cut_rhythm", "flicker", "temporal_stability",
    "film_grain_evolution",
})


def classify_dimension(dimension: str) -> DimensionType:
    """Return 'static' | 'still' | 'temporal'. Unknown dimensions default to 'still'."""
    if dimension in _STATIC:
        return "static"
    if dimension in _TEMPORAL:
        return "temporal"
    return "still"


def select_judges(
    judges: Sequence[JudgeBackend],
    *,
    dimension_type: DimensionType,
) -> list[JudgeBackend]:
    """Return judges that support a given dimension type."""
    return [j for j in judges if dimension_type in j.supported_dimension_types]
