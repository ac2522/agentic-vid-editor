"""Transition tools — parameter computation for timeline transitions.

Pure logic layer: no GES dependency.
"""

from dataclasses import dataclass
from enum import Enum


class TransitionError(Exception):
    """Raised when transition parameter validation fails."""


class TransitionType(Enum):
    """Supported transition types."""

    CROSSFADE = "crossfade"
    FADE_TO_BLACK = "fade-to-black"
    WIPE_LEFT = "wipe-left"
    WIPE_RIGHT = "wipe-right"
    WIPE_UP = "wipe-up"
    WIPE_DOWN = "wipe-down"


@dataclass(frozen=True)
class TransitionParams:
    """Computed transition parameters."""

    type: TransitionType
    duration_ns: int
    overlap_start_ns: int
    clip_a_new_end_ns: int
    clip_b_new_start_ns: int


def compute_transition(
    clip_a_end_ns: int,
    clip_b_start_ns: int,
    transition_type: TransitionType,
    duration_ns: int,
) -> TransitionParams:
    """Validate and compute transition parameters.

    Clips must be adjacent (clip_a ends where clip_b starts).
    The transition creates an overlap by moving clip_b earlier.

    Args:
        clip_a_end_ns: End position of clip A on timeline.
        clip_b_start_ns: Start position of clip B on timeline.
        transition_type: Type of transition effect.
        duration_ns: Duration of the transition in nanoseconds.

    Returns:
        TransitionParams with computed overlap positions.

    Raises:
        TransitionError: If parameters are invalid.
    """
    if duration_ns <= 0:
        raise TransitionError(f"Transition duration must be positive, got {duration_ns}")

    if clip_b_start_ns < clip_a_end_ns:
        raise TransitionError(
            f"Clips already overlap: clip_a ends at {clip_a_end_ns}, "
            f"clip_b starts at {clip_b_start_ns}"
        )

    # Tolerate tiny gaps (< 1ms) from nanosecond rounding at frame boundaries
    _ADJACENCY_TOLERANCE_NS = 1_000_000  # 1ms
    gap = clip_b_start_ns - clip_a_end_ns
    if gap > _ADJACENCY_TOLERANCE_NS:
        raise TransitionError(
            f"Clips must be adjacent for transition. Gap of {gap}ns between "
            f"clip_a (ends {clip_a_end_ns}) and clip_b (starts {clip_b_start_ns})"
        )

    # For a transition, clip_b moves earlier by the full transition
    # duration, creating an overlap region where both clips are visible.
    overlap_start = clip_a_end_ns - duration_ns

    if overlap_start < 0:
        raise TransitionError(f"Transition duration ({duration_ns}ns) exceeds available clip range")

    return TransitionParams(
        type=transition_type,
        duration_ns=duration_ns,
        overlap_start_ns=overlap_start,
        clip_a_new_end_ns=clip_a_end_ns,
        clip_b_new_start_ns=overlap_start,
    )
