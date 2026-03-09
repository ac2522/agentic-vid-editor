"""Edit tools — trim, split, concatenate parameter computation.

Pure logic layer: no GES dependency. Computes parameters that the GES
execution layer applies to the timeline.
"""

from dataclasses import dataclass


class EditError(Exception):
    """Raised when edit parameter validation fails."""


@dataclass(frozen=True)
class TrimParams:
    """Computed trim parameters."""

    in_ns: int
    out_ns: int
    duration_ns: int


@dataclass(frozen=True)
class SplitResult:
    """Result for one side of a split."""

    start_ns: int
    duration_ns: int
    inpoint_ns: int


@dataclass(frozen=True)
class ClipPosition:
    """Position for a clip in a concatenation."""

    start_ns: int
    duration_ns: int


def compute_trim(
    clip_duration_ns: int,
    in_ns: int,
    out_ns: int,
) -> TrimParams:
    """Validate and compute trim parameters.

    Args:
        clip_duration_ns: Current clip duration in nanoseconds.
        in_ns: New in-point (relative to clip start).
        out_ns: New out-point (relative to clip start).

    Returns:
        TrimParams with validated in/out/duration.

    Raises:
        EditError: If parameters are invalid.
    """
    if in_ns < 0 or out_ns < 0:
        raise EditError(f"Negative values not allowed: in_ns={in_ns}, out_ns={out_ns}")

    if in_ns >= out_ns:
        if in_ns == out_ns:
            raise EditError(f"Trim would result in zero duration: in_ns={in_ns}, out_ns={out_ns}")
        raise EditError(f"in_ns ({in_ns}) must be before out_ns ({out_ns})")

    if in_ns > clip_duration_ns:
        raise EditError(f"in_ns ({in_ns}) exceeds clip duration ({clip_duration_ns})")

    if out_ns > clip_duration_ns:
        raise EditError(f"out_ns ({out_ns}) exceeds clip duration ({clip_duration_ns})")

    return TrimParams(
        in_ns=in_ns,
        out_ns=out_ns,
        duration_ns=out_ns - in_ns,
    )


def compute_split(
    clip_start_ns: int,
    clip_duration_ns: int,
    split_position_ns: int,
    inpoint_ns: int = 0,
) -> tuple[SplitResult, SplitResult]:
    """Compute parameters for splitting a clip at a position.

    Args:
        clip_start_ns: Timeline position where clip starts.
        clip_duration_ns: Duration of the clip.
        split_position_ns: Timeline position to split at.
        inpoint_ns: Current in-point of the clip (media offset).

    Returns:
        Tuple of (left_result, right_result).

    Raises:
        EditError: If split position is invalid.
    """
    clip_end_ns = clip_start_ns + clip_duration_ns

    if split_position_ns <= clip_start_ns:
        if split_position_ns == clip_start_ns:
            raise EditError("Cannot split at the very start of clip")
        raise EditError(
            f"Split position ({split_position_ns}) is outside clip [{clip_start_ns}, {clip_end_ns})"
        )

    if split_position_ns >= clip_end_ns:
        if split_position_ns == clip_end_ns:
            raise EditError("Cannot split at the very end of clip")
        raise EditError(
            f"Split position ({split_position_ns}) is outside clip [{clip_start_ns}, {clip_end_ns})"
        )

    left_duration = split_position_ns - clip_start_ns
    right_duration = clip_end_ns - split_position_ns

    left = SplitResult(
        start_ns=clip_start_ns,
        duration_ns=left_duration,
        inpoint_ns=inpoint_ns,
    )

    right = SplitResult(
        start_ns=split_position_ns,
        duration_ns=right_duration,
        inpoint_ns=inpoint_ns + left_duration,
    )

    return left, right


def compute_concatenation(
    durations_ns: list[int],
    start_ns: int = 0,
) -> list[ClipPosition]:
    """Compute sequential positions for concatenating clips.

    Args:
        durations_ns: Duration of each clip in nanoseconds.
        start_ns: Timeline position for the first clip.

    Returns:
        List of ClipPosition with computed start positions.

    Raises:
        EditError: If the list is empty or contains zero durations.
    """
    if not durations_ns:
        raise EditError("Cannot concatenate empty list of clips")

    for i, d in enumerate(durations_ns):
        if d <= 0:
            raise EditError(f"Clip {i} has zero or negative duration: {d}")

    positions = []
    current = start_ns
    for d in durations_ns:
        positions.append(ClipPosition(start_ns=current, duration_ns=d))
        current += d

    return positions
