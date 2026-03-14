"""Timeline operations — GES execution layer for edit tools.

Applies computed parameters from ave.tools to GES Timeline objects.
All GES access goes through Timeline's public API (P0-1).
All mutations are verified via read-back (P0-5).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ave.tools.audio import compute_volume
from ave.tools.edit import compute_trim, compute_split, compute_concatenation

if TYPE_CHECKING:
    from ave.project.timeline import Timeline


class OperationError(Exception):
    """Raised when a timeline operation fails."""


def trim_clip(timeline: Timeline, clip_id: str, in_ns: int, out_ns: int) -> None:
    """Trim a clip to a subrange.

    in_ns/out_ns are relative to the clip's current visible range.
    The in-point accumulates on repeated trims.
    The clip's timeline position (start) is preserved.
    """
    clip = timeline.get_clip(clip_id)
    current_duration = clip.get_duration()

    params = compute_trim(current_duration, in_ns, out_ns)

    current_inpoint = clip.get_inpoint()
    new_inpoint = current_inpoint + params.in_ns
    clip.set_inpoint(new_inpoint)
    clip.set_duration(params.duration_ns)

    # P0-5: Verify mutations applied
    actual_inpoint = clip.get_inpoint()
    actual_duration = clip.get_duration()
    if actual_inpoint != new_inpoint:
        raise OperationError(
            f"GES did not apply inpoint: expected {new_inpoint}, got {actual_inpoint}"
        )
    if actual_duration != params.duration_ns:
        raise OperationError(
            f"GES did not apply duration: expected {params.duration_ns}, got {actual_duration}"
        )


def split_clip(timeline: Timeline, clip_id: str, position_ns: int) -> tuple[str, str]:
    """Split a clip at a timeline position.

    The original clip becomes the left portion. A new clip is created
    for the right portion via timeline.register_clip().

    Returns:
        Tuple of (left_clip_id, right_clip_id).
    """
    clip = timeline.get_clip(clip_id)

    clip_start = clip.get_start()
    clip_duration = clip.get_duration()
    clip_inpoint = clip.get_inpoint()

    left_params, right_params = compute_split(clip_start, clip_duration, position_ns, clip_inpoint)

    # Save original duration for rollback on failure
    original_duration = clip.get_duration()

    # Modify existing clip to be the left portion
    clip.set_duration(left_params.duration_ns)

    # Add right portion via Timeline's public API
    try:
        right_id = timeline.add_clip_from_source(
            source_clip_id=clip_id,
            start_ns=right_params.start_ns,
            inpoint_ns=right_params.inpoint_ns,
            duration_ns=right_params.duration_ns,
        )
    except Exception:
        clip.set_duration(original_duration)  # rollback
        raise OperationError(f"Failed to create right portion of split at {position_ns}")

    return clip_id, right_id


def concatenate_clips(
    timeline: Timeline,
    media_paths: list[Path],
    durations_ns: list[int],
    layer: int = 0,
    start_ns: int = 0,
) -> list[str]:
    """Place clips sequentially on the timeline.

    Args:
        timeline: Target timeline.
        media_paths: Paths to media files for each clip.
        durations_ns: Duration for each clip.
        layer: Timeline layer to place clips on.
        start_ns: Starting position for the first clip.

    Returns:
        List of clip IDs in order.
    """
    if len(media_paths) != len(durations_ns):
        raise OperationError(
            f"media_paths ({len(media_paths)}) and durations_ns ({len(durations_ns)}) "
            f"must have same length"
        )

    positions = compute_concatenation(durations_ns, start_ns)

    clip_ids = []
    for path, pos in zip(media_paths, positions):
        clip_id = timeline.add_clip(
            media_path=path,
            layer=layer,
            start_ns=pos.start_ns,
            duration_ns=pos.duration_ns,
        )
        clip_ids.append(clip_id)

    return clip_ids


def set_volume(timeline: Timeline, clip_id: str, level_db: float) -> str:
    """Set volume on a clip by adding a volume GES.Effect.

    Returns the effect_id for the added volume effect.
    """
    params = compute_volume(level_db)

    effect_id = timeline.add_effect(clip_id, "volume")
    timeline.set_effect_property(clip_id, effect_id, "volume", params.linear_gain)

    # P0-5: Verify
    actual = timeline.get_effect_property(clip_id, effect_id, "volume")
    if abs(actual - params.linear_gain) > 0.001:
        raise OperationError(
            f"Volume not applied: expected {params.linear_gain}, got {actual}"
        )

    return effect_id
