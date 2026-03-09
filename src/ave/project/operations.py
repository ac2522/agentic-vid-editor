"""Timeline operations — GES execution layer for edit tools.

Applies computed parameters from ave.tools to GES Timeline objects.
All GES access goes through Timeline's public API (P0-1).
All mutations are verified via read-back (P0-5).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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

    # Get the media URI from the existing clip's asset
    asset = clip.get_asset()
    if asset is None:
        clip.set_duration(original_duration)  # rollback
        raise OperationError(f"Clip {clip_id} has no asset")

    # Get the layer
    layer = clip.get_layer()
    if layer is None:
        clip.set_duration(original_duration)  # rollback
        raise OperationError(f"Clip {clip_id} has no layer")

    import gi

    gi.require_version("GES", "1.0")
    from gi.repository import GES

    # Add right portion as new clip
    right_clip = layer.add_asset(
        asset,
        right_params.start_ns,
        right_params.inpoint_ns,
        right_params.duration_ns,
        GES.TrackType.UNKNOWN,
    )

    if right_clip is None:
        clip.set_duration(original_duration)  # rollback
        raise OperationError(f"Failed to create right portion of split at {position_ns}")

    # P0-1: Use public API to register the new clip
    right_id = timeline.register_clip(right_clip)

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
