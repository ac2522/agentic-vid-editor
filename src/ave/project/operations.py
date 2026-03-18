"""Timeline operations — GES execution layer for edit tools.

Applies computed parameters from ave.tools to GES Timeline objects.
All GES access goes through Timeline's public API (P0-1).
All mutations are verified via read-back (P0-5).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ave.tools.audio import compute_volume, compute_fade
from ave.tools.edit import compute_trim, compute_split, compute_concatenation
from ave.tools.speed import compute_speed_change
from ave.tools.transitions import TransitionType, compute_transition

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
        raise OperationError(f"Volume not applied: expected {params.linear_gain}, got {actual}")

    return effect_id


def apply_fade(
    timeline: Timeline,
    clip_id: str,
    fade_in_ns: int,
    fade_out_ns: int,
) -> str:
    """Apply audio fade-in and/or fade-out to a clip.

    Creates a volume effect with keyframed control source.
    GstController.DirectControlBinding normalizes 0.0-1.0 to the element's
    property range (volume: 0.0-10.0), so unity gain = 0.1 in control source.

    Returns the effect_id for the added volume effect.
    """
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstController", "1.0")
    from gi.repository import GstController

    clip = timeline.get_clip(clip_id)
    clip_duration = clip.get_duration()

    params = compute_fade(clip_duration, fade_in_ns, fade_out_ns)

    # Add volume effect
    effect_id = timeline.add_effect(clip_id, "volume")

    # Get the GES.Effect object to attach control binding
    effect = timeline._get_effect(clip_id, effect_id)

    # Create interpolation control source
    cs = GstController.InterpolationControlSource()
    cs.set_property("mode", GstController.InterpolationMode.LINEAR)

    cb = GstController.DirectControlBinding.new(effect, "volume", cs)
    effect.add_control_binding(cb)

    # Unity gain in DirectControlBinding normalized space:
    # volume range is [0.0, 10.0], so 1.0 linear = 0.1 in control source
    UNITY = 0.1
    SILENCE = 0.0

    # Set keyframes
    if params.fade_in_ns > 0:
        cs.set(0, SILENCE)
        cs.set(params.fade_in_ns, UNITY)
    else:
        cs.set(0, UNITY)

    if params.fade_out_ns > 0:
        fade_out_start = clip_duration - params.fade_out_ns
        cs.set(fade_out_start, UNITY)
        cs.set(clip_duration, SILENCE)
    elif params.fade_in_ns == 0:
        cs.set(clip_duration, UNITY)

    return effect_id


def set_speed(
    timeline: Timeline,
    clip_id: str,
    rate: float,
    preserve_pitch: bool = True,
) -> None:
    """Set playback speed on a clip.

    Adds a pitch GStreamer effect for audio speed control and adjusts
    clip duration to match the new rate.
    """
    clip = timeline.get_clip(clip_id)
    current_duration = clip.get_duration()

    params = compute_speed_change(current_duration, rate, preserve_pitch)

    # Add pitch effect for audio speed control
    effect_id = timeline.add_effect(clip_id, "pitch")
    if preserve_pitch:
        timeline.set_effect_property(clip_id, effect_id, "tempo", rate)
    else:
        timeline.set_effect_property(clip_id, effect_id, "rate", rate)

    # Adjust clip duration
    clip.set_duration(params.new_duration_ns)

    # P0-5: Verify
    actual_duration = clip.get_duration()
    if actual_duration != params.new_duration_ns:
        raise OperationError(
            f"Speed duration not applied: expected {params.new_duration_ns}, got {actual_duration}"
        )


def apply_transition(
    timeline: Timeline,
    clip_a_id: str,
    clip_b_id: str,
    transition_type: TransitionType,
    duration_ns: int,
) -> None:
    """Apply a transition between two adjacent clips.

    Enables auto-transitions on the timeline, then moves clip_b earlier
    to create an overlap. GES auto-creates a TransitionClip in the overlap.
    For non-crossfade types, sets the vtype on the auto-created transition.
    """
    import gi

    gi.require_version("GES", "1.0")
    from gi.repository import GES as _GES

    clip_a = timeline.get_clip(clip_a_id)
    clip_b = timeline.get_clip(clip_b_id)

    clip_a_end = clip_a.get_start() + clip_a.get_duration()
    clip_b_start = clip_b.get_start()
    clip_b_duration = clip_b.get_duration()

    params = compute_transition(
        clip_a_end, clip_b_start, transition_type, duration_ns, clip_b_duration
    )

    # Enable auto-transitions
    timeline.enable_auto_transitions(True)

    # Move clip_b to create overlap
    clip_b.set_start(params.clip_b_new_start_ns)

    # P0-5: Verify clip_b moved
    actual_start = clip_b.get_start()
    if actual_start != params.clip_b_new_start_ns:
        raise OperationError(
            f"Transition not applied: clip_b start expected {params.clip_b_new_start_ns}, "
            f"got {actual_start}"
        )

    # For non-crossfade types, find the auto-created TransitionClip and set vtype
    if transition_type not in (TransitionType.CROSSFADE, TransitionType.FADE_TO_BLACK):
        _GES_TRANSITION_TYPES = {
            TransitionType.WIPE_LEFT: _GES.VideoStandardTransitionType.BAR_WIPE_LR,
            TransitionType.WIPE_RIGHT: _GES.VideoStandardTransitionType.BAR_WIPE_LR,
            TransitionType.WIPE_UP: _GES.VideoStandardTransitionType.BAR_WIPE_TB,
            TransitionType.WIPE_DOWN: _GES.VideoStandardTransitionType.BAR_WIPE_TB,
        }
        layer = clip_b.get_layer()
        if layer:
            for clip in layer.get_clips():
                if isinstance(clip, _GES.TransitionClip):
                    ges_vtype = _GES_TRANSITION_TYPES.get(transition_type)
                    if ges_vtype is not None:
                        clip.set_child_property("vtype", ges_vtype)
                    break
