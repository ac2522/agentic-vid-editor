"""Editing domain tool registration."""

from __future__ import annotations

from ave.agent.registry import ToolRegistry


def register_editing_tools(registry: ToolRegistry) -> None:
    """Register editing domain tools."""

    @registry.tool(
        domain="editing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["clip_trimmed"],
        tags=["cut", "shorten", "crop timeline", "in point", "out point", "subclip",
              "excerpt", "clip duration", "mark in", "mark out"],
    )
    def trim(clip_duration_ns: int, in_ns: int, out_ns: int):
        """Trim a clip to new in/out points in nanoseconds."""
        from ave.tools.edit import compute_trim

        return compute_trim(clip_duration_ns, in_ns, out_ns)

    @registry.tool(
        domain="editing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["clip_split"],
        tags=["razor", "blade", "cut at playhead", "divide", "slice", "bisect"],
    )
    def split(
        clip_start_ns: int,
        clip_duration_ns: int,
        split_position_ns: int,
        inpoint_ns: int = 0,
    ):
        """Split a clip at a timeline position into two parts."""
        from ave.tools.edit import compute_split

        return compute_split(clip_start_ns, clip_duration_ns, split_position_ns, inpoint_ns)

    @registry.tool(
        domain="editing",
        requires=["timeline_loaded"],
        provides=["clips_concatenated"],
        tags=["join", "append", "combine clips", "sequence", "assemble",
              "stitch together", "back to back"],
    )
    def concatenate(durations_ns: list, start_ns: int = 0):
        """Compute sequential positions for concatenating clips."""
        from ave.tools.edit import compute_concatenation

        return compute_concatenation(durations_ns, start_ns)

    @registry.tool(
        domain="editing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["clip_speed_changed"],
        tags=["slow motion", "fast forward", "ramp", "time stretch", "slo-mo",
              "speed ramp", "playback rate", "timelapse", "retiming"],
    )
    def speed(clip_duration_ns: int, rate: float, preserve_pitch: bool = True):
        """Change playback speed of a clip by a rate multiplier."""
        from ave.tools.speed import compute_speed_change

        return compute_speed_change(clip_duration_ns, rate, preserve_pitch)

    @registry.tool(
        domain="editing",
        requires=["timeline_loaded", "clip_exists"],
        provides=["transition_added"],
        tags=["crossfade", "dissolve", "wipe", "fade between", "smooth cut",
              "morph cut"],
    )
    def transition(
        clip_a_end_ns: int,
        clip_b_start_ns: int,
        transition_type: str,
        duration_ns: int,
        clip_b_duration_ns: int = 0,
    ):
        """Add a transition effect between two adjacent clips."""
        from ave.tools.transitions import TransitionType, compute_transition

        tt = TransitionType(transition_type)
        return compute_transition(
            clip_a_end_ns,
            clip_b_start_ns,
            tt,
            duration_ns,
            clip_b_duration_ns if clip_b_duration_ns > 0 else None,
        )
