"""Speed change tools — parameter computation for playback rate changes.

Pure logic layer: no GES dependency.
"""

from dataclasses import dataclass


class SpeedError(Exception):
    """Raised when speed parameter validation fails."""


MIN_RATE = 0.1
MAX_RATE = 100.0


@dataclass(frozen=True)
class SpeedParams:
    """Computed speed change parameters."""

    rate: float
    new_duration_ns: int
    preserve_pitch: bool


def compute_speed_change(
    clip_duration_ns: int,
    rate: float,
    preserve_pitch: bool = True,
) -> SpeedParams:
    """Validate and compute speed change parameters.

    Args:
        clip_duration_ns: Current clip duration in nanoseconds.
        rate: Playback rate multiplier (1.0 = normal, 2.0 = double speed).
        preserve_pitch: Whether to preserve audio pitch when changing speed.

    Returns:
        SpeedParams with computed new duration.

    Raises:
        SpeedError: If rate is invalid.
    """
    if rate <= 0:
        raise SpeedError(f"Rate must be positive, got {rate}")

    if rate < MIN_RATE or rate > MAX_RATE:
        raise SpeedError(f"Rate {rate} is outside allowed range [{MIN_RATE}, {MAX_RATE}]")

    new_duration_ns = int(clip_duration_ns / rate)

    return SpeedParams(
        rate=rate,
        new_duration_ns=new_duration_ns,
        preserve_pitch=preserve_pitch,
    )
