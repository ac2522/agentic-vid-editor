"""Audio tools — volume, fade, and normalization parameter computation.

Pure logic layer: no GES dependency.
"""

import math
from dataclasses import dataclass


class AudioError(Exception):
    """Raised when audio parameter validation fails."""


# Volume limits
MIN_VOLUME_DB = -80.0
MAX_VOLUME_DB = 24.0

# Normalization limits
MAX_TARGET_DB = 0.0


def db_to_linear(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to decibels.

    Raises AudioError for negative values.
    """
    if linear < 0:
        raise AudioError(f"Linear gain cannot be negative: {linear}")
    if linear == 0:
        return -math.inf
    return 20.0 * math.log10(linear)


@dataclass(frozen=True)
class VolumeParams:
    """Computed volume parameters."""

    level_db: float
    linear_gain: float


@dataclass(frozen=True)
class FadeParams:
    """Computed fade parameters."""

    fade_in_ns: int
    fade_out_ns: int


@dataclass(frozen=True)
class NormalizeParams:
    """Computed normalization parameters."""

    gain_db: float
    linear_gain: float


def compute_volume(level_db: float) -> VolumeParams:
    """Validate and compute volume parameters.

    Args:
        level_db: Target volume in decibels (0 dB = unity).

    Returns:
        VolumeParams with dB level and linear gain.

    Raises:
        AudioError: If level is outside allowed range.
    """
    if level_db < MIN_VOLUME_DB or level_db > MAX_VOLUME_DB:
        raise AudioError(
            f"Volume {level_db} dB is outside allowed range [{MIN_VOLUME_DB}, {MAX_VOLUME_DB}]"
        )

    return VolumeParams(
        level_db=level_db,
        linear_gain=db_to_linear(level_db),
    )


def compute_fade(
    clip_duration_ns: int,
    fade_in_ns: int,
    fade_out_ns: int,
) -> FadeParams:
    """Validate and compute fade parameters.

    Args:
        clip_duration_ns: Duration of the clip.
        fade_in_ns: Fade-in duration from start.
        fade_out_ns: Fade-out duration to end.

    Returns:
        FadeParams with validated durations.

    Raises:
        AudioError: If fades are invalid.
    """
    if fade_in_ns < 0 or fade_out_ns < 0:
        raise AudioError(f"Fade durations cannot be negative: in={fade_in_ns}, out={fade_out_ns}")

    if fade_in_ns + fade_out_ns > clip_duration_ns:
        raise AudioError(
            f"Fade durations (in={fade_in_ns} + out={fade_out_ns} = {fade_in_ns + fade_out_ns}) "
            f"exceed clip duration ({clip_duration_ns})"
        )

    return FadeParams(fade_in_ns=fade_in_ns, fade_out_ns=fade_out_ns)


def compute_normalize(
    current_peak_db: float,
    target_peak_db: float,
) -> NormalizeParams:
    """Compute gain adjustment for normalization.

    Args:
        current_peak_db: Current peak level in dB.
        target_peak_db: Desired peak level in dB.

    Returns:
        NormalizeParams with gain to apply.

    Raises:
        AudioError: If target is invalid.
    """
    if target_peak_db > MAX_TARGET_DB:
        raise AudioError(
            f"Normalization target {target_peak_db} dB exceeds maximum ({MAX_TARGET_DB} dB)"
        )

    gain_db = target_peak_db - current_peak_db

    return NormalizeParams(
        gain_db=gain_db,
        linear_gain=db_to_linear(gain_db),
    )
