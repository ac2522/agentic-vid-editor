"""Feedback loop orchestration with iteration cap and convergence detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from ave.tools.mask_eval import MaskQuality
from ave.tools.rotoscope import SegmentationMask


@dataclass(frozen=True)
class SegmentationResult:
    """Result of a feedback loop run."""

    converged: bool
    iterations: int
    final_quality: MaskQuality
    masks: list[SegmentationMask] = field(default_factory=list)
    reason: str = ""  # "converged", "max_iterations", "convergence_stalled"


DEFAULT_MAX_ITERATIONS = 5
DEFAULT_IMPROVEMENT_THRESHOLD = 0.05


def check_convergence(
    previous_quality: MaskQuality | None,
    current_quality: MaskQuality,
    iteration: int,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    quality_target: float = 0.8,
) -> tuple[bool, str]:
    """Check if the feedback loop should stop.

    Returns (should_stop, reason).
    """
    # Hit iteration cap
    if iteration >= max_iterations:
        return True, "max_iterations"

    # Quality target met
    if current_quality.confidence_mean >= quality_target and not current_quality.problem_frames:
        return True, "converged"

    # Convergence stalled — no meaningful improvement
    if previous_quality is not None:
        improvement = current_quality.confidence_mean - previous_quality.confidence_mean
        if improvement < improvement_threshold and iteration >= 2:
            return True, "convergence_stalled"

    return False, ""
