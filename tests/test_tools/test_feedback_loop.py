"""Tests for feedback loop convergence detection."""

from __future__ import annotations

from ave.tools.feedback_loop import (
    SegmentationResult,
    check_convergence,
    DEFAULT_MAX_ITERATIONS,
)
from ave.tools.mask_eval import MaskQuality


def _quality(confidence: float = 0.8, problems: list[int] | None = None) -> MaskQuality:
    return MaskQuality(
        edge_smoothness=0.9,
        temporal_stability=0.9,
        coverage_ratio=0.3,
        confidence_mean=confidence,
        problem_frames=problems or [],
    )


class TestCheckConvergence:
    def test_converged_when_quality_target_met(self):
        should_stop, reason = check_convergence(
            previous_quality=_quality(0.7),
            current_quality=_quality(0.85),
            iteration=2,
        )
        assert should_stop is True
        assert reason == "converged"

    def test_stops_at_max_iterations(self):
        should_stop, reason = check_convergence(
            previous_quality=_quality(0.5),
            current_quality=_quality(0.55),
            iteration=DEFAULT_MAX_ITERATIONS,
        )
        assert should_stop is True
        assert reason == "max_iterations"

    def test_stalled_when_no_improvement(self):
        should_stop, reason = check_convergence(
            previous_quality=_quality(0.6),
            current_quality=_quality(0.62),  # < 0.05 improvement
            iteration=3,
        )
        assert should_stop is True
        assert reason == "convergence_stalled"

    def test_continues_when_improving(self):
        should_stop, reason = check_convergence(
            previous_quality=_quality(0.5),
            current_quality=_quality(0.65),
            iteration=2,
        )
        assert should_stop is False

    def test_continues_on_first_iteration(self):
        should_stop, _ = check_convergence(
            previous_quality=None,
            current_quality=_quality(0.5),
            iteration=0,
        )
        assert should_stop is False

    def test_problem_frames_prevent_convergence(self):
        should_stop, reason = check_convergence(
            previous_quality=_quality(0.7),
            current_quality=_quality(0.85, problems=[3, 7]),
            iteration=2,
        )
        # Has problem frames, so doesn't count as converged
        assert should_stop is False


class TestSegmentationResult:
    def test_create(self):
        r = SegmentationResult(
            converged=True,
            iterations=3,
            final_quality=_quality(0.9),
            reason="converged",
        )
        assert r.converged is True
        assert r.iterations == 3
