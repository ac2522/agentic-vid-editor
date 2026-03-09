"""Unit tests for transition adjacency tolerance."""

import pytest


class TestTransitionNearAdjacent:
    """Test that transitions tolerate tiny gaps from nanosecond rounding."""

    def test_tiny_gap_accepted(self):
        """A gap of 1ns (nanosecond rounding artifact) should be tolerated."""
        from ave.tools.transitions import compute_transition, TransitionType

        result = compute_transition(
            clip_a_end_ns=3_000_000_000,
            clip_b_start_ns=3_000_000_001,  # 1ns gap
            transition_type=TransitionType.CROSSFADE,
            duration_ns=500_000_000,
        )
        assert result.type == TransitionType.CROSSFADE

    def test_one_frame_gap_rejected(self):
        """A gap of 1 frame (41.6ms at 24fps) should still be rejected."""
        from ave.tools.transitions import compute_transition, TransitionType, TransitionError

        with pytest.raises(TransitionError, match="adjacent"):
            compute_transition(
                clip_a_end_ns=3_000_000_000,
                clip_b_start_ns=3_041_666_667,  # ~1 frame at 24fps
                transition_type=TransitionType.CROSSFADE,
                duration_ns=500_000_000,
            )
