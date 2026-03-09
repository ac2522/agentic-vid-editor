"""Unit tests for transition tools — pure logic, no GES required."""

import pytest


class TestTransitionParams:
    """Test transition parameter validation."""

    def test_crossfade_valid(self):
        from ave.tools.transitions import compute_transition, TransitionType

        result = compute_transition(
            clip_a_end_ns=3_000_000_000,
            clip_b_start_ns=3_000_000_000,
            transition_type=TransitionType.CROSSFADE,
            duration_ns=500_000_000,
        )
        assert result.type == TransitionType.CROSSFADE
        assert result.duration_ns == 500_000_000
        assert result.overlap_start_ns == 2_500_000_000
        assert result.clip_a_new_end_ns == 3_000_000_000
        assert result.clip_b_new_start_ns == 2_500_000_000

    def test_fade_to_black_valid(self):
        from ave.tools.transitions import compute_transition, TransitionType

        result = compute_transition(
            clip_a_end_ns=3_000_000_000,
            clip_b_start_ns=3_000_000_000,
            transition_type=TransitionType.FADE_TO_BLACK,
            duration_ns=1_000_000_000,
        )
        assert result.type == TransitionType.FADE_TO_BLACK
        assert result.duration_ns == 1_000_000_000

    def test_transition_duration_too_long(self):
        from ave.tools.transitions import compute_transition, TransitionType, TransitionError

        with pytest.raises(TransitionError, match="exceeds"):
            compute_transition(
                clip_a_end_ns=1_000_000_000,
                clip_b_start_ns=1_000_000_000,
                transition_type=TransitionType.CROSSFADE,
                duration_ns=3_000_000_000,
            )

    def test_transition_zero_duration(self):
        from ave.tools.transitions import compute_transition, TransitionType, TransitionError

        with pytest.raises(TransitionError, match="positive"):
            compute_transition(
                clip_a_end_ns=3_000_000_000,
                clip_b_start_ns=3_000_000_000,
                transition_type=TransitionType.CROSSFADE,
                duration_ns=0,
            )

    def test_transition_negative_duration(self):
        from ave.tools.transitions import compute_transition, TransitionType, TransitionError

        with pytest.raises(TransitionError, match="positive"):
            compute_transition(
                clip_a_end_ns=3_000_000_000,
                clip_b_start_ns=3_000_000_000,
                transition_type=TransitionType.CROSSFADE,
                duration_ns=-500_000_000,
            )

    def test_clips_not_adjacent(self):
        """Clips with a gap between them."""
        from ave.tools.transitions import compute_transition, TransitionType, TransitionError

        with pytest.raises(TransitionError, match="adjacent"):
            compute_transition(
                clip_a_end_ns=2_000_000_000,
                clip_b_start_ns=4_000_000_000,
                transition_type=TransitionType.CROSSFADE,
                duration_ns=500_000_000,
            )

    def test_clips_already_overlapping(self):
        """Clips that already overlap."""
        from ave.tools.transitions import compute_transition, TransitionType, TransitionError

        with pytest.raises(TransitionError, match="overlap"):
            compute_transition(
                clip_a_end_ns=4_000_000_000,
                clip_b_start_ns=2_000_000_000,
                transition_type=TransitionType.CROSSFADE,
                duration_ns=500_000_000,
            )

    def test_wipe_transition(self):
        from ave.tools.transitions import compute_transition, TransitionType

        result = compute_transition(
            clip_a_end_ns=5_000_000_000,
            clip_b_start_ns=5_000_000_000,
            transition_type=TransitionType.WIPE_LEFT,
            duration_ns=750_000_000,
        )
        assert result.type == TransitionType.WIPE_LEFT

    def test_all_transition_types_exist(self):
        from ave.tools.transitions import TransitionType

        expected = {"CROSSFADE", "FADE_TO_BLACK", "WIPE_LEFT", "WIPE_RIGHT", "WIPE_UP", "WIPE_DOWN"}
        actual = {t.name for t in TransitionType}
        assert expected.issubset(actual)
