"""Unit tests for edit tools — pure logic, no GES required."""

import pytest


class TestTrimParams:
    """Test trim parameter validation and computation."""

    def test_trim_valid_params(self):
        from ave.tools.edit import compute_trim

        result = compute_trim(
            clip_duration_ns=5_000_000_000,
            in_ns=1_000_000_000,
            out_ns=4_000_000_000,
        )
        assert result.in_ns == 1_000_000_000
        assert result.out_ns == 4_000_000_000
        assert result.duration_ns == 3_000_000_000

    def test_trim_in_exceeds_duration(self):
        from ave.tools.edit import compute_trim, EditError

        with pytest.raises(EditError, match="in_ns.*exceeds.*duration"):
            compute_trim(clip_duration_ns=5_000_000_000, in_ns=6_000_000_000, out_ns=7_000_000_000)

    def test_trim_out_exceeds_duration(self):
        from ave.tools.edit import compute_trim, EditError

        with pytest.raises(EditError, match="out_ns.*exceeds.*duration"):
            compute_trim(clip_duration_ns=5_000_000_000, in_ns=0, out_ns=6_000_000_000)

    def test_trim_in_after_out(self):
        from ave.tools.edit import compute_trim, EditError

        with pytest.raises(EditError, match="in_ns.*must be before.*out_ns"):
            compute_trim(clip_duration_ns=5_000_000_000, in_ns=3_000_000_000, out_ns=2_000_000_000)

    def test_trim_zero_duration(self):
        from ave.tools.edit import compute_trim, EditError

        with pytest.raises(EditError, match="zero"):
            compute_trim(clip_duration_ns=5_000_000_000, in_ns=2_000_000_000, out_ns=2_000_000_000)

    def test_trim_negative_values(self):
        from ave.tools.edit import compute_trim, EditError

        with pytest.raises(EditError):
            compute_trim(clip_duration_ns=5_000_000_000, in_ns=-1, out_ns=3_000_000_000)

    def test_trim_full_clip(self):
        """Trimming to full clip range should work."""
        from ave.tools.edit import compute_trim

        result = compute_trim(clip_duration_ns=5_000_000_000, in_ns=0, out_ns=5_000_000_000)
        assert result.duration_ns == 5_000_000_000


class TestSplitParams:
    """Test split parameter validation."""

    def test_split_valid_position(self):
        from ave.tools.edit import compute_split

        left, right = compute_split(
            clip_start_ns=0,
            clip_duration_ns=4_000_000_000,
            split_position_ns=2_000_000_000,
        )
        assert left.duration_ns == 2_000_000_000
        assert right.duration_ns == 2_000_000_000
        assert right.start_ns == 2_000_000_000

    def test_split_at_start_fails(self):
        from ave.tools.edit import compute_split, EditError

        with pytest.raises(EditError, match="at the very start"):
            compute_split(clip_start_ns=0, clip_duration_ns=4_000_000_000, split_position_ns=0)

    def test_split_at_end_fails(self):
        from ave.tools.edit import compute_split, EditError

        with pytest.raises(EditError, match="at the very end"):
            compute_split(
                clip_start_ns=0, clip_duration_ns=4_000_000_000, split_position_ns=4_000_000_000
            )

    def test_split_outside_clip_fails(self):
        from ave.tools.edit import compute_split, EditError

        with pytest.raises(EditError, match="outside"):
            compute_split(
                clip_start_ns=1_000_000_000,
                clip_duration_ns=2_000_000_000,
                split_position_ns=5_000_000_000,
            )

    def test_split_with_offset_clip(self):
        """Split a clip that starts at 2s, duration 4s, split at 4s."""
        from ave.tools.edit import compute_split

        left, right = compute_split(
            clip_start_ns=2_000_000_000,
            clip_duration_ns=4_000_000_000,
            split_position_ns=4_000_000_000,
        )
        assert left.duration_ns == 2_000_000_000
        assert right.start_ns == 4_000_000_000
        assert right.duration_ns == 2_000_000_000

    def test_split_preserves_inpoint(self):
        """When splitting, the right portion's inpoint should be offset."""
        from ave.tools.edit import compute_split

        left, right = compute_split(
            clip_start_ns=0,
            clip_duration_ns=4_000_000_000,
            split_position_ns=1_000_000_000,
            inpoint_ns=500_000_000,
        )
        assert left.inpoint_ns == 500_000_000
        assert left.duration_ns == 1_000_000_000
        assert right.inpoint_ns == 1_500_000_000
        assert right.duration_ns == 3_000_000_000


class TestConcatenateParams:
    """Test concatenate parameter validation."""

    def test_concatenate_computes_positions(self):
        from ave.tools.edit import compute_concatenation

        durations = [2_000_000_000, 3_000_000_000, 1_000_000_000]
        positions = compute_concatenation(durations, start_ns=0)

        assert len(positions) == 3
        assert positions[0].start_ns == 0
        assert positions[1].start_ns == 2_000_000_000
        assert positions[2].start_ns == 5_000_000_000
        assert sum(p.duration_ns for p in positions) == 6_000_000_000

    def test_concatenate_with_offset(self):
        from ave.tools.edit import compute_concatenation

        durations = [1_000_000_000, 2_000_000_000]
        positions = compute_concatenation(durations, start_ns=5_000_000_000)

        assert positions[0].start_ns == 5_000_000_000
        assert positions[1].start_ns == 6_000_000_000

    def test_concatenate_empty_list(self):
        from ave.tools.edit import compute_concatenation, EditError

        with pytest.raises(EditError, match="empty"):
            compute_concatenation([], start_ns=0)

    def test_concatenate_zero_duration_rejected(self):
        from ave.tools.edit import compute_concatenation, EditError

        with pytest.raises(EditError, match="zero.*duration"):
            compute_concatenation([1_000_000_000, 0, 2_000_000_000], start_ns=0)

    def test_concatenate_single_clip(self):
        from ave.tools.edit import compute_concatenation

        positions = compute_concatenation([3_000_000_000], start_ns=0)
        assert len(positions) == 1
        assert positions[0].start_ns == 0
        assert positions[0].duration_ns == 3_000_000_000
