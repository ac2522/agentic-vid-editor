"""Unit tests for speed change tools — pure logic, no GES required."""

import pytest


class TestSpeedParams:
    """Test speed change parameter validation."""

    def test_speed_double(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(
            clip_duration_ns=4_000_000_000,
            rate=2.0,
        )
        assert result.rate == 2.0
        assert result.new_duration_ns == 2_000_000_000
        assert result.preserve_pitch is True

    def test_speed_half(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(
            clip_duration_ns=4_000_000_000,
            rate=0.5,
        )
        assert result.rate == 0.5
        assert result.new_duration_ns == 8_000_000_000

    def test_speed_normal(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(
            clip_duration_ns=4_000_000_000,
            rate=1.0,
        )
        assert result.new_duration_ns == 4_000_000_000

    def test_speed_zero_rejected(self):
        from ave.tools.speed import compute_speed_change, SpeedError

        with pytest.raises(SpeedError, match="positive"):
            compute_speed_change(clip_duration_ns=4_000_000_000, rate=0.0)

    def test_speed_negative_rejected(self):
        from ave.tools.speed import compute_speed_change, SpeedError

        with pytest.raises(SpeedError, match="positive"):
            compute_speed_change(clip_duration_ns=4_000_000_000, rate=-1.0)

    def test_speed_too_slow(self):
        from ave.tools.speed import compute_speed_change, SpeedError

        with pytest.raises(SpeedError, match="range"):
            compute_speed_change(clip_duration_ns=4_000_000_000, rate=0.01)

    def test_speed_too_fast(self):
        from ave.tools.speed import compute_speed_change, SpeedError

        with pytest.raises(SpeedError, match="range"):
            compute_speed_change(clip_duration_ns=4_000_000_000, rate=200.0)

    def test_speed_boundary_slow(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(clip_duration_ns=4_000_000_000, rate=0.1)
        assert result.rate == 0.1
        assert result.new_duration_ns == 40_000_000_000

    def test_speed_boundary_fast(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(clip_duration_ns=4_000_000_000, rate=100.0)
        assert result.rate == 100.0
        assert result.new_duration_ns == 40_000_000

    def test_speed_no_pitch_preserve(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(
            clip_duration_ns=4_000_000_000,
            rate=2.0,
            preserve_pitch=False,
        )
        assert result.preserve_pitch is False

    def test_speed_fractional_rate(self):
        from ave.tools.speed import compute_speed_change

        result = compute_speed_change(
            clip_duration_ns=3_000_000_000,
            rate=1.5,
        )
        assert result.new_duration_ns == 2_000_000_000
