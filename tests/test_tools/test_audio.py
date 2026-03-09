"""Unit tests for audio tools — pure logic, no GES required."""

import math

import pytest


class TestVolumeConversion:
    """Test dB/linear conversion utilities."""

    def test_db_to_linear_0db(self):
        from ave.tools.audio import db_to_linear

        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_db_to_linear_positive(self):
        from ave.tools.audio import db_to_linear

        assert db_to_linear(6.0) == pytest.approx(2.0, rel=0.01)

    def test_db_to_linear_negative(self):
        from ave.tools.audio import db_to_linear

        assert db_to_linear(-6.0) == pytest.approx(0.5, rel=0.01)

    def test_db_to_linear_minus_inf(self):
        from ave.tools.audio import db_to_linear

        assert db_to_linear(-100.0) == pytest.approx(0.0, abs=1e-5)

    def test_linear_to_db_unity(self):
        from ave.tools.audio import linear_to_db

        assert linear_to_db(1.0) == pytest.approx(0.0)

    def test_linear_to_db_double(self):
        from ave.tools.audio import linear_to_db

        assert linear_to_db(2.0) == pytest.approx(6.02, abs=0.1)

    def test_linear_to_db_half(self):
        from ave.tools.audio import linear_to_db

        assert linear_to_db(0.5) == pytest.approx(-6.02, abs=0.1)

    def test_linear_to_db_zero(self):
        from ave.tools.audio import linear_to_db

        result = linear_to_db(0.0)
        assert result == -math.inf or result < -100

    def test_linear_to_db_negative_rejected(self):
        from ave.tools.audio import linear_to_db, AudioError

        with pytest.raises(AudioError):
            linear_to_db(-1.0)

    def test_roundtrip_conversion(self):
        from ave.tools.audio import db_to_linear, linear_to_db

        for db in [-20.0, -6.0, 0.0, 3.0, 12.0]:
            assert linear_to_db(db_to_linear(db)) == pytest.approx(db, abs=0.001)


class TestVolumeParams:
    """Test volume parameter validation."""

    def test_set_volume_valid(self):
        from ave.tools.audio import compute_volume

        result = compute_volume(level_db=-6.0)
        assert result.level_db == -6.0
        assert result.linear_gain == pytest.approx(0.5, rel=0.01)

    def test_set_volume_mute(self):
        from ave.tools.audio import compute_volume

        result = compute_volume(level_db=-80.0)
        assert result.linear_gain < 0.001

    def test_set_volume_boost(self):
        from ave.tools.audio import compute_volume

        result = compute_volume(level_db=6.0)
        assert result.linear_gain == pytest.approx(2.0, rel=0.01)

    def test_set_volume_too_loud(self):
        from ave.tools.audio import compute_volume, AudioError

        with pytest.raises(AudioError, match="range"):
            compute_volume(level_db=25.0)

    def test_set_volume_too_quiet(self):
        from ave.tools.audio import compute_volume, AudioError

        with pytest.raises(AudioError, match="range"):
            compute_volume(level_db=-100.0)


class TestFadeParams:
    """Test audio fade parameter validation."""

    def test_fade_in_only(self):
        from ave.tools.audio import compute_fade

        result = compute_fade(
            clip_duration_ns=5_000_000_000,
            fade_in_ns=1_000_000_000,
            fade_out_ns=0,
        )
        assert result.fade_in_ns == 1_000_000_000
        assert result.fade_out_ns == 0

    def test_fade_out_only(self):
        from ave.tools.audio import compute_fade

        result = compute_fade(
            clip_duration_ns=5_000_000_000,
            fade_in_ns=0,
            fade_out_ns=2_000_000_000,
        )
        assert result.fade_in_ns == 0
        assert result.fade_out_ns == 2_000_000_000

    def test_fade_in_and_out(self):
        from ave.tools.audio import compute_fade

        result = compute_fade(
            clip_duration_ns=5_000_000_000,
            fade_in_ns=1_000_000_000,
            fade_out_ns=1_000_000_000,
        )
        assert result.fade_in_ns == 1_000_000_000
        assert result.fade_out_ns == 1_000_000_000

    def test_fade_exceeds_duration(self):
        from ave.tools.audio import compute_fade, AudioError

        with pytest.raises(AudioError, match="exceed"):
            compute_fade(
                clip_duration_ns=2_000_000_000,
                fade_in_ns=1_500_000_000,
                fade_out_ns=1_500_000_000,
            )

    def test_fade_negative_rejected(self):
        from ave.tools.audio import compute_fade, AudioError

        with pytest.raises(AudioError, match="negative"):
            compute_fade(
                clip_duration_ns=5_000_000_000,
                fade_in_ns=-1,
                fade_out_ns=0,
            )

    def test_fade_exactly_clip_duration(self):
        """Fades that sum to exactly clip duration should work."""
        from ave.tools.audio import compute_fade

        result = compute_fade(
            clip_duration_ns=4_000_000_000,
            fade_in_ns=2_000_000_000,
            fade_out_ns=2_000_000_000,
        )
        assert result.fade_in_ns + result.fade_out_ns == 4_000_000_000

    def test_no_fade(self):
        from ave.tools.audio import compute_fade

        result = compute_fade(
            clip_duration_ns=5_000_000_000,
            fade_in_ns=0,
            fade_out_ns=0,
        )
        assert result.fade_in_ns == 0
        assert result.fade_out_ns == 0


class TestNormalizeParams:
    """Test normalization parameter computation."""

    def test_normalize_quiet_clip(self):
        from ave.tools.audio import compute_normalize

        result = compute_normalize(
            current_peak_db=-20.0,
            target_peak_db=-1.0,
        )
        assert result.gain_db == pytest.approx(19.0)

    def test_normalize_loud_clip(self):
        from ave.tools.audio import compute_normalize

        result = compute_normalize(
            current_peak_db=0.0,
            target_peak_db=-3.0,
        )
        assert result.gain_db == pytest.approx(-3.0)

    def test_normalize_already_at_target(self):
        from ave.tools.audio import compute_normalize

        result = compute_normalize(
            current_peak_db=-3.0,
            target_peak_db=-3.0,
        )
        assert result.gain_db == pytest.approx(0.0)

    def test_normalize_invalid_target(self):
        from ave.tools.audio import compute_normalize, AudioError

        with pytest.raises(AudioError, match="target"):
            compute_normalize(current_peak_db=-10.0, target_peak_db=10.0)

    def test_normalize_silent_clip_raises(self):
        """Normalizing a silent clip (peak=-inf) must raise AudioError."""
        from ave.tools.audio import compute_normalize, AudioError

        with pytest.raises(AudioError, match="silent"):
            compute_normalize(current_peak_db=-math.inf, target_peak_db=-1.0)

    def test_normalize_extremely_quiet_clip_raises(self):
        """Normalizing a near-silent clip (peak < -120 dB) must raise AudioError."""
        from ave.tools.audio import compute_normalize, AudioError

        with pytest.raises(AudioError, match="silent"):
            compute_normalize(current_peak_db=-150.0, target_peak_db=-1.0)
