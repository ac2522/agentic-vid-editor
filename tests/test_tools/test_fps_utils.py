"""Unit tests for FPS comparison utility."""


class TestFpsClose:
    """Test that FPS comparison handles float precision correctly."""

    def test_exact_match(self):
        from ave.utils import fps_close

        assert fps_close(24.0, 24.0) is True

    def test_ntsc_23976_variants(self):
        """23.976 expressed differently should be considered equal."""
        from ave.utils import fps_close

        # 24000/1001 = 23.976023976...
        assert fps_close(23.976, 24000 / 1001) is True

    def test_ntsc_2997_variants(self):
        from ave.utils import fps_close

        assert fps_close(29.97, 30000 / 1001) is True

    def test_different_rates(self):
        from ave.utils import fps_close

        assert fps_close(24.0, 25.0) is False
        assert fps_close(29.97, 30.0) is False

    def test_zero(self):
        from ave.utils import fps_close

        assert fps_close(0.0, 0.0) is True


class TestPathToUri:
    """Test shared path_to_uri utility."""

    def test_converts_path(self, tmp_path):
        from ave.utils import path_to_uri

        p = tmp_path / "test.mp4"
        p.touch()
        uri = path_to_uri(p)
        assert uri.startswith("file:///")
        assert "test.mp4" in uri

    def test_handles_spaces(self, tmp_path):
        from ave.utils import path_to_uri

        p = tmp_path / "my file.mp4"
        p.touch()
        uri = path_to_uri(p)
        assert "my%20file.mp4" in uri


class TestFpsToFraction:
    """Test FPS to fraction conversion for GES restriction caps."""

    def test_integer_fps(self):
        from ave.utils import fps_to_fraction as _fps_to_fraction

        assert _fps_to_fraction(24.0) == (24, 1)
        assert _fps_to_fraction(30.0) == (30, 1)
        assert _fps_to_fraction(60.0) == (60, 1)

    def test_ntsc_23976(self):
        from ave.utils import fps_to_fraction as _fps_to_fraction

        num, den = _fps_to_fraction(23.976)
        assert num == 24000
        assert den == 1001

    def test_ntsc_2997(self):
        from ave.utils import fps_to_fraction as _fps_to_fraction

        num, den = _fps_to_fraction(29.97)
        assert num == 30000
        assert den == 1001

    def test_ntsc_5994(self):
        from ave.utils import fps_to_fraction as _fps_to_fraction

        num, den = _fps_to_fraction(59.94)
        assert num == 60000
        assert den == 1001

    def test_double_ntsc_47952(self):
        """2x 23.976 should produce a proper fraction, not 47952/1000."""
        from ave.utils import fps_to_fraction as _fps_to_fraction
        from fractions import Fraction

        num, den = _fps_to_fraction(47.952)
        # Must be a reasonable fraction, not int(47952)/1000
        actual_fps = Fraction(num, den)
        expected_fps = Fraction(48000, 1001)
        assert abs(float(actual_fps) - float(expected_fps)) < 0.001
