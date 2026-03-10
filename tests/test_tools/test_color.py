"""Unit tests for color tools — pure logic, no GES required."""

import pytest


class TestColorError:
    """Test that ColorError is a proper exception."""

    def test_color_error_is_exception(self):
        from ave.tools.color import ColorError

        assert issubclass(ColorError, Exception)

    def test_color_error_message(self):
        from ave.tools.color import ColorError

        err = ColorError("something went wrong")
        assert str(err) == "something went wrong"


class TestParseCubeLUT:
    """Test .cube LUT file parsing."""

    def test_valid_3d_lut(self, tmp_path):
        from ave.tools.color import parse_cube_lut

        lut_file = tmp_path / "test.cube"
        lut_file.write_text(
            "TITLE \"Test LUT\"\n"
            "LUT_3D_SIZE 2\n"
            "DOMAIN_MIN 0.0 0.0 0.0\n"
            "DOMAIN_MAX 1.0 1.0 1.0\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n"
            "1.0 0.0 1.0\n"
            "0.0 1.0 1.0\n"
            "1.0 1.0 1.0\n"
        )

        result = parse_cube_lut(str(lut_file))
        assert result.title == "Test LUT"
        assert result.size == 2
        assert result.domain_min == (0.0, 0.0, 0.0)
        assert result.domain_max == (1.0, 1.0, 1.0)
        assert len(result.table) == 8  # 2^3
        assert result.table[0] == (0.0, 0.0, 0.0)
        assert result.table[7] == (1.0, 1.0, 1.0)

    def test_valid_1d_lut(self, tmp_path):
        from ave.tools.color import parse_cube_lut

        lut_file = tmp_path / "test1d.cube"
        lut_file.write_text(
            "LUT_1D_SIZE 3\n"
            "0.0 0.0 0.0\n"
            "0.5 0.5 0.5\n"
            "1.0 1.0 1.0\n"
        )

        result = parse_cube_lut(str(lut_file))
        assert result.size == 3
        assert len(result.table) == 3

    def test_file_not_found(self):
        from ave.tools.color import parse_cube_lut, ColorError

        with pytest.raises(ColorError, match="not found"):
            parse_cube_lut("/nonexistent/path/to/file.cube")

    def test_invalid_format(self, tmp_path):
        from ave.tools.color import parse_cube_lut, ColorError

        lut_file = tmp_path / "bad.cube"
        lut_file.write_text("this is not a valid cube file\n")

        with pytest.raises(ColorError):
            parse_cube_lut(str(lut_file))

    def test_custom_domain(self, tmp_path):
        from ave.tools.color import parse_cube_lut

        lut_file = tmp_path / "custom_domain.cube"
        lut_file.write_text(
            "LUT_3D_SIZE 2\n"
            "DOMAIN_MIN 0.0 0.1 0.2\n"
            "DOMAIN_MAX 0.8 0.9 1.0\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n"
            "1.0 0.0 1.0\n"
            "0.0 1.0 1.0\n"
            "1.0 1.0 1.0\n"
        )

        result = parse_cube_lut(str(lut_file))
        assert result.domain_min == (0.0, 0.1, 0.2)
        assert result.domain_max == (0.8, 0.9, 1.0)

    def test_lut_with_title(self, tmp_path):
        from ave.tools.color import parse_cube_lut

        lut_file = tmp_path / "titled.cube"
        lut_file.write_text(
            'TITLE "My Cool LUT"\n'
            "LUT_3D_SIZE 2\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n"
            "1.0 0.0 1.0\n"
            "0.0 1.0 1.0\n"
            "1.0 1.0 1.0\n"
        )

        result = parse_cube_lut(str(lut_file))
        assert result.title == "My Cool LUT"

    def test_lut_without_title(self, tmp_path):
        from ave.tools.color import parse_cube_lut

        lut_file = tmp_path / "notitled.cube"
        lut_file.write_text(
            "LUT_3D_SIZE 2\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n"
            "1.0 0.0 1.0\n"
            "0.0 1.0 1.0\n"
            "1.0 1.0 1.0\n"
        )

        result = parse_cube_lut(str(lut_file))
        assert result.title == ""


class TestComputeColorGrade:
    """Test color grading parameter computation."""

    def test_valid_grade(self):
        from ave.tools.color import compute_color_grade

        result = compute_color_grade(
            lift=(0.1, 0.0, -0.1),
            gamma=(1.0, 1.2, 0.8),
            gain=(1.0, 1.1, 0.9),
            saturation=1.2,
            offset=(0.01, 0.0, -0.01),
        )
        assert result.lift == (0.1, 0.0, -0.1)
        assert result.gamma == (1.0, 1.2, 0.8)
        assert result.gain == (1.0, 1.1, 0.9)
        assert result.saturation == 1.2
        assert result.offset == (0.01, 0.0, -0.01)

    def test_default_saturation_and_offset(self):
        from ave.tools.color import compute_color_grade

        result = compute_color_grade(
            lift=(0.0, 0.0, 0.0),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.0, 1.0, 1.0),
        )
        assert result.saturation == 1.0
        assert result.offset == (0, 0, 0)

    def test_lift_out_of_range(self):
        from ave.tools.color import compute_color_grade, ColorError

        with pytest.raises(ColorError, match="lift"):
            compute_color_grade(
                lift=(2.0, 0.0, 0.0),
                gamma=(1.0, 1.0, 1.0),
                gain=(1.0, 1.0, 1.0),
            )

    def test_lift_below_range(self):
        from ave.tools.color import compute_color_grade, ColorError

        with pytest.raises(ColorError, match="lift"):
            compute_color_grade(
                lift=(0.0, -1.5, 0.0),
                gamma=(1.0, 1.0, 1.0),
                gain=(1.0, 1.0, 1.0),
            )

    def test_gamma_zero(self):
        from ave.tools.color import compute_color_grade, ColorError

        with pytest.raises(ColorError, match="gamma"):
            compute_color_grade(
                lift=(0.0, 0.0, 0.0),
                gamma=(0.0, 1.0, 1.0),
                gain=(1.0, 1.0, 1.0),
            )

    def test_gain_negative(self):
        from ave.tools.color import compute_color_grade, ColorError

        with pytest.raises(ColorError, match="gain"):
            compute_color_grade(
                lift=(0.0, 0.0, 0.0),
                gamma=(1.0, 1.0, 1.0),
                gain=(-0.1, 1.0, 1.0),
            )

    def test_saturation_negative(self):
        from ave.tools.color import compute_color_grade, ColorError

        with pytest.raises(ColorError, match="saturation"):
            compute_color_grade(
                lift=(0.0, 0.0, 0.0),
                gamma=(1.0, 1.0, 1.0),
                gain=(1.0, 1.0, 1.0),
                saturation=-0.1,
            )


class TestComputeCDL:
    """Test ASC CDL parameter computation."""

    def test_valid_cdl(self):
        from ave.tools.color import compute_cdl

        result = compute_cdl(
            slope=(1.2, 1.0, 0.8),
            offset=(0.01, 0.0, -0.01),
            power=(1.0, 1.1, 0.9),
            saturation=1.1,
        )
        assert result.slope == (1.2, 1.0, 0.8)
        assert result.offset == (0.01, 0.0, -0.01)
        assert result.power == (1.0, 1.1, 0.9)
        assert result.saturation == 1.1

    def test_identity_cdl(self):
        from ave.tools.color import compute_cdl

        result = compute_cdl(
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
            saturation=1.0,
        )
        assert result.slope == (1.0, 1.0, 1.0)
        assert result.offset == (0.0, 0.0, 0.0)
        assert result.power == (1.0, 1.0, 1.0)
        assert result.saturation == 1.0

    def test_slope_negative(self):
        from ave.tools.color import compute_cdl, ColorError

        with pytest.raises(ColorError, match="slope"):
            compute_cdl(
                slope=(-0.1, 1.0, 1.0),
                offset=(0.0, 0.0, 0.0),
                power=(1.0, 1.0, 1.0),
            )

    def test_power_zero(self):
        from ave.tools.color import compute_cdl, ColorError

        with pytest.raises(ColorError, match="power"):
            compute_cdl(
                slope=(1.0, 1.0, 1.0),
                offset=(0.0, 0.0, 0.0),
                power=(0.0, 1.0, 1.0),
            )


class TestGenerateGradeGLSL:
    """Test GLSL shader generation for color grading."""

    def test_contains_main(self):
        from ave.tools.color import compute_color_grade, generate_grade_glsl

        grade = compute_color_grade(
            lift=(0.0, 0.0, 0.0),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.0, 1.0, 1.0),
        )
        glsl = generate_grade_glsl(grade)
        assert "void main" in glsl

    def test_contains_frag_color(self):
        from ave.tools.color import compute_color_grade, generate_grade_glsl

        grade = compute_color_grade(
            lift=(0.0, 0.0, 0.0),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.0, 1.0, 1.0),
        )
        glsl = generate_grade_glsl(grade)
        assert "gl_FragColor" in glsl or "fragColor" in glsl

    def test_identity_grade_shader(self):
        from ave.tools.color import compute_color_grade, generate_grade_glsl

        grade = compute_color_grade(
            lift=(0.0, 0.0, 0.0),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.0, 1.0, 1.0),
            saturation=1.0,
            offset=(0.0, 0.0, 0.0),
        )
        glsl = generate_grade_glsl(grade)
        # Identity grade should still produce a valid shader
        assert isinstance(glsl, str)
        assert len(glsl) > 0
        assert "void main" in glsl


class TestGenerateCDLGLSL:
    """Test GLSL shader generation for ASC CDL."""

    def test_returns_valid_glsl(self):
        from ave.tools.color import compute_cdl, generate_cdl_glsl

        cdl = compute_cdl(
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
        )
        glsl = generate_cdl_glsl(cdl)
        assert isinstance(glsl, str)
        assert "void main" in glsl

    def test_contains_slope_offset_power(self):
        from ave.tools.color import compute_cdl, generate_cdl_glsl

        cdl = compute_cdl(
            slope=(1.2, 1.0, 0.8),
            offset=(0.01, 0.0, -0.01),
            power=(1.0, 1.1, 0.9),
        )
        glsl = generate_cdl_glsl(cdl)
        assert "slope" in glsl.lower() or "slope" in glsl
        assert "offset" in glsl.lower() or "offset" in glsl
        assert "power" in glsl.lower() or "power" in glsl


class TestComputeLUTApplication:
    """Test LUT application parameter computation."""

    def test_valid_params(self, tmp_path):
        from ave.tools.color import compute_lut_application

        lut_file = tmp_path / "test.cube"
        lut_file.write_text("LUT_3D_SIZE 2\n" + "0.0 0.0 0.0\n" * 8)

        result = compute_lut_application(str(lut_file), intensity=0.8)
        assert result.path == str(lut_file)
        assert result.intensity == 0.8

    def test_missing_file(self):
        from ave.tools.color import compute_lut_application, ColorError

        with pytest.raises(ColorError, match="not found"):
            compute_lut_application("/nonexistent/file.cube")

    def test_wrong_extension(self, tmp_path):
        from ave.tools.color import compute_lut_application, ColorError

        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not a lut")

        with pytest.raises(ColorError, match="extension"):
            compute_lut_application(str(bad_file))

    def test_intensity_out_of_range(self, tmp_path):
        from ave.tools.color import compute_lut_application, ColorError

        lut_file = tmp_path / "test.cube"
        lut_file.write_text("LUT_3D_SIZE 2\n" + "0.0 0.0 0.0\n" * 8)

        with pytest.raises(ColorError, match="intensity"):
            compute_lut_application(str(lut_file), intensity=1.5)

    def test_intensity_negative(self, tmp_path):
        from ave.tools.color import compute_lut_application, ColorError

        lut_file = tmp_path / "test.cube"
        lut_file.write_text("LUT_3D_SIZE 2\n" + "0.0 0.0 0.0\n" * 8)

        with pytest.raises(ColorError, match="intensity"):
            compute_lut_application(str(lut_file), intensity=-0.1)

    def test_intensity_zero_valid(self, tmp_path):
        from ave.tools.color import compute_lut_application

        lut_file = tmp_path / "test.cube"
        lut_file.write_text("LUT_3D_SIZE 2\n" + "0.0 0.0 0.0\n" * 8)

        result = compute_lut_application(str(lut_file), intensity=0.0)
        assert result.intensity == 0.0


class TestComputeColorTransform:
    """Test OCIO color transform parameter computation."""

    def test_valid_transform(self):
        from ave.tools.color import compute_color_transform

        result = compute_color_transform(
            src_colorspace="ACEScg",
            dst_colorspace="sRGB",
        )
        assert result.src_colorspace == "ACEScg"
        assert result.dst_colorspace == "sRGB"
        assert result.config_path is None

    def test_empty_src_raises(self):
        from ave.tools.color import compute_color_transform, ColorError

        with pytest.raises(ColorError, match="src"):
            compute_color_transform(src_colorspace="", dst_colorspace="sRGB")

    def test_empty_dst_raises(self):
        from ave.tools.color import compute_color_transform, ColorError

        with pytest.raises(ColorError, match="dst"):
            compute_color_transform(src_colorspace="ACEScg", dst_colorspace="")

    def test_config_path_not_found(self, tmp_path):
        from ave.tools.color import compute_color_transform, ColorError

        with pytest.raises(ColorError, match="config"):
            compute_color_transform(
                src_colorspace="ACEScg",
                dst_colorspace="sRGB",
                config_path="/nonexistent/config.ocio",
            )

    def test_none_config_path_valid(self):
        from ave.tools.color import compute_color_transform

        result = compute_color_transform(
            src_colorspace="ACEScg",
            dst_colorspace="sRGB",
            config_path=None,
        )
        assert result.config_path is None

    def test_valid_config_path(self, tmp_path):
        from ave.tools.color import compute_color_transform

        config_file = tmp_path / "config.ocio"
        config_file.write_text("ocio_profile_version: 2\n")

        result = compute_color_transform(
            src_colorspace="ACEScg",
            dst_colorspace="sRGB",
            config_path=str(config_file),
        )
        assert result.config_path == str(config_file)
