"""Tests for color GES operations layer."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestApplyLut:
    """Test LUT application via GES."""

    def test_apply_lut_validates_path(self, tmp_path):
        """Should raise ColorError for missing LUT file."""
        from ave.tools.color_ops import apply_lut
        from ave.tools.color import ColorError

        timeline = MagicMock()
        with pytest.raises(ColorError):
            apply_lut(timeline, "clip_0000", str(tmp_path / "missing.cube"))

    def test_apply_lut_validates_extension(self, tmp_path):
        """Should raise ColorError for wrong extension."""
        from ave.tools.color_ops import apply_lut
        from ave.tools.color import ColorError

        bad_file = tmp_path / "test.txt"
        bad_file.touch()
        timeline = MagicMock()
        with pytest.raises(ColorError):
            apply_lut(timeline, "clip_0000", str(bad_file))

    def test_apply_lut_validates_intensity(self, tmp_path):
        """Should raise ColorError for intensity > 1.0."""
        from ave.tools.color_ops import apply_lut
        from ave.tools.color import ColorError

        lut = tmp_path / "test.cube"
        lut.write_text("LUT_3D_SIZE 2\n" + "0.0 0.0 0.0\n" * 8)
        timeline = MagicMock()
        with pytest.raises(ColorError):
            apply_lut(timeline, "clip_0000", str(lut), intensity=1.5)

    def test_apply_lut_calls_add_effect(self, tmp_path):
        """Should add placebofilter effect to timeline."""
        from ave.tools.color_ops import apply_lut

        lut = tmp_path / "test.cube"
        lut.write_text("LUT_3D_SIZE 2\n" + "0.0 0.0 0.0\n" * 8)
        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        apply_lut(timeline, "clip_0000", str(lut))
        timeline.add_effect.assert_called_once_with("clip_0000", "placebofilter")


class TestApplyColorGrade:
    """Test color grade application via GES."""

    def test_grade_validates_params(self):
        """Should raise ColorError for invalid lift."""
        from ave.tools.color_ops import apply_color_grade
        from ave.tools.color import ColorError

        timeline = MagicMock()
        with pytest.raises(ColorError):
            apply_color_grade(
                timeline, "clip_0000",
                lift=(5.0, 0.0, 0.0),  # out of range
                gamma=(1.0, 1.0, 1.0),
                gain=(1.0, 1.0, 1.0),
            )

    def test_grade_adds_glshader_effect(self):
        """Should add glshader effect with GLSL fragment source."""
        from ave.tools.color_ops import apply_color_grade

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        apply_color_grade(
            timeline, "clip_0000",
            lift=(0.0, 0.0, 0.0),
            gamma=(1.0, 1.0, 1.0),
            gain=(1.0, 1.0, 1.0),
        )
        timeline.add_effect.assert_called_once_with("clip_0000", "glshader")
        # Should set fragment property with GLSL
        timeline.set_effect_property.assert_called()
        call_args = timeline.set_effect_property.call_args_list
        prop_names = [c[0][2] for c in call_args]
        assert "fragment" in prop_names


class TestApplyCDL:
    """Test CDL application via GES."""

    def test_cdl_validates_params(self):
        """Should raise ColorError for negative slope."""
        from ave.tools.color_ops import apply_cdl
        from ave.tools.color import ColorError

        timeline = MagicMock()
        with pytest.raises(ColorError):
            apply_cdl(
                timeline, "clip_0000",
                slope=(-1.0, 1.0, 1.0),
                offset=(0.0, 0.0, 0.0),
                power=(1.0, 1.0, 1.0),
            )

    def test_cdl_adds_glshader_effect(self):
        """Should add glshader effect."""
        from ave.tools.color_ops import apply_cdl

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        apply_cdl(
            timeline, "clip_0000",
            slope=(1.0, 1.0, 1.0),
            offset=(0.0, 0.0, 0.0),
            power=(1.0, 1.0, 1.0),
        )
        timeline.add_effect.assert_called_once_with("clip_0000", "glshader")


class TestApplyColorTransform:
    """Test OCIO color transform application."""

    def test_transform_validates_empty_src(self):
        """Should raise ColorError for empty src."""
        from ave.tools.color_ops import apply_color_transform
        from ave.tools.color import ColorError

        timeline = MagicMock()
        with pytest.raises(ColorError):
            apply_color_transform(timeline, "clip_0000", "", "sRGB")

    def test_transform_adds_ociofilter_effect(self):
        """Should add ociofilter effect with properties."""
        from ave.tools.color_ops import apply_color_transform

        timeline = MagicMock()
        timeline.add_effect.return_value = "clip_0000_fx_0"
        apply_color_transform(timeline, "clip_0000", "ACEScg", "sRGB")
        timeline.add_effect.assert_called_once_with("clip_0000", "ociofilter")
        # Check properties were set
        call_args = timeline.set_effect_property.call_args_list
        prop_names = [c[0][2] for c in call_args]
        assert "src-colorspace" in prop_names
        assert "dst-colorspace" in prop_names


class TestApplyIDT:
    """Test IDT application from clip metadata."""

    def test_idt_reads_clip_metadata(self, tmp_path):
        """Should read camera color space from clip metadata."""
        from ave.tools.color_ops import apply_idt

        ocio_cfg = tmp_path / "ocio.yaml"
        ocio_cfg.write_text("ocio_profile_version: 2\n")
        timeline = MagicMock()
        timeline.get_clip_metadata.return_value = "V-Gamut"
        timeline.add_effect.return_value = "clip_0000_fx_0"
        apply_idt(timeline, "clip_0000", str(ocio_cfg))
        timeline.get_clip_metadata.assert_called_with("clip_0000", "agent:camera-color-space")

    def test_idt_raises_without_metadata(self, tmp_path):
        """Should raise when clip has no color space metadata."""
        from ave.tools.color_ops import apply_idt
        from ave.tools.color import ColorError

        ocio_cfg = tmp_path / "ocio.yaml"
        ocio_cfg.write_text("ocio_profile_version: 2\n")
        timeline = MagicMock()
        timeline.get_clip_metadata.return_value = None
        with pytest.raises(ColorError, match="color space"):
            apply_idt(timeline, "clip_0000", str(ocio_cfg))

    def test_idt_applies_transform_to_acescg(self, tmp_path):
        """Should transform from camera space to ACEScg."""
        from ave.tools.color_ops import apply_idt

        ocio_cfg = tmp_path / "ocio.yaml"
        ocio_cfg.write_text("ocio_profile_version: 2\n")
        timeline = MagicMock()
        timeline.get_clip_metadata.return_value = "S-Gamut3.Cine"
        timeline.add_effect.return_value = "clip_0000_fx_0"
        apply_idt(timeline, "clip_0000", str(ocio_cfg))
        # Verify src-colorspace was set to camera space
        call_args = timeline.set_effect_property.call_args_list
        src_calls = [c for c in call_args if c[0][2] == "src-colorspace"]
        assert len(src_calls) == 1
        assert src_calls[0][0][3] == "S-Gamut3.Cine"


class TestRemoveColorEffect:
    """Test color effect removal."""

    def test_remove_delegates_to_timeline(self):
        """Should call timeline.remove_effect."""
        from ave.tools.color_ops import remove_color_effect

        timeline = MagicMock()
        remove_color_effect(timeline, "clip_0000", "clip_0000_fx_0")
        timeline.remove_effect.assert_called_once_with("clip_0000", "clip_0000_fx_0")
