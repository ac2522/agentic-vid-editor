"""Unit tests for compositing blend mode shaders — pure logic, no GES required."""

import pytest


class TestGenerateOverlayGlsl:
    """Test GLSL overlay shader generation."""

    def test_returns_valid_glsl_string(self):
        from ave.tools.compositing import generate_overlay_glsl

        glsl = generate_overlay_glsl()
        assert isinstance(glsl, str)
        assert len(glsl) > 0

    def test_contains_version_directive(self):
        from ave.tools.compositing import generate_overlay_glsl

        glsl = generate_overlay_glsl()
        assert "#version 120" in glsl

    def test_contains_texture2d_calls(self):
        from ave.tools.compositing import generate_overlay_glsl

        glsl = generate_overlay_glsl()
        assert "texture2D" in glsl

    def test_contains_overlay_formula(self):
        from ave.tools.compositing import generate_overlay_glsl

        glsl = generate_overlay_glsl()
        # Overlay: conditional on dst < 0.5
        assert "0.5" in glsl
        # Should contain the multiply and screen parts
        assert "2.0" in glsl

    def test_contains_sampler_uniforms(self):
        from ave.tools.compositing import generate_overlay_glsl

        glsl = generate_overlay_glsl()
        assert "uniform sampler2D" in glsl

    def test_writes_to_gl_fragcolor(self):
        from ave.tools.compositing import generate_overlay_glsl

        glsl = generate_overlay_glsl()
        assert "gl_FragColor" in glsl


class TestGenerateSoftLightGlsl:
    """Test GLSL soft light shader generation."""

    def test_returns_valid_glsl_string(self):
        from ave.tools.compositing import generate_soft_light_glsl

        glsl = generate_soft_light_glsl()
        assert isinstance(glsl, str)
        assert len(glsl) > 0

    def test_contains_version_directive(self):
        from ave.tools.compositing import generate_soft_light_glsl

        glsl = generate_soft_light_glsl()
        assert "#version 120" in glsl

    def test_contains_texture2d_calls(self):
        from ave.tools.compositing import generate_soft_light_glsl

        glsl = generate_soft_light_glsl()
        assert "texture2D" in glsl

    def test_contains_soft_light_formula(self):
        from ave.tools.compositing import generate_soft_light_glsl

        glsl = generate_soft_light_glsl()
        # Soft light uses sqrt
        assert "sqrt" in glsl
        assert "0.5" in glsl

    def test_contains_sampler_uniforms(self):
        from ave.tools.compositing import generate_soft_light_glsl

        glsl = generate_soft_light_glsl()
        assert "uniform sampler2D" in glsl

    def test_writes_to_gl_fragcolor(self):
        from ave.tools.compositing import generate_soft_light_glsl

        glsl = generate_soft_light_glsl()
        assert "gl_FragColor" in glsl


class TestBlendFuncParamsRequiresShader:
    """Test requires_shader field on BlendFuncParams."""

    def test_overlay_requires_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.OVERLAY)
        assert params.requires_shader is True

    def test_soft_light_requires_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.SOFT_LIGHT)
        assert params.requires_shader is True

    def test_source_does_not_require_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.SOURCE)
        assert params.requires_shader is False

    def test_over_does_not_require_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.OVER)
        assert params.requires_shader is False

    def test_multiply_does_not_require_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.MULTIPLY)
        assert params.requires_shader is False

    def test_screen_does_not_require_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.SCREEN)
        assert params.requires_shader is False

    def test_add_does_not_require_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.ADD)
        assert params.requires_shader is False


class TestBlendShaderInfo:
    """Test BlendShaderInfo dataclass and compute_blend_info."""

    def test_overlay_returns_shader_info_with_glsl(self):
        from ave.tools.compositing import BlendMode, compute_blend_info

        info = compute_blend_info(BlendMode.OVERLAY)
        assert info.blend_mode == BlendMode.OVERLAY
        assert info.requires_shader is True
        assert info.glsl_source is not None
        assert "texture2D" in info.glsl_source
        assert info.blend_params is None

    def test_soft_light_returns_shader_info_with_glsl(self):
        from ave.tools.compositing import BlendMode, compute_blend_info

        info = compute_blend_info(BlendMode.SOFT_LIGHT)
        assert info.blend_mode == BlendMode.SOFT_LIGHT
        assert info.requires_shader is True
        assert info.glsl_source is not None
        assert "sqrt" in info.glsl_source
        assert info.blend_params is None

    def test_multiply_returns_blend_params_no_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_info

        info = compute_blend_info(BlendMode.MULTIPLY)
        assert info.blend_mode == BlendMode.MULTIPLY
        assert info.requires_shader is False
        assert info.glsl_source is None
        assert info.blend_params is not None
        assert info.blend_params.src_rgb == 0x0306  # GL_DST_COLOR

    def test_source_returns_blend_params_no_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_info

        info = compute_blend_info(BlendMode.SOURCE)
        assert info.requires_shader is False
        assert info.glsl_source is None
        assert info.blend_params is not None

    def test_over_returns_blend_params_no_shader(self):
        from ave.tools.compositing import BlendMode, compute_blend_info

        info = compute_blend_info(BlendMode.OVER)
        assert info.requires_shader is False
        assert info.blend_params is not None

    def test_all_blend_modes_return_valid_info(self):
        from ave.tools.compositing import BlendMode, BlendShaderInfo, compute_blend_info

        for mode in BlendMode:
            info = compute_blend_info(mode)
            assert isinstance(info, BlendShaderInfo)
            assert info.blend_mode == mode
            # Either has shader or blend_params, not both
            if info.requires_shader:
                assert info.glsl_source is not None
                assert info.blend_params is None
            else:
                assert info.glsl_source is None
                assert info.blend_params is not None

    def test_blend_shader_info_is_frozen(self):
        from ave.tools.compositing import BlendMode, compute_blend_info

        info = compute_blend_info(BlendMode.OVERLAY)
        with pytest.raises(AttributeError):
            info.blend_mode = BlendMode.OVER  # type: ignore[misc]
