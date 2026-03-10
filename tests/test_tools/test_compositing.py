"""Unit tests for compositing tools — pure logic, no GES required."""

import pytest


class TestBlendMode:
    """Test BlendMode enum members."""

    def test_all_blend_modes_exist(self):
        from ave.tools.compositing import BlendMode

        expected = {"SOURCE", "OVER", "MULTIPLY", "SCREEN", "OVERLAY", "SOFT_LIGHT", "ADD"}
        actual = {m.name for m in BlendMode}
        assert actual == expected


class TestComputeLayerParams:
    """Test layer parameter validation and computation."""

    def test_valid_single_layer(self):
        from ave.tools.compositing import BlendMode, compute_layer_params

        layers = [
            {
                "layer_index": 0,
                "alpha": 1.0,
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            }
        ]
        result = compute_layer_params(layers)
        assert len(result) == 1
        assert result[0].layer_index == 0
        assert result[0].alpha == 1.0
        assert result[0].blend_mode == BlendMode.OVER
        assert result[0].position_x == 0
        assert result[0].position_y == 0
        assert result[0].width is None
        assert result[0].height is None

    def test_valid_multiple_layers(self):
        from ave.tools.compositing import BlendMode, compute_layer_params

        layers = [
            {
                "layer_index": 0,
                "alpha": 1.0,
                "blend_mode": BlendMode.SOURCE,
                "position_x": 0,
                "position_y": 0,
                "width": 1920,
                "height": 1080,
            },
            {
                "layer_index": 1,
                "alpha": 0.5,
                "blend_mode": BlendMode.OVER,
                "position_x": 100,
                "position_y": 200,
                "width": 640,
                "height": 480,
            },
        ]
        result = compute_layer_params(layers)
        assert len(result) == 2
        assert result[0].layer_index == 0
        assert result[1].layer_index == 1
        assert result[1].alpha == 0.5

    def test_duplicate_layer_index_raises(self):
        from ave.tools.compositing import BlendMode, CompositingError, compute_layer_params

        layers = [
            {
                "layer_index": 0,
                "alpha": 1.0,
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            },
            {
                "layer_index": 0,
                "alpha": 0.5,
                "blend_mode": BlendMode.OVER,
                "position_x": 10,
                "position_y": 10,
                "width": None,
                "height": None,
            },
        ]
        with pytest.raises(CompositingError, match="[Dd]uplicate"):
            compute_layer_params(layers)

    def test_alpha_out_of_range_raises(self):
        from ave.tools.compositing import BlendMode, CompositingError, compute_layer_params

        layers = [
            {
                "layer_index": 0,
                "alpha": 1.5,
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            },
        ]
        with pytest.raises(CompositingError, match="[Aa]lpha"):
            compute_layer_params(layers)

    def test_negative_layer_index_raises(self):
        from ave.tools.compositing import BlendMode, CompositingError, compute_layer_params

        layers = [
            {
                "layer_index": -1,
                "alpha": 1.0,
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            },
        ]
        with pytest.raises(CompositingError, match="[Ll]ayer.*index"):
            compute_layer_params(layers)

    def test_empty_list_raises(self):
        from ave.tools.compositing import CompositingError, compute_layer_params

        with pytest.raises(CompositingError, match="[Ee]mpty"):
            compute_layer_params([])

    def test_default_width_height_none(self):
        from ave.tools.compositing import BlendMode, compute_layer_params

        layers = [
            {
                "layer_index": 0,
                "alpha": 1.0,
                "blend_mode": BlendMode.OVER,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            },
        ]
        result = compute_layer_params(layers)
        assert result[0].width is None
        assert result[0].height is None

    def test_layers_sorted_by_index(self):
        from ave.tools.compositing import BlendMode, compute_layer_params

        layers = [
            {
                "layer_index": 2,
                "alpha": 0.8,
                "blend_mode": BlendMode.ADD,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            },
            {
                "layer_index": 0,
                "alpha": 1.0,
                "blend_mode": BlendMode.SOURCE,
                "position_x": 0,
                "position_y": 0,
                "width": None,
                "height": None,
            },
            {
                "layer_index": 1,
                "alpha": 0.5,
                "blend_mode": BlendMode.OVER,
                "position_x": -50,
                "position_y": -100,
                "width": None,
                "height": None,
            },
        ]
        result = compute_layer_params(layers)
        assert [r.layer_index for r in result] == [0, 1, 2]


class TestComputeBlendParams:
    """Test blend mode to GL blend function mapping."""

    def test_over_mode(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.OVER)
        # Standard alpha blending: src_alpha, one_minus_src_alpha
        assert params.src_rgb == 0x0302  # GL_SRC_ALPHA
        assert params.dst_rgb == 0x0303  # GL_ONE_MINUS_SRC_ALPHA

    def test_multiply_mode(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.MULTIPLY)
        # result = src * dst_color + dst * 0 = src * dst
        assert params.src_rgb == 0x0306  # GL_DST_COLOR
        assert params.dst_rgb == 0  # GL_ZERO

    def test_add_mode(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        params = compute_blend_params(BlendMode.ADD)
        assert params.src_rgb == 1  # GL_ONE
        assert params.dst_rgb == 1  # GL_ONE

    def test_all_blend_modes_return_valid_params(self):
        from ave.tools.compositing import BlendMode, compute_blend_params

        for mode in BlendMode:
            params = compute_blend_params(mode)
            assert params.src_rgb is not None
            assert params.dst_rgb is not None
            assert params.src_alpha is not None
            assert params.dst_alpha is not None
            assert params.equation_rgb is not None
            assert params.equation_alpha is not None
