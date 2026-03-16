"""Tests for rotoscope protocol and data models."""

from __future__ import annotations

import numpy as np

from ave.tools.rotoscope import SegmentPrompt, SegmentationMask, MaskCorrection


class TestSegmentPrompt:
    def test_point_prompt(self):
        p = SegmentPrompt(kind="point", value=(100, 200))
        assert p.kind == "point"

    def test_box_prompt(self):
        p = SegmentPrompt(kind="box", value=(10, 20, 300, 400))
        assert p.kind == "box"

    def test_text_prompt(self):
        p = SegmentPrompt(kind="text", value="the person in the red shirt")
        assert p.kind == "text"


class TestSegmentationMask:
    def test_create_binary_mask(self):
        data = np.zeros((480, 640), dtype=np.uint8)
        data[100:300, 200:500] = 255
        m = SegmentationMask(mask=data, confidence=0.95, frame_index=0, metadata={})
        assert m.mask.shape == (480, 640)

    def test_create_alpha_mask(self):
        data = np.zeros((480, 640), dtype=np.float32)
        data[100:300, 200:500] = 0.8
        m = SegmentationMask(mask=data, confidence=0.9, frame_index=42, metadata={})
        assert m.mask.dtype == np.float32


class TestMaskCorrection:
    def test_include_point(self):
        c = MaskCorrection(kind="include_point", value=(150, 250))
        assert c.kind == "include_point"

    def test_exclude_region(self):
        c = MaskCorrection(kind="exclude_region", value=(0, 0, 100, 100))
        assert c.kind == "exclude_region"
