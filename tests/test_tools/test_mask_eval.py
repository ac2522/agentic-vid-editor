"""Tests for mask quality evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from ave.tools.mask_eval import MaskQuality, MaskEvaluator
from ave.tools.rotoscope import SegmentationMask


class TestMaskEvaluator:
    def _make_mask(
        self, frame_idx: int, fill_region: tuple = (100, 200, 300, 500)
    ) -> SegmentationMask:
        data = np.zeros((480, 640), dtype=np.float32)
        y1, x1, y2, x2 = fill_region
        data[y1:y2, x1:x2] = 1.0
        return SegmentationMask(
            mask=data, confidence=0.9, frame_index=frame_idx, metadata={}
        )

    def test_perfect_masks_score_high(self):
        masks = [self._make_mask(i) for i in range(5)]
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate(masks, frames)
        assert quality.temporal_stability > 0.9
        assert quality.confidence_mean == pytest.approx(0.9)
        assert len(quality.problem_frames) == 0

    def test_inconsistent_masks_flag_problems(self):
        masks = [self._make_mask(i) for i in range(5)]
        masks[3] = SegmentationMask(
            mask=np.ones((480, 640), dtype=np.float32),
            confidence=0.3,
            frame_index=3,
            metadata={},
        )
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate(masks, frames)
        assert 3 in quality.problem_frames
        assert quality.temporal_stability < 0.9

    def test_empty_masks_list(self):
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate([], [])
        assert quality.confidence_mean == 0.0

    def test_single_mask(self):
        masks = [self._make_mask(0)]
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        evaluator = MaskEvaluator()
        quality = evaluator.evaluate(masks, frames)
        assert quality.temporal_stability == 1.0  # Single frame = stable
        assert quality.confidence_mean == 0.9
