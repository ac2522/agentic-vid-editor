"""End-to-end tests for rotoscope feedback loop."""

from __future__ import annotations

import numpy as np

from ave.tools.rotoscope import SegmentPrompt, MaskCorrection
from ave.tools.rotoscope_sam2 import SamBackend
from ave.tools.rotoscope_chroma import ChromaKeyBackend
from ave.tools.mask_eval import MaskEvaluator


class TestFeedbackLoop:
    def test_segment_evaluate_refine_cycle(self):
        """Simulate the agent's feedback loop."""
        backend = SamBackend(model_size="tiny")
        evaluator = MaskEvaluator(quality_threshold=0.6)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        prompts = [SegmentPrompt(kind="point", value=(320, 240))]

        # Segment keyframes
        masks = [backend.segment_frame(f, prompts) for f in frames]
        for i, m in enumerate(masks):
            m.frame_index = i

        # Evaluate
        quality = evaluator.evaluate(masks, frames)
        assert quality.confidence_mean > 0

        # Refine problem frames if any
        if quality.problem_frames:
            for idx in quality.problem_frames:
                masks[idx] = backend.refine_mask(
                    frames[idx],
                    masks[idx],
                    [MaskCorrection(kind="include_point", value=(200, 300))],
                )

        # Re-evaluate
        quality_after = evaluator.evaluate(masks, frames)
        assert quality_after.confidence_mean >= quality.confidence_mean

    def test_chroma_key_full_pipeline(self):
        """Green screen through full pipeline."""
        backend = ChromaKeyBackend(tolerance=0.3)
        evaluator = MaskEvaluator()

        frames = []
        for _ in range(3):
            f = np.zeros((480, 640, 3), dtype=np.uint8)
            f[:, :] = [0, 255, 0]
            f[100:380, 200:440] = [200, 200, 200]
            frames.append(f)

        prompts = [SegmentPrompt(kind="text", value="green")]
        masks = [backend.segment_frame(f, prompts) for f in frames]
        for i, m in enumerate(masks):
            m.frame_index = i

        quality = evaluator.evaluate(masks, frames)
        assert quality.temporal_stability > 0.9
        assert quality.confidence_mean > 0.9
