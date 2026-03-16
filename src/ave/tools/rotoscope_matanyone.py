"""MatAnyone 2 video matting backend (CVPR 2026).

Successor to Robust Video Matting. Features:
- Learned quality evaluator (maps to MaskEvaluator concept)
- Handles camera movement, changing lighting, partial occlusion
- Works on both camera footage and AI-generated video
- Dual-region treatment (core vs. boundary) for fine detail

Requires: torch, matanyone2
Stub implementation for testing.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ave.tools.rotoscope import (
    MaskCorrection,
    SegmentationMask,
    SegmentPrompt,
)


class MatAnyoneBackend:
    """MatAnyone 2 human video matting backend (stub for testing).

    Produces soft alpha mattes with superior edge quality.
    Model downloaded on first use via ModelManager with user consent.
    """

    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask:
        h, w = frame.shape[:2]
        # Soft alpha matte with gradual falloff (simulating real output)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.float32)
        max_dist = float(min(h, w)) * 0.4
        mask = np.clip(1.0 - dist / max_dist, 0.0, 1.0).astype(np.float32)

        return SegmentationMask(
            mask=mask,
            confidence=0.92,
            frame_index=0,
            metadata={"backend": "matanyone2"},
        )

    def segment_video(
        self,
        frames: Iterator[np.ndarray],
        prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
        chunk_size: int = 300,
    ) -> Iterator[SegmentationMask]:
        for i, frame in enumerate(frames):
            m = self.segment_frame(frame, prompts)
            m.frame_index = i
            yield m

    def refine_mask(
        self,
        frame: np.ndarray,
        mask: SegmentationMask,
        corrections: list[MaskCorrection],
    ) -> SegmentationMask:
        return SegmentationMask(
            mask=mask.mask.copy(),
            confidence=min(1.0, mask.confidence + 0.03),
            frame_index=mask.frame_index,
            metadata=mask.metadata,
        )
