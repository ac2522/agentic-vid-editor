"""Robust Video Matting backend.

Purpose-built for human foreground/background separation.
Produces soft alpha mattes. Stub for testing.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ave.tools.rotoscope import (
    MaskCorrection,
    SegmentationMask,
    SegmentPrompt,
)


class RvmBackend:
    """Robust Video Matting backend (stub for testing)."""

    def segment_frame(self, frame: np.ndarray, prompts: list[SegmentPrompt]) -> SegmentationMask:
        h, w = frame.shape[:2]
        # Soft alpha matte — center region with gradual falloff
        mask = np.zeros((h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_dist = min(h, w) * 0.4
        mask = np.clip(1.0 - dist / max_dist, 0.0, 1.0).astype(np.float32)

        return SegmentationMask(
            mask=mask,
            confidence=0.9,
            frame_index=0,
            metadata={"backend": "rvm"},
        )

    def segment_video(
        self,
        frames: Iterator[np.ndarray],
        prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
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
