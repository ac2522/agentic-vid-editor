"""SAM 2 video segmentation backend.

Requires: torch, segment-anything-2
Stub implementation for testing — returns synthetic masks.
Real implementation loads model on first use inside Docker with CUDA.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ave.tools.rotoscope import (
    MaskCorrection,
    SegmentationMask,
    SegmentPrompt,
)


class Sam2Backend:
    """SAM 2 video segmentation backend (stub for testing)."""

    def __init__(self, model_size: str = "large") -> None:
        self._model_size = model_size

    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
        return SegmentationMask(
            mask=mask,
            confidence=0.85,
            frame_index=0,
            metadata={"backend": "sam2", "model": self._model_size},
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
        refined = mask.mask.copy()
        return SegmentationMask(
            mask=refined,
            confidence=min(1.0, mask.confidence + 0.05),
            frame_index=mask.frame_index,
            metadata=mask.metadata,
        )
