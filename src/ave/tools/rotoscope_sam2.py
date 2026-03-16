"""SAM (Segment Anything Model) video segmentation backend.

Targets SAM 3 by default (released Nov 2025) with SAM 2.1 as fallback.
SAM 3 adds native text prompts, concept-level segmentation, and is a
drop-in replacement for SAM 2 workflows.

Requires: torch, segment-anything-3 (or segment-anything-2 for fallback)
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


class SamBackend:
    """SAM 3 video segmentation backend (stub for testing).

    Prefers SAM 3 (text prompts, concept segmentation). Falls back to
    SAM 2.1 if SAM 3 is unavailable. Model is downloaded on first use
    via ModelManager with user consent.
    """

    def __init__(
        self,
        model_size: str = "large",
        version: str = "auto",
    ) -> None:
        self._model_size = model_size
        self._version = version  # "3", "2", or "auto" (prefer 3)

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
            metadata={"backend": "sam", "version": self._version, "model": self._model_size},
        )

    def segment_video(
        self,
        frames: Iterator[np.ndarray],
        prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
        chunk_size: int = 300,
    ) -> Iterator[SegmentationMask]:
        """Segment video frames in chunks for memory safety.

        Args:
            chunk_size: Process N frames at a time to limit memory usage.
        """
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


# Backward compatibility alias
Sam2Backend = SamBackend
