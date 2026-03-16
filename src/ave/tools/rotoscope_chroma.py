"""Chroma key backend — deterministic color-space keying."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ave.tools.rotoscope import (
    MaskCorrection,
    SegmentationMask,
    SegmentPrompt,
)

_KEY_COLORS = {
    "green": np.array([0, 255, 0], dtype=np.float32),
    "blue": np.array([255, 0, 0], dtype=np.float32),  # BGR
}


class ChromaKeyBackend:
    """Deterministic chroma keying — not ML-based."""

    def __init__(
        self,
        tolerance: float = 0.3,
        spill_suppression: float = 0.5,
    ) -> None:
        self._tolerance = tolerance
        self._spill = spill_suppression

    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask:
        key_color_name = "green"
        for p in prompts:
            if p.kind == "text" and isinstance(p.value, str):
                val = p.value.lower()
                if "blue" in val:
                    key_color_name = "blue"

        key_color = _KEY_COLORS[key_color_name]
        frame_f = frame.astype(np.float32)
        diff = np.linalg.norm(frame_f - key_color, axis=2) / 441.67
        mask = np.where(diff > self._tolerance, 1.0, 0.0).astype(np.float32)

        return SegmentationMask(
            mask=mask,
            confidence=0.95,
            frame_index=0,
            metadata={"key_color": key_color_name, "tolerance": self._tolerance},
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
        for c in corrections:
            if c.kind == "include_point" and isinstance(c.value, tuple):
                y, x = c.value
                refined[max(0, y - 5) : y + 5, max(0, x - 5) : x + 5] = 1.0
            elif c.kind == "exclude_point" and isinstance(c.value, tuple):
                y, x = c.value
                refined[max(0, y - 5) : y + 5, max(0, x - 5) : x + 5] = 0.0
        return SegmentationMask(
            mask=refined,
            confidence=mask.confidence,
            frame_index=mask.frame_index,
            metadata=mask.metadata,
        )
