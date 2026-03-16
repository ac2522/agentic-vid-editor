"""Rotoscope backend protocol and data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class SegmentPrompt:
    kind: Literal["point", "box", "text"]
    value: Any


@dataclass
class SegmentationMask:
    mask: np.ndarray
    confidence: float
    frame_index: int
    metadata: dict


@dataclass(frozen=True)
class MaskCorrection:
    kind: Literal["include_point", "exclude_point", "include_region", "exclude_region"]
    value: Any


class RotoscopeBackend(Protocol):
    """Protocol for rotoscoping/segmentation backends."""

    def segment_frame(
        self, frame: np.ndarray, prompts: list[SegmentPrompt]
    ) -> SegmentationMask: ...

    def segment_video(
        self,
        frames: Iterator[np.ndarray],
        prompts: list[SegmentPrompt],
        keyframes: list[int] | None = None,
    ) -> Iterator[SegmentationMask]: ...

    def refine_mask(
        self,
        frame: np.ndarray,
        mask: SegmentationMask,
        corrections: list[MaskCorrection],
    ) -> SegmentationMask: ...
