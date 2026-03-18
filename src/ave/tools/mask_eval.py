"""Mask quality evaluator for rotoscoping feedback loop."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from ave.tools.rotoscope import SegmentationMask


@dataclass(frozen=True)
class MaskQuality:
    edge_smoothness: float
    temporal_stability: float
    coverage_ratio: float
    confidence_mean: float
    problem_frames: list[int] = field(default_factory=list)


class MaskEvaluator:
    """Evaluate segmentation mask quality across keyframes."""

    def __init__(self, quality_threshold: float = 0.6) -> None:
        self._threshold = quality_threshold

    def evaluate(self, masks: list[SegmentationMask], frames: list[np.ndarray]) -> MaskQuality:
        if not masks:
            return MaskQuality(
                edge_smoothness=0.0,
                temporal_stability=0.0,
                coverage_ratio=0.0,
                confidence_mean=0.0,
            )

        confidences = [m.confidence for m in masks]
        confidence_mean = sum(confidences) / len(confidences)

        # Temporal stability: IoU between consecutive masks
        ious: list[float] = []
        for i in range(len(masks) - 1):
            iou = self._mask_iou(masks[i].mask, masks[i + 1].mask)
            ious.append(iou)
        temporal_stability = sum(ious) / len(ious) if ious else 1.0

        # Edge smoothness
        smoothness_scores = [self._edge_smoothness(m.mask) for m in masks]
        edge_smoothness = sum(smoothness_scores) / len(smoothness_scores)

        # Coverage ratio
        coverage_scores: list[float] = []
        for m in masks:
            total = m.mask.size
            fg = int(np.count_nonzero(m.mask > 0.5))
            coverage_scores.append(fg / total if total > 0 else 0.0)
        coverage_ratio = sum(coverage_scores) / len(coverage_scores)

        # Problem frames
        problem_frames: list[int] = []
        for m in masks:
            if m.confidence < self._threshold:
                problem_frames.append(m.frame_index)
        for i, iou in enumerate(ious):
            if iou < self._threshold:
                problem_frames.append(masks[i + 1].frame_index)

        return MaskQuality(
            edge_smoothness=edge_smoothness,
            temporal_stability=temporal_stability,
            coverage_ratio=coverage_ratio,
            confidence_mean=confidence_mean,
            problem_frames=sorted(set(problem_frames)),
        )

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        a_bin = a > 0.5
        b_bin = b > 0.5
        intersection = int(np.logical_and(a_bin, b_bin).sum())
        union = int(np.logical_or(a_bin, b_bin).sum())
        return float(intersection / union) if union > 0 else 1.0

    @staticmethod
    def _edge_smoothness(mask: np.ndarray) -> float:
        """Higher = smoother edges."""
        binary = (mask > 0.5).astype(np.float32)
        gy = np.diff(binary, axis=0)
        gx = np.diff(binary, axis=1)
        edge_pixels = float(np.abs(gy).sum() + np.abs(gx).sum())
        if edge_pixels == 0:
            return 1.0
        h, w = mask.shape
        ideal = 2.0 * (h + w)
        return float(min(1.0, ideal / (edge_pixels + 1)))
