"""Tests for chroma key backend."""

from __future__ import annotations

import numpy as np

from ave.tools.rotoscope_chroma import ChromaKeyBackend
from ave.tools.rotoscope import SegmentPrompt


class TestChromaKeyBackend:
    def test_green_screen_keying(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 255, 0]  # Green background
        frame[30:70, 30:70] = [255, 255, 255]  # White foreground

        backend = ChromaKeyBackend()
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="text", value="green")]
        )
        assert mask.mask[50, 50] > 0.5  # foreground
        assert mask.mask[5, 5] < 0.5  # green = keyed out

    def test_blue_screen_keying(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]  # Blue (BGR)
        frame[30:70, 30:70] = [255, 255, 255]

        backend = ChromaKeyBackend()
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="text", value="blue")]
        )
        assert mask.mask[50, 50] > 0.5
        assert mask.mask[5, 5] < 0.5

    def test_refine_mask(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = [0, 255, 0]
        backend = ChromaKeyBackend()
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="text", value="green")]
        )
        from ave.tools.rotoscope import MaskCorrection

        refined = backend.refine_mask(
            frame, mask, [MaskCorrection(kind="include_point", value=(5, 5))]
        )
        assert refined.mask[5, 5] == 1.0

    def test_segment_video(self):
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        backend = ChromaKeyBackend()
        masks = list(
            backend.segment_video(
                iter(frames), [SegmentPrompt(kind="text", value="green")]
            )
        )
        assert len(masks) == 3
        assert masks[0].frame_index == 0
        assert masks[2].frame_index == 2
