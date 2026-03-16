"""Tests for SAM 2 and RVM backend protocol compliance."""

from __future__ import annotations

import numpy as np

from ave.tools.rotoscope import SegmentPrompt, RotoscopeBackend


class TestSam2BackendProtocol:
    def test_implements_protocol(self):
        from ave.tools.rotoscope_sam2 import Sam2Backend

        backend: RotoscopeBackend = Sam2Backend(model_size="tiny")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(
            frame, [SegmentPrompt(kind="point", value=(320, 240))]
        )
        assert mask.mask.shape == (480, 640)
        assert 0.0 <= mask.confidence <= 1.0

    def test_segment_video(self):
        from ave.tools.rotoscope_sam2 import Sam2Backend

        backend = Sam2Backend(model_size="tiny")
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
        masks = list(
            backend.segment_video(
                iter(frames), [SegmentPrompt(kind="point", value=(160, 120))]
            )
        )
        assert len(masks) == 3


class TestRvmBackendProtocol:
    def test_implements_protocol(self):
        from ave.tools.rotoscope_rvm import RvmBackend

        backend: RotoscopeBackend = RvmBackend()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        assert mask.mask.shape == (480, 640)
        assert mask.mask.dtype == np.float32

    def test_soft_alpha_matte(self):
        from ave.tools.rotoscope_rvm import RvmBackend

        backend = RvmBackend()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        # RVM should produce soft edges (values between 0 and 1)
        unique_values = np.unique(mask.mask)
        assert len(unique_values) > 2  # Not just 0 and 1
