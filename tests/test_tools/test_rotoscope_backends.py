"""Tests for SAM, MatAnyone, and RVM backend protocol compliance."""

from __future__ import annotations

np = __import__("pytest").importorskip("numpy")

from ave.tools.rotoscope import SegmentPrompt, RotoscopeBackend  # noqa: E402


class TestSamBackendProtocol:
    def test_implements_protocol(self):
        from ave.tools.rotoscope_sam2 import SamBackend

        backend: RotoscopeBackend = SamBackend(model_size="tiny")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [SegmentPrompt(kind="point", value=(320, 240))])
        assert mask.mask.shape == (480, 640)
        assert 0.0 <= mask.confidence <= 1.0

    def test_segment_video_with_chunk_size(self):
        from ave.tools.rotoscope_sam2 import SamBackend

        backend = SamBackend(model_size="tiny")
        frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
        masks = list(
            backend.segment_video(
                iter(frames),
                [SegmentPrompt(kind="point", value=(160, 120))],
                chunk_size=2,
            )
        )
        assert len(masks) == 3

    def test_backward_compat_alias(self):
        from ave.tools.rotoscope_sam2 import Sam2Backend, SamBackend

        assert Sam2Backend is SamBackend

    def test_metadata_includes_version(self):
        from ave.tools.rotoscope_sam2 import SamBackend

        backend = SamBackend(version="3")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        assert mask.metadata["version"] == "3"
        assert mask.metadata["backend"] == "sam"


class TestMatAnyoneBackendProtocol:
    def test_implements_protocol(self):
        from ave.tools.rotoscope_matanyone import MatAnyoneBackend

        backend: RotoscopeBackend = MatAnyoneBackend()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        assert mask.mask.shape == (480, 640)
        assert mask.mask.dtype == np.float32

    def test_soft_alpha_matte(self):
        from ave.tools.rotoscope_matanyone import MatAnyoneBackend

        backend = MatAnyoneBackend()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        unique_values = np.unique(mask.mask)
        assert len(unique_values) > 2  # Soft edges

    def test_higher_confidence_than_rvm(self):
        from ave.tools.rotoscope_matanyone import MatAnyoneBackend

        backend = MatAnyoneBackend()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = backend.segment_frame(frame, [])
        assert mask.confidence >= 0.9


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
        unique_values = np.unique(mask.mask)
        assert len(unique_values) > 2
