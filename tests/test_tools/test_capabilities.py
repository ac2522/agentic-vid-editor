"""Tests for system capability detection and graceful degradation."""

from __future__ import annotations

from ave.tools.capabilities import SystemCapabilities


class TestSystemCapabilities:
    """Test capability detection and fallback logic."""

    def test_detect_returns_capabilities(self):
        caps = SystemCapabilities.detect()
        assert isinstance(caps.gpu_available, bool)
        assert isinstance(caps.ges_available, bool)
        assert isinstance(caps.ffmpeg_available, bool)
        assert isinstance(caps.whisper_available, bool)
        # cuda_version may be None
        assert caps.cuda_version is None or isinstance(caps.cuda_version, str)

    def test_fallback_for_gpu(self):
        caps = SystemCapabilities(
            gpu_available=False,
            cuda_version=None,
            ges_available=True,
            ffmpeg_available=True,
            whisper_available=False,
        )
        fallback = caps.fallback_for("gpu")
        assert fallback is not None
        assert "cpu" in fallback.lower() or "software" in fallback.lower()

    def test_fallback_for_available_capability(self):
        caps = SystemCapabilities(
            gpu_available=True,
            cuda_version="12.0",
            ges_available=True,
            ffmpeg_available=True,
            whisper_available=True,
        )
        assert caps.fallback_for("gpu") is None
        assert caps.fallback_for("ffmpeg") is None

    def test_fallback_for_whisper(self):
        caps = SystemCapabilities(
            gpu_available=True,
            cuda_version="12.0",
            ges_available=True,
            ffmpeg_available=True,
            whisper_available=False,
        )
        fallback = caps.fallback_for("whisper")
        assert fallback is not None

    def test_fallback_for_unknown_capability(self):
        caps = SystemCapabilities(
            gpu_available=True,
            cuda_version="12.0",
            ges_available=True,
            ffmpeg_available=True,
            whisper_available=True,
        )
        assert caps.fallback_for("nonexistent") is None

    def test_to_dict(self):
        caps = SystemCapabilities(
            gpu_available=False,
            cuda_version=None,
            ges_available=True,
            ffmpeg_available=True,
            whisper_available=False,
        )
        d = caps.to_dict()
        assert d["gpu_available"] is False
        assert d["ges_available"] is True
        assert "cuda_version" in d
