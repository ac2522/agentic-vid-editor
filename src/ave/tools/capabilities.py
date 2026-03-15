"""System capability detection for graceful degradation.

Detects available hardware and software capabilities at runtime,
mirroring the conftest.py skip markers but packaged for production use.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class SystemCapabilities:
    """Available system capabilities."""

    gpu_available: bool
    cuda_version: str | None
    ges_available: bool
    ffmpeg_available: bool
    whisper_available: bool

    @classmethod
    def detect(cls) -> SystemCapabilities:
        """Detect current system capabilities."""
        return cls(
            gpu_available=_gpu_available(),
            cuda_version=_cuda_version(),
            ges_available=_ges_available(),
            ffmpeg_available=_ffmpeg_available(),
            whisper_available=_whisper_available(),
        )

    def fallback_for(self, capability: str) -> str | None:
        """Return fallback description for a missing capability.

        Returns None if the capability is available or unknown.
        """
        fallbacks = {
            "gpu": (self.gpu_available, "CPU software rendering (slower but functional)"),
            "ges": (self.ges_available, "XGES file manipulation without live GES pipeline"),
            "ffmpeg": (self.ffmpeg_available, "No media processing available"),
            "whisper": (self.whisper_available, "Transcription unavailable; manual transcript input supported"),
        }
        entry = fallbacks.get(capability)
        if entry is None:
            return None
        available, description = entry
        return None if available else description

    def to_dict(self) -> dict:
        """Serializable capability summary."""
        return {
            "gpu_available": self.gpu_available,
            "cuda_version": self.cuda_version,
            "ges_available": self.ges_available,
            "ffmpeg_available": self.ffmpeg_available,
            "whisper_available": self.whisper_available,
        }


def _gpu_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cuda_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _ges_available() -> bool:
    try:
        import gi  # noqa: F401

        gi.require_version("Gst", "1.0")
        gi.require_version("GES", "1.0")
        from gi.repository import Gst, GES  # noqa: F401

        return True
    except (ImportError, ValueError):
        return False


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _whisper_available() -> bool:
    try:
        import pywhispercpp  # noqa: F401

        return True
    except ImportError:
        return False
