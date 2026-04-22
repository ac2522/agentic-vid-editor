"""Shared test fixtures for ave test suite."""

import subprocess
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _ges_available() -> bool:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GES", "1.0")
        from gi.repository import Gst, GES

        Gst.init(None)
        GES.init()
        return True
    except (ImportError, ValueError):
        return False


def _whisper_available() -> bool:
    try:
        from pywhispercpp.model import Model  # noqa: F401

        return True
    except ImportError:
        return False


def _scenedetect_available() -> bool:
    try:
        import scenedetect  # noqa: F401

        return True
    except ImportError:
        return False


def _vision_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _aiohttp_available() -> bool:
    try:
        import aiohttp  # noqa: F401

        return True
    except ImportError:
        return False


def _gpu_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, check=True, text=True)
        return "NVIDIA" in result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# Skip markers
requires_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="FFmpeg not found")
requires_ges = pytest.mark.skipif(not _ges_available(), reason="GES not available")
requires_gpu = pytest.mark.skipif(not _gpu_available(), reason="NVIDIA GPU not available")
requires_whisper = pytest.mark.skipif(not _whisper_available(), reason="pywhispercpp not available")
requires_scenedetect = pytest.mark.skipif(
    not _scenedetect_available(), reason="PySceneDetect not available"
)
requires_vision = pytest.mark.skipif(not _vision_available(), reason="ONNX Runtime not available")
requires_aiohttp = pytest.mark.skipif(not _aiohttp_available(), reason="aiohttp not installed")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to test fixtures directory. Generates fixtures if missing."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory structure."""
    project = tmp_path / "test_project"
    (project / "assets" / "media" / "working").mkdir(parents=True)
    (project / "assets" / "media" / "proxy").mkdir(parents=True)
    (project / "cache" / "segments").mkdir(parents=True)
    (project / "cache" / "thumbnails").mkdir(parents=True)
    (project / "luts").mkdir(parents=True)
    (project / "transcriptions").mkdir(parents=True)
    (project / "exports").mkdir(parents=True)
    return project


def _inspect_available() -> bool:
    try:
        import inspect_ai  # noqa: F401

        return True
    except ImportError:
        return False


def _pyyaml_available() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except ImportError:
        return False


requires_inspect = pytest.mark.skipif(
    not _inspect_available(), reason="inspect-ai not installed (pip install ave[harness])"
)
requires_pyyaml = pytest.mark.skipif(not _pyyaml_available(), reason="pyyaml not installed")
