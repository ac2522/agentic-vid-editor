"""Model download manager with user consent for large ML models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about a downloadable model."""

    name: str
    description: str
    size_bytes: int
    url: str
    checksum_sha256: str | None = None

    @property
    def size_human(self) -> str:
        """Human-readable size string."""
        if self.size_bytes >= 1_000_000_000:
            return f"{self.size_bytes / 1_000_000_000:.1f} GB"
        if self.size_bytes >= 1_000_000:
            return f"{self.size_bytes / 1_000_000:.0f} MB"
        return f"{self.size_bytes / 1_000:.0f} KB"


# Well-known model registry
KNOWN_MODELS: dict[str, ModelInfo] = {
    "sam3-large": ModelInfo(
        name="SAM 3 Large",
        description="Segment Anything Model 3 — state-of-the-art video segmentation with text prompts",
        size_bytes=2_400_000_000,  # ~2.4 GB
        url="https://dl.fbaipublicfiles.com/segment_anything_3/sam3_hiera_large.pt",
    ),
    "sam3-base": ModelInfo(
        name="SAM 3 Base+",
        description="SAM 3 Base+ — smaller, faster variant",
        size_bytes=400_000_000,  # ~400 MB
        url="https://dl.fbaipublicfiles.com/segment_anything_3/sam3_hiera_base_plus.pt",
    ),
    "sam2-large": ModelInfo(
        name="SAM 2.1 Large (legacy)",
        description="Segment Anything Model 2.1 — legacy fallback",
        size_bytes=2_400_000_000,
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    ),
    "matanyone2": ModelInfo(
        name="MatAnyone 2",
        description="Human video matting with learned quality evaluator (CVPR 2026)",
        size_bytes=800_000_000,  # ~800 MB
        url="https://huggingface.co/pq-yang/MatAnyone2/resolve/main/matanyone2.pth",
    ),
    "rvm-mobilenetv3": ModelInfo(
        name="Robust Video Matting (MobileNetV3)",
        description="Lightweight human matting — real-time on modest GPUs (legacy)",
        size_bytes=15_000_000,  # ~15 MB
        url="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth",
    ),
}


ConsentCallback = Callable[[ModelInfo], bool]


def _default_consent(model: ModelInfo) -> bool:
    """Default consent: always True (non-interactive / bypass mode)."""
    return True


class ModelManager:
    """Manages model downloads with user consent.

    Models are downloaded to a cache directory on first use.
    The consent_callback is called before each download — return True
    to proceed, False to skip. In interactive mode (web UI, CLI), this
    should prompt the user. In agent/bypass mode, it auto-consents.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        consent_callback: ConsentCallback = _default_consent,
    ) -> None:
        self._cache_dir = cache_dir or Path.home() / ".ave" / "models"
        self._consent = consent_callback

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def model_path(self, model_id: str) -> Path | None:
        """Return path to cached model, or None if not downloaded."""
        info = KNOWN_MODELS.get(model_id)
        if info is None:
            return None
        path = self._cache_dir / f"{model_id}.pt"
        return path if path.exists() else None

    def is_available(self, model_id: str) -> bool:
        """Check if model is downloaded and ready."""
        return self.model_path(model_id) is not None

    def ensure_model(self, model_id: str) -> Path:
        """Ensure model is downloaded. Asks consent if needed.

        Returns path to the model file.
        Raises RuntimeError if consent denied or download fails.
        """
        # Check cache first
        cached = self.model_path(model_id)
        if cached is not None:
            return cached

        info = KNOWN_MODELS.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")

        # Ask consent
        if not self._consent(info):
            raise RuntimeError(f"Download consent denied for {info.name} ({info.size_human})")

        # Download
        return self._download(info, model_id)

    def _download(self, info: ModelInfo, model_id: str) -> Path:
        """Download model to cache. Stub — real impl uses urllib/aiohttp."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{model_id}.pt"
        # Stub: create empty file (real impl downloads from info.url)
        path.write_bytes(b"")
        return path
