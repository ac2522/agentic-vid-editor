"""Whisper internal word aligner — models and protocol.

Based on "Whisper Has an Internal Word Aligner" (arXiv 2509.09987).
Requires PyTorch + transformers (heavy optional dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class WordAlignment:
    """Word-level timestamp from alignment."""

    word: str
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclass(frozen=True)
class WhisperAlignerConfig:
    """Configuration for Whisper internal aligner."""

    model_size: str = "large-v3"
    head_selection: str = "auto"
    dtw_backend: str = "numpy"


@runtime_checkable
class WhisperAlignerBackend(Protocol):
    """Protocol for Whisper attention-head word alignment.

    Based on "Whisper Has an Internal Word Aligner" (arXiv 2509.09987).
    Requires PyTorch + transformers (heavy optional dependency).
    """

    def align_words(
        self,
        audio_path: Path,
        transcript: str,
        language: str = "en",
    ) -> list[WordAlignment]: ...
