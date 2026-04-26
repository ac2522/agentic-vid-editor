"""JudgeBackend Protocol — pluggable single-dimension judges.

A judge takes (rendered_path, dimension, prompt, pass_threshold) and returns
a JudgeVerdict. Different judges specialize by dimension type:
- "static": deterministic checks (ffprobe duration/resolution/RMS)
- "still": frame-sampling VLMs (Claude, GLM)
- "temporal": video-native VLMs (Gemini, Kimi, Qwen3-VL, Molmo 2)

The Ensemble (judges/ensemble.py) routes a scenario's rubric dimensions
to judges that declare the matching tier in supported_dimension_types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


DimensionType = str  # "static" | "still" | "temporal"


@dataclass(frozen=True)
class JudgeVerdict:
    judge_name: str
    dimension: str
    score: float
    passed: bool
    explanation: str
    metadata: dict = field(default_factory=dict)


class JudgeBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def supported_dimension_types(self) -> tuple[DimensionType, ...]: ...

    def judge_dimension(
        self,
        *,
        rendered_path: Path,
        dimension: str,
        prompt: str,
        pass_threshold: float,
    ) -> JudgeVerdict: ...
