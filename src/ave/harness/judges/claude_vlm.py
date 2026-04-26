"""Claude VLM judge — frame-sampling for still-composition dimensions."""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from typing import Any

from ave.harness.judges._protocol import JudgeVerdict


def _extract_frames(video_path: Path, n: int, output_dir: Path) -> list[Path]:
    """Extract N evenly-spaced JPEG frames using ffmpeg.

    Picks first/middle/last frames via select expressions; if the video has
    fewer than N frames the eq() terms simply yield no output for the missing
    indices.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%04d.jpg"
    if n <= 0:
        return []
    if n == 1:
        select_expr = "eq(n,0)"
    else:
        select_expr = "+".join(f"eq(n,{i})" for i in (0, n // 2, n - 1))
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select='{select_expr}'",
        "-vsync", "vfr",
        "-frames:v", str(n),
        str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, check=False)
    return sorted(output_dir.glob("frame_*.jpg"))


def _build_prompt(dimension: str, prompt: str) -> str:
    return (
        f"Score this video on dimension '{dimension}' (0.0 to 1.0).\n"
        f"Rubric: {prompt}\n\n"
        f'Respond with JSON: {{"score": float, "explanation": string}}.\n'
        f"Higher score = better adherence to rubric."
    )


class ClaudeVlmJudge:
    """JudgeBackend that scores still-composition dimensions via Claude vision."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", n_frames: int = 3):
        self._model = model
        self._n_frames = n_frames

    @property
    def name(self) -> str:
        return f"claude_vlm:{self._model}"

    @property
    def supported_dimension_types(self) -> tuple[str, ...]:
        return ("still",)

    def judge_dimension(
        self,
        *,
        rendered_path: Path,
        dimension: str,
        prompt: str,
        pass_threshold: float,
    ) -> JudgeVerdict:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                frames = _extract_frames(rendered_path, self._n_frames, Path(td))
                response = self._call_api(frames, dimension, prompt)
                parsed = json.loads(response)
                score = float(parsed.get("score", 0.0))
                explanation = str(parsed.get("explanation", ""))
                return JudgeVerdict(
                    judge_name=self.name, dimension=dimension,
                    score=score, passed=score >= pass_threshold,
                    explanation=explanation,
                    metadata={"model": self._model, "n_frames": len(frames)},
                )
        except Exception as exc:
            return JudgeVerdict(
                judge_name=self.name, dimension=dimension,
                score=0.0, passed=False,
                explanation=f"claude_vlm api error: {exc}",
            )

    def _call_api(self, frames: list[Path], dimension: str, prompt: str) -> str:
        """Call Anthropic API with frames as image content blocks. Returns JSON response text."""
        from anthropic import Anthropic
        client = Anthropic()
        content: list[dict[str, Any]] = []
        for f in frames:
            data = base64.b64encode(f.read_bytes()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
            })
        content.append({"type": "text", "text": _build_prompt(dimension, prompt)})
        msg = client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text  # type: ignore[union-attr]
