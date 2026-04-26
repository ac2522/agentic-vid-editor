"""Deterministic ffprobe-backed judge.

Scores static rubric dimensions (duration, resolution, aspect_ratio,
audio_rms, format) by calling ffprobe/ffmpeg on the rendered output.
No model inference, no API cost.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from ave.harness.judges._protocol import JudgeVerdict


DETERMINISTIC_DIMENSIONS = ("duration", "resolution", "aspect_ratio", "audio_rms", "format")


def _parse_kv(prompt: str) -> dict[str, str]:
    """Parse a 'key=value,key=value' style prompt suffix.

    Tolerates leading free-form text by only collecting tokens that match the
    'key=value' shape. Whitespace around tokens is stripped.
    """
    out: dict[str, str] = {}
    for token in re.split(r"[,\s]+", prompt.strip()):
        if "=" in token:
            key, _, value = token.partition("=")
            key = key.strip()
            value = value.strip()
            if key:
                out[key] = value
    return out


def _ffprobe_format_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def _ffprobe_video_stream(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_streams",
        "-of", "json",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("no video stream found")
    return streams[0]


def _ffmpeg_volumedetect_mean(path: Path) -> float:
    """Run ffmpeg volumedetect and return mean_volume in dB."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not on PATH")
    cmd = [
        ffmpeg, "-nostats", "-i", str(path),
        "-af", "volumedetect",
        "-vn", "-sn", "-dn",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", r.stderr)
    if not match:
        raise RuntimeError("ffmpeg volumedetect produced no mean_volume")
    return float(match.group(1))


class DeterministicJudge:
    """Judge backend that scores static rubric dimensions via ffprobe/ffmpeg."""

    @property
    def name(self) -> str:
        return "ffprobe"

    @property
    def supported_dimension_types(self) -> tuple[str, ...]:
        return ("static",)

    def judge_dimension(
        self,
        *,
        rendered_path: Path,
        dimension: str,
        prompt: str,
        pass_threshold: float,
    ) -> JudgeVerdict:
        if dimension == "duration":
            return self._judge_duration(rendered_path, prompt, pass_threshold)
        if dimension == "resolution":
            return self._judge_resolution(rendered_path, prompt, pass_threshold)
        if dimension == "aspect_ratio":
            return self._judge_aspect_ratio(rendered_path, prompt, pass_threshold)
        if dimension == "audio_rms":
            return self._judge_audio_rms(rendered_path, prompt, pass_threshold)
        if dimension == "format":
            return self._judge_format(rendered_path, prompt, pass_threshold)
        return JudgeVerdict(
            judge_name=self.name,
            dimension=dimension,
            score=0.0,
            passed=False,
            explanation=f"unsupported dimension: {dimension}",
        )

    def _judge_duration(self, path: Path, prompt: str, pass_threshold: float) -> JudgeVerdict:
        kv = _parse_kv(prompt)
        try:
            target = float(kv["target"])
        except (KeyError, ValueError):
            return JudgeVerdict(
                judge_name=self.name, dimension="duration", score=0.0, passed=False,
                explanation="duration prompt missing 'target=<float>'",
            )
        tolerance = float(kv.get("tolerance", "0.1"))
        try:
            actual = _ffprobe_format_duration(path)
        except (subprocess.CalledProcessError, ValueError, RuntimeError, OSError) as exc:
            return JudgeVerdict(
                judge_name=self.name, dimension="duration", score=0.0, passed=False,
                explanation=f"ffprobe duration failed: {exc}",
            )
        delta = abs(actual - target)
        score = max(0.0, 1.0 - (delta / tolerance)) if tolerance > 0 else (1.0 if delta == 0 else 0.0)
        passed = delta <= tolerance and score >= pass_threshold
        return JudgeVerdict(
            judge_name=self.name, dimension="duration",
            score=score, passed=passed,
            explanation=f"duration actual={actual:.3f}s target={target:.3f}s tolerance={tolerance}",
            metadata={"actual": actual, "target": target, "tolerance": tolerance},
        )

    def _judge_resolution(self, path: Path, prompt: str, pass_threshold: float) -> JudgeVerdict:
        kv = _parse_kv(prompt)
        expect = kv.get("expect", "")
        match = re.match(r"^(\d+)x(\d+)$", expect)
        if not match:
            return JudgeVerdict(
                judge_name=self.name, dimension="resolution", score=0.0, passed=False,
                explanation="resolution prompt missing 'expect=<W>x<H>'",
            )
        want_w, want_h = int(match.group(1)), int(match.group(2))
        try:
            stream = _ffprobe_video_stream(path)
        except (subprocess.CalledProcessError, ValueError, RuntimeError, OSError) as exc:
            return JudgeVerdict(
                judge_name=self.name, dimension="resolution", score=0.0, passed=False,
                explanation=f"ffprobe resolution failed: {exc}",
            )
        actual_w = int(stream.get("width", 0))
        actual_h = int(stream.get("height", 0))
        passed = actual_w == want_w and actual_h == want_h
        score = 1.0 if passed else 0.0
        return JudgeVerdict(
            judge_name=self.name, dimension="resolution",
            score=score, passed=passed and score >= pass_threshold,
            explanation=f"resolution actual={actual_w}x{actual_h} expect={want_w}x{want_h}",
            metadata={"actual_w": actual_w, "actual_h": actual_h, "expect_w": want_w, "expect_h": want_h},
        )

    def _judge_aspect_ratio(self, path: Path, prompt: str, pass_threshold: float) -> JudgeVerdict:
        kv = _parse_kv(prompt)
        expect_raw = kv.get("expect", "")
        try:
            if ":" in expect_raw:
                a, b = expect_raw.split(":", 1)
                want = float(a) / float(b)
            else:
                want = float(expect_raw)
        except (ValueError, ZeroDivisionError):
            return JudgeVerdict(
                judge_name=self.name, dimension="aspect_ratio", score=0.0, passed=False,
                explanation="aspect_ratio prompt missing 'expect=<float>' or '<W>:<H>'",
            )
        tolerance = float(kv.get("tolerance", "0.02"))
        try:
            stream = _ffprobe_video_stream(path)
        except (subprocess.CalledProcessError, ValueError, RuntimeError, OSError) as exc:
            return JudgeVerdict(
                judge_name=self.name, dimension="aspect_ratio", score=0.0, passed=False,
                explanation=f"ffprobe aspect_ratio failed: {exc}",
            )
        actual_w = int(stream.get("width", 0))
        actual_h = int(stream.get("height", 0))
        if actual_h == 0:
            return JudgeVerdict(
                judge_name=self.name, dimension="aspect_ratio", score=0.0, passed=False,
                explanation="ffprobe reported zero height",
            )
        actual = actual_w / actual_h
        delta = abs(actual - want)
        score = max(0.0, 1.0 - (delta / tolerance)) if tolerance > 0 else (1.0 if delta == 0 else 0.0)
        passed = delta <= tolerance and score >= pass_threshold
        return JudgeVerdict(
            judge_name=self.name, dimension="aspect_ratio",
            score=score, passed=passed,
            explanation=f"aspect_ratio actual={actual:.4f} expect={want:.4f} tolerance={tolerance}",
            metadata={"actual": actual, "expect": want, "tolerance": tolerance},
        )

    def _judge_audio_rms(self, path: Path, prompt: str, pass_threshold: float) -> JudgeVerdict:
        kv = _parse_kv(prompt)
        try:
            min_db = float(kv["min"])
            max_db = float(kv["max"])
        except (KeyError, ValueError):
            return JudgeVerdict(
                judge_name=self.name, dimension="audio_rms", score=0.0, passed=False,
                explanation="audio_rms prompt missing 'min=<dB>,max=<dB>'",
            )
        try:
            mean_db = _ffmpeg_volumedetect_mean(path)
        except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
            return JudgeVerdict(
                judge_name=self.name, dimension="audio_rms", score=0.0, passed=False,
                explanation=f"audio_rms unsupported (volumedetect failed): {exc}",
            )
        passed = min_db <= mean_db <= max_db
        score = 1.0 if passed else 0.0
        return JudgeVerdict(
            judge_name=self.name, dimension="audio_rms",
            score=score, passed=passed and score >= pass_threshold,
            explanation=f"audio_rms mean={mean_db:.2f}dB window=[{min_db},{max_db}]",
            metadata={"mean_db": mean_db, "min_db": min_db, "max_db": max_db},
        )

    def _judge_format(self, path: Path, prompt: str, pass_threshold: float) -> JudgeVerdict:
        kv = _parse_kv(prompt)
        expect = kv.get("expect", "").lower()
        if not expect:
            return JudgeVerdict(
                judge_name=self.name, dimension="format", score=0.0, passed=False,
                explanation="format prompt missing 'expect=<container>'",
            )
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=format_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            return JudgeVerdict(
                judge_name=self.name, dimension="format", score=0.0, passed=False,
                explanation=f"ffprobe format failed: {exc}",
            )
        actual = r.stdout.strip().lower()
        passed = expect in {p.strip() for p in actual.split(",")} or expect == actual
        score = 1.0 if passed else 0.0
        return JudgeVerdict(
            judge_name=self.name, dimension="format",
            score=score, passed=passed and score >= pass_threshold,
            explanation=f"format actual={actual} expect={expect}",
            metadata={"actual": actual, "expect": expect},
        )
