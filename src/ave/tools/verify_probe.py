"""Probe-based edit verifier — uses ffprobe to extract media metrics.

Implements VerificationBackend protocol for verifying edits against
rendered segments without requiring GStreamer bindings.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ave.tools.verify import EditIntent, VerificationResult, compare_metrics


class ProbeError(Exception):
    """Raised when ffprobe operations fail."""


class ProbeVerifier:
    """Verifies edits by probing rendered segments with ffprobe.

    Implements VerificationBackend protocol.
    """

    def verify(self, intent: EditIntent, segment_path: Path) -> VerificationResult:
        """Probe the segment and compare against intent."""
        actual = self.probe_segment(segment_path)
        passed, discrepancies = compare_metrics(intent.expected_changes, actual)
        # Confidence: 1.0 when probe succeeds (deterministic measurement)
        return VerificationResult(
            passed=passed,
            intent=intent,
            actual_metrics=actual,
            discrepancies=discrepancies,
            confidence=1.0,
        )

    def probe_segment(self, path: Path) -> dict:
        """Extract metrics from a media file via ffprobe.

        Returns dict with keys: duration_seconds, width, height, has_audio,
        has_video, video_codec, audio_codec, frame_rate.
        """
        if not path.exists():
            raise ProbeError(f"File not found: {path}")

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise ProbeError(f"ffprobe failed: {e.stderr}") from e

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = float(fmt.get("duration", 0))

        frame_rate = 0.0
        if video_stream:
            r_frame_rate = video_stream.get("r_frame_rate", "0/1")
            num, den = r_frame_rate.split("/")
            if int(den) > 0:
                frame_rate = int(num) / int(den)

        return {
            "duration_seconds": duration,
            "width": int(video_stream["width"]) if video_stream else 0,
            "height": int(video_stream["height"]) if video_stream else 0,
            "has_video": video_stream is not None,
            "has_audio": audio_stream is not None,
            "video_codec": video_stream.get("codec_name", "") if video_stream else "",
            "audio_codec": audio_stream.get("codec_name", "") if audio_stream else "",
            "frame_rate": frame_rate,
        }
