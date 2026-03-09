"""Transcription tools — speech-to-text with word-level timestamps.

Pure logic layer for data models and validation.
Transcription engine (whispercpp) integration is conditional.
"""

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel


class TranscribeError(Exception):
    """Raised when transcription fails."""


SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".aac", ".mxf", ".mov"}


class TranscriptSegment(BaseModel):
    """A segment of transcribed text with timestamps."""

    start: float
    end: float
    text: str
    words: list[dict] = []


class Transcript(BaseModel):
    """Complete transcription result."""

    language: str
    duration: float
    segments: list[TranscriptSegment]

    @property
    def full_text(self) -> str:
        """Concatenate all segment texts."""
        return " ".join(seg.text for seg in self.segments)


def validate_transcribe_input(path: Path) -> None:
    """Validate input file for transcription.

    Raises TranscribeError if the file doesn't exist or has wrong format.
    """
    if not path.exists():
        raise TranscribeError(f"Audio file does not exist: {path}")

    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise TranscribeError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
        )


def save_transcript(transcript: Transcript, path: Path) -> None:
    """Save transcript to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript.model_dump(), indent=2))


def load_transcript(path: Path) -> Transcript:
    """Load transcript from JSON file."""
    data = json.loads(path.read_text())
    return Transcript(**data)


def extract_audio_for_transcription(
    source: Path,
    output: Path,
    sample_rate: int = 16000,
) -> None:
    """Extract and resample audio to WAV for transcription.

    Whisper models expect 16kHz mono WAV input.

    Args:
        source: Input media file (video or audio).
        output: Output WAV path.
        sample_rate: Target sample rate (default 16000 for Whisper).
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-vn",  # No video
                "-ac",
                "1",  # Mono
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise TranscribeError(f"Audio extraction failed: {e.stderr.decode()}") from e


def transcribe(
    audio_path: Path,
    model_name: str = "large-v3-turbo",
    language: str | None = None,
) -> Transcript:
    """Transcribe audio file to text with word-level timestamps.

    Uses whispercpp (GGML) when available, raises TranscribeError if not installed.

    Args:
        audio_path: Path to WAV audio (16kHz mono recommended).
        model_name: Whisper model name.
        language: Language code or None for auto-detection.

    Returns:
        Transcript with segments and word-level timestamps.
    """
    validate_transcribe_input(audio_path)

    try:
        from pywhispercpp.model import Model
    except ImportError:
        raise TranscribeError("pywhispercpp not installed. Install with: pip install pywhispercpp")

    model = Model(model_name)

    segments_raw = model.transcribe(str(audio_path), language=language)

    segments = []
    for seg in segments_raw:
        segments.append(
            TranscriptSegment(
                start=seg.t0 / 100.0,  # centiseconds to seconds
                end=seg.t1 / 100.0,
                text=seg.text.strip(),
                words=[],  # Word-level alignment depends on model support
            )
        )

    # Detect language (use provided or default to English)
    detected_language = language or "en"

    # Get audio duration from last segment
    duration = segments[-1].end if segments else 0.0

    return Transcript(
        language=detected_language,
        duration=duration,
        segments=segments,
    )
