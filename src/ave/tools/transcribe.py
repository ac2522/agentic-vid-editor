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


# Recommended GGML models — fast with CUDA, accurate, small footprint.
# All use Q5_0 quantization (best speed-to-accuracy ratio).
RECOMMENDED_MODELS = {
    "large-v3-turbo-q5_0": {
        "size_mb": 547,
        "description": "Best speed-to-accuracy ratio. Oct 2024, pruned decoder (4 layers vs 32).",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
    },
    "medium-q5_0": {
        "size_mb": 515,
        "description": "Good accuracy, similar size to turbo. Use if turbo has issues.",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
    },
    "large-v3-turbo": {
        "size_mb": 1620,
        "description": "Full precision turbo. Use if Q5_0 quality is insufficient.",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    },
}

DEFAULT_MODEL = "large-v3-turbo-q5_0"

# Model cache directory — shared across projects, persists in Docker volume
MODEL_CACHE_DIR = Path.home() / ".cache" / "ave" / "models"


def resolve_model(model: str) -> str:
    """Resolve a model name or path for pywhispercpp.

    Accepts:
        - A path to a .bin GGML file (returned as-is if exists)
        - A model name like "large-v3-turbo-q5_0" (auto-downloads via pywhispercpp)
        - A short alias like "large-v3-turbo" (maps to known model)

    Returns the model identifier string for pywhispercpp.Model().
    """
    # Direct path to a .bin file
    model_path = Path(model)
    if model_path.suffix == ".bin" and model_path.exists():
        return str(model_path)

    # Check cache dir for pre-downloaded GGML files
    cached = MODEL_CACHE_DIR / f"ggml-{model}.bin"
    if cached.exists():
        return str(cached)

    # Pass through to pywhispercpp (it handles download)
    return model


def transcribe(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> Transcript:
    """Transcribe audio file to text with word-level timestamps.

    Uses whispercpp (GGML) with CUDA acceleration when available.
    Default model: large-v3-turbo-q5_0 (~547MB, ~6-9s per minute of audio on GPU).

    Args:
        audio_path: Path to WAV audio (16kHz mono recommended).
        model: Model name, alias, or path to .bin GGML file.
            Recommended: "large-v3-turbo-q5_0" (default, best speed/accuracy).
        language: Language code or None for auto-detection.

    Returns:
        Transcript with segments and word-level timestamps.
    """
    validate_transcribe_input(audio_path)

    try:
        from pywhispercpp.model import Model
    except ImportError:
        raise TranscribeError(
            "Missing optional dependency 'pywhispercpp'. Install with: pip install pywhispercpp"
        ) from None

    resolved = resolve_model(model)
    m = Model(resolved)

    segments_raw = m.transcribe(str(audio_path), language=language)

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

    detected_language = language or "en"
    duration = segments[-1].end if segments else 0.0

    return Transcript(
        language=detected_language,
        duration=duration,
        segments=segments,
    )
