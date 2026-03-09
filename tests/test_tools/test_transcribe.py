"""Unit tests for transcription tools — data models and output parsing."""

import json
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


class TestTranscriptModel:
    """Test transcript data model."""

    def test_create_segment(self):
        from ave.tools.transcribe import TranscriptSegment

        seg = TranscriptSegment(
            start=0.0,
            end=2.5,
            text="Hello world",
            words=[
                {"start": 0.0, "end": 1.0, "word": "Hello"},
                {"start": 1.1, "end": 2.5, "word": "world"},
            ],
        )
        assert seg.start == 0.0
        assert seg.end == 2.5
        assert seg.text == "Hello world"
        assert len(seg.words) == 2

    def test_create_transcript(self):
        from ave.tools.transcribe import Transcript, TranscriptSegment

        transcript = Transcript(
            language="en",
            duration=10.0,
            segments=[
                TranscriptSegment(start=0.0, end=5.0, text="First sentence.", words=[]),
                TranscriptSegment(start=5.0, end=10.0, text="Second sentence.", words=[]),
            ],
        )
        assert transcript.language == "en"
        assert transcript.duration == 10.0
        assert len(transcript.segments) == 2

    def test_transcript_full_text(self):
        from ave.tools.transcribe import Transcript, TranscriptSegment

        transcript = Transcript(
            language="en",
            duration=10.0,
            segments=[
                TranscriptSegment(start=0.0, end=5.0, text="Hello.", words=[]),
                TranscriptSegment(start=5.0, end=10.0, text="World.", words=[]),
            ],
        )
        assert transcript.full_text == "Hello. World."

    def test_transcript_to_json(self):
        from ave.tools.transcribe import Transcript, TranscriptSegment

        transcript = Transcript(
            language="en",
            duration=5.0,
            segments=[
                TranscriptSegment(
                    start=0.0,
                    end=2.5,
                    text="Hello world",
                    words=[
                        {"start": 0.0, "end": 1.0, "word": "Hello"},
                        {"start": 1.1, "end": 2.5, "word": "world"},
                    ],
                ),
            ],
        )
        data = transcript.model_dump()
        assert data["language"] == "en"
        assert len(data["segments"]) == 1
        assert data["segments"][0]["text"] == "Hello world"

    def test_transcript_from_json(self):
        from ave.tools.transcribe import Transcript

        data = {
            "language": "en",
            "duration": 5.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "words": [
                        {"start": 0.0, "end": 1.0, "word": "Hello"},
                    ],
                },
            ],
        }
        transcript = Transcript(**data)
        assert transcript.segments[0].text == "Hello world"

    def test_transcript_save_load(self, tmp_path: Path):
        from ave.tools.transcribe import (
            Transcript,
            TranscriptSegment,
            save_transcript,
            load_transcript,
        )

        transcript = Transcript(
            language="en",
            duration=5.0,
            segments=[
                TranscriptSegment(start=0.0, end=5.0, text="Test.", words=[]),
            ],
        )

        path = tmp_path / "transcript.json"
        save_transcript(transcript, path)

        assert path.exists()

        loaded = load_transcript(path)
        assert loaded.language == "en"
        assert loaded.segments[0].text == "Test."
        assert loaded.duration == 5.0


class TestTranscribeValidation:
    """Test transcription input validation."""

    def test_validate_audio_path_missing(self):
        from ave.tools.transcribe import validate_transcribe_input, TranscribeError

        with pytest.raises(TranscribeError, match="does not exist"):
            validate_transcribe_input(Path("/nonexistent/audio.wav"))

    def test_validate_audio_path_wrong_extension(self, tmp_path: Path):
        from ave.tools.transcribe import validate_transcribe_input, TranscribeError

        bad = tmp_path / "file.txt"
        bad.write_text("not audio")

        with pytest.raises(TranscribeError, match="format"):
            validate_transcribe_input(bad)

    def test_validate_audio_path_valid(self, tmp_path: Path):
        from ave.tools.transcribe import validate_transcribe_input

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        # Should not raise
        validate_transcribe_input(audio)

    def test_validate_accepts_mp4(self, tmp_path: Path):
        from ave.tools.transcribe import validate_transcribe_input

        audio = tmp_path / "test.mp4"
        audio.write_bytes(b"\x00" * 100)

        validate_transcribe_input(audio)

    def test_validate_accepts_mp3(self, tmp_path: Path):
        from ave.tools.transcribe import validate_transcribe_input

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 100)

        validate_transcribe_input(audio)


class TestModelResolution:
    """Test model name/path resolution."""

    def test_default_model_is_turbo_q5(self):
        from ave.tools.transcribe import DEFAULT_MODEL

        assert DEFAULT_MODEL == "large-v3-turbo-q5_0"

    def test_resolve_model_passthrough(self):
        from ave.tools.transcribe import resolve_model

        assert resolve_model("large-v3-turbo-q5_0") == "large-v3-turbo-q5_0"

    def test_resolve_model_bin_path(self, tmp_path: Path):
        from ave.tools.transcribe import resolve_model

        model_file = tmp_path / "ggml-custom.bin"
        model_file.write_bytes(b"\x00" * 100)

        assert resolve_model(str(model_file)) == str(model_file)

    def test_resolve_model_missing_bin_passthrough(self):
        from ave.tools.transcribe import resolve_model

        # Non-existent .bin path falls through to name-based resolution
        result = resolve_model("/nonexistent/model.bin")
        assert result == "/nonexistent/model.bin"

    def test_resolve_model_cached_file(self, tmp_path: Path, monkeypatch):
        import ave.tools.transcribe as mod
        from ave.tools.transcribe import resolve_model

        monkeypatch.setattr(mod, "MODEL_CACHE_DIR", tmp_path)

        cached = tmp_path / "ggml-medium-q5_0.bin"
        cached.write_bytes(b"\x00" * 100)

        assert resolve_model("medium-q5_0") == str(cached)

    def test_recommended_models_have_required_fields(self):
        from ave.tools.transcribe import RECOMMENDED_MODELS

        for name, info in RECOMMENDED_MODELS.items():
            assert "size_mb" in info, f"{name} missing size_mb"
            assert "url" in info, f"{name} missing url"
            assert info["url"].endswith(".bin"), f"{name} url should point to .bin"


@requires_ffmpeg
class TestExtractAudio:
    """Test audio extraction for transcription."""

    def test_extract_audio_from_video(self, fixtures_dir: Path, tmp_path: Path):
        """Extract WAV audio from a video file for transcription."""
        source = fixtures_dir / "av_clip_1080p24.mp4"
        if not source.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(source)

        from ave.tools.transcribe import extract_audio_for_transcription

        output = tmp_path / "audio.wav"
        extract_audio_for_transcription(source, output)

        assert output.exists()
        assert output.stat().st_size > 0

        # Verify it's a valid WAV
        import subprocess

        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(output)],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        audio_stream = next(s for s in data["streams"] if s["codec_type"] == "audio")
        assert audio_stream["codec_name"] == "pcm_s16le"
        assert int(audio_stream["sample_rate"]) == 16000  # Whisper expects 16kHz

    def test_extract_audio_from_audio(self, fixtures_dir: Path, tmp_path: Path):
        """Extract (resample) audio from an audio-only file."""
        source = fixtures_dir / "test_tone_1khz.wav"
        if not source.exists():
            from tests.fixtures.generate import generate_test_tone

            generate_test_tone(source)

        from ave.tools.transcribe import extract_audio_for_transcription

        output = tmp_path / "resampled.wav"
        extract_audio_for_transcription(source, output)

        assert output.exists()
