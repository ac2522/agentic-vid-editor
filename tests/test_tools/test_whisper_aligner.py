"""Tests for Whisper internal aligner models and protocol."""

from pathlib import Path

import pytest

from ave.tools.whisper_aligner import (
    WhisperAlignerBackend,
    WhisperAlignerConfig,
    WordAlignment,
)


class TestWordAlignment:
    def test_creation_and_fields(self):
        wa = WordAlignment(word="hello", start_seconds=1.0, end_seconds=1.5, confidence=0.95)
        assert wa.word == "hello"
        assert wa.start_seconds == 1.0
        assert wa.end_seconds == 1.5
        assert wa.confidence == 0.95

    def test_frozen(self):
        wa = WordAlignment(word="hi", start_seconds=0.0, end_seconds=0.1, confidence=0.9)
        with pytest.raises(AttributeError):
            wa.word = "bye"  # type: ignore[misc]


class TestWhisperAlignerConfig:
    def test_defaults(self):
        cfg = WhisperAlignerConfig()
        assert cfg.model_size == "large-v3"
        assert cfg.head_selection == "auto"
        assert cfg.dtw_backend == "numpy"

    def test_custom_values(self):
        cfg = WhisperAlignerConfig(model_size="medium", head_selection="manual", dtw_backend="scipy")
        assert cfg.model_size == "medium"
        assert cfg.head_selection == "manual"
        assert cfg.dtw_backend == "scipy"


class TestWhisperAlignerBackendProtocol:
    def test_concrete_class_satisfies_protocol(self):
        """A concrete class implementing align_words satisfies the protocol via structural typing."""

        class _MockAligner:
            def align_words(
                self,
                audio_path: Path,
                transcript: str,
                language: str = "en",
            ) -> list[WordAlignment]:
                return [
                    WordAlignment(word="test", start_seconds=0.0, end_seconds=0.5, confidence=1.0)
                ]

        aligner: WhisperAlignerBackend = _MockAligner()
        result = aligner.align_words(Path("/tmp/audio.wav"), "test")
        assert len(result) == 1
        assert result[0].word == "test"
        assert isinstance(aligner, WhisperAlignerBackend)
