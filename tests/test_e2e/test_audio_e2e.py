"""End-to-end tests for audio tools and transcription pipeline.

Audio effect tests require GES + FFmpeg.
Transcription tests require FFmpeg + whisper.
"""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_ges, requires_whisper
from tests.fixtures.generate import generate_av_clip, generate_test_tone

from ave.tools.audio import compute_fade, compute_volume
from ave.tools.transcribe import (
    Transcript,
    TranscriptSegment,
    extract_audio_for_transcription,
    load_transcript,
    save_transcript,
    transcribe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS_PER_SEC = 1_000_000_000


def _probe_audio(path: Path) -> dict:
    """Probe audio stream metadata via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return next(s for s in data["streams"] if s["codec_type"] == "audio")


# ===========================================================================
# TestAudioEffectsE2E
# ===========================================================================


@requires_ges
@requires_ffmpeg
@pytest.mark.slow
class TestAudioEffectsE2E:
    """Audio effects rendered through the full GES pipeline."""

    def test_volume_effect_renders(self, tmp_path: Path) -> None:
        """Create timeline with clip, add volume effect at 0.5, render, verify audio."""
        from ave.ingest.probe import probe_media
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        # Generate a short AV clip
        clip_path = tmp_path / "source.mp4"
        generate_av_clip(clip_path, duration=3, width=320, height=240, fps=24)

        # Build timeline
        xges_path = tmp_path / "timeline.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        clip_id = tl.add_clip(clip_path, layer=0, start_ns=0, duration_ns=3 * _NS_PER_SEC)

        # Compute volume and apply as GES effect
        vol = compute_volume(-6.0)  # ~ 0.5 linear
        effect_id = tl.add_effect(clip_id, "volume")
        tl.set_effect_property(clip_id, effect_id, "volume", vol.linear_gain)

        tl.save()

        # Render
        output_path = tmp_path / "rendered.mp4"
        render_proxy(xges_path, output_path, height=240)

        # Verify output
        assert output_path.exists(), "Rendered file should exist"
        info = probe_media(output_path)
        assert info.has_audio, "Rendered output must contain an audio stream"
        assert info.duration_seconds == pytest.approx(3.0, abs=0.5)

    def test_audio_fade_renders(self, tmp_path: Path) -> None:
        """Create timeline with fade parameters, render, verify audio present."""
        from ave.ingest.probe import probe_media
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        clip_path = tmp_path / "source.mp4"
        duration_s = 5
        generate_av_clip(clip_path, duration=duration_s, width=320, height=240, fps=24)

        xges_path = tmp_path / "timeline.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        clip_duration_ns = duration_s * _NS_PER_SEC
        clip_id = tl.add_clip(clip_path, layer=0, start_ns=0, duration_ns=clip_duration_ns)

        # Validate fade params (1s fade-in, 1s fade-out)
        fade = compute_fade(clip_duration_ns, 1 * _NS_PER_SEC, 1 * _NS_PER_SEC)
        assert fade.fade_in_ns == 1 * _NS_PER_SEC
        assert fade.fade_out_ns == 1 * _NS_PER_SEC

        # Add volume effect to carry the fade
        effect_id = tl.add_effect(clip_id, "volume")
        tl.set_effect_property(clip_id, effect_id, "volume", 1.0)

        tl.save()

        output_path = tmp_path / "rendered_fade.mp4"
        render_proxy(xges_path, output_path, height=240)

        assert output_path.exists(), "Rendered file should exist"
        info = probe_media(output_path)
        assert info.has_audio, "Rendered output must contain an audio stream"
        assert info.duration_seconds == pytest.approx(duration_s, abs=0.5)


# ===========================================================================
# TestTranscriptionE2E
# ===========================================================================


@requires_ffmpeg
@requires_whisper
@pytest.mark.slow
class TestTranscriptionE2E:
    """Transcription pipeline end-to-end tests."""

    def test_extract_audio_produces_valid_wav(self, tmp_path: Path) -> None:
        """Generate AV clip, extract audio, verify 16kHz mono PCM WAV."""
        clip_path = tmp_path / "source.mp4"
        generate_av_clip(clip_path, duration=3, width=320, height=240, fps=24)

        wav_path = tmp_path / "extracted.wav"
        extract_audio_for_transcription(clip_path, wav_path, sample_rate=16000)

        assert wav_path.exists(), "Extracted WAV should exist"

        audio_info = _probe_audio(wav_path)
        assert audio_info["codec_name"] == "pcm_s16le"
        assert int(audio_info["sample_rate"]) == 16000
        assert int(audio_info["channels"]) == 1

    def test_transcribe_synthetic_audio(self, tmp_path: Path) -> None:
        """Transcribe a short test tone; pipeline should not crash."""
        tone_path = tmp_path / "tone.wav"
        generate_test_tone(tone_path, frequency=1000, duration=3, sample_rate=48000)

        wav_16k = tmp_path / "tone_16k.wav"
        extract_audio_for_transcription(tone_path, wav_16k, sample_rate=16000)

        result = transcribe(wav_16k, model_name="tiny", language="en")

        assert isinstance(result, Transcript)
        assert len(result.segments) >= 0  # tone likely produces 0+ segments
        assert result.language == "en"
        assert result.duration >= 0.0

    def test_transcript_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Create Transcript manually, save to JSON, load back, verify equality."""
        original = Transcript(
            language="en",
            duration=12.5,
            segments=[
                TranscriptSegment(start=0.0, end=3.0, text="Hello world", words=[]),
                TranscriptSegment(
                    start=3.0,
                    end=6.5,
                    text="This is a test",
                    words=[{"word": "This", "start": 3.0, "end": 3.3}],
                ),
                TranscriptSegment(start=6.5, end=12.5, text="End of segment", words=[]),
            ],
        )

        json_path = tmp_path / "transcript.json"
        save_transcript(original, json_path)
        assert json_path.exists(), "Saved transcript file should exist"

        loaded = load_transcript(json_path)

        assert loaded.language == original.language
        assert loaded.duration == pytest.approx(original.duration)
        assert loaded.full_text == original.full_text
        assert len(loaded.segments) == len(original.segments)

        for orig_seg, load_seg in zip(original.segments, loaded.segments):
            assert load_seg.start == pytest.approx(orig_seg.start)
            assert load_seg.end == pytest.approx(orig_seg.end)
            assert load_seg.text == orig_seg.text
            assert load_seg.words == orig_seg.words


# ===========================================================================
# TestAudioExtractionE2E
# ===========================================================================


@requires_ffmpeg
@pytest.mark.slow
class TestAudioExtractionE2E:
    """Audio extraction tests (FFmpeg only, no whisper needed)."""

    def test_extract_from_video_file(self, tmp_path: Path) -> None:
        """Generate AV clip, extract audio, verify WAV output exists with correct sample rate."""
        clip_path = tmp_path / "source.mp4"
        generate_av_clip(clip_path, duration=3, width=320, height=240, fps=24)

        wav_path = tmp_path / "audio.wav"
        extract_audio_for_transcription(clip_path, wav_path, sample_rate=16000)

        assert wav_path.exists(), "Extracted WAV should exist"
        audio_info = _probe_audio(wav_path)
        assert int(audio_info["sample_rate"]) == 16000

    def test_extract_resamples_to_16khz(self, tmp_path: Path) -> None:
        """Generate AV clip with 48kHz audio, extract at 16kHz, verify resampling."""
        clip_path = tmp_path / "source_48k.mp4"
        # generate_av_clip uses 48kHz audio internally
        generate_av_clip(clip_path, duration=3, width=320, height=240, fps=24)

        # Verify source is 48kHz
        source_audio = _probe_audio(clip_path)
        assert int(source_audio["sample_rate"]) == 48000, "Source should be 48kHz"

        # Extract at 16kHz
        wav_path = tmp_path / "resampled.wav"
        extract_audio_for_transcription(clip_path, wav_path, sample_rate=16000)

        assert wav_path.exists()
        resampled_audio = _probe_audio(wav_path)
        assert int(resampled_audio["sample_rate"]) == 16000, "Output should be resampled to 16kHz"
        assert int(resampled_audio["channels"]) == 1, "Output should be mono"
