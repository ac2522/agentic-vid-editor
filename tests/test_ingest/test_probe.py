"""Tests for media probe utility."""

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestProbe:
    @pytest.fixture(autouse=True)
    def _setup_fixture(self, fixtures_dir: Path):
        """Ensure color bars fixture exists."""
        self.fixture = fixtures_dir / "color_bars_1080p24.mp4"
        if not self.fixture.exists():
            from tests.fixtures.generate import generate_color_bars

            generate_color_bars(self.fixture)

    def test_probe_returns_media_info(self):
        from ave.ingest.probe import probe_media

        info = probe_media(self.fixture)

        assert info.path == self.fixture
        assert info.duration_seconds > 0
        assert info.has_video
        assert info.video is not None

    def test_probe_video_stream(self):
        from ave.ingest.probe import probe_media

        info = probe_media(self.fixture)

        assert info.video.width == 1920
        assert info.video.height == 1080
        assert info.video.codec == "h264"
        assert info.video.fps == pytest.approx(24.0, abs=0.1)
        assert info.video.pix_fmt == "yuv420p"

    def test_probe_audio_stream(self, fixtures_dir: Path):
        av_clip = fixtures_dir / "av_clip_1080p24.mp4"
        if not av_clip.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(av_clip)

        from ave.ingest.probe import probe_media

        info = probe_media(av_clip)

        assert info.has_audio
        assert info.audio is not None
        assert info.audio.sample_rate == 48000
        assert info.audio.channels >= 1

    def test_probe_nonexistent_file(self):
        from ave.ingest.probe import probe_media, ProbeError

        with pytest.raises(ProbeError, match="does not exist"):
            probe_media(Path("/nonexistent/file.mp4"))

    def test_probe_invalid_file(self, tmp_path: Path):
        bad_file = tmp_path / "not_media.txt"
        bad_file.write_text("this is not media")

        from ave.ingest.probe import probe_media, ProbeError

        with pytest.raises(ProbeError, match="ffprobe failed"):
            probe_media(bad_file)


class TestProbeStreamDuration:
    """Stream-level duration_seconds should be None when not available."""

    def test_parse_video_stream_missing_duration(self):
        """When stream has no 'duration' key, duration_seconds should be None."""
        from ave.ingest.probe import _parse_video_stream

        streams = [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "r_frame_rate": "24/1",
                "bits_per_raw_sample": "8",
                # no "duration" key
            }
        ]
        video = _parse_video_stream(streams)
        assert video is not None
        assert video.duration_seconds is None

    def test_parse_audio_stream_missing_duration(self):
        """When stream has no 'duration' key, duration_seconds should be None."""
        from ave.ingest.probe import _parse_audio_stream

        streams = [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                # no "duration" key
            }
        ]
        audio = _parse_audio_stream(streams)
        assert audio is not None
        assert audio.duration_seconds is None

    def test_parse_video_stream_with_duration(self):
        """When stream has 'duration' key, it should be parsed."""
        from ave.ingest.probe import _parse_video_stream

        streams = [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "r_frame_rate": "24/1",
                "bits_per_raw_sample": "8",
                "duration": "5.0",
            }
        ]
        video = _parse_video_stream(streams)
        assert video.duration_seconds == pytest.approx(5.0)
