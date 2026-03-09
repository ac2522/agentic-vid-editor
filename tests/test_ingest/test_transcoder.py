"""Tests for ingest transcoder — any format to DNxHR HQX + proxy."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg


def _probe(path: Path) -> dict:
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
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    return next(s for s in probe["streams"] if s["codec_type"] == "video")


@requires_ffmpeg
class TestTranscoder:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.source = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.source.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.source)
        self.project = tmp_project

    def test_transcode_to_dnxhr(self):
        from ave.ingest.transcoder import transcode_to_working

        output = self.project / "assets" / "media" / "working" / "clip.mxf"
        transcode_to_working(self.source, output, codec="dnxhd", profile="dnxhr_hqx")

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["codec_name"] == "dnxhd"
        assert video["width"] == 1920
        assert video["height"] == 1080

    def test_transcode_to_proxy(self):
        from ave.ingest.transcoder import transcode_to_proxy

        output = self.project / "assets" / "media" / "proxy" / "clip.mp4"
        transcode_to_proxy(self.source, output, height=480)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["codec_name"] == "h264"
        assert video["height"] == 480

    def test_transcode_with_fps_conforming(self, fixtures_dir: Path):
        """30fps source conformed to 24fps project."""
        source_30 = fixtures_dir / "color_bars_720p30.mp4"
        if not source_30.exists():
            from tests.fixtures.generate import generate_color_bars

            generate_color_bars(source_30, duration=3, width=1280, height=720, fps=30)

        from ave.ingest.transcoder import transcode_to_working
        from fractions import Fraction

        output = self.project / "assets" / "media" / "working" / "conformed.mxf"
        transcode_to_working(
            source_30,
            output,
            codec="dnxhd",
            profile="dnxhr_hqx",
            target_fps=24,
        )

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        fps_str = video.get("r_frame_rate", "0/1")
        fps = float(Fraction(fps_str))
        assert fps == pytest.approx(24.0, abs=0.1)

    @pytest.mark.slow
    def test_full_ingest(self):
        """Full ingest: probe, transcode working + proxy, register."""
        from ave.ingest.transcoder import ingest
        from ave.ingest.registry import AssetRegistry

        registry = AssetRegistry(self.project / "assets" / "registry.json")

        entry = ingest(
            source=self.source,
            project_dir=self.project,
            asset_id="clip_001",
            registry=registry,
            project_fps=24.0,
        )

        assert entry.asset_id == "clip_001"
        assert entry.working_path.exists()
        assert entry.proxy_path.exists()
        assert registry.count() == 1
        assert registry.get("clip_001") == entry
