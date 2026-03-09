"""Tests for proxy rendering via GES pipeline."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg, requires_ges


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


@requires_ges
@requires_ffmpeg
class TestProxyRender:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_render_single_clip(self):
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        tl.save()

        output = self.project / "exports" / "proxy.mp4"
        render_proxy(self.project / "project.xges", output, height=480)

        assert output.exists()
        probe = _probe(output)
        video = _video_stream(probe)
        assert video["height"] == 480
        assert video["codec_name"] == "h264"
        assert float(probe["format"]["duration"]) > 0

    def test_render_creates_valid_output(self):
        from ave.project.timeline import Timeline
        from ave.render.proxy import render_proxy

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=1 * 1_000_000_000,
        )
        tl.save()

        output = self.project / "exports" / "test_render.mp4"
        render_proxy(self.project / "project.xges", output)

        assert output.exists()
        assert output.stat().st_size > 0
