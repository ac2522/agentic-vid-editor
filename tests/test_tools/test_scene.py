"""Unit tests for scene detection — data models, pure logic, and backends."""

import subprocess

import pytest

from ave.tools.scene import SceneBoundary, SceneError
from tests.conftest import requires_ffmpeg, requires_scenedetect


class TestSceneBoundary:
    def test_create_boundary(self):
        b = SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0)
        assert b.start_ns == 0
        assert b.end_ns == 2_000_000_000
        assert b.fps == 24.0

    def test_duration_ns(self):
        b = SceneBoundary(start_ns=1_000_000_000, end_ns=3_000_000_000, fps=24.0)
        assert b.duration_ns == 2_000_000_000

    def test_start_frame_derived(self):
        b = SceneBoundary(start_ns=0, end_ns=1_000_000_000, fps=24.0)
        assert b.start_frame == 0

    def test_end_frame_derived(self):
        b = SceneBoundary(start_ns=0, end_ns=1_000_000_000, fps=24.0)
        assert b.end_frame == 24

    def test_mid_frame_derived(self):
        b = SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0)
        assert b.mid_frame == 24  # middle of 0-48

    def test_boundary_with_offset(self):
        b = SceneBoundary(start_ns=5_000_000_000, end_ns=7_000_000_000, fps=30.0)
        assert b.start_frame == 150
        assert b.end_frame == 210

    def test_metadata_key_constants_exist(self):
        from ave.tools.scene import AGENT_META_SCENE_ID

        assert AGENT_META_SCENE_ID == "agent:scene-id"

    def test_scene_error_is_exception(self):
        err = SceneError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"


@requires_ffmpeg
class TestExtractKeyframes:
    def test_extract_middle_keyframes(self, tmp_path):
        from ave.tools.scene import extract_keyframes
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=4, width=320, height=240)

        boundaries = [
            SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0),
            SceneBoundary(start_ns=2_000_000_000, end_ns=4_000_000_000, fps=24.0),
        ]

        output_dir = tmp_path / "keyframes"
        paths = extract_keyframes(clip, boundaries, output_dir, strategy="middle")

        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".jpg"

    def test_extract_first_keyframes(self, tmp_path):
        from ave.tools.scene import extract_keyframes
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=2, width=320, height=240)

        boundaries = [
            SceneBoundary(start_ns=0, end_ns=2_000_000_000, fps=24.0),
        ]

        output_dir = tmp_path / "keyframes"
        paths = extract_keyframes(clip, boundaries, output_dir, strategy="first")

        assert len(paths) == 1
        assert paths[0].exists()

    def test_invalid_strategy_raises(self):
        from pathlib import Path

        from ave.tools.scene import extract_keyframes

        with pytest.raises(SceneError, match="Unknown strategy"):
            extract_keyframes(
                Path("/nonexistent.mp4"),
                [SceneBoundary(start_ns=0, end_ns=1_000_000_000, fps=24.0)],
                Path("/tmp/out"),
                strategy="invalid",
            )


@requires_scenedetect
@requires_ffmpeg
class TestPySceneDetectBackend:
    def test_detect_scenes_on_synthetic_video(self, tmp_path):
        """Detect cuts in a video with known scene changes."""
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_color_bars

        # Create two visually different clips and concatenate them
        clip_a = tmp_path / "clip_a.mp4"
        clip_b = tmp_path / "clip_b.mp4"
        combined = tmp_path / "combined.mp4"

        generate_color_bars(clip_a, duration=2, width=320, height=240)
        generate_color_bars(clip_b, duration=2, width=320, height=240, fps=24)

        # Concatenate with FFmpeg (creates a hard cut)
        filelist = tmp_path / "filelist.txt"
        filelist.write_text(f"file '{clip_a}'\nfile '{clip_b}'\n")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(filelist),
                "-c",
                "copy",
                str(combined),
            ],
            capture_output=True,
            check=True,
        )

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(combined, threshold=27.0)
        # Should find at least 1 scene (the whole video) or 2 if cut detected
        assert len(scenes) >= 1
        assert all(isinstance(s, SceneBoundary) for s in scenes)
        assert scenes[0].start_ns == 0

    def test_detect_scenes_returns_valid_timestamps(self, tmp_path):
        """All returned boundaries have valid timestamp ranges."""
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=3, width=320, height=240)

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(clip, threshold=27.0)
        assert len(scenes) >= 1
        for scene in scenes:
            assert scene.start_ns >= 0
            assert scene.end_ns > scene.start_ns
            assert scene.fps > 0

    def test_detect_scenes_adaptive_detector(self, tmp_path):
        """Adaptive detector runs without error and returns results."""
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=3, width=320, height=240)

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(clip, threshold=3.0, detector="adaptive")
        assert len(scenes) >= 1

    def test_invalid_detector_raises(self, tmp_path):
        """Unknown detector type raises SceneError."""
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend
        from tests.fixtures.generate import generate_av_clip

        clip = tmp_path / "clip.mp4"
        generate_av_clip(clip, duration=1, width=320, height=240)

        backend = PySceneDetectBackend()
        with pytest.raises(SceneError, match="Unknown detector"):
            backend.detect_scenes(clip, threshold=27.0, detector="invalid")
