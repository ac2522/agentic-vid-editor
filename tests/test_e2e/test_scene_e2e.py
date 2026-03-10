"""E2E tests for scene detection pipeline."""

import subprocess

import pytest

from tests.conftest import requires_ffmpeg, requires_scenedetect
from ave.tools.scene import extract_keyframes


@requires_scenedetect
@requires_ffmpeg
@pytest.mark.slow
class TestSceneDetectionE2E:
    def test_detect_and_extract_keyframes(self, tmp_path):
        """Full pipeline: generate video with cuts -> detect scenes -> extract keyframes."""
        from tests.fixtures.generate import generate_color_bars
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend

        # Create video with a hard cut (two different clips concatenated)
        clip_a = tmp_path / "a.mp4"
        combined = tmp_path / "combined.mp4"

        generate_color_bars(clip_a, duration=2, width=320, height=240)

        # Generate a solid red clip for contrast
        clip_b = tmp_path / "b.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:size=320x240:rate=24:duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(clip_b),
            ],
            capture_output=True,
            check=True,
        )

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

        # Detect scenes
        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(combined, threshold=27.0)
        assert len(scenes) >= 1

        # Extract keyframes
        kf_dir = tmp_path / "keyframes"
        keyframes = extract_keyframes(combined, scenes, kf_dir, strategy="middle")
        assert len(keyframes) == len(scenes)
        for kf in keyframes:
            assert kf.exists()
            assert kf.stat().st_size > 0

    def test_single_scene_video(self, tmp_path):
        """A single-shot video should return one scene boundary."""
        from tests.fixtures.generate import generate_av_clip
        from ave.tools.scene_pyscenedetect import PySceneDetectBackend

        clip = tmp_path / "single.mp4"
        generate_av_clip(clip, duration=3, width=320, height=240)

        backend = PySceneDetectBackend()
        scenes = backend.detect_scenes(clip, threshold=27.0)
        assert len(scenes) >= 1
        assert scenes[0].start_ns == 0
        assert scenes[-1].end_ns > 0
