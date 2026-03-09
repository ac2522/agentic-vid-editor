"""Tests for GES timeline interface."""

from pathlib import Path

import pytest

from tests.conftest import requires_ges, requires_ffmpeg


@requires_ges
class TestTimelineCreate:
    def test_create_empty_timeline(self, tmp_project: Path):
        from ave.project.timeline import Timeline

        tl = Timeline.create(tmp_project / "project.xges", fps=24.0)

        assert tl.fps == 24.0
        assert tl.duration_ns == 0
        assert tl.clip_count == 0

    def test_save_and_load(self, tmp_project: Path):
        from ave.project.timeline import Timeline

        xges_path = tmp_project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        tl.set_metadata("agent:project-name", "Test Project")
        tl.save()

        assert xges_path.exists()

        tl2 = Timeline.load(xges_path)
        assert tl2.fps == 24.0
        assert tl2.get_metadata("agent:project-name") == "Test Project"


@requires_ges
@requires_ffmpeg
class TestTimelineClips:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_add_clip(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        assert clip_id is not None
        assert tl.clip_count == 1
        assert tl.duration_ns > 0

    def test_add_clip_with_inpoint(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            inpoint_ns=1_000_000_000,  # start 1s into clip
            duration_ns=2 * 1_000_000_000,
        )

        assert clip_id is not None
        assert tl.clip_count == 1

    def test_remove_clip(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        tl.remove_clip(clip_id)
        assert tl.clip_count == 0

    def test_add_multiple_clips_sequentially(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        dur = 2 * 1_000_000_000

        tl.add_clip(media_path=self.clip_path, layer=0, start_ns=0, duration_ns=dur)
        tl.add_clip(media_path=self.clip_path, layer=0, start_ns=dur, duration_ns=dur)

        assert tl.clip_count == 2

    def test_clip_metadata(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        tl.set_clip_metadata(clip_id, "agent:edit-intent", "Opening shot")
        value = tl.get_clip_metadata(clip_id, "agent:edit-intent")
        assert value == "Opening shot"

    def test_save_load_with_clips(self):
        from ave.project.timeline import Timeline

        xges_path = self.project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)
        tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        tl.save()

        tl2 = Timeline.load(xges_path)
        assert tl2.clip_count == 1


@requires_ges
@requires_ffmpeg
class TestTimelineEffects:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip
            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_add_effect_to_clip(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        effect_id = tl.add_effect(clip_id, "videobalance")
        assert effect_id is not None

    def test_set_effect_property(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        effect_id = tl.add_effect(clip_id, "videobalance")

        tl.set_effect_property(clip_id, effect_id, "saturation", 0.5)
        value = tl.get_effect_property(clip_id, effect_id, "saturation")
        assert value == pytest.approx(0.5, abs=0.01)

    def test_remove_effect(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path, layer=0, start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        effect_id = tl.add_effect(clip_id, "videobalance")
        tl.remove_effect(clip_id, effect_id)

        # No exception means success — clip has no effects
