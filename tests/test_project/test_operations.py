"""Integration tests for timeline operations — requires GES."""

from pathlib import Path

import pytest

from tests.conftest import requires_ges, requires_ffmpeg


@requires_ges
@requires_ffmpeg
class TestTimelineTrim:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_trim_clip(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import trim_clip

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=5_000_000_000,
        )

        trim_clip(tl, clip_id, in_ns=1_000_000_000, out_ns=4_000_000_000)

        # Verify the clip was trimmed
        clip = tl._get_clip(clip_id)
        assert clip.get_duration() == 3_000_000_000
        assert clip.get_inpoint() == 1_000_000_000

    def test_trim_preserves_position(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import trim_clip

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=2_000_000_000,
            duration_ns=5_000_000_000,
        )

        trim_clip(tl, clip_id, in_ns=1_000_000_000, out_ns=3_000_000_000)

        clip = tl._get_clip(clip_id)
        assert clip.get_start() == 2_000_000_000  # Position unchanged
        assert clip.get_duration() == 2_000_000_000


@requires_ges
@requires_ffmpeg
class TestTimelineSplit:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_split_clip(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import split_clip

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=4_000_000_000,
        )

        left_id, right_id = split_clip(tl, clip_id, position_ns=2_000_000_000)

        assert tl.clip_count == 2
        left = tl._get_clip(left_id)
        right = tl._get_clip(right_id)
        assert left.get_duration() == 2_000_000_000
        assert right.get_duration() == 2_000_000_000
        assert right.get_start() == 2_000_000_000

    def test_split_clip_unequal(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import split_clip

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=4_000_000_000,
        )

        left_id, right_id = split_clip(tl, clip_id, position_ns=1_000_000_000)

        left = tl._get_clip(left_id)
        right = tl._get_clip(right_id)
        assert left.get_duration() == 1_000_000_000
        assert right.get_duration() == 3_000_000_000


@requires_ges
@requires_ffmpeg
class TestTimelineConcatenate:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_concatenate_clips(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import concatenate_clips

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_ids = concatenate_clips(
            tl,
            media_paths=[self.clip_path, self.clip_path, self.clip_path],
            durations_ns=[1_000_000_000, 2_000_000_000, 1_000_000_000],
            layer=0,
        )

        assert len(clip_ids) == 3
        assert tl.clip_count == 3
        assert tl.duration_ns == 4_000_000_000

    def test_concatenate_sequential_positions(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import concatenate_clips

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_ids = concatenate_clips(
            tl,
            media_paths=[self.clip_path, self.clip_path],
            durations_ns=[2_000_000_000, 3_000_000_000],
            layer=0,
        )

        clip_0 = tl._get_clip(clip_ids[0])
        clip_1 = tl._get_clip(clip_ids[1])
        assert clip_0.get_start() == 0
        assert clip_1.get_start() == 2_000_000_000
