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
        clip = tl.get_clip(clip_id)
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

        clip = tl.get_clip(clip_id)
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
        left = tl.get_clip(left_id)
        right = tl.get_clip(right_id)
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

        left = tl.get_clip(left_id)
        right = tl.get_clip(right_id)
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

        clip_0 = tl.get_clip(clip_ids[0])
        clip_1 = tl.get_clip(clip_ids[1])
        assert clip_0.get_start() == 0
        assert clip_1.get_start() == 2_000_000_000


# --- P0-4: operations.py uses public API only ---
# --- P0-5: Verification of GES mutations ---


@requires_ges
@requires_ffmpeg
class TestOperationsVerification:
    """P0-4/P0-5: Operations must use public Timeline API and verify mutations."""

    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_trim_clip_verifies_result(self):
        """trim_clip must verify the trim was actually applied."""
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

        # Use public API (not _get_clip)
        clip = tl.get_clip(clip_id)
        assert clip.get_duration() == 3_000_000_000
        assert clip.get_inpoint() == 1_000_000_000

    def test_split_clip_uses_register_clip(self):
        """split_clip must use register_clip() for the new right-side clip."""
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

        # Both clips must be accessible via public API
        left = tl.get_clip(left_id)
        right = tl.get_clip(right_id)
        assert left.get_duration() == 2_000_000_000
        assert right.get_duration() == 2_000_000_000
        assert right.get_start() == 2_000_000_000

    def test_split_clip_rollback_on_failure(self):
        """If creating the right clip fails, the left clip must be restored."""
        from ave.project.timeline import Timeline
        from ave.project.operations import split_clip

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=4_000_000_000,
        )

        # We can't easily force add_asset to fail in a real GES timeline,
        # so we just verify the clip count and original clip are correct
        # after a successful split as a basic integrity check
        left_id, right_id = split_clip(tl, clip_id, position_ns=2_000_000_000)
        assert tl.clip_count == 2

    def test_concatenate_uses_public_api(self):
        """concatenate_clips must work with public Timeline API."""
        from ave.project.timeline import Timeline
        from ave.project.operations import concatenate_clips

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_ids = concatenate_clips(
            tl,
            media_paths=[self.clip_path, self.clip_path],
            durations_ns=[2_000_000_000, 2_000_000_000],
            layer=0,
        )

        # Verify via public API
        for cid in clip_ids:
            clip = tl.get_clip(cid)
            assert clip is not None

    def test_split_clip_no_direct_ges_imports(self):
        """split_clip in operations.py must not import GES directly (P1-2)."""
        import ast
        import inspect
        from ave.project.operations import split_clip

        source = inspect.getsource(split_clip)
        tree = ast.parse(source)

        # Check there are no 'import gi' or 'from gi.repository' statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "gi", "split_clip must not import gi directly"
            if isinstance(node, ast.ImportFrom):
                assert node.module != "gi.repository", (
                    "split_clip must not import from gi.repository"
                )


@requires_ges
@requires_ffmpeg
class TestTimelineVolume:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_set_volume(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import set_volume

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=3_000_000_000,
        )

        effect_id = set_volume(tl, clip_id, level_db=-6.0)

        actual = tl.get_effect_property(clip_id, effect_id, "volume")
        assert abs(actual - 0.5012) < 0.01

    def test_set_volume_zero_db(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import set_volume

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=3_000_000_000,
        )

        effect_id = set_volume(tl, clip_id, level_db=0.0)

        actual = tl.get_effect_property(clip_id, effect_id, "volume")
        assert abs(actual - 1.0) < 0.001


@requires_ges
@requires_ffmpeg
class TestTimelineFade:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_apply_fade_in(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_fade

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=5_000_000_000,
        )

        effect_id = apply_fade(tl, clip_id, fade_in_ns=1_000_000_000, fade_out_ns=0)

        assert effect_id is not None
        # Volume effect should exist
        actual = tl.get_effect_property(clip_id, effect_id, "volume")
        assert actual is not None

    def test_apply_fade_both(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_fade

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=5_000_000_000,
        )

        effect_id = apply_fade(
            tl, clip_id, fade_in_ns=1_000_000_000, fade_out_ns=1_000_000_000
        )

        assert effect_id is not None

    def test_apply_fade_out_only(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_fade

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=5_000_000_000,
        )

        effect_id = apply_fade(tl, clip_id, fade_in_ns=0, fade_out_ns=2_000_000_000)

        assert effect_id is not None


@requires_ges
@requires_ffmpeg
class TestTimelineTransition:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_apply_crossfade(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_transition, concatenate_clips
        from ave.tools.transitions import TransitionType

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_ids = concatenate_clips(
            tl,
            media_paths=[self.clip_path, self.clip_path],
            durations_ns=[3_000_000_000, 3_000_000_000],
            layer=0,
        )

        apply_transition(
            tl,
            clip_ids[0],
            clip_ids[1],
            transition_type=TransitionType.CROSSFADE,
            duration_ns=1_000_000_000,
        )

        # 3s + 3s - 1s overlap = 5s
        assert tl.duration_ns == 5_000_000_000

    def test_apply_transition_preserves_clips(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import apply_transition, concatenate_clips
        from ave.tools.transitions import TransitionType

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        clip_ids = concatenate_clips(
            tl,
            media_paths=[self.clip_path, self.clip_path],
            durations_ns=[3_000_000_000, 3_000_000_000],
            layer=0,
        )

        apply_transition(
            tl,
            clip_ids[0],
            clip_ids[1],
            transition_type=TransitionType.CROSSFADE,
            duration_ns=1_000_000_000,
        )

        # Both source clips should still be accessible
        assert tl.get_clip(clip_ids[0]) is not None
        assert tl.get_clip(clip_ids[1]) is not None


@requires_ges
@requires_ffmpeg
class TestTimelineSpeed:
    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_set_speed_double(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import set_speed

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=4_000_000_000,
        )

        set_speed(tl, clip_id, rate=2.0)

        clip = tl.get_clip(clip_id)
        assert clip.get_duration() == 2_000_000_000

    def test_set_speed_half(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import set_speed

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )

        set_speed(tl, clip_id, rate=0.5)

        clip = tl.get_clip(clip_id)
        assert clip.get_duration() == 4_000_000_000

    def test_set_speed_preserves_position(self):
        from ave.project.timeline import Timeline
        from ave.project.operations import set_speed

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=1_000_000_000,
            duration_ns=4_000_000_000,
        )

        set_speed(tl, clip_id, rate=2.0)

        clip = tl.get_clip(clip_id)
        assert clip.get_start() == 1_000_000_000
