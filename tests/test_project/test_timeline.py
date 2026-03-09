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
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
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
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
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
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
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
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )

        effect_id = tl.add_effect(clip_id, "videobalance")
        assert effect_id is not None

    def test_set_effect_property(self):
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
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
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2 * 1_000_000_000,
        )
        effect_id = tl.add_effect(clip_id, "videobalance")
        tl.remove_effect(clip_id, effect_id)

        # No exception means success — clip has no effects


# --- P0-1: register_clip() and get_clip() public API ---


@requires_ges
@requires_ffmpeg
class TestTimelinePublicAPI:
    """P0-1: Timeline must expose public methods for clip access."""

    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_get_clip_returns_ges_clip(self):
        """get_clip() should return the GES clip object by ID."""
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )

        clip = tl.get_clip(clip_id)
        assert clip is not None
        assert clip.get_duration() == 2_000_000_000

    def test_get_clip_nonexistent_raises(self):
        """get_clip() should raise TimelineError for unknown IDs."""
        from ave.project.timeline import Timeline, TimelineError

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        with pytest.raises(TimelineError, match="Clip not found"):
            tl.get_clip("clip_9999")

    def test_register_clip_returns_new_id(self):
        """register_clip() should assign and return a new clip ID."""
        from ave.project.timeline import Timeline

        import gi

        gi.require_version("GES", "1.0")
        from gi.repository import GES

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        # Add a clip via normal API to get an asset, then create a second
        # clip manually via GES to test register_clip
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )
        original_clip = tl.get_clip(clip_id)
        asset = original_clip.get_asset()
        layer = original_clip.get_layer()

        # Create a new GES clip manually
        new_ges_clip = layer.add_asset(
            asset,
            3_000_000_000,
            0,
            1_000_000_000,
            GES.TrackType.UNKNOWN,
        )
        assert new_ges_clip is not None

        new_id = tl.register_clip(new_ges_clip)
        assert new_id != clip_id
        assert new_id.startswith("clip_")
        assert tl.get_clip(new_id) is new_ges_clip


# --- P0-2: Effect ID scheme (index-based) ---


@requires_ges
@requires_ffmpeg
class TestEffectIDScheme:
    """P0-2: Effect IDs must be unique and index-based."""

    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_duplicate_effects_get_unique_ids(self):
        """Two effects of same type on same clip must get different IDs."""
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )

        fx1 = tl.add_effect(clip_id, "videobalance")
        fx2 = tl.add_effect(clip_id, "videobalance")

        assert fx1 != fx2

    def test_set_property_on_specific_duplicate_effect(self):
        """Setting property on one duplicate effect must not affect the other."""
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )

        fx1 = tl.add_effect(clip_id, "videobalance")
        fx2 = tl.add_effect(clip_id, "videobalance")

        tl.set_effect_property(clip_id, fx1, "saturation", 0.3)
        tl.set_effect_property(clip_id, fx2, "saturation", 0.8)

        val1 = tl.get_effect_property(clip_id, fx1, "saturation")
        val2 = tl.get_effect_property(clip_id, fx2, "saturation")

        assert val1 == pytest.approx(0.3, abs=0.01)
        assert val2 == pytest.approx(0.8, abs=0.01)

    def test_remove_specific_duplicate_effect(self):
        """Removing one duplicate effect must leave the other intact."""
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )

        fx1 = tl.add_effect(clip_id, "videobalance")
        fx2 = tl.add_effect(clip_id, "videobalance")

        tl.set_effect_property(clip_id, fx1, "saturation", 0.2)
        tl.set_effect_property(clip_id, fx2, "saturation", 0.9)

        tl.remove_effect(clip_id, fx1)

        # fx2 should still work with its value
        val2 = tl.get_effect_property(clip_id, fx2, "saturation")
        assert val2 == pytest.approx(0.9, abs=0.01)

    def test_effect_id_is_index_based(self):
        """Effect IDs should follow {clip_id}_fx_{index} pattern."""
        from ave.project.timeline import Timeline

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )

        fx1 = tl.add_effect(clip_id, "videobalance")
        fx2 = tl.add_effect(clip_id, "audioamplify")

        assert fx1 == f"{clip_id}_fx_0"
        assert fx2 == f"{clip_id}_fx_1"


# --- P0-3: Clip ID stability across save/load ---


@requires_ges
@requires_ffmpeg
class TestClipIDStability:
    """P0-3: Clip IDs must survive save/load round-trips."""

    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_clip_ids_stable_after_save_load(self):
        """Clip IDs must be identical after save/load."""
        from ave.project.timeline import Timeline

        xges_path = self.project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        id_a = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )
        id_b = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=2_000_000_000,
            duration_ns=2_000_000_000,
        )
        tl.save()

        tl2 = Timeline.load(xges_path)
        # Both IDs must resolve to clips with same positions
        clip_a = tl2.get_clip(id_a)
        clip_b = tl2.get_clip(id_b)
        assert clip_a.get_start() == 0
        assert clip_b.get_start() == 2_000_000_000

    def test_clip_ids_stable_with_out_of_order_positions(self):
        """IDs must be stable even when clips are not in position order."""
        from ave.project.timeline import Timeline

        xges_path = self.project / "project.xges"
        tl = Timeline.create(xges_path, fps=24.0)

        # Add clips out of position order
        id_first = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=5_000_000_000,
            duration_ns=1_000_000_000,
        )
        id_second = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=1_000_000_000,
        )
        tl.save()

        tl2 = Timeline.load(xges_path)
        # IDs must still map to the correct clips
        clip_first = tl2.get_clip(id_first)
        clip_second = tl2.get_clip(id_second)
        assert clip_first.get_start() == 5_000_000_000
        assert clip_second.get_start() == 0

    def test_fps_survives_save_load(self):
        """Timeline fps must be correct after save/load."""
        from ave.project.timeline import Timeline

        xges_path = self.project / "project.xges"
        tl = Timeline.create(xges_path, fps=29.97)
        tl.save()

        tl2 = Timeline.load(xges_path)
        assert tl2.fps == pytest.approx(29.97, abs=0.01)


# --- P0-5: Verification (remove_clip return check) ---


@requires_ges
@requires_ffmpeg
class TestTimelineVerification:
    """P0-5: GES mutations must be verified."""

    @pytest.fixture(autouse=True)
    def _setup(self, fixtures_dir: Path, tmp_project: Path):
        self.clip_path = fixtures_dir / "av_clip_1080p24.mp4"
        if not self.clip_path.exists():
            from tests.fixtures.generate import generate_av_clip

            generate_av_clip(self.clip_path)
        self.project = tmp_project

    def test_remove_clip_nonexistent_raises(self):
        """Removing a nonexistent clip must raise TimelineError."""
        from ave.project.timeline import Timeline, TimelineError

        tl = Timeline.create(self.project / "project.xges", fps=24.0)

        with pytest.raises(TimelineError, match="Clip not found"):
            tl.remove_clip("clip_9999")

    def test_set_effect_property_invalid_raises(self):
        """Setting invalid effect property must raise TimelineError."""
        from ave.project.timeline import Timeline, TimelineError

        tl = Timeline.create(self.project / "project.xges", fps=24.0)
        clip_id = tl.add_clip(
            media_path=self.clip_path,
            layer=0,
            start_ns=0,
            duration_ns=2_000_000_000,
        )
        fx_id = tl.add_effect(clip_id, "videobalance")

        with pytest.raises(TimelineError, match="Failed to set property"):
            tl.set_effect_property(clip_id, fx_id, "nonexistent_prop", 1.0)
