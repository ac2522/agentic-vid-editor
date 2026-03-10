"""Unit tests for scene detection — data models and pure logic."""

from ave.tools.scene import SceneBoundary, SceneError


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
