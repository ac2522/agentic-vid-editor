"""Tests for timeline info — clip identity and timeline state queries."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_XGES = """\
<?xml version="1.0" encoding="UTF-8"?>
<ges version="0.7">
  <project>
    <timeline properties="framerate=(fraction)24/1;">
      <layer priority="0">
        <clip id="0" asset-id="file:///media/interview.mov"
              start="0" duration="5000000000" inpoint="0"
              track-types="6"
              metadatas="agent:clip-id=(string)clip_001;">
        </clip>
        <clip id="1" asset-id="file:///media/broll.mp4"
              start="5000000000" duration="3000000000" inpoint="500000000"
              track-types="4"
              metadatas="agent:clip-id=(string)clip_002;">
          <effect asset-id="agingtv"/>
        </clip>
      </layer>
      <layer priority="1">
        <clip id="2" asset-id="file:///media/music.wav"
              start="0" duration="8000000000" inpoint="0"
              track-types="2"
              metadatas="agent:clip-id=(string)clip_003;">
        </clip>
      </layer>
    </timeline>
  </project>
</ges>
"""

EMPTY_XGES = """\
<?xml version="1.0" encoding="UTF-8"?>
<ges version="0.7">
  <project>
    <timeline properties="framerate=(fraction)30/1;">
    </timeline>
  </project>
</ges>
"""


class TestListTimelineClips:
    """Test list_timeline_clips pure function."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text(SAMPLE_XGES)
        return p

    @pytest.fixture()
    def empty_xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "empty.xges"
        p.write_text(EMPTY_XGES)
        return p

    def test_returns_all_clips(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        assert len(clips) == 3

    def test_clip_ids_present(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        ids = {c["clip_id"] for c in clips}
        assert ids == {"clip_001", "clip_002", "clip_003"}

    def test_clip_fields(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        clip1 = next(c for c in clips if c["clip_id"] == "clip_001")

        assert clip1["name"] == "interview.mov"
        assert clip1["start_ns"] == 0
        assert clip1["duration_ns"] == 5_000_000_000
        assert clip1["end_ns"] == 5_000_000_000
        assert clip1["inpoint_ns"] == 0
        assert clip1["layer"] == 0
        assert clip1["has_video"] is True
        assert clip1["has_audio"] is True

    def test_video_only_clip(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        clip2 = next(c for c in clips if c["clip_id"] == "clip_002")

        assert clip2["has_video"] is True
        assert clip2["has_audio"] is False
        assert clip2["inpoint_ns"] == 500_000_000

    def test_audio_only_clip(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        clip3 = next(c for c in clips if c["clip_id"] == "clip_003")

        assert clip3["has_video"] is False
        assert clip3["has_audio"] is True
        assert clip3["layer"] == 1

    def test_effects_included(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        clip2 = next(c for c in clips if c["clip_id"] == "clip_002")
        assert clip2["effects"] == ["agingtv"]

    def test_empty_timeline(self, empty_xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(empty_xges_file))
        assert clips == []

    def test_clips_sorted_by_start_then_layer(self, xges_file):
        from ave.tools.timeline_info import list_timeline_clips

        clips = list_timeline_clips(str(xges_file))
        # clip_001 starts at 0 layer 0, clip_003 starts at 0 layer 1,
        # clip_002 starts at 5s layer 0
        starts = [(c["start_ns"], c["layer"]) for c in clips]
        assert starts == sorted(starts)

    def test_nonexistent_file_raises(self):
        from ave.tools.timeline_info import list_timeline_clips, TimelineInfoError

        with pytest.raises(TimelineInfoError, match="not found"):
            list_timeline_clips("/nonexistent/path.xges")


class TestGetTimelineInfo:
    """Test get_timeline_info pure function."""

    @pytest.fixture()
    def xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "project.xges"
        p.write_text(SAMPLE_XGES)
        return p

    @pytest.fixture()
    def empty_xges_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "empty.xges"
        p.write_text(EMPTY_XGES)
        return p

    def test_returns_summary(self, xges_file):
        from ave.tools.timeline_info import get_timeline_info

        info = get_timeline_info(str(xges_file))
        assert info["fps"] == 24.0
        assert info["duration_ns"] == 8_000_000_000
        assert info["layer_count"] == 2
        assert info["clip_count"] == 3

    def test_duration_seconds(self, xges_file):
        from ave.tools.timeline_info import get_timeline_info

        info = get_timeline_info(str(xges_file))
        assert info["duration_seconds"] == pytest.approx(8.0, abs=0.01)

    def test_empty_timeline(self, empty_xges_file):
        from ave.tools.timeline_info import get_timeline_info

        info = get_timeline_info(str(empty_xges_file))
        assert info["fps"] == 30.0
        assert info["duration_ns"] == 0
        assert info["clip_count"] == 0
        assert info["layer_count"] == 0

    def test_nonexistent_file_raises(self):
        from ave.tools.timeline_info import get_timeline_info, TimelineInfoError

        with pytest.raises(TimelineInfoError, match="not found"):
            get_timeline_info("/nonexistent/path.xges")


class TestToolRegistration:
    """Test that clip identity tools integrate with the agent registry."""

    def test_list_clips_registered(self):
        from ave.agent.session import EditingSession

        session = EditingSession()
        results = session.search_tools("clip list timeline")
        names = [r.name for r in results]
        assert "list_clips" in names

    def test_timeline_info_registered(self):
        from ave.agent.session import EditingSession

        session = EditingSession()
        results = session.search_tools("timeline duration info")
        names = [r.name for r in results]
        assert "timeline_info" in names

    def test_list_clips_provides_clip_exists(self):
        from ave.agent.session import EditingSession

        session = EditingSession()
        schema = session.get_tool_schema("list_clips")
        assert "clip_exists" in schema.provides

    def test_list_clips_requires_timeline_loaded(self):
        from ave.agent.session import EditingSession

        session = EditingSession()
        schema = session.get_tool_schema("list_clips")
        assert "timeline_loaded" in schema.requires

    def test_list_clips_callable_with_xges(self, tmp_path):
        from ave.agent.session import EditingSession

        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        session = EditingSession()
        session.load_project(xges_file)
        result = session.call_tool("list_clips", {"xges_path": str(xges_file)})
        assert isinstance(result, list)
        assert len(result) == 3

    def test_timeline_info_callable_with_xges(self, tmp_path):
        from ave.agent.session import EditingSession

        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        session = EditingSession()
        session.load_project(xges_file)
        result = session.call_tool("timeline_info", {"xges_path": str(xges_file)})
        assert isinstance(result, dict)
        assert result["clip_count"] == 3

    def test_list_clips_sets_clip_exists_provision(self, tmp_path):
        from ave.agent.session import EditingSession

        xges_file = tmp_path / "project.xges"
        xges_file.write_text(SAMPLE_XGES)

        session = EditingSession()
        session.load_project(xges_file)
        assert not session.state.has("clip_exists")

        session.call_tool("list_clips", {"xges_path": str(xges_file)})
        assert session.state.has("clip_exists")
