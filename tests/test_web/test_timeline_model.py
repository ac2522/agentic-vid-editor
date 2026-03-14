"""Tests for TimelineModel — pure Python timeline state."""

from __future__ import annotations

import json

from ave.web.timeline_model import ClipState, LayerState, TimelineModel


# ---------------------------------------------------------------------------
# Task 1: Data classes and serialization
# ---------------------------------------------------------------------------


class TestClipState:
    def test_clip_defaults(self):
        clip = ClipState(
            clip_id="c1",
            asset_uri="file:///media/a.mov",
            name="a.mov",
            layer_index=0,
            start_ns=0,
            duration_ns=1_000_000_000,
            inpoint_ns=0,
            track_types=6,
        )
        assert clip.clip_id == "c1"
        assert clip.effects == []

    def test_clip_end_ns(self):
        clip = ClipState(
            clip_id="c1",
            asset_uri="file:///media/a.mov",
            name="a.mov",
            layer_index=0,
            start_ns=500,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        assert clip.end_ns == 1500

    def test_clip_to_dict(self):
        clip = ClipState(
            clip_id="c1",
            asset_uri="file:///media/a.mov",
            name="a.mov",
            layer_index=0,
            start_ns=0,
            duration_ns=1_000_000_000,
            inpoint_ns=0,
            track_types=6,
        )
        d = clip.to_dict()
        assert d["clip_id"] == "c1"
        assert d["name"] == "a.mov"
        assert d["track_types"] == 6
        assert d["effects"] == []

    def test_clip_track_type_helpers(self):
        video_only = ClipState(
            clip_id="v", asset_uri="", name="v", layer_index=0,
            start_ns=0, duration_ns=1000, inpoint_ns=0, track_types=4,
        )
        audio_only = ClipState(
            clip_id="a", asset_uri="", name="a", layer_index=0,
            start_ns=0, duration_ns=1000, inpoint_ns=0, track_types=2,
        )
        both = ClipState(
            clip_id="b", asset_uri="", name="b", layer_index=0,
            start_ns=0, duration_ns=1000, inpoint_ns=0, track_types=6,
        )
        assert video_only.has_video and not video_only.has_audio
        assert audio_only.has_audio and not audio_only.has_video
        assert both.has_video and both.has_audio


class TestLayerState:
    def test_empty_layer(self):
        layer = LayerState(layer_index=0)
        assert layer.clips == []
        assert layer.to_dict() == {"layer_index": 0, "clips": []}


class TestTimelineModelSerialization:
    def test_empty_timeline(self):
        tm = TimelineModel()
        d = tm.to_dict()
        assert d == {"layers": [], "duration_ns": 0, "fps": 24.0}

    def test_empty_timeline_json_roundtrip(self):
        tm = TimelineModel()
        s = json.dumps(tm.to_dict())
        parsed = json.loads(s)
        assert parsed["layers"] == []
        assert parsed["duration_ns"] == 0

    def test_duration_calculated_from_clips(self):
        tm = TimelineModel()
        tm.add_clip(ClipState(
            clip_id="c1", asset_uri="", name="c1", layer_index=0,
            start_ns=0, duration_ns=2_000_000_000, inpoint_ns=0, track_types=6,
        ))
        tm.add_clip(ClipState(
            clip_id="c2", asset_uri="", name="c2", layer_index=1,
            start_ns=1_000_000_000, duration_ns=3_000_000_000,
            inpoint_ns=0, track_types=6,
        ))
        assert tm.duration_ns == 4_000_000_000

    def test_duration_empty(self):
        assert TimelineModel().duration_ns == 0

    def test_custom_fps(self):
        tm = TimelineModel(fps=30.0)
        assert tm.to_dict()["fps"] == 30.0

    def test_to_dict_with_clips(self):
        tm = TimelineModel()
        tm.add_clip(ClipState(
            clip_id="c1", asset_uri="file:///a.mov", name="a.mov",
            layer_index=0, start_ns=0, duration_ns=1000,
            inpoint_ns=0, track_types=6,
        ))
        d = tm.to_dict()
        assert len(d["layers"]) == 1
        assert len(d["layers"][0]["clips"]) == 1
        assert d["layers"][0]["clips"][0]["clip_id"] == "c1"
