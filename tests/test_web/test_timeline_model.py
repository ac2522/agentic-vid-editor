"""Tests for TimelineModel — pure Python timeline state."""

from __future__ import annotations

import json
import textwrap
import pytest

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
            clip_id="v",
            asset_uri="",
            name="v",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=4,
        )
        audio_only = ClipState(
            clip_id="a",
            asset_uri="",
            name="a",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=2,
        )
        both = ClipState(
            clip_id="b",
            asset_uri="",
            name="b",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
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
        tm.add_clip(
            ClipState(
                clip_id="c1",
                asset_uri="",
                name="c1",
                layer_index=0,
                start_ns=0,
                duration_ns=2_000_000_000,
                inpoint_ns=0,
                track_types=6,
            )
        )
        tm.add_clip(
            ClipState(
                clip_id="c2",
                asset_uri="",
                name="c2",
                layer_index=1,
                start_ns=1_000_000_000,
                duration_ns=3_000_000_000,
                inpoint_ns=0,
                track_types=6,
            )
        )
        assert tm.duration_ns == 4_000_000_000

    def test_duration_empty(self):
        assert TimelineModel().duration_ns == 0

    def test_custom_fps(self):
        tm = TimelineModel(fps=30.0)
        assert tm.to_dict()["fps"] == 30.0

    def test_to_dict_with_clips(self):
        tm = TimelineModel()
        tm.add_clip(
            ClipState(
                clip_id="c1",
                asset_uri="file:///a.mov",
                name="a.mov",
                layer_index=0,
                start_ns=0,
                duration_ns=1000,
                inpoint_ns=0,
                track_types=6,
            )
        )
        d = tm.to_dict()
        assert len(d["layers"]) == 1
        assert len(d["layers"][0]["clips"]) == 1
        assert d["layers"][0]["clips"][0]["clip_id"] == "c1"


# ---------------------------------------------------------------------------
# Task 2: Clip CRUD
# ---------------------------------------------------------------------------


class TestClipCRUD:
    def test_add_clip_creates_layer(self):
        tm = TimelineModel()
        clip = ClipState(
            clip_id="c1",
            asset_uri="",
            name="c1",
            layer_index=2,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        tm.add_clip(clip)
        # Layers 0, 1 should be created as empty; layer 2 has the clip
        assert len(tm.layers) == 3
        assert len(tm.layers[0].clips) == 0
        assert len(tm.layers[1].clips) == 0
        assert len(tm.layers[2].clips) == 1

    def test_add_clip_to_existing_layer(self):
        tm = TimelineModel()
        c1 = ClipState(
            clip_id="c1",
            asset_uri="",
            name="c1",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        c2 = ClipState(
            clip_id="c2",
            asset_uri="",
            name="c2",
            layer_index=0,
            start_ns=1000,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        tm.add_clip(c1)
        tm.add_clip(c2)
        assert len(tm.layers[0].clips) == 2

    def test_add_clip_duplicate_id_raises(self):
        tm = TimelineModel()
        clip = ClipState(
            clip_id="c1",
            asset_uri="",
            name="c1",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        tm.add_clip(clip)
        with pytest.raises(ValueError, match="Duplicate clip_id"):
            tm.add_clip(clip)

    def test_remove_clip(self):
        tm = TimelineModel()
        clip = ClipState(
            clip_id="c1",
            asset_uri="",
            name="c1",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        tm.add_clip(clip)
        tm.remove_clip("c1")
        assert len(tm.layers[0].clips) == 0

    def test_remove_clip_not_found(self):
        tm = TimelineModel()
        with pytest.raises(KeyError):
            tm.remove_clip("nonexistent")

    def test_update_clip(self):
        tm = TimelineModel()
        clip = ClipState(
            clip_id="c1",
            asset_uri="",
            name="c1",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        tm.add_clip(clip)
        tm.update_clip("c1", start_ns=500, duration_ns=2000)
        updated = tm.get_clip("c1")
        assert updated.start_ns == 500
        assert updated.duration_ns == 2000

    def test_update_clip_not_found(self):
        tm = TimelineModel()
        with pytest.raises(KeyError):
            tm.update_clip("nonexistent", start_ns=0)

    def test_get_clip(self):
        tm = TimelineModel()
        clip = ClipState(
            clip_id="c1",
            asset_uri="",
            name="c1",
            layer_index=0,
            start_ns=0,
            duration_ns=1000,
            inpoint_ns=0,
            track_types=6,
        )
        tm.add_clip(clip)
        assert tm.get_clip("c1") is clip

    def test_get_clip_not_found(self):
        tm = TimelineModel()
        with pytest.raises(KeyError):
            tm.get_clip("nonexistent")


# ---------------------------------------------------------------------------
# Task 3: XGES parser
# ---------------------------------------------------------------------------

SAMPLE_XGES = textwrap.dedent("""\
    <ges version='0.7'>
      <project properties='properties;' metadatas='metadatas;'>
        <ressources>
          <asset id='file:///media/clip_a.mov' extractable-type-name='GESUriClip' />
          <asset id='file:///media/clip_b.mp4' extractable-type-name='GESUriClip' />
        </ressources>
        <timeline properties='properties, framerate=(fraction)24/1;'
                  metadatas='metadatas;'>
          <track caps='video/x-raw(ANY)' track-type='4'
                 properties='properties;' metadatas='metadatas;'/>
          <track caps='audio/x-raw(ANY)' track-type='2'
                 properties='properties;' metadatas='metadatas;'/>
          <layer priority='0' properties='properties;' metadatas='metadatas;'>
            <clip id='0' asset-id='file:///media/clip_a.mov'
                  type-name='GESUriClip' layer-priority='0'
                  track-types='6' start='0' duration='2000000000'
                  inpoint='0' rate='0'
                  properties='properties;'
                  metadatas='metadatas, agent:clip-id=(string)clip_001;'>
              <effect asset-id='agingtv' clip-id='0'
                      type-name='GESEffect' track-type='4'
                      properties='properties;' metadatas='metadatas;'>
              </effect>
            </clip>
          </layer>
          <layer priority='1' properties='properties;' metadatas='metadatas;'>
            <clip id='1' asset-id='file:///media/clip_b.mp4'
                  type-name='GESUriClip' layer-priority='1'
                  track-types='4' start='1000000000' duration='3000000000'
                  inpoint='500000000' rate='0'
                  properties='properties;'
                  metadatas='metadatas;'>
            </clip>
          </layer>
        </timeline>
      </project>
    </ges>
""")


class TestXGESParser:
    def test_load_from_xges_string(self):
        tm = TimelineModel.load_from_xges_string(SAMPLE_XGES)
        assert len(tm.layers) == 2
        assert tm.fps == 24.0

    def test_clip_attributes_parsed(self):
        tm = TimelineModel.load_from_xges_string(SAMPLE_XGES)
        c1 = tm.get_clip("clip_001")
        assert c1.asset_uri == "file:///media/clip_a.mov"
        assert c1.name == "clip_a.mov"
        assert c1.start_ns == 0
        assert c1.duration_ns == 2_000_000_000
        assert c1.inpoint_ns == 0
        assert c1.track_types == 6
        assert c1.layer_index == 0

    def test_fallback_clip_id(self):
        tm = TimelineModel.load_from_xges_string(SAMPLE_XGES)
        c2 = tm.get_clip("clip_1")  # fallback: clip_{xml_id}
        assert c2.asset_uri == "file:///media/clip_b.mp4"
        assert c2.name == "clip_b.mp4"
        assert c2.inpoint_ns == 500_000_000

    def test_effects_parsed(self):
        tm = TimelineModel.load_from_xges_string(SAMPLE_XGES)
        c1 = tm.get_clip("clip_001")
        assert len(c1.effects) == 1
        assert c1.effects[0] == "agingtv"

    def test_duration_computed(self):
        tm = TimelineModel.load_from_xges_string(SAMPLE_XGES)
        # clip_b: start=1e9, duration=3e9 => end=4e9
        assert tm.duration_ns == 4_000_000_000

    def test_framerate_parsed(self):
        xges_30fps = SAMPLE_XGES.replace("framerate=(fraction)24/1", "framerate=(fraction)30/1")
        tm = TimelineModel.load_from_xges_string(xges_30fps)
        assert tm.fps == 30.0

    def test_framerate_default_when_missing(self):
        xges_no_fps = SAMPLE_XGES.replace("properties, framerate=(fraction)24/1;", "properties;")
        tm = TimelineModel.load_from_xges_string(xges_no_fps)
        assert tm.fps == 24.0

    def test_load_from_xges_file(self, tmp_path):
        p = tmp_path / "test.xges"
        p.write_text(SAMPLE_XGES)
        tm = TimelineModel.load_from_xges(p)
        assert len(tm.layers) == 2
        assert tm.get_clip("clip_001").name == "clip_a.mov"

    def test_reload_from_xges(self, tmp_path):
        p = tmp_path / "test.xges"
        p.write_text(SAMPLE_XGES)
        tm = TimelineModel.load_from_xges(p)
        assert len(tm.layers) == 2

        # Write a version with only layer 0
        modified = textwrap.dedent("""\
            <ges version='0.7'>
              <project properties='properties;' metadatas='metadatas;'>
                <timeline properties='properties, framerate=(fraction)24/1;'>
                  <layer priority='0'>
                    <clip id='0' asset-id='file:///media/clip_a.mov'
                          type-name='GESUriClip' layer-priority='0'
                          track-types='6' start='0' duration='2000000000'
                          inpoint='0' rate='0'
                          properties='properties;'
                          metadatas='metadatas, agent:clip-id=(string)clip_001;'/>
                  </layer>
                </timeline>
              </project>
            </ges>
        """)
        p.write_text(modified)
        tm.reload_from_xges()
        assert len(tm.layers) == 1

    def test_reload_no_path_is_noop(self):
        tm = TimelineModel()
        tm.reload_from_xges()  # should not raise
        assert tm.layers == []

    def test_malformed_xml_raises(self):
        with pytest.raises(ValueError, match="[Mm]alformed"):
            TimelineModel.load_from_xges_string("<not valid xml>>>")

    def test_empty_timeline_xges(self):
        xges = textwrap.dedent("""\
            <ges version='0.7'>
              <project>
                <timeline properties='properties;'>
                </timeline>
              </project>
            </ges>
        """)
        tm = TimelineModel.load_from_xges_string(xges)
        assert tm.layers == []
        assert tm.duration_ns == 0

    def test_layer_priority_gap(self):
        """Layers with priority gap (0, 2) should fill in layer 1."""
        xges = textwrap.dedent("""\
            <ges version='0.7'>
              <project>
                <timeline properties='properties, framerate=(fraction)24/1;'>
                  <layer priority='0'>
                    <clip id='0' asset-id='file:///a.mov' type-name='GESUriClip'
                          layer-priority='0' track-types='6' start='0'
                          duration='1000' inpoint='0' rate='0'
                          properties='properties;'
                          metadatas='metadatas;'/>
                  </layer>
                  <layer priority='2'>
                    <clip id='1' asset-id='file:///b.mov' type-name='GESUriClip'
                          layer-priority='2' track-types='6' start='0'
                          duration='1000' inpoint='0' rate='0'
                          properties='properties;'
                          metadatas='metadatas;'/>
                  </layer>
                </timeline>
              </project>
            </ges>
        """)
        tm = TimelineModel.load_from_xges_string(xges)
        assert len(tm.layers) == 3
        assert len(tm.layers[1].clips) == 0  # gap layer is empty

    def test_track_types_video_only(self):
        tm = TimelineModel.load_from_xges_string(SAMPLE_XGES)
        c2 = tm.get_clip("clip_1")
        assert c2.track_types == 4
        assert c2.has_video
        assert not c2.has_audio
