"""Tests for rough_cut tool — scene-to-timeline rough cut workflow."""

from __future__ import annotations

import pytest

from ave.tools.rough_cut import (
    ClipPlacement,
    RoughCutError,
    RoughCutParams,
    compute_rough_cut,
    select_scenes_by_duration,
    select_scenes_by_tags,
)
from ave.tools.scene import SceneBoundary
from ave.tools.vision import SceneTag

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NS = 1_000_000_000  # 1 second in ns


def _make_scenes(count: int = 10, fps: float = 24.0) -> list[SceneBoundary]:
    """Create *count* sequential 1-second scenes."""
    return [
        SceneBoundary(start_ns=i * _NS, end_ns=(i + 1) * _NS, fps=fps)
        for i in range(count)
    ]


def _make_tags(scenes: list[SceneBoundary], labels_cycle: list[str]) -> list[SceneTag]:
    """Create SceneTag objects cycling through the given labels."""
    tags = []
    for i, _scene in enumerate(scenes):
        label = labels_cycle[i % len(labels_cycle)]
        # top_label will be `label` since it has the highest score
        tags.append(
            SceneTag(
                scene_id=f"scene_{i}",
                labels={label: 0.9, "other": 0.1},
            )
        )
    return tags


# ---------------------------------------------------------------------------
# select_scenes_by_tags
# ---------------------------------------------------------------------------


class TestSelectScenesByTags:
    def test_include_only(self):
        scenes = _make_scenes(10)
        tags = _make_tags(scenes, ["action", "dialogue", "landscape"])
        # "action" at indices 0, 3, 6, 9
        result = select_scenes_by_tags(scenes, tags, include_labels={"action"})
        assert result == [0, 3, 6, 9]

    def test_exclude_only(self):
        scenes = _make_scenes(10)
        tags = _make_tags(scenes, ["action", "dialogue", "landscape"])
        # exclude "action" → keep 1,2,4,5,7,8
        result = select_scenes_by_tags(scenes, tags, exclude_labels={"action"})
        assert result == [1, 2, 4, 5, 7, 8]

    def test_include_and_exclude_include_wins(self):
        scenes = _make_scenes(10)
        tags = _make_tags(scenes, ["action", "dialogue", "landscape"])
        # include "action", exclude "action" → include wins, so action is kept
        # but "dialogue" and "landscape" are NOT in include, so they are excluded
        result = select_scenes_by_tags(
            scenes, tags, include_labels={"action"}, exclude_labels={"action"}
        )
        assert result == [0, 3, 6, 9]

    def test_include_multiple_labels(self):
        scenes = _make_scenes(10)
        tags = _make_tags(scenes, ["action", "dialogue", "landscape"])
        result = select_scenes_by_tags(
            scenes, tags, include_labels={"action", "dialogue"}
        )
        # action: 0,3,6,9  dialogue: 1,4,7
        assert result == [0, 1, 3, 4, 6, 7, 9]

    def test_no_filters_returns_all(self):
        scenes = _make_scenes(5)
        tags = _make_tags(scenes, ["action"])
        result = select_scenes_by_tags(scenes, tags)
        assert result == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# select_scenes_by_duration
# ---------------------------------------------------------------------------


class TestSelectScenesByDuration:
    def test_filter_short_scenes(self):
        scenes = [
            SceneBoundary(start_ns=0, end_ns=500_000_000, fps=24.0),  # 0.5s
            SceneBoundary(start_ns=500_000_000, end_ns=2_000_000_000, fps=24.0),  # 1.5s
            SceneBoundary(start_ns=2_000_000_000, end_ns=2_200_000_000, fps=24.0),  # 0.2s
        ]
        result = select_scenes_by_duration(scenes, min_duration_ns=_NS)
        assert result == [1]

    def test_filter_long_scenes(self):
        scenes = [
            SceneBoundary(start_ns=0, end_ns=_NS, fps=24.0),
            SceneBoundary(start_ns=_NS, end_ns=5 * _NS, fps=24.0),  # 4s
            SceneBoundary(start_ns=5 * _NS, end_ns=6 * _NS, fps=24.0),
        ]
        result = select_scenes_by_duration(scenes, max_duration_ns=2 * _NS)
        assert result == [0, 2]

    def test_filter_both_min_and_max(self):
        scenes = [
            SceneBoundary(start_ns=0, end_ns=500_000_000, fps=24.0),  # 0.5s too short
            SceneBoundary(start_ns=_NS, end_ns=2 * _NS, fps=24.0),  # 1s ok
            SceneBoundary(start_ns=2 * _NS, end_ns=5 * _NS, fps=24.0),  # 3s too long
        ]
        result = select_scenes_by_duration(
            scenes, min_duration_ns=800_000_000, max_duration_ns=2 * _NS
        )
        assert result == [1]

    def test_no_filters_returns_all(self):
        scenes = _make_scenes(3)
        result = select_scenes_by_duration(scenes)
        assert result == [0, 1, 2]


# ---------------------------------------------------------------------------
# compute_rough_cut
# ---------------------------------------------------------------------------


class TestComputeRoughCut:
    def test_chronological_order(self):
        scenes = _make_scenes(5)
        # Select scenes out of order; chronological should sort by start_ns
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[3, 1, 4],
            order="chronological",
            gap_ns=0,
        )
        result = compute_rough_cut(params)
        assert len(result) == 3
        # Chronological: sorted by start_ns → indices 1, 3, 4
        assert result[0].scene_index == 1
        assert result[1].scene_index == 3
        assert result[2].scene_index == 4

    def test_custom_order(self):
        scenes = _make_scenes(5)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[3, 1, 4],
            order="custom",
            gap_ns=0,
        )
        result = compute_rough_cut(params)
        assert len(result) == 3
        assert result[0].scene_index == 3
        assert result[1].scene_index == 1
        assert result[2].scene_index == 4

    def test_timeline_positions_no_gap(self):
        scenes = _make_scenes(5)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[0, 1, 2],
            order="chronological",
            gap_ns=0,
        )
        result = compute_rough_cut(params)
        # Each scene is 1 second; clips should be contiguous
        assert result[0].timeline_position_ns == 0
        assert result[1].timeline_position_ns == _NS
        assert result[2].timeline_position_ns == 2 * _NS

    def test_timeline_positions_with_gap(self):
        scenes = _make_scenes(3)
        gap = 500_000_000  # 0.5s
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[0, 1, 2],
            order="chronological",
            gap_ns=gap,
        )
        result = compute_rough_cut(params)
        assert result[0].timeline_position_ns == 0
        assert result[1].timeline_position_ns == _NS + gap
        assert result[2].timeline_position_ns == 2 * _NS + 2 * gap

    def test_source_timestamps(self):
        scenes = _make_scenes(5)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[2],
            order="chronological",
            gap_ns=0,
        )
        result = compute_rough_cut(params)
        assert len(result) == 1
        assert result[0].source_start_ns == 2 * _NS
        assert result[0].source_end_ns == 3 * _NS
        assert result[0].timeline_position_ns == 0

    def test_empty_selection_raises(self):
        scenes = _make_scenes(5)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[],
            order="chronological",
            gap_ns=0,
        )
        with pytest.raises(RoughCutError, match="empty"):
            compute_rough_cut(params)

    def test_invalid_index_raises(self):
        scenes = _make_scenes(5)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[0, 10],
            order="chronological",
            gap_ns=0,
        )
        with pytest.raises(RoughCutError, match="out of range"):
            compute_rough_cut(params)

    def test_negative_index_raises(self):
        scenes = _make_scenes(5)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[-1],
            order="chronological",
            gap_ns=0,
        )
        with pytest.raises(RoughCutError, match="out of range"):
            compute_rough_cut(params)

    def test_empty_scenes_raises(self):
        params = RoughCutParams(
            scenes=[],
            selected_indices=[0],
            order="chronological",
            gap_ns=0,
        )
        with pytest.raises(RoughCutError):
            compute_rough_cut(params)

    def test_zero_gap_contiguous(self):
        """With zero gap, end of one clip == start of next."""
        scenes = _make_scenes(3)
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[0, 1, 2],
            order="chronological",
            gap_ns=0,
        )
        result = compute_rough_cut(params)
        for i in range(len(result) - 1):
            clip_end = result[i].timeline_position_ns + (
                result[i].source_end_ns - result[i].source_start_ns
            )
            assert clip_end == result[i + 1].timeline_position_ns

    def test_custom_order_with_gap(self):
        scenes = _make_scenes(4)
        gap = 100_000_000  # 0.1s
        params = RoughCutParams(
            scenes=scenes,
            selected_indices=[3, 0],
            order="custom",
            gap_ns=gap,
        )
        result = compute_rough_cut(params)
        assert result[0].scene_index == 3
        assert result[0].timeline_position_ns == 0
        # scene 3 duration is 1s, so next clip starts at 1s + gap
        assert result[1].timeline_position_ns == _NS + gap
        assert result[1].scene_index == 0
