"""Pure state-diff evaluator tests — no Inspect AI or GES dependency."""

import pytest


_XGES_TWO_CLIPS = """<?xml version="1.0" encoding="utf-8"?>
<ges version="0.3">
  <project>
    <timeline>
      <layer priority="0">
        <clip id="c1" asset-id="file:///a.mp4" type-name="GESUriClip"
              start="0" duration="60000000000" inpoint="0" rate="0">
          <effect asset-id="frei0r-filter-color-distance" type="effect"/>
        </clip>
        <clip id="c2" asset-id="file:///b.mp4" type-name="GESUriClip"
              start="60000000000" duration="30000000000" inpoint="0" rate="0"/>
      </layer>
    </timeline>
  </project>
</ges>"""

_XGES_EMPTY = """<?xml version="1.0" encoding="utf-8"?>
<ges version="0.3"><project><timeline><layer priority="0"/></timeline></project></ges>"""


def test_extract_metrics_two_clips():
    from ave.harness.evaluators.state_diff import extract_timeline_metrics
    m = extract_timeline_metrics(_XGES_TWO_CLIPS)
    assert m.clip_count == 2
    assert abs(m.duration_seconds - 90.0) < 0.01
    assert "frei0r-filter-color-distance" in m.effect_names


def test_extract_metrics_empty_timeline():
    from ave.harness.evaluators.state_diff import extract_timeline_metrics
    m = extract_timeline_metrics(_XGES_EMPTY)
    assert m.clip_count == 0
    assert m.duration_seconds == 0.0
    assert m.effect_names == frozenset()


def test_evaluate_state_clip_count_in_range():
    from ave.harness.evaluators.state_diff import evaluate_execute_state, TimelineMetrics
    from ave.harness.schema import ExecuteExpected, MinMax, TimelineBounds
    metrics = TimelineMetrics(clip_count=2, duration_seconds=90.0, effect_names=frozenset())
    expected = ExecuteExpected(timeline=TimelineBounds(clip_count=MinMax(min=1, max=10)))
    v = evaluate_execute_state(metrics, expected)
    assert v.passed is True


def test_evaluate_state_clip_count_below_min():
    from ave.harness.evaluators.state_diff import evaluate_execute_state, TimelineMetrics
    from ave.harness.schema import ExecuteExpected, MinMax, TimelineBounds
    metrics = TimelineMetrics(clip_count=0, duration_seconds=0.0, effect_names=frozenset())
    expected = ExecuteExpected(timeline=TimelineBounds(clip_count=MinMax(min=1)))
    v = evaluate_execute_state(metrics, expected)
    assert v.passed is False
    assert "clip_count" in v.reason.lower()


def test_evaluate_state_duration_out_of_range():
    from ave.harness.evaluators.state_diff import evaluate_execute_state, TimelineMetrics
    from ave.harness.schema import ExecuteExpected, MinMax, TimelineBounds
    metrics = TimelineMetrics(clip_count=1, duration_seconds=100.0, effect_names=frozenset())
    expected = ExecuteExpected(timeline=TimelineBounds(duration_seconds=MinMax(max=35.0)))
    v = evaluate_execute_state(metrics, expected)
    assert v.passed is False
    assert "duration" in v.reason.lower()


def test_evaluate_state_forbidden_effect_present():
    from ave.harness.evaluators.state_diff import evaluate_execute_state, TimelineMetrics
    from ave.harness.schema import ExecuteExpected, TimelineBounds
    metrics = TimelineMetrics(clip_count=1, duration_seconds=10.0, effect_names=frozenset(["cdl"]))
    expected = ExecuteExpected(timeline=TimelineBounds(effects_forbidden=("cdl",)))
    v = evaluate_execute_state(metrics, expected)
    assert v.passed is False
    assert "forbidden" in v.reason.lower()


def test_evaluate_state_required_effect_missing():
    from ave.harness.evaluators.state_diff import evaluate_execute_state, TimelineMetrics
    from ave.harness.schema import ExecuteExpected, TimelineBounds
    metrics = TimelineMetrics(clip_count=1, duration_seconds=10.0, effect_names=frozenset())
    expected = ExecuteExpected(timeline=TimelineBounds(effects_applied=("blur",)))
    v = evaluate_execute_state(metrics, expected)
    assert v.passed is False
    assert "blur" in v.reason


def test_evaluate_state_no_constraints_passes():
    from ave.harness.evaluators.state_diff import evaluate_execute_state, TimelineMetrics
    from ave.harness.schema import ExecuteExpected
    metrics = TimelineMetrics(clip_count=5, duration_seconds=42.0, effect_names=frozenset(["x"]))
    v = evaluate_execute_state(metrics, ExecuteExpected())
    assert v.passed is True
    assert v.rule == "state_ok"
