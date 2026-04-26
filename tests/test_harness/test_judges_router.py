"""Tests for the dimension-type router."""


def test_router_classifies_static_dimensions():
    from ave.harness.judges.router import classify_dimension
    assert classify_dimension("duration") == "static"
    assert classify_dimension("resolution") == "static"
    assert classify_dimension("aspect_ratio") == "static"
    assert classify_dimension("audio_rms") == "static"
    assert classify_dimension("format") == "static"


def test_router_classifies_still_dimensions():
    from ave.harness.judges.router import classify_dimension
    assert classify_dimension("framing") == "still"
    assert classify_dimension("speaker_framing") == "still"
    assert classify_dimension("caption_legibility") == "still"
    assert classify_dimension("color_palette") == "still"
    assert classify_dimension("visual_balance") == "still"


def test_router_classifies_temporal_dimensions():
    from ave.harness.judges.router import classify_dimension
    assert classify_dimension("pacing") == "temporal"
    assert classify_dimension("motion_blur") == "temporal"
    assert classify_dimension("audio_continuity") == "temporal"
    assert classify_dimension("animation_smoothness") == "temporal"


def test_router_unknown_dimension_defaults_to_still():
    """Unknown dimensions get 'still' so a frame-sampling VLM can still try."""
    from ave.harness.judges.router import classify_dimension
    assert classify_dimension("totally_made_up") == "still"


def test_router_select_judges_filters_by_type():
    from ave.harness.judges.router import select_judges

    class FakeJudge:
        def __init__(self, name, types):
            self._name = name
            self._types = types
        @property
        def name(self): return self._name
        @property
        def supported_dimension_types(self): return self._types

    judges = [
        FakeJudge("ff", ("static",)),
        FakeJudge("claude", ("still",)),
        FakeJudge("gemini", ("temporal",)),
    ]

    selected = select_judges(judges, dimension_type="still")
    assert len(selected) == 1
    assert selected[0].name == "claude"
