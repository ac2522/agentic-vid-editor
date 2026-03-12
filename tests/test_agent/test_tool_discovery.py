"""Tests for natural language tool discovery across diverse user backgrounds."""

from __future__ import annotations

import pytest

from ave.agent.session import EditingSession


@pytest.fixture
def session():
    return EditingSession()


# ---------------------------------------------------------------------------
# 1. Technical terms -- professional/technical users
# ---------------------------------------------------------------------------


class TestTechnicalQueries:
    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("ASC CDL correction", "cdl"),
            ("lift gamma gain", "color_grade"),
            ("LUFS normalization", "normalize"),
            ("crossfade between clips", "transition"),
            ("OpenTimelineIO export", "export_otio"),
            ("fragmented MP4 render", "render_segment"),
            ("slope offset power grading", "cdl"),
            ("peak normalize audio", "normalize"),
            ("lookup table cube file", "lut_parse"),
            ("playback rate change", "speed"),
        ],
    )
    def test_technical_finds_tool(self, session, query, expected_tool):
        results = session.search_tools(query)
        found_names = [r.name for r in results]
        assert expected_tool in found_names, (
            f"Query '{query}' should find '{expected_tool}', got {found_names}"
        )


# ---------------------------------------------------------------------------
# 2. Casual / non-technical -- users who don't know editing terms
# ---------------------------------------------------------------------------


class TestCasualQueries:
    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("make the video brighter", "color_grade"),
            ("remove the ums and uhs", "find_fillers"),
            ("put text on the screen", "add_text_overlay"),
            ("make it see through", "set_clip_alpha"),
            ("slow motion effect", "speed"),
            ("join two clips together", "concatenate"),
            ("show speaker name", "add_lower_third"),
            ("what formats can I export", "list_render_presets"),
            ("make the audio louder", "volume"),
            ("add words on the video", "add_text_overlay"),
            ("stitch together clips", "concatenate"),
            ("turn down the volume", "volume"),
        ],
    )
    def test_casual_finds_tool(self, session, query, expected_tool):
        results = session.search_tools(query)
        found_names = [r.name for r in results]
        assert expected_tool in found_names, (
            f"Query '{query}' should find '{expected_tool}', got {found_names}"
        )


# ---------------------------------------------------------------------------
# 3. Editor jargon -- professional video editors
# ---------------------------------------------------------------------------


class TestEditorJargon:
    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("razor blade at playhead", "split"),
            ("J-cut transition", "transition"),
            ("chyron for interview", "add_lower_third"),
            ("selects reel assembly", "create_rough_cut"),
            ("picture in picture", "set_clip_position"),
            ("DaVinci round trip", "export_otio"),
            ("printer lights", "cdl"),
            ("assembly edit from scenes", "create_rough_cut"),
            ("dissolve between shots", "transition"),
            ("name super graphic", "add_lower_third"),
        ],
    )
    def test_jargon_finds_tool(self, session, query, expected_tool):
        results = session.search_tools(query)
        found_names = [r.name for r in results]
        assert expected_tool in found_names, (
            f"Query '{query}' should find '{expected_tool}', got {found_names}"
        )


# ---------------------------------------------------------------------------
# 4. Task-oriented -- users describing what they want to accomplish
# ---------------------------------------------------------------------------


class TestTaskOrientedQueries:
    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("I need to cut out a section", "trim"),
            ("find where someone says hello", "search_transcript"),
            ("detect all the scene changes", "detect_scenes"),
            ("render for YouTube 4K", "render_with_preset"),
            ("render for Instagram", "render_with_preset"),
            ("add movie title at the start", "add_title_card"),
            ("import footage into project", "ingest_media"),
            ("check video resolution", "probe_media"),
            ("split the clip in two", "split"),
            ("find the filler words", "find_fillers"),
            ("make a rough cut automatically", "create_rough_cut"),
            ("export for Premiere", "export_otio"),
        ],
    )
    def test_task_oriented_finds_tool(self, session, query, expected_tool):
        results = session.search_tools(query)
        found_names = [r.name for r in results]
        assert expected_tool in found_names, (
            f"Query '{query}' should find '{expected_tool}', got {found_names}"
        )


# ---------------------------------------------------------------------------
# 5. Multi-word concept matching -- queries that match tag phrases
# ---------------------------------------------------------------------------


class TestMultiWordConceptMatching:
    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("log to rec709 transform", "lut_parse"),
            ("film emulation LUT", "lut_parse"),
            ("warm up the colors", "color_grade"),
            ("bring clip to front", "set_layer_order"),
            ("send to back layer", "set_layer_order"),
            ("auto detect scenes", "detect_scenes"),
            ("color decision list", "cdl"),
            ("fade in audio", "fade"),
            ("burn in text overlay", "add_text_overlay"),
            ("clean up speech hesitations", "find_fillers"),
        ],
    )
    def test_multi_word_finds_tool(self, session, query, expected_tool):
        results = session.search_tools(query)
        found_names = [r.name for r in results]
        assert expected_tool in found_names, (
            f"Query '{query}' should find '{expected_tool}', got {found_names}"
        )


# ---------------------------------------------------------------------------
# 6. Domain browsing -- verify domain search returns all expected tools
# ---------------------------------------------------------------------------


class TestDomainBrowsing:
    @pytest.mark.parametrize(
        "domain, expected_tools",
        [
            ("editing", {"trim", "split", "concatenate", "speed", "transition"}),
            ("audio", {"volume", "fade", "normalize"}),
            ("color", {"color_grade", "cdl", "lut_parse"}),
            (
                "compositing",
                {
                    "set_layer_order",
                    "apply_blend_mode",
                    "set_clip_position",
                    "set_clip_alpha",
                },
            ),
            (
                "motion_graphics",
                {"add_text_overlay", "add_lower_third", "add_title_card"},
            ),
            ("scene", {"detect_scenes", "classify_shots", "create_rough_cut"}),
            ("interchange", {"export_otio", "import_otio"}),
            (
                "render",
                {
                    "render_proxy",
                    "render_segment",
                    "compute_segments",
                    "render_with_preset",
                    "list_render_presets",
                },
            ),
        ],
    )
    def test_domain_returns_all_tools(self, session, domain, expected_tools):
        results = session.search_tools(domain=domain)
        found_names = {r.name for r in results}
        assert found_names == expected_tools, (
            f"Domain '{domain}' should have {expected_tools}, got {found_names}"
        )


# ---------------------------------------------------------------------------
# 7. Negative tests -- queries that should NOT match certain tools
# ---------------------------------------------------------------------------


class TestNegativeQueries:
    @pytest.mark.parametrize(
        "query",
        [
            "quantum physics simulation",
            "kubernetes orchestration",
            "differential equations",
            "cryptocurrency blockchain",
            "photosynthesis chlorophyll",
            "SQL database migration",
            "thermodynamics entropy",
            "terraform infrastructure",
        ],
    )
    def test_unrelated_query_returns_empty(self, session, query):
        results = session.search_tools(query)
        assert results == [], (
            f"Query '{query}' should return no results, got "
            f"{[r.name for r in results]}"
        )

    @pytest.mark.parametrize(
        "query, unexpected_domain",
        [
            ("audio loudness gain", "editing"),
            ("blend multiply screen", "audio"),
            ("title card intro", "color"),
        ],
    )
    def test_query_does_not_match_wrong_domain(self, session, query, unexpected_domain):
        """Top result should not come from the unexpected domain."""
        results = session.search_tools(query)
        if results:
            assert results[0].domain != unexpected_domain, (
                f"Query '{query}' top result should NOT be from '{unexpected_domain}', "
                f"got '{results[0].name}' in domain '{results[0].domain}'"
            )


# ---------------------------------------------------------------------------
# 8. Top-result ranking -- expected tool should be in top 3
# ---------------------------------------------------------------------------


class TestTopResultRanking:
    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("trim clip", "trim"),
            ("split at playhead", "split"),
            ("blend mode overlay", "apply_blend_mode"),
            ("scene detection", "detect_scenes"),
            ("text overlay", "add_text_overlay"),
            ("lower third", "add_lower_third"),
            ("export OTIO", "export_otio"),
            ("volume level", "volume"),
            ("color grade", "color_grade"),
            ("render proxy", "render_proxy"),
        ],
    )
    def test_expected_tool_in_top_3(self, session, query, expected_tool):
        results = session.search_tools(query)
        top_3 = [r.name for r in results[:3]]
        assert expected_tool in top_3, (
            f"Query '{query}' should have '{expected_tool}' in top 3, "
            f"got {top_3}"
        )
