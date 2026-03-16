"""Cross-module integration tests for the Agent Tool Architecture (Phase 4).

Verifies that registry, dependencies, session, and orchestrator modules
work coherently together end-to-end. All tests are pure logic — no GES,
no FFmpeg, no server dependencies.
"""

from __future__ import annotations

import json

import pytest

from ave.agent.orchestrator import Orchestrator
from ave.agent.registry import PrerequisiteError
from ave.agent.session import EditingSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xges(tmp_path):
    """Create a minimal .xges file and return its path."""
    p = tmp_path / "test.xges"
    p.write_text("<ges version='0.7'></ges>")
    return p


def _make_transcript_json() -> str:
    """Return a minimal transcript JSON string with filler words."""
    transcript = {
        "language": "en",
        "duration": 10.0,
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Hello um world like okay",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "um", "start": 0.6, "end": 0.8},
                    {"word": "world", "start": 1.0, "end": 1.5},
                    {"word": "like", "start": 2.0, "end": 2.3},
                    {"word": "okay", "start": 3.0, "end": 3.5},
                ],
            }
        ],
    }
    return json.dumps(transcript)


def _session_with_project(tmp_path):
    """Create an EditingSession with a project loaded and clip_exists set."""
    session = EditingSession()
    session.load_project(_make_xges(tmp_path))
    # Many editing tools require clip_exists; simulate that state.
    session.state.add("clip_exists")
    return session


# ---------------------------------------------------------------------------
# TestDiscoveryFlow
# ---------------------------------------------------------------------------


class TestDiscoveryFlow:
    """Tests for tool discovery through the session/registry pipeline."""

    def test_full_discovery_flow(self, tmp_path):
        """Create EditingSession -> search_tools('trim') -> get_tool_schema('trim')
        -> verify schema has correct params and dependencies."""
        session = EditingSession()

        # Search
        results = session.search_tools("trim")
        assert len(results) >= 1
        names = [r.name for r in results]
        assert "trim" in names

        # Schema
        schema = session.get_tool_schema("trim")
        assert schema.name == "trim"
        assert schema.domain == "editing"
        param_names = [p.name for p in schema.params]
        assert "clip_duration_ns" in param_names
        assert "in_ns" in param_names
        assert "out_ns" in param_names
        assert "timeline_loaded" in schema.requires
        assert "clip_trimmed" in schema.provides

    def test_all_domains_searchable(self):
        """For each domain, search(domain=X) returns non-empty results."""
        session = EditingSession()
        domains = session.registry.list_domains()
        assert len(domains) >= 6

        for entry in domains:
            domain_name = entry["domain"]
            results = session.search_tools(domain=domain_name)
            assert len(results) > 0, f"Domain '{domain_name}' returned no tools"
            for r in results:
                assert r.domain == domain_name

    def test_all_tools_have_valid_schemas(self):
        """Iterate all tools across all domains -> get_tool_schema for each
        -> verify non-empty params and description."""
        session = EditingSession()
        domains = session.registry.list_domains()

        for entry in domains:
            tools = session.search_tools(domain=entry["domain"])
            for summary in tools:
                schema = session.get_tool_schema(summary.name)
                assert schema.description, f"Tool '{summary.name}' has no description"
                assert isinstance(schema.params, list), f"Tool '{summary.name}' params not a list"

    def test_transcript_tools_in_registry(self):
        """search('filler') finds find_fillers tool -> get schema
        -> call with transcript data -> returns FillerMatch results."""
        session = EditingSession()
        # Provide transcript_loaded so we can call the tool
        session.state.add("transcript_loaded")

        results = session.search_tools("filler")
        names = [r.name for r in results]
        assert "find_fillers" in names

        schema = session.get_tool_schema("find_fillers")
        assert schema.domain == "transcription"

        transcript_json = _make_transcript_json()
        result = session.call_tool(
            "find_fillers",
            {
                "transcript_json": transcript_json,
            },
        )
        # Should find "um" and "like"
        assert isinstance(result, list)
        assert len(result) >= 2
        words_found = {m.word for m in result}
        assert "um" in words_found
        assert "like" in words_found


# ---------------------------------------------------------------------------
# TestSessionWorkflow
# ---------------------------------------------------------------------------


class TestSessionWorkflow:
    """Tests for session state progression, undo, and reset."""

    def test_session_state_progression(self, tmp_path):
        """Create session -> verify trim requires timeline_loaded
        -> load_project -> state has timeline_loaded -> call trim
        -> state has clip_trimmed."""
        session = EditingSession()

        # Verify trim requires timeline_loaded
        schema = session.get_tool_schema("trim")
        assert "timeline_loaded" in schema.requires

        # Load project
        session.load_project(_make_xges(tmp_path))
        assert session.state.has("timeline_loaded")

        # Also need clip_exists for trim
        session.state.add("clip_exists")

        # Call trim
        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        assert session.state.has("clip_trimmed")

    def test_prerequisite_chain_enforcement(self):
        """Create session (no project loaded) -> attempt call_tool('trim', ...)
        -> raises PrerequisiteError with 'timeline_loaded' in message."""
        session = EditingSession()

        with pytest.raises(PrerequisiteError, match="timeline_loaded"):
            session.call_tool(
                "trim",
                {
                    "clip_duration_ns": 10_000_000_000,
                    "in_ns": 0,
                    "out_ns": 5_000_000_000,
                },
            )

    def test_multi_tool_workflow(self, tmp_path):
        """Create session -> load project -> call trim -> call split
        -> call volume -> verify history has 3 entries with correct tool names."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 9_000_000_000,
            },
        )
        session.call_tool(
            "split",
            {
                "clip_start_ns": 0,
                "clip_duration_ns": 8_000_000_000,
                "split_position_ns": 4_000_000_000,
            },
        )
        session.call_tool("volume", {"level_db": -6.0})

        history = session.history
        assert len(history) == 3
        assert history[0].tool_name == "trim"
        assert history[1].tool_name == "split"
        assert history[2].tool_name == "volume"

    def test_undo_reverts_state(self, tmp_path):
        """Create session -> load project -> call trim (adds clip_trimmed)
        -> verify state has clip_trimmed -> undo -> verify clip_trimmed removed."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        assert session.state.has("clip_trimmed")

        undone = session.undo_last()
        assert undone is not None
        assert undone.tool_name == "trim"
        assert not session.state.has("clip_trimmed")

    def test_session_reset_clears_everything(self, tmp_path):
        """Create session -> load project -> call tools -> reset
        -> verify empty state, empty history, no project."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        session.call_tool("volume", {"level_db": -3.0})
        assert len(session.history) == 2
        assert session.state.has("timeline_loaded")

        session.reset()

        assert len(session.history) == 0
        assert not session.state.has("timeline_loaded")
        assert not session.state.has("clip_trimmed")
        data = session.to_dict()
        assert data["project_path"] is None
        assert data["history_length"] == 0
        assert data["state"] == []


# ---------------------------------------------------------------------------
# TestOrchestratorIntegration
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """Tests for orchestrator routing and error handling."""

    def test_orchestrator_discovery_to_execution(self, tmp_path):
        """Create Orchestrator -> handle_tool_call('search_tools', {'query': 'volume'})
        -> handle_tool_call('get_tool_schema', {'tool_name': 'volume'})
        -> handle_tool_call('call_tool', {'tool_name': 'volume', 'params': {'level_db': -6.0}})
        -> verify formatted result contains VolumeParams."""
        session = _session_with_project(tmp_path)
        orch = Orchestrator(session)

        # Step 1: search
        search_result = orch.handle_tool_call("search_tools", {"query": "volume"})
        assert "volume" in search_result
        assert "Found" in search_result

        # Step 2: get schema
        schema_result = orch.handle_tool_call("get_tool_schema", {"tool_name": "volume"})
        assert "level_db" in schema_result
        assert "audio" in schema_result

        # Step 3: call tool
        call_result = orch.handle_tool_call(
            "call_tool",
            {
                "tool_name": "volume",
                "params": {"level_db": -6.0},
            },
        )
        assert "VolumeParams" in call_result
        assert "executed successfully" in call_result

    def test_orchestrator_error_handling(self):
        """Orchestrator handles errors gracefully: call nonexistent tool
        -> returns 'Error:' string, not exception."""
        session = EditingSession()
        orch = Orchestrator(session)

        result = orch.handle_tool_call(
            "call_tool",
            {
                "tool_name": "nonexistent_tool",
                "params": {},
            },
        )
        assert "Error:" in result
        # Should not raise — error is returned as a string

    def test_orchestrator_system_prompt_complete(self):
        """System prompt contains all 6 domains with correct tool counts."""
        session = EditingSession()
        orch = Orchestrator(session)

        prompt = orch.get_system_prompt()
        expected_domains = ["editing", "audio", "color", "transcription", "render", "project"]
        for domain in expected_domains:
            assert domain in prompt, f"Domain '{domain}' not found in system prompt"

        # Verify it mentions tool counts (e.g. "editing (5 tools)")
        domains = session.registry.list_domains()
        for entry in domains:
            count_str = f"{entry['domain']} ({entry['count']} tools)"
            assert count_str in prompt, f"Expected '{count_str}' in system prompt"

    def test_orchestrator_unknown_meta_tool(self):
        """Calling an unknown meta-tool returns an error string."""
        session = EditingSession()
        orch = Orchestrator(session)

        result = orch.handle_tool_call("unknown_meta_tool", {})
        assert "Error:" in result
        assert "unknown_meta_tool" in result


# ---------------------------------------------------------------------------
# TestCrossPhase
# ---------------------------------------------------------------------------


class TestCrossPhase:
    """Cross-phase integration tests."""

    def test_render_tools_compute_segments(self, tmp_path):
        """Call compute_segments tool through orchestrator
        -> verify returns SegmentBoundary-like results."""
        session = _session_with_project(tmp_path)
        orch = Orchestrator(session)

        result = orch.handle_tool_call(
            "call_tool",
            {
                "tool_name": "compute_segments",
                "params": {
                    "duration_ns": 15_000_000_000,
                    "segment_duration_ns": 5_000_000_000,
                },
            },
        )
        assert "executed successfully" in result
        assert "SegmentBoundary" in result

        # Also verify via direct session call
        boundaries = session.call_tool(
            "compute_segments",
            {
                "duration_ns": 15_000_000_000,
                "segment_duration_ns": 5_000_000_000,
            },
        )
        assert isinstance(boundaries, list)
        assert len(boundaries) == 3
        assert boundaries[0].start_ns == 0
        assert boundaries[0].end_ns == 5_000_000_000
        assert boundaries[2].end_ns == 15_000_000_000

    def test_session_serialization_roundtrip(self, tmp_path):
        """Create session -> load project -> call tools -> to_dict()
        -> verify all fields present and correct types."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        session.call_tool("volume", {"level_db": -3.0})

        data = session.to_dict()

        # Verify all fields present
        assert "tool_count" in data
        assert "state" in data
        assert "history_length" in data
        assert "project_path" in data

        # Verify types
        assert isinstance(data["tool_count"], int)
        assert isinstance(data["state"], list)
        assert isinstance(data["history_length"], int)
        assert isinstance(data["project_path"], str)

        # Verify values
        assert data["tool_count"] >= 6  # at least 6 domains worth of tools
        assert data["history_length"] == 2
        assert "timeline_loaded" in data["state"]
        assert "clip_trimmed" in data["state"]
        assert "volume_set" in data["state"]
        assert "test.xges" in data["project_path"]


# ---------------------------------------------------------------------------
# TestMultiToolWorkflow
# ---------------------------------------------------------------------------


class TestMultiToolWorkflow:
    """Sequential tool chains with state tracking."""

    def test_ingest_then_trim_workflow(self, tmp_path):
        """probe_media -> ingest_media -> trim: state accumulates correctly.

        Since probe_media and ingest_media call real backends that may not
        be available in test, we simulate their state provisions and test
        the chain enforcement + trim execution.
        """
        session = EditingSession()

        # probe_media has no prerequisites — verify schema
        schema = session.get_tool_schema("probe_media")
        assert schema.requires == []
        assert "media_probed" in schema.provides

        # ingest_media requires media_probed
        schema = session.get_tool_schema("ingest_media")
        assert "media_probed" in schema.requires
        assert "media_ingested" in schema.provides

        # Cannot ingest without probing first
        with pytest.raises(PrerequisiteError, match="media_probed"):
            session.call_tool(
                "ingest_media",
                {
                    "source": "/tmp/test.mp4",
                    "project_dir": "/tmp/proj",
                    "asset_id": "a1",
                    "registry_path": "/tmp/reg.json",
                },
            )

        # Simulate probe + ingest provisions, load project, then trim
        session.state.add("media_probed", "media_ingested")
        session.load_project(_make_xges(tmp_path))
        session.state.add("clip_exists")

        result = session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 0,
                "out_ns": 5_000_000_000,
            },
        )
        assert result is not None
        assert session.state.has("media_probed")
        assert session.state.has("media_ingested")
        assert session.state.has("clip_trimmed")

    def test_scene_pipeline_workflow(self, tmp_path):
        """detect_scenes -> classify_shots: prerequisite chain enforced."""
        session = EditingSession()
        session.load_project(_make_xges(tmp_path))

        # classify_shots requires scenes_detected — should fail
        with pytest.raises(PrerequisiteError, match="scenes_detected"):
            session.call_tool(
                "classify_shots",
                {
                    "video_path": "/tmp/test.mp4",
                    "scenes_json": "[]",
                    "output_dir": "/tmp/out",
                },
            )

        # Simulate detect_scenes provision (the real backend needs ffmpeg)
        session.state.add("scenes_detected")

        # Now classify should work
        result = session.call_tool(
            "classify_shots",
            {
                "video_path": "/tmp/test.mp4",
                "scenes_json": "[]",
                "output_dir": "/tmp/out",
            },
        )
        assert result is not None
        assert session.state.has("shots_classified")

    def test_editing_audio_color_chain(self, tmp_path):
        """trim -> volume -> color_grade: multi-domain chain accumulates state."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 9_000_000_000,
            },
        )
        assert session.state.has("clip_trimmed")

        session.call_tool("volume", {"level_db": -3.0})
        assert session.state.has("volume_set")

        session.call_tool(
            "color_grade",
            {
                "lift_r": 0.0,
                "lift_g": 0.0,
                "lift_b": 0.0,
                "gamma_r": 1.0,
                "gamma_g": 1.0,
                "gamma_b": 1.0,
                "gain_r": 1.0,
                "gain_g": 1.0,
                "gain_b": 1.0,
            },
        )
        assert session.state.has("color_graded")

        # All three provisions accumulated
        assert session.state.has("clip_trimmed")
        assert session.state.has("volume_set")
        assert session.state.has("color_graded")

        history = session.history
        assert len(history) == 3
        assert [h.tool_name for h in history] == ["trim", "volume", "color_grade"]


# ---------------------------------------------------------------------------
# TestUndoStateRollback
# ---------------------------------------------------------------------------


class TestUndoStateRollback:
    """Undo with provision-aware rollback."""

    def test_undo_removes_unique_provisions(self, tmp_path):
        """Undo removes provisions not covered by remaining history."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        assert session.state.has("clip_trimmed")

        session.call_tool("volume", {"level_db": -6.0})
        assert session.state.has("volume_set")

        # Undo volume — volume_set should be removed, clip_trimmed preserved
        undone = session.undo_last()
        assert undone.tool_name == "volume"
        assert not session.state.has("volume_set")
        assert session.state.has("clip_trimmed")

    def test_undo_preserves_shared_provisions(self, tmp_path):
        """Undo preserves provisions that other history entries also provide.

        Both trim and split provide their own provisions, but both require
        timeline_loaded which comes from load_project. We test with two
        tools that share a provision by calling compute_segments twice
        (both provide segments_computed).
        """
        session = _session_with_project(tmp_path)

        # Call compute_segments twice — both provide "segments_computed"
        session.call_tool(
            "compute_segments",
            {
                "duration_ns": 10_000_000_000,
                "segment_duration_ns": 5_000_000_000,
            },
        )
        assert session.state.has("segments_computed")

        session.call_tool(
            "compute_segments",
            {
                "duration_ns": 20_000_000_000,
                "segment_duration_ns": 5_000_000_000,
            },
        )
        assert session.state.has("segments_computed")
        assert len(session.history) == 2

        # Undo second call — segments_computed should still be set
        # because the first call also provides it
        undone = session.undo_last()
        assert undone is not None
        assert session.state.has("segments_computed")

        # Undo first call — now segments_computed should be removed
        undone = session.undo_last()
        assert undone is not None
        assert not session.state.has("segments_computed")

    def test_multiple_undos_restore_initial_state(self, tmp_path):
        """Undoing all calls restores empty state (except timeline_loaded from load_project)."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        session.call_tool("volume", {"level_db": -3.0})
        session.call_tool(
            "fade",
            {
                "clip_duration_ns": 10_000_000_000,
                "fade_in_ns": 500_000_000,
                "fade_out_ns": 500_000_000,
            },
        )

        assert len(session.history) == 3
        assert session.state.has("clip_trimmed")
        assert session.state.has("volume_set")
        assert session.state.has("fade_applied")

        # Undo all three
        session.undo_last()
        session.undo_last()
        session.undo_last()

        assert len(session.history) == 0
        assert not session.state.has("clip_trimmed")
        assert not session.state.has("volume_set")
        assert not session.state.has("fade_applied")
        # timeline_loaded and clip_exists were set before tool calls,
        # so they remain (not provided by any tool call in history)
        assert session.state.has("timeline_loaded")
        assert session.state.has("clip_exists")

    def test_undo_on_empty_history_returns_none(self):
        """Undo on empty session returns None without error."""
        session = EditingSession()
        assert session.undo_last() is None


# ---------------------------------------------------------------------------
# TestSearchToCallFlow
# ---------------------------------------------------------------------------


class TestSearchToCallFlow:
    """Simulates LLM agent pattern: search -> schema -> call."""

    def test_search_schema_call_pattern(self, tmp_path):
        """Simulate: user asks question -> agent searches -> gets schema -> calls tool."""
        session = EditingSession()

        # Step 1: Search
        results = session.search_tools("trim")
        assert len(results) > 0
        tool_name = results[0].name

        # Step 2: Get schema
        schema = session.get_tool_schema(tool_name)
        assert schema.requires  # has prerequisites
        assert len(schema.params) > 0

        # Step 3: Check prerequisites
        missing = [r for r in schema.requires if not session.state.has(r)]
        assert len(missing) > 0  # can't call yet

        # Step 4: Satisfy prerequisites
        for req in schema.requires:
            session.state.add(req)

        # Step 5: Call with params derived from schema
        result = session.call_tool(
            tool_name,
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 0,
                "out_ns": 5_000_000_000,
            },
        )
        assert result is not None

    def test_domain_browse_then_pick(self):
        """Simulate: agent browses domains -> picks domain -> searches within it."""
        session = EditingSession()
        domains = session.registry.list_domains()
        [d["domain"] for d in domains]
        assert len(domains) >= 10

        # Browse color domain
        color_tools = session.search_tools(domain="color")
        assert len(color_tools) == 3
        color_names = {t.name for t in color_tools}
        assert "color_grade" in color_names
        assert "cdl" in color_names
        assert "lut_parse" in color_names

        # Pick color_grade, get schema
        schema = session.get_tool_schema("color_grade")
        assert "timeline_loaded" in schema.requires
        assert "clip_exists" in schema.requires
        assert "color_graded" in schema.provides

    def test_search_by_natural_query(self):
        """Search with natural language terms returns relevant tools."""
        session = EditingSession()

        # Search for "slow motion" should find speed tool
        results = session.search_tools("slow motion")
        names = [r.name for r in results]
        assert "speed" in names

        # Search for "fade" should find fade tool
        results = session.search_tools("fade")
        names = [r.name for r in results]
        assert "fade" in names

        # Search for "subtitle" should find text overlay
        results = session.search_tools("subtitle")
        names = [r.name for r in results]
        assert "add_text_overlay" in names

    def test_search_returns_correct_domain(self):
        """Each search result has the correct domain field."""
        session = EditingSession()

        for domain_entry in session.registry.list_domains():
            domain = domain_entry["domain"]
            tools = session.search_tools(domain=domain)
            for tool in tools:
                assert tool.domain == domain, (
                    f"Tool '{tool.name}' reports domain '{tool.domain}' "
                    f"but was found under '{domain}'"
                )


# ---------------------------------------------------------------------------
# TestSessionSerialization
# ---------------------------------------------------------------------------


class TestSessionSerialization:
    """Session state roundtrip and serialization."""

    def test_session_to_dict_initial(self):
        """Session serialization captures tool count, state, history on fresh session."""
        session = EditingSession()
        d = session.to_dict()
        assert d["tool_count"] >= 34
        assert d["state"] == []
        assert d["history_length"] == 0
        assert d["project_path"] is None

    def test_session_tracks_history(self, tmp_path):
        """Each call_tool adds to history with timestamp and provisions."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )
        session.call_tool("volume", {"level_db": -6.0})

        history = session.history
        assert len(history) == 2

        # First entry
        assert history[0].tool_name == "trim"
        assert history[0].provisions == ["clip_trimmed"]
        assert isinstance(history[0].timestamp, float)
        assert history[0].timestamp > 0

        # Second entry
        assert history[1].tool_name == "volume"
        assert history[1].provisions == ["volume_set"]
        assert history[1].timestamp >= history[0].timestamp

    def test_session_to_dict_after_tools(self, tmp_path):
        """Serialization reflects accumulated state after tool calls."""
        session = _session_with_project(tmp_path)

        session.call_tool(
            "trim",
            {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 1_000_000_000,
                "out_ns": 5_000_000_000,
            },
        )

        d = session.to_dict()
        assert d["history_length"] == 1
        assert "clip_trimmed" in d["state"]
        assert "timeline_loaded" in d["state"]
        assert d["project_path"] is not None

    def test_history_records_params(self, tmp_path):
        """History entries store the params that were passed."""
        session = _session_with_project(tmp_path)

        params = {"level_db": -9.5}
        session.call_tool("volume", params)

        entry = session.history[0]
        assert entry.params == params
        assert entry.result is not None


# ---------------------------------------------------------------------------
# TestPrerequisiteEnforcement
# ---------------------------------------------------------------------------


class TestPrerequisiteEnforcement:
    """Various prerequisite failure scenarios."""

    def test_cannot_trim_without_timeline(self):
        """Trim requires timeline_loaded — fails on fresh session."""
        session = EditingSession()
        with pytest.raises(PrerequisiteError, match="timeline_loaded"):
            session.call_tool(
                "trim",
                {
                    "clip_duration_ns": 10_000_000_000,
                    "in_ns": 0,
                    "out_ns": 5_000_000_000,
                },
            )

    def test_cannot_search_transcript_without_loaded(self):
        """search_transcript requires transcript_loaded."""
        session = EditingSession()
        with pytest.raises(PrerequisiteError, match="transcript_loaded"):
            session.call_tool(
                "search_transcript",
                {
                    "transcript_json": "{}",
                    "query": "hello",
                },
            )

    def test_cannot_classify_without_scenes_detected(self):
        """classify_shots requires scenes_detected."""
        session = EditingSession()
        session.state.add("timeline_loaded")
        with pytest.raises(PrerequisiteError, match="scenes_detected"):
            session.call_tool(
                "classify_shots",
                {
                    "video_path": "/tmp/v.mp4",
                    "scenes_json": "[]",
                    "output_dir": "/tmp/o",
                },
            )

    def test_cannot_export_otio_without_timeline(self):
        """export_otio requires timeline_loaded."""
        session = EditingSession()
        with pytest.raises(PrerequisiteError, match="timeline_loaded"):
            session.call_tool(
                "export_otio",
                {
                    "timeline_data_json": "{}",
                    "output_path": "/tmp/out.otio",
                },
            )

    def test_import_otio_has_no_prerequisites(self):
        """import_otio has no prerequisites — should not raise PrerequisiteError."""
        schema = EditingSession().get_tool_schema("import_otio")
        assert schema.requires == []
        assert "timeline_loaded" in schema.provides

    def test_list_render_presets_has_no_prerequisites(self):
        """list_render_presets has no prerequisites or provisions."""
        session = EditingSession()
        schema = session.get_tool_schema("list_render_presets")
        assert schema.requires == []
        assert schema.provides == []

        # Should be callable on a fresh session
        result = session.call_tool("list_render_presets", {})
        assert result is not None

    def test_cannot_color_grade_without_clip(self, tmp_path):
        """color_grade requires both timeline_loaded and clip_exists."""
        session = EditingSession()
        session.load_project(_make_xges(tmp_path))
        # Has timeline_loaded but not clip_exists
        with pytest.raises(PrerequisiteError, match="clip_exists"):
            session.call_tool(
                "color_grade",
                {
                    "lift_r": 0.0,
                    "lift_g": 0.0,
                    "lift_b": 0.0,
                    "gamma_r": 1.0,
                    "gamma_g": 1.0,
                    "gamma_b": 1.0,
                    "gain_r": 1.0,
                    "gain_g": 1.0,
                    "gain_b": 1.0,
                },
            )

    def test_cannot_normalize_without_clip(self):
        """normalize requires timeline_loaded and clip_exists."""
        session = EditingSession()
        with pytest.raises(PrerequisiteError):
            session.call_tool(
                "normalize",
                {
                    "current_peak_db": -3.0,
                    "target_peak_db": -1.0,
                },
            )
