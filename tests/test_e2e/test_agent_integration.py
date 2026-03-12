"""Cross-module integration tests for the Agent Tool Architecture (Phase 4).

Verifies that registry, dependencies, session, and orchestrator modules
work coherently together end-to-end. All tests are pure logic — no GES,
no FFmpeg, no server dependencies.
"""

from __future__ import annotations

import json

import pytest

from ave.agent.dependencies import SessionState
from ave.agent.orchestrator import Orchestrator
from ave.agent.registry import PrerequisiteError, ToolRegistry
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
        result = session.call_tool("find_fillers", {
            "transcript_json": transcript_json,
        })
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
        result = session.call_tool("trim", {
            "clip_duration_ns": 10_000_000_000,
            "in_ns": 1_000_000_000,
            "out_ns": 5_000_000_000,
        })
        assert session.state.has("clip_trimmed")

    def test_prerequisite_chain_enforcement(self):
        """Create session (no project loaded) -> attempt call_tool('trim', ...)
        -> raises PrerequisiteError with 'timeline_loaded' in message."""
        session = EditingSession()

        with pytest.raises(PrerequisiteError, match="timeline_loaded"):
            session.call_tool("trim", {
                "clip_duration_ns": 10_000_000_000,
                "in_ns": 0,
                "out_ns": 5_000_000_000,
            })

    def test_multi_tool_workflow(self, tmp_path):
        """Create session -> load project -> call trim -> call split
        -> call volume -> verify history has 3 entries with correct tool names."""
        session = _session_with_project(tmp_path)

        session.call_tool("trim", {
            "clip_duration_ns": 10_000_000_000,
            "in_ns": 1_000_000_000,
            "out_ns": 9_000_000_000,
        })
        session.call_tool("split", {
            "clip_start_ns": 0,
            "clip_duration_ns": 8_000_000_000,
            "split_position_ns": 4_000_000_000,
        })
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

        session.call_tool("trim", {
            "clip_duration_ns": 10_000_000_000,
            "in_ns": 1_000_000_000,
            "out_ns": 5_000_000_000,
        })
        assert session.state.has("clip_trimmed")

        undone = session.undo_last()
        assert undone is not None
        assert undone.tool_name == "trim"
        assert not session.state.has("clip_trimmed")

    def test_session_reset_clears_everything(self, tmp_path):
        """Create session -> load project -> call tools -> reset
        -> verify empty state, empty history, no project."""
        session = _session_with_project(tmp_path)

        session.call_tool("trim", {
            "clip_duration_ns": 10_000_000_000,
            "in_ns": 1_000_000_000,
            "out_ns": 5_000_000_000,
        })
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
        call_result = orch.handle_tool_call("call_tool", {
            "tool_name": "volume",
            "params": {"level_db": -6.0},
        })
        assert "VolumeParams" in call_result
        assert "executed successfully" in call_result

    def test_orchestrator_error_handling(self):
        """Orchestrator handles errors gracefully: call nonexistent tool
        -> returns 'Error:' string, not exception."""
        session = EditingSession()
        orch = Orchestrator(session)

        result = orch.handle_tool_call("call_tool", {
            "tool_name": "nonexistent_tool",
            "params": {},
        })
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
            assert count_str in prompt, (
                f"Expected '{count_str}' in system prompt"
            )

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

        result = orch.handle_tool_call("call_tool", {
            "tool_name": "compute_segments",
            "params": {
                "duration_ns": 15_000_000_000,
                "segment_duration_ns": 5_000_000_000,
            },
        })
        assert "executed successfully" in result
        assert "SegmentBoundary" in result

        # Also verify via direct session call
        boundaries = session.call_tool("compute_segments", {
            "duration_ns": 15_000_000_000,
            "segment_duration_ns": 5_000_000_000,
        })
        assert isinstance(boundaries, list)
        assert len(boundaries) == 3
        assert boundaries[0].start_ns == 0
        assert boundaries[0].end_ns == 5_000_000_000
        assert boundaries[2].end_ns == 15_000_000_000

    def test_session_serialization_roundtrip(self, tmp_path):
        """Create session -> load project -> call tools -> to_dict()
        -> verify all fields present and correct types."""
        session = _session_with_project(tmp_path)

        session.call_tool("trim", {
            "clip_duration_ns": 10_000_000_000,
            "in_ns": 1_000_000_000,
            "out_ns": 5_000_000_000,
        })
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
